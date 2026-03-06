import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from vector_store import VectorStore
from query_normalizer import normalize_query as normalize_user_query, expand_query_for_vasanth

load_dotenv()

logger = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self, vector_store: VectorStore, model_name: str = None):
        self.vector_store = vector_store

        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.use_groq = True

        self.vertex_ai_project = os.getenv("VERTEX_AI_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.vertex_ai_location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.use_vertex = False

        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not configured. Set it as an environment variable.")
        else:
            logger.info(f"RAG Pipeline initialized with Groq model: {self.groq_model}")

        if self.vertex_ai_project:
            logger.info(f"Vertex AI backup configured. Model: {self.gemini_model}")

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.vector_store.search_hybrid(
            query,
            top_k=top_k,
            min_similarity=0.12,
            vector_weight=0.55,
            keyword_weight=0.45,
        )

        if not results:
            return ""

        context_parts = []
        seen_texts = set()
        for result in results:
            text = result["text"]
            score = result.get("score", 0.0)
            if score > 0 and text not in seen_texts:
                seen_texts.add(text)
                context_parts.append(text)

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    def _expand_query(self, query: str) -> List[str]:
        queries = [query]

        vasanth_expanded = expand_query_for_vasanth(query)
        if vasanth_expanded != query and vasanth_expanded not in queries:
            queries.append(vasanth_expanded)

        query_lower = query.lower()

        if "vasanth" in query_lower or "vasanthkumar" in query_lower:
            if "vasanthkumar" not in query_lower:
                queries.append(query.replace("Vasanth", "Vasanthkumar").replace("vasanth", "Vasanthkumar"))
            if "vasanth" not in query_lower:
                queries.append(query.replace("Vasanthkumar", "Vasanth").replace("vasanthkumar", "Vasanth"))

        if query.endswith("?"):
            queries.append(query[:-1].strip())

        seen = set()
        unique = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:5]

    def generate_response_groq(self, query: str, context: str) -> str:
        try:
            from groq import Groq

            if not self.groq_api_key:
                raise Exception("Groq API key not configured. Set GROQ_API_KEY environment variable.")

            client = Groq(api_key=self.groq_api_key)

            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Vasanth's portfolio assistant. The user is ALWAYS asking about Vasanth (Vasanthkumar)—even when they do not say his name. Interpret questions like 'What is the total experience?', 'What are his skills?', 'Tell me about education' as about Vasanth and answer using the context. Ignore typos and spelling errors (e.g. experiance, relevent) and understand what the user means. Answer based only on the provided context. Be clear and friendly. Refer to Vasanthkumar as 'Vasanth'. Do not mention documents or sources; answer naturally. Example: if asked 'What is the total experiance?' answer with Vasanth's total experience from the context (e.g. 'Total experience of Vasanth is X years with relevant experience in...')."
                    },
                    {
                        "role": "user",
                        "content": self._create_prompt(query, context)
                    }
                ],
                temperature=0.7,
                max_tokens=1024
            )

            return response.choices[0].message.content.strip()

        except ImportError:
            raise Exception("Groq library not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Groq API error: {e}", exc_info=True)
            raise Exception(f"Groq error: {str(e)}")

    def generate_response_vertex_ai(self, query: str, context: str) -> str:
        try:
            from vertexai.preview.generative_models import GenerativeModel
            import vertexai

            if not self.vertex_ai_project:
                raise Exception("Vertex AI project not configured")

            vertexai.init(project=self.vertex_ai_project, location=self.vertex_ai_location)

            model = GenerativeModel(self.gemini_model)

            response = model.generate_content(
                self._create_prompt(query, context),
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                }
            )

            return response.text.strip()

        except ImportError:
            raise Exception("Vertex AI library not installed. Run: pip install google-cloud-aiplatform")
        except Exception as e:
            logger.error(f"Vertex AI error: {e}", exc_info=True)
            raise Exception(f"Vertex AI error: {str(e)}")

    def _create_prompt(self, query: str, context: str) -> str:
        return f"""You are Vasanth's personal assistant. The user is asking about Vasanth. Even if the question has typos (e.g. experiance, relevent) or does not mention Vasanth by name, interpret it as about Vasanth and answer from the context below. Example: "What is the total experiance?" means Vasanth's total experience—answer with his experience in years and relevant areas (e.g. AI, Python Backend).

Guidelines:
- Treat every question as about Vasanth's experience, skills, projects, education, etc.
- Ignore spelling errors; understand intent (experiance→experience, relevent→relevant).
- Answer naturally and concisely (2-3 sentences). Use only the context below.
- Refer to Vasanthkumar as "Vasanth". Do not mention documents or sources.
- If information isn't in the context, say you don't have that detail. Be friendly and professional.

Information about Vasanth:
{context}

Question: {query}

Answer:"""

    def generate_response(self, query: str, context: str) -> str:
        if self.groq_api_key:
            try:
                self.use_groq = True
                self.use_vertex = False
                response = self.generate_response_groq(query, context)
                logger.info("Response generated using Groq")
                return response
            except Exception as e:
                logger.warning(f"Groq failed: {str(e)}")
                logger.info("Attempting fallback to Vertex AI Gemini...")

        if self.vertex_ai_project:
            try:
                self.use_groq = False
                self.use_vertex = True
                response = self.generate_response_vertex_ai(query, context)
                logger.info("Response generated using Vertex AI Gemini (backup)")
                return response
            except Exception as e:
                error_msg = f"Both Groq and Vertex AI failed. Vertex AI error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return f"I found relevant information, but couldn't generate a response. {error_msg}"

        error_msg = "No LLM provider configured. Set GROQ_API_KEY or VERTEX_AI_PROJECT."
        logger.error(error_msg)
        return f"I found relevant information, but couldn't generate a response. {error_msg}"

    def _is_greeting(self, query: str) -> bool:
        greeting_words = [
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
            "good evening", "hi there", "hello there", "hey there", "what's up",
            "howdy", "sup", "hola", "namaste"
        ]
        query_lower = query.lower().strip()
        query_clean = ''.join(c for c in query_lower if c.isalnum() or c.isspace())
        query_words = query_clean.split()

        if len(query_words) <= 3:
            return any(greeting in query_lower for greeting in greeting_words)
        else:
            first_word = query_words[0] if query_words else ""
            return first_word in greeting_words or any(query_lower.startswith(g) for g in greeting_words)

    def chat(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        normalized_query, _ = normalize_user_query(query)
        query_for_flow = normalized_query or query

        if self._is_greeting(query):
            greeting_responses = [
                "Hello! I'm Vasanth's portfolio assistant. How can I help you today?",
                "Hi there! I can help you learn about Vasanth's experience, skills, projects, and more. What would you like to know?",
                "Hello! Welcome! I'm here to answer questions about Vasanth. What can I help you with?",
                "Hi! I'm Vasanth's assistant. Feel free to ask me anything about his background, experience, or skills!"
            ]
            import random
            return {
                "response": random.choice(greeting_responses),
                "context": "",
                "sources": [],
                "model": self.gemini_model if self.use_vertex else self.groq_model,
                "provider": "vertex-ai" if self.use_vertex else "groq"
            }

        context = self.retrieve_context(query_for_flow, top_k=top_k)

        if not context or not context.strip():
            return {
                "response": "I don't have specific information about that in my knowledge base. Could you try rephrasing your question or ask about Vasanth's experience, skills, projects, or education?",
                "context": "",
                "sources": [],
                "model": self.gemini_model if self.use_vertex else self.groq_model,
                "provider": "vertex-ai" if self.use_vertex else "groq"
            }

        response = self.generate_response(query_for_flow, context)

        retrieval_results = self.vector_store.search_hybrid(query_for_flow, top_k=top_k)

        return {
            "response": response,
            "context": context[:500] + "..." if len(context) > 500 else context,
            "sources": [
                {
                    "text": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
                    "metadata": r["metadata"],
                    "score": round(r["score"], 3)
                }
                for r in retrieval_results[:3]
            ],
            "model": self.gemini_model if self.use_vertex else self.groq_model,
            "provider": "vertex-ai" if self.use_vertex else "groq"
        }

    def test_groq_connection(self) -> bool:
        try:
            if not self.groq_api_key:
                logger.warning("Groq API key not configured")
                return False

            from groq import Groq
            client = Groq(api_key=self.groq_api_key)

            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )

            logger.info(f"Groq API connection successful. Model: {self.groq_model}")
            return True

        except Exception as e:
            logger.error(f"Groq API connection failed: {e}")
            return False

    def test_vertex_ai_connection(self) -> bool:
        try:
            if not self.vertex_ai_project:
                logger.warning("Vertex AI project not configured")
                return False

            from vertexai.preview.generative_models import GenerativeModel
            import vertexai

            vertexai.init(project=self.vertex_ai_project, location=self.vertex_ai_location)
            model = GenerativeModel(self.gemini_model)

            response = model.generate_content("Hi", generation_config={"max_output_tokens": 5})

            logger.info(f"Vertex AI connection successful. Model: {self.gemini_model}")
            return True

        except Exception as e:
            logger.error(f"Vertex AI connection failed: {e}")
            return False

    def test_llm_connection(self) -> bool:
        groq_ok = False
        vertex_ok = False

        if self.groq_api_key:
            groq_ok = self.test_groq_connection()

        if self.vertex_ai_project:
            vertex_ok = self.test_vertex_ai_connection()

        if groq_ok:
            logger.info("Primary LLM: Groq")
        if vertex_ok:
            logger.info("Backup LLM: Vertex AI Gemini")

        return groq_ok or vertex_ok
