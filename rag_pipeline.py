import ollama
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from vector_store import VectorStore
from embedding_service import EmbeddingService

load_dotenv()


class RAGPipeline:
    
    def __init__(self, vector_store: VectorStore, model_name: str = None):

        self.vector_store = vector_store

        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Groq configuration (fallback)
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.use_groq = False  # Flag to track if we're using Groq
        
        try:
            ollama.Client(host=self.ollama_base_url)
            self.ollama_available = True
            print(f"RAG Pipeline initialized with Ollama model: {self.model_name}")
        except Exception as e:
            print(f"Ollama initialization warning: {str(e)}")
            self.ollama_available = False
            if self.groq_api_key:
                print(f"Groq fallback will be used. Model: {self.groq_model}")
            else:
                print("Warning: No Groq API key found. Groq fallback will not be available.")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:

        results = self.vector_store.search(query, top_k=top_k)
        
        if not results:
            return "There is no relevant information,try asking something else."

        context_parts = []
        for i, result in enumerate(results, 1):
            text = result["text"]
            metadata = result.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            
            context_parts.append(f"[Document {i} from {filename}]\n{text}")
        
        context = "\n\n".join(context_parts)
        return context
    
    def generate_response_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama (primary method)"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=self._create_prompt(query, context),
                stream=False
            )
            return response['response'].strip()
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    def generate_response_groq(self, query: str, context: str) -> str:
        """Generate response using Groq API (fallback method)"""
        try:
            from groq import Groq
            
            if not self.groq_api_key:
                raise Exception("Groq API key not configured")
            
            client = Groq(api_key=self.groq_api_key)
            
            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context from documents. Do not make any answers by yourself that is irrelevant to the question or the document and there has to be clarity in answering the question. Make sure to check the document which should contain all the informations related to technologies, projects, experiences, education, skills, etc of Vasanthkumar. You can mention Vasanthkumar as Vasanth while answering the question. Your answer should be in a friendly and engaging manner."
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
            raise Exception(f"Groq error: {str(e)}")
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for LLM"""
        return f"""You are a helpful assistant that answers questions based on the provided context from documents.
         Do not make any answers by your self that is irrelevant to the question or the document and there has to clarity in the answering the question.
         Make sure to check the document which should contain all the informations related to technologies, projects, experiences, education, skills, etc of Vasanthkumar.

         You can mention Vasanthkumar as Vasanth while answering the question.
         Your answer should be in a friendly and engaging manner.

Context from documents:
{context}

User Question: {query}

Instructions:
- Answer the question based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Use the information from the context to provide a helpful answer

Answer:"""
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response with automatic fallback: Ollama -> Groq"""
        
        # Try Ollama first (if available)
        if self.ollama_available:
            try:
                self.use_groq = False
                response = self.generate_response_ollama(query, context)
                return response
            except Exception as e:
                print(f"Ollama failed: {str(e)}")
                print("Falling back to Groq...")
                self.ollama_available = False  # Mark Ollama as unavailable
        
        # Fallback to Groq if Ollama failed or unavailable
        if self.groq_api_key:
            try:
                self.use_groq = True
                response = self.generate_response_groq(query, context)
                print("Using Groq for response generation")
                return response
            except Exception as e:
                error_msg = f"Both Ollama and Groq failed. Groq error: {str(e)}"
                print(error_msg)
                return f"I found relevant information, but couldn't generate a response. Error: {str(e)}. Please check if Ollama is running or Groq API key is configured."
        else:
            error_msg = "Ollama is not available and Groq API key is not configured."
            print(error_msg)
            return f"I found relevant information, but couldn't generate a response. {error_msg} Please check if Ollama is running or configure Groq API key."

    def chat(self, query: str, top_k: int = 3) -> Dict[str, Any]:

        context = self.retrieve_context(query, top_k=top_k)

        response = self.generate_response(query, context)

        retrieval_results = self.vector_store.search(query, top_k=top_k)
        
        return {
            "response": response,
            "context": context,
            "sources": [
                {
                    "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                    "metadata": r["metadata"],
                    "score": r["score"]
                }
                for r in retrieval_results
            ],
            "model": self.groq_model if self.use_groq else self.model_name,
            "provider": "groq" if self.use_groq else "ollama"
        }
    
    def test_ollama_connection(self) -> bool:

        try:
            models = ollama.list()
            model_names = [model['name'] for model in models.get('models', [])]
            
            if self.model_name in model_names:
                print(f"✓ Ollama connection successful. Model '{self.model_name}' is available.")
                self.ollama_available = True
                return True
            else:
                print(f"⚠ Ollama is running, but model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Please run: ollama pull {self.model_name}")
                self.ollama_available = False
                return False
        
        except Exception as e:
            print(f"✗ Ollama connection failed: {str(e)}")
            print(f"Make sure Ollama is running on {self.ollama_base_url}")
            self.ollama_available = False
            return False
