import ollama
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from vector_store import VectorStore
from embedding_service import EmbeddingService
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

load_dotenv()


class RAGPipeline:
    
    def __init__(self, vector_store: VectorStore, model_name: str = None):

        self.vector_store = vector_store

        # Get model name from env, but ensure it's not a path
        env_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        # Fix: If env var contains a path (has backslash or starts with drive), use default
        if "\\" in env_model or (len(env_model) > 1 and env_model[1] == ":"):
            print(f"Warning: OLLAMA_MODEL env var contains path '{env_model}', using default model instead")
            env_model = "llama3.2:3b"
        self.model_name = model_name or env_model

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Groq configuration (fallback)
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.use_groq = False  # Flag to track if we're using Groq
        
        # Initialize Ollama client
        try:
            self.ollama_client = ollama.Client(host=self.ollama_base_url)
            self.ollama_available = True
            print(f"RAG Pipeline initialized with Ollama model: {self.model_name}")
        except Exception as e:
            print(f"Ollama initialization warning: {str(e)}")
            self.ollama_client = None
            self.ollama_available = False
            if self.groq_api_key:
                print(f"Groq fallback will be used. Model: {self.groq_model}")
            else:
                print("Warning: No Groq API key found. Groq fallback will not be available.")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Enhanced context retrieval with query expansion and intelligent filtering.
        """
        # Expand query with variations for better retrieval
        expanded_queries = self._expand_query(query)
        
        # Search with each query variation and combine results
        all_results = []
        seen_indices = set()
        
        for expanded_query in expanded_queries:
            # Lower threshold to catch more relevant results
            results = self.vector_store.search(expanded_query, top_k=top_k * 2, min_similarity=0.15)
            
            for result in results:
                # Use text as unique identifier to avoid duplicates
                text_hash = hash(result["text"])
                if text_hash not in seen_indices:
                    seen_indices.add(text_hash)
                    all_results.append(result)
        
        # Sort by score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        results = all_results[:top_k]
        
        if not results:
            return ""

        # Build context - include all results, even lower scores
        # The LLM can filter out irrelevant content
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result["text"]
            score = result.get("score", 0.0)
            
            # Lower threshold - include results above 0.15
            # This ensures we get context even if similarity is moderate
            if score >= 0.15:
                context_parts.append(text)
        
        if not context_parts:
            return ""
        
        context = "\n\n".join(context_parts)
        return context
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations for better retrieval"""
        queries = [query]  # Original query
        
        # Add variations for common question patterns
        query_lower = query.lower()
        
        # If asking about "Vasanth" or "Vasanthkumar", add variations
        if "vasanth" in query_lower or "vasanthkumar" in query_lower:
            queries.append(query.replace("Vasanth", "Vasanthkumar"))
            queries.append(query.replace("Vasanthkumar", "Vasanth"))
        
        # Add question variations
        if query.endswith("?"):
            queries.append(query[:-1])  # Without question mark
        
        return queries[:3]  # Limit to 3 variations
    
    def generate_response_ollama(self, query: str, context: str, timeout: int = 5) -> str:
        """Generate response using Ollama (primary method) with timeout"""
        def _call_ollama():
            """Internal function to call Ollama"""
            if not self.ollama_client:
                # Reinitialize client if needed
                self.ollama_client = ollama.Client(host=self.ollama_base_url)
            
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=self._create_prompt(query, context),
                stream=False
            )
            return response['response'].strip()
        
        try:
            # Use ThreadPoolExecutor for timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_ollama)
                response = future.result(timeout=timeout)
                return response
        except FutureTimeoutError:
            raise Exception(f"Ollama timeout: No response within {timeout} seconds")
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
        """Create optimized prompt for LLM"""
        return f"""You are Vasanth's personal assistant. Answer questions about Vasanth based on the information provided below.

Guidelines:
- Only If anyone say hi or hellow then greet the user with welcome message
- Answer naturally and conversationally, as if you personally know this information
- Keep answers concise (2-3 sentences maximum)
- Only use information from the provided context below
- If the information isn't available, politely say you don't have that specific detail
- Refer to Vasanthkumar as "Vasanth" in your responses
- Do NOT mention documents, files, or sources - just answer naturally
- Be friendly and professional

Information about Vasanth:
{context}

Question: {query}

Answer:"""
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response with automatic fallback: Ollama -> Groq (with 5s timeout)"""
        
        # Always try Ollama first if it's available (check connection first)
        # This ensures we use Ollama if it's working, even if it failed before
        if self.ollama_available or self.test_ollama_connection():
            try:
                self.use_groq = False
                # Try Ollama with 5 second timeout
                response = self.generate_response_ollama(query, context, timeout=5)
                # If successful, ensure ollama_available is True
                self.ollama_available = True
                return response
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    print(f"Ollama timeout: No response within 5 seconds. Switching to Groq...")
                else:
                    print(f"Ollama failed: {error_msg}")
                    print("Falling back to Groq...")
                # Don't permanently mark as unavailable - might be temporary
                # Only mark as unavailable if connection test also fails
                if not self.test_ollama_connection():
                    self.ollama_available = False
        
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

    def chat(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Enhanced chat with better context retrieval and response generation.
        """
        # Retrieve context with expanded search
        context = self.retrieve_context(query, top_k=top_k)
        
        # If no context found, return early
        if not context or not context.strip():
            return {
                "response": "I don't have specific information about that in my knowledge base. Could you try rephrasing your question or ask about Vasanth's experience, skills, projects, or education?",
                "context": "",
                "sources": [],
                "model": self.groq_model if self.use_groq else self.model_name,
                "provider": "groq" if self.use_groq else "ollama"
            }
        
        # Generate response
        response = self.generate_response(query, context)
        
        # Get retrieval results for sources
        retrieval_results = self.vector_store.search(query, top_k=top_k, min_similarity=0.25)
        
        return {
            "response": response,
            "context": context[:500] + "..." if len(context) > 500 else context,  # Truncate for response
            "sources": [
                {
                    "text": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"],
                    "metadata": r["metadata"],
                    "score": round(r["score"], 3)
                }
                for r in retrieval_results[:3]  # Top 3 sources
            ],
            "model": self.groq_model if self.use_groq else self.model_name,
            "provider": "groq" if self.use_groq else "ollama"
        }
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is accessible and model is available"""
        try:
            # Reinitialize client if needed
            if not self.ollama_client:
                self.ollama_client = ollama.Client(host=self.ollama_base_url)
            
            models_response = self.ollama_client.list()
            
            # Handle ListResponse object (new ollama library format)
            if hasattr(models_response, 'models'):
                model_names = [model.model for model in models_response.models]
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_names = [m.get('name') or m.get('model') for m in models_response['models']]
            else:
                model_names = []
            
            if self.model_name in model_names:
                print(f"Ollama connection successful. Model '{self.model_name}' is available.")
                self.ollama_available = True
                return True
            else:
                print(f"Ollama is running, but model '{self.model_name}' not found.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Please run: ollama pull {self.model_name}")
                self.ollama_available = False
                return False
        
        except Exception as e:
            print(f"Ollama connection failed: {str(e)}")
            print(f"Make sure Ollama is running on {self.ollama_base_url}")
            self.ollama_client = None
            self.ollama_available = False
            return False
