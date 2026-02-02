"""
Portfolio Backend - FastAPI application with RAG capabilities
Optimized for Google Cloud Run deployment
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pdf_processor import process_pdf
from vector_store import VectorStore
from embedding_service import EmbeddingService
from rag_pipeline import RAGPipeline
from config import settings

# Load environment variables
load_dotenv()

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services (initialized in lifespan)
embedding_service: Optional[EmbeddingService] = None
vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    global embedding_service, vector_store, rag_pipeline
    
    # Startup
    logger.info("Starting Portfolio Backend API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
    
    try:
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        logger.info("Initializing vector store...")
        vector_store = VectorStore(embedding_service)
        
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(vector_store)
        
        stats = vector_store.get_stats()
        logger.info(f"Vector store ready. Documents: {stats['total_documents']}")
        
        # Defer LLM connection test to background (saves ~10s startup - Cloud Run timeout)
        import threading
        def _test_llm_async():
            try:
                rag_pipeline.test_llm_connection()
            except Exception as e:
                logger.warning(f"Background LLM test: {e}")
        threading.Thread(target=_test_llm_async, daemon=True).start()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Portfolio Backend API...")
    try:
        if vector_store:
            vector_store.save()
            logger.info("Vector store saved to disk")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Rate limiter: use X-Forwarded-For when behind proxy (e.g. Cloud Run), else client IP
def get_client_identifier(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)

_rate = f"{getattr(settings, 'rate_limit_requests', 60)}/{getattr(settings, 'rate_limit_minutes', 1)} minute"
limiter = Limiter(key_func=get_client_identifier, default_limits=[_rate])

app = FastAPI(
    title="Portfolio Backend API",
    description="Backend API for portfolio with RAG capabilities - Deployed on Cloud Run",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware configuration
# Update these origins with your actual frontend domains
ALLOWED_ORIGINS = [
    "https://vasanthkumarr.com",  # Replace with your actual domain
    "https://www.vasanthkumarr.com",  # Replace with your actual domain
    "http://localhost:3000",  # Local development
    "http://localhost:8080",  # Local development
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
]

# For development/testing, you can add "*" to allow all origins
# In production, restrict to specific domains
if os.getenv("ENVIRONMENT", "production") == "development":
    ALLOWED_ORIGINS.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    model: str
    provider: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    results: List[dict]
    query: str


class UploadResponse(BaseModel):
    message: str
    chunks_processed: int
    filename: str


class HealthResponse(BaseModel):
    status: str
    groq_configured: bool
    vertex_ai_configured: bool
    current_provider: str
    vector_store_documents: int
    environment: str


# Exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("ENVIRONMENT") == "development" else "An error occurred"
        }
    )


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Portfolio Backend API is running",
        "status": "healthy",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "upload": "/upload",
            "query": "/query",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Cloud Run uses this to verify the service is healthy.
    """
    if not rag_pipeline or not vector_store:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    groq_available = bool(rag_pipeline.groq_api_key)
    vertex_available = bool(rag_pipeline.vertex_ai_project)
    
    return HealthResponse(
        status="healthy" if (groq_available or vertex_available) else "degraded",
        groq_configured=groq_available,
        vertex_ai_configured=vertex_available,
        current_provider="vertex-ai" if rag_pipeline.use_vertex else "groq",
        vector_store_documents=vector_store.index.ntotal,
        environment=os.getenv("ENVIRONMENT", "production")
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(_rate)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint with RAG (Retrieval Augmented Generation).
    Takes user message, retrieves relevant context, and generates response using Groq.
    Rate limited per client IP to prevent abuse.
    """
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        logger.info(f"Processing chat request: {chat_request.message[:50]}...")
        
        # Use RAG pipeline to generate response (handles typos and Vasanth context)
        result = rag_pipeline.chat(chat_request.message, top_k=3)
        
        logger.info(f"Chat response generated using {result['provider']}")
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            model=result["model"],
            provider=result["provider"]
        )
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    Extracts text, chunks it, creates embeddings, and stores in vector database.
    
    Note: In Cloud Run, uploaded files are stored temporarily.
    For persistent storage, consider using Cloud Storage.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF - extract text and chunk it (smaller chunks for better RAG retrieval)
        chunks = process_pdf(file_path, chunk_size=700, chunk_overlap=180)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Create metadata for each chunk
        metadata = [{"filename": file.filename, "chunk_id": i} for i in range(len(chunks))]
        
        # Add documents to vector store (creates embeddings and stores them)
        vector_store.add_documents(chunks, metadata)
        
        # Save vector store to disk
        vector_store.save()
        
        # Clean up uploaded file
        os.remove(file_path)
        
        logger.info(f"Successfully processed {file.filename}: {len(chunks)} chunks")
        
        return UploadResponse(
            message="PDF processed and added to vector store",
            chunks_processed=len(chunks),
            filename=file.filename
        )
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
@limiter.limit(_rate)
async def query_documents(request: Request, query_request: QueryRequest):
    """
    Query the document store using semantic search.
    Returns the most relevant chunks based on the query (without LLM generation).
    Rate limited per client IP.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not query_request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = vector_store.search(query_request.query, top_k=query_request.top_k)
        
        return QueryResponse(
            query=query_request.query,
            results=results
        )
    
    except Exception as e:
        logger.error(f"Error querying documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get statistics about the chatbot service"""
    if not vector_store or not rag_pipeline:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    stats = vector_store.get_stats()
    return {
        **stats,
        "groq_configured": bool(rag_pipeline.groq_api_key),
        "groq_model": rag_pipeline.groq_model,
        "vertex_ai_configured": bool(rag_pipeline.vertex_ai_project),
        "gemini_model": rag_pipeline.gemini_model,
        "current_provider": "vertex-ai" if rag_pipeline.use_vertex else "groq",
        "environment": os.getenv("ENVIRONMENT", "production")
    }


@app.delete("/clear")
async def clear_vector_store():
    """Clear all documents from the vector store"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    vector_store.clear()
    logger.info("Vector store cleared")
    return {"message": "Vector store cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
