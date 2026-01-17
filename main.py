"""
Portfolio Backend - FastAPI application with RAG capabilities
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from pdf_processor import process_pdf
from vector_store import VectorStore
from embedding_service import EmbeddingService
from rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Portfolio Backend API",
    description="Backend API for portfolio with RAG capabilities",
    version="1.0.0"
)

# CORS middleware - allows frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
print("Initializing services...")
embedding_service = EmbeddingService()
vector_store = VectorStore(embedding_service)
rag_pipeline = RAGPipeline(vector_store)

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


@app.on_event("startup")
async def startup_event():
    """Load existing vector store and test Ollama connection on startup"""
    if os.path.exists("data/faiss_index/index.faiss"):
        vector_store.load()
        print("[OK] Vector store loaded from disk")
    
    # Test Ollama connection
    rag_pipeline.test_ollama_connection()


@app.on_event("shutdown")
async def shutdown_event():
    """Save vector store on shutdown"""
    vector_store.save()
    print("Vector store saved to disk")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Portfolio Backend API is running",
        "status": "healthy",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "upload": "/upload",
            "query": "/query",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    # Check if Ollama is accessible
    ollama_ok = rag_pipeline.test_ollama_connection()
    
    # Check Groq availability
    groq_available = bool(rag_pipeline.groq_api_key)
    
    return {
        "status": "healthy" if (ollama_ok or groq_available) else "degraded",
        "ollama_connected": ollama_ok,
        "groq_configured": groq_available,
        "current_provider": "groq" if rag_pipeline.use_groq else ("ollama" if ollama_ok else "none"),
        "vector_store_documents": vector_store.index.ntotal
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG (Retrieval Augmented Generation).
    Takes user message, retrieves relevant context, and generates response using Ollama.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Use RAG pipeline to generate response
        result = rag_pipeline.chat(request.message, top_k=3)
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            model=result["model"],
            provider=result["provider"] 
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file.
    Extracts text, chunks it, creates embeddings, and stores in vector database.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF - extract text and chunk it
        chunks = process_pdf(file_path, chunk_size=1500, chunk_overlap=400)
        
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
        
        return UploadResponse(
            message="PDF processed and added to vector store",
            chunks_processed=len(chunks),
            filename=file.filename
        )
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document store using semantic search.
    Returns the most relevant chunks based on the query (without LLM generation).
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = vector_store.search(request.query, top_k=request.top_k)
        
        return QueryResponse(
            query=request.query,
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")


@app.get("/stats")
async def get_stats():
    stats = vector_store.get_stats()
    return {
        **stats,
        "ollama_model": rag_pipeline.model_name,
        "ollama_url": rag_pipeline.ollama_base_url,
        "ollama_available": rag_pipeline.ollama_available,
        "groq_configured": bool(rag_pipeline.groq_api_key),
        "groq_model": rag_pipeline.groq_model,
        "current_provider": "groq" if rag_pipeline.use_groq else "ollama"
    }


@app.delete("/clear")
async def clear_vector_store():

    vector_store.clear()
    return {"message": "Vector store cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
