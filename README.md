# Portfolio Backend API

A FastAPI-based backend with RAG (Retrieval Augmented Generation) capabilities for portfolio websites.

## Features

- 📄 PDF document processing and text extraction
- 🔍 Semantic search using FAISS vector database
- 🤖 AI-powered chat with Ollama (local) and Groq (cloud fallback)
- 🔄 Automatic fallback from Ollama to Groq when local server is unavailable
- 📊 RESTful API endpoints for document management

## Tech Stack

- **FastAPI** - Modern Python web framework
- **FAISS** - Vector similarity search
- **sentence-transformers** - Text embeddings
- **Ollama** - Local LLM inference
- **Groq** - Cloud LLM API (fallback)

## Setup

### 1. Install Dependencies

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt### 2. Configure Environment

Create a `.env` file:
v
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Groq Configuration (Fallback)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant### 3. Start Ollama (Optional)

ollama serve
ollama pull llama3.2:3b
### 4. Process Your PDF

python process_pdf.py data/your_document.pdf### 5. Start the Server

uvicorn main:app --reload --host 0.0.0.0 --port 8000## API Endpoints

### `POST /chat`
Chat with RAG - generates intelligent responses based on your documents.

**Request:**on
{
  "message": "What is your experience?"
}**Response:**
{
  "response": "...",
  "sources": [...],
  "model": "llama3.2:3b",
  "provider": "ollama"
}### `POST /upload`
Upload and process a PDF file.

### `POST /query`
Semantic search without LLM generation.

### `GET /health`
Health check with service status.

### `GET /stats`
Get vector store statistics.

### `DELETE /clear`
Clear all documents from vector store.

## How It Works

1. **Document Processing**: PDFs are extracted, chunked, and converted to embeddings
2. **Vector Storage**: Embeddings stored in FAISS for fast similarity search
3. **Query Processing**: User queries are converted to embeddings and matched against stored documents
4. **Response Generation**: 
   - Primary: Uses Ollama (local) if available
   - Fallback: Automatically switches to Groq if Ollama is unavailable

## Development

# Run with auto-reload
uvicorn main:app --reload

# Run on specific port
uvicorn main:app --host 0.0.0.0 --port 8000## License

MIT