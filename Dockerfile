# Dockerfile for FastAPI Portfolio Chatbot
# Optimized for Google Cloud Run deployment

# Use official Python runtime as base image
FROM python:3.11-slim

# Set maintainer label
LABEL maintainer="vasanthkumarr"
LABEL description="FastAPI Portfolio Chatbot with RAG capabilities"

# Set environment variables
# Prevents Python from writing pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
# Set working directory
ENV APP_HOME=/app
# Default port for Cloud Run
ENV PORT=8080

# Set working directory in container
WORKDIR $APP_HOME

# Install system dependencies required for FAISS and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download ML models at build time into $APP_HOME so they persist for appuser at runtime
# (Root's ~/.cache is not used when container runs as appuser, causing re-download + 429 rate limit)
ENV HF_HOME=$APP_HOME/.cache/huggingface
ENV TRANSFORMERS_CACHE=$APP_HOME/.cache/huggingface
RUN mkdir -p $APP_HOME/.cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY main.py .
COPY config.py .
COPY embedding_service.py .
COPY vector_store.py .
COPY rag_pipeline.py .
COPY pdf_processor.py .
COPY query_normalizer.py .

# Copy data directory (includes your PDF documents and pre-built index if any)
COPY data/ ./data/

# Create non-root user for security (Cloud Run best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

# Expose the port Cloud Run will use
EXPOSE $PORT

# Health check (Cloud Run uses this to verify service health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Run the application
# Cloud Run sets PORT env variable automatically
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
