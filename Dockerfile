FROM python:3.11-slim

LABEL maintainer="vasanthkumarr"
LABEL description="FastAPI Portfolio Chatbot with RAG capabilities"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app
ENV PORT=8080

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=$APP_HOME/.cache/huggingface
ENV TRANSFORMERS_CACHE=$APP_HOME/.cache/huggingface
RUN mkdir -p $APP_HOME/.cache && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY main.py .
COPY config.py .
COPY embedding_service.py .
COPY vector_store.py .
COPY rag_pipeline.py .
COPY pdf_processor.py .
COPY query_normalizer.py .

COPY data/ ./data/

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
