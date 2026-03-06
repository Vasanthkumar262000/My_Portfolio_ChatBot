import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    environment: str = Field(default="production", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    port: int = Field(default=8080, alias="PORT")

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")

    google_cloud_project: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")

    vertex_ai_project: Optional[str] = Field(default=None, alias="VERTEX_AI_PROJECT")
    vertex_ai_location: str = Field(default="us-central1", alias="VERTEX_AI_LOCATION")
    gemini_model: str = Field(default="gemini-1.5-flash", alias="GEMINI_MODEL")

    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=400, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    min_similarity: float = Field(default=0.15, alias="MIN_SIMILARITY")

    vector_store_path: str = Field(default="data/faiss_index", alias="VECTOR_STORE_PATH")

    rate_limit_requests: int = Field(default=60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_minutes: int = Field(default=1, alias="RATE_LIMIT_MINUTES")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        populate_by_name = True


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
