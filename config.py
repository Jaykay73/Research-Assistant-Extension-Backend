"""
AI Research Paper Helper - Configuration
Environment-based configuration with sensible defaults.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Mode: 'local' uses HuggingFace models, 'api' uses external LLMs
    # Note: Embeddings always run locally (lightweight), only LLM calls go to API
    api_mode: Literal['local', 'api'] = Field(
        default='api',
        description="Mode for LLM inference: 'local' uses HuggingFace models, 'api' uses Groq/OpenRouter"
    )
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: list[str] = ["*"]
    
    # Model settings
    model_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "ai-research-helper",
        description="Directory for caching downloaded models"
    )
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Summarization model
    summarization_model: str = "facebook/bart-large-cnn"
    max_summary_length: int = 300
    min_summary_length: int = 50
    
    # External API settings (for hybrid mode)
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-3-haiku"
    
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.1-8b-instant"
    
    # RAG settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # tokens (12.5% overlap)
    top_k_retrieval: int = 5
    faiss_index_type: str = "IndexFlatIP"  # Inner product for cosine similarity
    
    # Equation explanation
    use_llm_for_equations: bool = True
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: int = 60
    
    class Config:
        env_prefix = "AIHELPER_"
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure cache directory exists
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)


def get_llm_api_key() -> str | None:
    """Get the first available LLM API key."""
    # Check for valid keys (not placeholders)
    groq_valid = settings.groq_api_key and settings.groq_api_key.startswith('gsk_')
    openrouter_valid = settings.openrouter_api_key and settings.openrouter_api_key.startswith('sk-or-')
    
    if groq_valid:
        return settings.groq_api_key
    if openrouter_valid:
        return settings.openrouter_api_key
    return None


def get_llm_config() -> dict | None:
    """Get LLM API configuration."""
    # Check for valid keys (not placeholders)
    groq_valid = settings.groq_api_key and settings.groq_api_key.startswith('gsk_')
    openrouter_valid = settings.openrouter_api_key and settings.openrouter_api_key.startswith('sk-or-')
    
    # Prefer Groq (faster) if available
    if groq_valid:
        return {
            "api_key": settings.groq_api_key,
            "base_url": settings.groq_base_url,
            "model": settings.groq_model,
            "provider": "groq"
        }
    elif openrouter_valid:
        return {
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
            "model": settings.openrouter_model,
            "provider": "openrouter"
        }
    return None
