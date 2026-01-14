"""
AI Research Paper Helper - Model Manager
Lightweight version: Only handles local embeddings.
LLM inference goes through API (Groq/OpenRouter).
"""

import asyncio
import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for ML models with lazy loading and caching.
    
    Only loads the lightweight embedding model locally.
    All LLM inference is delegated to external APIs.
    """
    
    _instance: Optional['ModelManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self):
        self.device = "cpu"  # Embeddings run fine on CPU
        logger.info(f"ModelManager initialized (embeddings only, device: {self.device})")
        
        # Only embedding model (lightweight, ~80MB)
        self._embedding_model: Optional[SentenceTransformer] = None
    
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @property
    def embedding_model(self) -> Optional[SentenceTransformer]:
        return self._embedding_model
    
    async def load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model asynchronously."""
        if self._embedding_model is not None:
            return self._embedding_model
        
        async with self._lock:
            if self._embedding_model is not None:
                return self._embedding_model
            
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            self._embedding_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    settings.embedding_model,
                    cache_folder=str(settings.model_cache_dir),
                    device=self.device
                )
            )
            
            logger.info("Embedding model loaded successfully")
            return self._embedding_model
    
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = await self.load_embedding_model()
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        )
        
        return embeddings.tolist()
    
    async def cleanup(self):
        """Clean up model resources."""
        logger.info("Cleaning up model resources...")
        
        if self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
        
        logger.info("Model cleanup complete")
