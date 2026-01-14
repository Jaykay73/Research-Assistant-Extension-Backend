"""
AI Research Paper Helper - Embeddings Module
Handles text embedding generation and similarity calculations.
"""

import numpy as np
from typing import List
import logging

from ml.models import ModelManager

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating and comparing text embeddings."""
    
    def __init__(self):
        self.model_manager = ModelManager.get_instance()
    
    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        embeddings = await self.model_manager.get_embeddings(texts)
        return np.array(embeddings)
    
    async def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if a.size == 0 or b.size == 0:
            return 0.0
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all corpus vectors."""
        if query.size == 0 or corpus.size == 0:
            return np.array([])
        
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        corpus_norms = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(corpus_norms, query_norm)
        return similarities
    
    async def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """Find the most similar texts to a query."""
        if not candidates:
            return []
        
        # Embed query and candidates
        query_embedding = await self.embed_single(query)
        candidate_embeddings = await self.embed_texts(candidates)
        
        # Calculate similarities
        similarities = self.batch_cosine_similarity(query_embedding, candidate_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (candidates[i], float(similarities[i]), i)
            for i in top_indices
        ]
        
        return results


# Singleton instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
