"""
AI Research Paper Helper - RAG Engine Service
Retrieval-Augmented Generation for paper Q&A.
"""

import logging
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import httpx

from config import settings, get_llm_config
from ml.embeddings import get_embedding_service
from ml.text_processor import get_text_processor, TextChunk

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A source chunk used to answer a question."""
    chunk_id: str
    text: str
    section: Optional[str]
    score: float


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: List[Source]
    confidence: float


class PaperIndex:
    """Index for a single paper."""
    
    def __init__(self, paper_id: str, title: str):
        self.paper_id = paper_id
        self.title = title
        self.chunks: List[TextChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
    
    def is_indexed(self) -> bool:
        return self.embeddings is not None and len(self.chunks) > 0


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for paper Q&A.
    
    Features:
    - FAISS-based vector similarity search
    - Section-aware chunking
    - Answer grounding with citations
    """
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.text_processor = get_text_processor()
        self.paper_indices: Dict[str, PaperIndex] = {}
    
    async def index_paper(
        self,
        paper_id: str,
        title: str,
        content: str,
        abstract: Optional[str] = None,
        sections: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Index a paper for RAG queries.
        
        Args:
            paper_id: Unique identifier for the paper
            title: Paper title
            content: Full paper content
            abstract: Paper abstract
            sections: Pre-extracted sections
        
        Returns:
            Dict with indexing status and stats
        """
        logger.info(f"Indexing paper: {paper_id}")
        
        # Create chunks
        chunks = self.text_processor.chunk_document(
            content=content,
            abstract=abstract,
            sections=sections
        )
        
        if not chunks:
            logger.warning(f"No chunks created for paper {paper_id}")
            return {"success": False, "error": "No content to index"}
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.embed_texts(chunk_texts)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
        faiss_index.add(embeddings_array)
        
        # Store index
        paper_index = PaperIndex(paper_id, title)
        paper_index.chunks = chunks
        paper_index.embeddings = embeddings_array
        paper_index.faiss_index = faiss_index
        
        self.paper_indices[paper_id] = paper_index
        
        logger.info(f"Indexed {len(chunks)} chunks for paper {paper_id}")
        
        return {
            "success": True,
            "paper_id": paper_id,
            "chunks_indexed": len(chunks),
            "embedding_dimension": dimension
        }
    
    async def query(
        self,
        query: str,
        paper_id: str,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Query a paper using RAG.
        
        Args:
            query: Natural language question
            paper_id: ID of the indexed paper
            top_k: Number of chunks to retrieve
        
        Returns:
            RAGResponse with answer and sources
        """
        # Get paper index
        paper_index = self.paper_indices.get(paper_id)
        if not paper_index or not paper_index.is_indexed():
            return RAGResponse(
                answer="Paper not indexed. Please analyze the paper first.",
                sources=[],
                confidence=0.0
            )
        
        # Embed query
        query_embedding = await self.embedding_service.embed_single(query)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = paper_index.faiss_index.search(query_embedding, min(top_k, len(paper_index.chunks)))
        
        # Get relevant chunks
        sources = []
        context_chunks = []
        
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(paper_index.chunks):
                continue
            
            chunk = paper_index.chunks[idx]
            sources.append(Source(
                chunk_id=chunk.id,
                text=chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                section=chunk.section,
                score=float(score)
            ))
            context_chunks.append(chunk.text)
        
        if not context_chunks:
            return RAGResponse(
                answer="No relevant content found for your question.",
                sources=[],
                confidence=0.0
            )
        
        # Generate answer
        answer = await self._generate_answer(query, context_chunks, paper_index.title)
        
        # Calculate confidence based on top retrieval scores
        confidence = float(np.mean(scores[0][:3])) if len(scores[0]) >= 3 else float(np.mean(scores[0]))
        confidence = min(1.0, max(0.0, confidence))
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    
    async def _generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        title: str
    ) -> str:
        """Generate answer from retrieved context."""
        context = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Use LLM if available
        config = get_llm_config()
        if config and settings.api_mode in ['api', 'hybrid']:
            return await self._answer_with_llm(query, context, title, config)
        else:
            return self._answer_extractive(query, context_chunks)
    
    async def _answer_with_llm(
        self,
        query: str,
        context: str,
        title: str,
        config: dict
    ) -> str:
        """Generate answer using LLM."""
        prompt = f"""Based on the following excerpts from the paper "{title}", answer the question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I couldn't find this information in the paper"
- Be concise but complete
- Reference source numbers [1], [2] etc. when citing specific information

ANSWER:"""

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                headers = {
                    "Authorization": f"Bearer {config['api_key']}",
                    "Content-Type": "application/json"
                }
                
                if config['provider'] == 'openrouter':
                    headers["HTTP-Referer"] = "https://ai-research-helper.local"
                
                response = await client.post(
                    f"{config['base_url']}/chat/completions",
                    headers=headers,
                    json={
                        "model": config['model'],
                        "messages": [
                            {"role": "system", "content": "You are a helpful research assistant answering questions about academic papers. Only answer based on the provided context."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                
                return response.json()['choices'][0]['message']['content']
                
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return self._answer_extractive(query, context.split("\n\n"))
    
    def _answer_extractive(self, query: str, chunks: List[str]) -> str:
        """Simple extractive answer when LLM is not available."""
        # Return the most relevant chunk as the answer
        if chunks:
            # Combine top chunks
            combined = " ".join(chunks[:2])[:1000]
            return f"Based on the paper: {combined}"
        return "Unable to find relevant information."
    
    def is_paper_indexed(self, paper_id: str) -> bool:
        """Check if a paper is indexed."""
        paper_index = self.paper_indices.get(paper_id)
        return paper_index is not None and paper_index.is_indexed()
    
    def get_indexed_papers(self) -> List[str]:
        """Get list of indexed paper IDs."""
        return [pid for pid, idx in self.paper_indices.items() if idx.is_indexed()]
    
    def clear_index(self, paper_id: str) -> bool:
        """Clear index for a paper."""
        if paper_id in self.paper_indices:
            del self.paper_indices[paper_id]
            return True
        return False


# Singleton instance
_rag_engine = None

def get_rag_engine() -> RAGEngine:
    """Get the singleton RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for RAG. Install with: pip install faiss-cpu")
        _rag_engine = RAGEngine()
    return _rag_engine
