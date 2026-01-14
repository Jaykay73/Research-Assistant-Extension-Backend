"""
AI Research Paper Helper - RAG Router
Endpoints for paper indexing and question answering.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging

from services.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class IndexRequest(BaseModel):
    """Request model for paper indexing."""
    paper_id: str = Field(..., description="Unique paper identifier (e.g., URL)")
    title: str = Field(..., description="Paper title")
    content: str = Field(..., min_length=100, description="Paper content")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    sections: Optional[List[Dict]] = Field(None, description="Pre-extracted sections")


class IndexResponse(BaseModel):
    """Response model for paper indexing."""
    success: bool
    paper_id: str
    chunks_indexed: int
    embedding_dimension: int


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    query: str = Field(..., min_length=3, description="Question about the paper")
    paper_id: str = Field(..., description="ID of the indexed paper")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceInfo(BaseModel):
    """Source chunk information."""
    chunk_id: str
    text: str
    section: Optional[str]
    score: float


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: List[SourceInfo]
    confidence: float


class StatusResponse(BaseModel):
    """Response model for RAG status."""
    indexed_papers: List[str]
    total_papers: int


@router.post("/index", response_model=IndexResponse)
async def index_paper(request: IndexRequest):
    """
    Index a paper for RAG-based Q&A.
    
    The paper content is:
    1. Split into overlapping chunks (512 tokens, 64 token overlap)
    2. Each chunk is embedded using sentence-transformers
    3. Embeddings are stored in a FAISS index for fast retrieval
    
    After indexing, use /rag/query to ask questions about the paper.
    """
    try:
        engine = get_rag_engine()
        result = await engine.index_paper(
            paper_id=request.paper_id,
            title=request.title,
            content=request.content,
            abstract=request.abstract,
            sections=request.sections
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Indexing failed")
            )
        
        return IndexResponse(
            success=True,
            paper_id=result["paper_id"],
            chunks_indexed=result["chunks_indexed"],
            embedding_dimension=result["embedding_dimension"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper indexing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Paper indexing failed: {str(e)}"
        )


@router.post("/query", response_model=QueryResponse)
async def query_paper(request: QueryRequest):
    """
    Query an indexed paper using natural language.
    
    The query is:
    1. Embedded using the same model as the paper chunks
    2. Similar chunks are retrieved using FAISS cosine similarity
    3. Retrieved chunks are used to generate a grounded answer
    4. Sources are cited for transparency
    
    **Important**: The paper must be indexed first using /rag/index
    """
    try:
        engine = get_rag_engine()
        
        # Check if paper is indexed
        if not engine.is_paper_indexed(request.paper_id):
            raise HTTPException(
                status_code=404,
                detail=f"Paper not indexed: {request.paper_id}. Please index the paper first."
            )
        
        result = await engine.query(
            query=request.query,
            paper_id=request.paper_id,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceInfo(
                    chunk_id=s.chunk_id,
                    text=s.text,
                    section=s.section,
                    score=s.score
                )
                for s in result.sources
            ],
            confidence=result.confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@router.get("/status", response_model=StatusResponse)
async def get_rag_status():
    """
    Get the current status of the RAG engine.
    
    Returns list of indexed paper IDs.
    """
    try:
        engine = get_rag_engine()
        indexed = engine.get_indexed_papers()
        
        return StatusResponse(
            indexed_papers=indexed,
            total_papers=len(indexed)
        )
        
    except Exception as e:
        logger.error(f"RAG status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )


@router.delete("/{paper_id}")
async def clear_paper_index(paper_id: str):
    """
    Clear the index for a specific paper.
    """
    try:
        engine = get_rag_engine()
        success = engine.clear_index(paper_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Paper not found: {paper_id}"
            )
        
        return {"success": True, "paper_id": paper_id, "message": "Index cleared"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index clearing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Index clearing failed: {str(e)}"
        )
