"""
Tests for the RAG engine.
"""

import pytest
from services.rag_engine import get_rag_engine


@pytest.fixture
def rag_engine():
    return get_rag_engine()


@pytest.mark.asyncio
async def test_index_paper(rag_engine):
    """Test paper indexing."""
    result = await rag_engine.index_paper(
        paper_id="test_paper_001",
        title="Test Paper",
        content="This is a test paper about machine learning. " * 50,
        abstract="We present a novel approach to testing.",
        sections=None
    )
    
    assert result["success"] is True
    assert result["chunks_indexed"] > 0
    assert result["embedding_dimension"] == 384


@pytest.mark.asyncio
async def test_query_indexed_paper(rag_engine):
    """Test querying an indexed paper."""
    # First index
    await rag_engine.index_paper(
        paper_id="test_paper_002",
        title="Machine Learning Paper",
        content="This paper discusses neural networks and deep learning. " * 50,
        abstract="A study of neural network architectures.",
    )
    
    # Then query
    result = await rag_engine.query(
        query="What does this paper discuss?",
        paper_id="test_paper_002",
        top_k=3
    )
    
    assert result.answer != ""
    assert len(result.sources) > 0
    assert result.confidence >= 0


@pytest.mark.asyncio
async def test_query_unindexed_paper(rag_engine):
    """Test querying a paper that hasn't been indexed."""
    result = await rag_engine.query(
        query="What is this about?",
        paper_id="nonexistent_paper",
        top_k=3
    )
    
    assert "not indexed" in result.answer.lower()
    assert result.confidence == 0


@pytest.mark.asyncio
async def test_is_paper_indexed(rag_engine):
    """Test checking if a paper is indexed."""
    await rag_engine.index_paper(
        paper_id="test_paper_003",
        title="Another Test",
        content="Test content. " * 100,
    )
    
    assert rag_engine.is_paper_indexed("test_paper_003") is True
    assert rag_engine.is_paper_indexed("not_indexed") is False


@pytest.mark.asyncio
async def test_clear_index(rag_engine):
    """Test clearing a paper index."""
    await rag_engine.index_paper(
        paper_id="test_paper_004",
        title="To Be Deleted",
        content="This will be deleted. " * 100,
    )
    
    assert rag_engine.is_paper_indexed("test_paper_004") is True
    
    success = rag_engine.clear_index("test_paper_004")
    assert success is True
    assert rag_engine.is_paper_indexed("test_paper_004") is False
