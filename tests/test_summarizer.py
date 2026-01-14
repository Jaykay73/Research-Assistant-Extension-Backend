"""
Tests for the summarization service.
"""

import pytest
from services.summarizer import get_summarizer_service


@pytest.fixture
def summarizer():
    return get_summarizer_service()


@pytest.mark.asyncio
async def test_summarize_short_text(summarizer):
    """Test summarization with a short abstract."""
    result = await summarizer.summarize(
        content="This paper presents a novel approach to machine learning.",
        abstract="We propose a new method for training neural networks efficiently.",
        title="Efficient Neural Network Training",
        page_type="arxiv"
    )
    
    assert result is not None
    assert isinstance(result.tldr, list)
    assert len(result.tldr) > 0
    assert isinstance(result.technical, str)
    assert isinstance(result.beginner, str)


@pytest.mark.asyncio
async def test_summarize_empty_content():
    """Test handling of empty content."""
    summarizer = get_summarizer_service()
    
    # Should handle gracefully
    result = await summarizer.summarize(
        content="",
        abstract="Short abstract for testing.",
        title="Test",
        page_type="arxiv"
    )
    
    assert result is not None


@pytest.mark.asyncio
async def test_bullet_extraction(summarizer):
    """Test that TL;DR produces bullet points."""
    long_abstract = """
    This paper introduces a groundbreaking approach to natural language processing.
    We propose a novel transformer architecture that significantly reduces computational costs.
    Our method achieves state-of-the-art results on multiple benchmarks.
    The key innovation is a sparse attention mechanism that scales linearly with sequence length.
    We demonstrate improvements of 30% in efficiency while maintaining accuracy.
    Extensive experiments validate our approach across diverse datasets.
    """
    
    result = await summarizer.summarize(
        content=long_abstract,
        abstract=long_abstract,
        title="Efficient Transformers",
        page_type="arxiv"
    )
    
    assert len(result.tldr) >= 1
    assert len(result.tldr) <= 6
