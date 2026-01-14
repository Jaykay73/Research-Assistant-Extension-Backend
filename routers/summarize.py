"""
AI Research Paper Helper - Summarization Router
Endpoint for multi-level paper summarization.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from services.summarizer import get_summarizer_service

logger = logging.getLogger(__name__)
router = APIRouter()


class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    title: Optional[str] = Field(None, description="Paper title")
    content: str = Field(..., min_length=100, description="Paper content")
    abstract: Optional[str] = Field(None, description="Paper abstract")
    page_type: str = Field("arxiv", description="Source page type")


class SummaryResponse(BaseModel):
    """Response model for summarization."""
    tldr: List[str] = Field(..., description="5-6 bullet point summary")
    technical: str = Field(..., description="Technical summary for researchers")
    beginner: str = Field(..., description="Beginner-friendly explanation")
    section_summaries: dict = Field(default_factory=dict, description="Per-section summaries")


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_paper(request: SummarizeRequest):
    """
    Generate multi-level summaries for a research paper.
    
    Returns:
    - **tldr**: 5-6 bullet points capturing key insights
    - **technical**: Detailed summary for ML researchers
    - **beginner**: Plain English explanation for non-experts
    """
    try:
        service = get_summarizer_service()
        result = await service.summarize(
            content=request.content,
            abstract=request.abstract,
            title=request.title,
            page_type=request.page_type
        )
        
        return SummaryResponse(
            tldr=result.tldr,
            technical=result.technical,
            beginner=result.beginner,
            section_summaries=result.section_summaries
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )
