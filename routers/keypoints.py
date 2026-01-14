"""
AI Research Paper Helper - Key Points Router
Endpoint for extracting key contributions and concepts.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from services.keyword_extractor import get_keyword_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ConceptInfo(BaseModel):
    """Concept information model."""
    name: str
    category: str
    description: str
    frequency: int


class KeypointsRequest(BaseModel):
    """Request model for keypoints extraction."""
    title: Optional[str] = Field(None, description="Paper title")
    content: str = Field(..., min_length=100, description="Paper content")
    abstract: Optional[str] = Field(None, description="Paper abstract")


class KeypointsResponse(BaseModel):
    """Response model for keypoints extraction."""
    contributions: List[str] = Field(..., description="Novel contributions")
    datasets: List[str] = Field(..., description="Datasets mentioned")
    metrics: List[str] = Field(..., description="Evaluation metrics")
    concepts: List[ConceptInfo] = Field(..., description="Key concepts")
    algorithms: List[str] = Field(..., description="Algorithms/architectures")
    assumptions: List[str] = Field(..., description="Stated assumptions")


@router.post("/extract-key-points", response_model=KeypointsResponse)
async def extract_keypoints(request: KeypointsRequest):
    """
    Extract key contributions and concepts from a research paper.
    
    Extracts:
    - **contributions**: Novel contributions claimed by authors
    - **datasets**: Datasets used in experiments
    - **metrics**: Evaluation metrics reported
    - **concepts**: Key ML concepts and terminology
    - **algorithms**: Referenced algorithms and architectures
    - **assumptions**: Stated assumptions and limitations
    """
    try:
        service = get_keyword_service()
        result = await service.extract(
            content=request.content,
            abstract=request.abstract,
            title=request.title
        )
        
        return KeypointsResponse(
            contributions=result.contributions,
            datasets=result.datasets,
            metrics=result.metrics,
            concepts=[
                ConceptInfo(
                    name=c.name,
                    category=c.category,
                    description=c.description,
                    frequency=c.frequency
                )
                for c in result.concepts
            ],
            algorithms=result.algorithms,
            assumptions=result.assumptions
        )
        
    except Exception as e:
        logger.error(f"Keypoints extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Keypoints extraction failed: {str(e)}"
        )
