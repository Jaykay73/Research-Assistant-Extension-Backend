"""
AI Research Paper Helper - Equations Router
Endpoint for LaTeX equation explanation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from services.equation_explainer import get_equation_service

logger = logging.getLogger(__name__)
router = APIRouter()


class VariableInfo(BaseModel):
    """Variable information model."""
    symbol: str
    latex: str
    description: str


class EquationRequest(BaseModel):
    """Request model for equation explanation."""
    equation: str = Field(..., min_length=1, description="LaTeX or MathML equation")
    context: Optional[str] = Field(None, description="Surrounding text for context")
    format: str = Field("latex", description="Format: 'latex' or 'mathml'")


class EquationResponse(BaseModel):
    """Response model for equation explanation."""
    readable: str = Field(..., description="Human-readable form")
    meaning: str = Field(..., description="What the equation represents")
    variables: List[VariableInfo] = Field(..., description="Variable explanations")
    importance: str = Field(..., description="Why it matters")
    equation_type: str = Field(..., description="Type of equation")


@router.post("/explain-equations", response_model=EquationResponse)
async def explain_equation(request: EquationRequest):
    """
    Explain a mathematical equation in plain English.
    
    Provides:
    - **readable**: Human-readable form of the equation
    - **meaning**: What the equation represents conceptually
    - **variables**: Explanation of each variable/symbol
    - **importance**: Why this equation matters in context
    """
    try:
        service = get_equation_service()
        result = await service.explain(
            equation=request.equation,
            context=request.context,
            format=request.format
        )
        
        return EquationResponse(
            readable=result.readable,
            meaning=result.meaning,
            variables=[
                VariableInfo(
                    symbol=v.symbol,
                    latex=v.latex,
                    description=v.description
                )
                for v in result.variables
            ],
            importance=result.importance,
            equation_type=result.equation_type
        )
        
    except Exception as e:
        logger.error(f"Equation explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Equation explanation failed: {str(e)}"
        )
