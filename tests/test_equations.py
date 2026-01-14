"""
Tests for the equation explainer service.
"""

import pytest
from services.equation_explainer import get_equation_service


@pytest.fixture
def equation_service():
    return get_equation_service()


@pytest.mark.asyncio
async def test_explain_simple_equation(equation_service):
    """Test explanation of a simple loss function."""
    result = await equation_service.explain(
        equation=r"\mathcal{L} = -\sum_i y_i \log(\hat{y}_i)",
        context="This is the cross-entropy loss function.",
        format="latex"
    )
    
    assert result is not None
    assert result.readable != ""
    assert result.meaning != ""
    assert result.equation_type == "loss"


@pytest.mark.asyncio
async def test_explain_gradient_equation(equation_service):
    """Test explanation of a gradient equation."""
    result = await equation_service.explain(
        equation=r"\nabla_\theta J(\theta)",
        context="Computing the gradient of the cost function.",
        format="latex"
    )
    
    assert result.equation_type == "gradient"
    assert len(result.variables) > 0


@pytest.mark.asyncio
async def test_variable_extraction(equation_service):
    """Test that variables are properly extracted."""
    result = await equation_service.explain(
        equation=r"y = Wx + b",
        context="Linear transformation.",
        format="latex"
    )
    
    variable_symbols = [v.symbol for v in result.variables]
    assert 'W' in variable_symbols or 'w' in variable_symbols.lower()
    assert 'x' in variable_symbols
    assert 'b' in variable_symbols


@pytest.mark.asyncio
async def test_readable_conversion(equation_service):
    """Test LaTeX to readable conversion."""
    result = await equation_service.explain(
        equation=r"\alpha + \beta = \gamma",
        format="latex"
    )
    
    # Should contain Greek letters
    assert 'α' in result.readable or 'alpha' in result.readable.lower()
