"""
AI Research Paper Helper - Equation Explainer Service
Parses and explains LaTeX equations in plain English.
"""

import re
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass
import httpx

from config import settings, get_llm_config

logger = logging.getLogger(__name__)


@dataclass
class VariableExplanation:
    """Explanation of a variable in an equation."""
    symbol: str
    latex: str
    description: str


@dataclass
class EquationExplanation:
    """Complete explanation of an equation."""
    readable: str              # Human-readable form
    meaning: str               # What it represents
    variables: List[VariableExplanation]
    importance: str            # Why it matters
    equation_type: str         # loss, gradient, probability, etc.


class EquationExplainerService:
    """Service for explaining mathematical equations."""
    
    # LaTeX symbol mappings
    SYMBOL_MAP = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\mu': 'μ',
        r'\sigma': 'σ', r'\phi': 'φ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Sigma': 'Σ', r'\Pi': 'Π', r'\Omega': 'Ω',
        r'\sum': 'Σ', r'\prod': 'Π', r'\int': '∫',
        r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
        r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
        r'\in': '∈', r'\forall': '∀', r'\exists': '∃',
        r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
        r'\cdot': '·', r'\times': '×', r'\pm': '±'
    }
    
    # Common ML equation patterns
    EQUATION_PATTERNS = {
        'loss': [r'\\mathcal\{L\}', r'\\text\{loss\}', r'loss', r'J\(', r'L\('],
        'gradient': [r'\\nabla', r'\\partial', r'\\frac\{\\partial'],
        'probability': [r'\\mathbb\{P\}', r'\\Pr', r'p\(', r'P\('],
        'expectation': [r'\\mathbb\{E\}', r'\\E\[', r'E\['],
        'softmax': [r'softmax', r'\\text\{softmax\}', r'\\frac\{e\^'],
        'attention': [r'attention', r'Attention', r'\\text\{Attention\}'],
        'norm': [r'\|.*?\|', r'\\|.*?\\|', r'\\lVert', r'\\rVert']
    }
    
    # Common variable meanings in ML
    COMMON_VARIABLES = {
        'x': 'input data or features',
        'y': 'target or label',
        'w': 'weight parameter',
        'W': 'weight matrix',
        'b': 'bias term',
        'θ': 'model parameters',
        'α': 'learning rate or scaling factor',
        'β': 'momentum coefficient or scaling factor',
        'λ': 'regularization coefficient',
        'σ': 'sigmoid function or standard deviation',
        'μ': 'mean',
        'ε': 'small constant for numerical stability',
        'η': 'learning rate',
        'L': 'loss function',
        'J': 'cost function',
        'n': 'number of samples',
        'm': 'batch size or number of features',
        'h': 'hidden state or layer',
        'z': 'latent variable or pre-activation',
        'a': 'activation',
        'p': 'probability',
        'q': 'approximate distribution',
        'K': 'number of classes',
        'T': 'temperature or time steps',
        'd': 'dimension'
    }
    
    async def explain(
        self,
        equation: str,
        context: Optional[str] = None,
        format: str = 'latex'
    ) -> EquationExplanation:
        """
        Explain an equation in plain English.
        
        Args:
            equation: LaTeX or MathML equation string
            context: Surrounding text for better understanding
            format: 'latex' or 'mathml'
        
        Returns:
            EquationExplanation with all components
        """
        # Clean the equation
        clean_eq = self._clean_equation(equation)
        
        # Detect equation type
        eq_type = self._detect_type(clean_eq)
        
        # Convert to readable form
        readable = self._to_readable(clean_eq)
        
        # Extract variables
        variables = self._extract_variables(clean_eq, context)
        
        # Generate explanations
        if settings.api_mode == 'api' and get_llm_config():
            return await self._explain_with_llm(clean_eq, context, eq_type, readable, variables)
        else:
            return self._explain_local(clean_eq, eq_type, readable, variables)
    
    def _clean_equation(self, equation: str) -> str:
        """Clean and normalize LaTeX equation."""
        # Remove display mode markers
        clean = equation.strip()
        clean = re.sub(r'\\\[|\\\]', '', clean)
        clean = re.sub(r'\$\$?', '', clean)
        clean = re.sub(r'\\begin\{[^}]+\}|\\end\{[^}]+\}', '', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
    
    def _detect_type(self, equation: str) -> str:
        """Detect the type of equation."""
        eq_lower = equation.lower()
        
        for eq_type, patterns in self.EQUATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, equation, re.IGNORECASE):
                    return eq_type
        
        # Check for common structures
        if '=' in equation:
            if re.search(r'\\frac', equation):
                return 'definition'
            return 'equation'
        elif re.search(r'[<>≤≥]', equation):
            return 'inequality'
        
        return 'expression'
    
    def _to_readable(self, equation: str) -> str:
        """Convert LaTeX to human-readable form."""
        readable = equation
        
        # Replace symbols
        for latex, symbol in self.SYMBOL_MAP.items():
            readable = readable.replace(latex, symbol)
        
        # Handle fractions
        readable = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', readable)
        
        # Handle superscripts/subscripts
        readable = re.sub(r'\^{([^}]+)}', r'^(\1)', readable)
        readable = re.sub(r'_{([^}]+)}', r'_(\1)', readable)
        readable = re.sub(r'\^(\w)', r'^(\1)', readable)
        readable = re.sub(r'_(\w)', r'_(\1)', readable)
        
        # Handle text
        readable = re.sub(r'\\text\{([^}]+)\}', r'\1', readable)
        readable = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', readable)
        readable = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', readable)
        
        # Handle sqrt
        readable = re.sub(r'\\sqrt\{([^}]+)\}', r'√(\1)', readable)
        
        # Clean remaining latex commands
        readable = re.sub(r'\\[a-zA-Z]+', '', readable)
        readable = re.sub(r'[{}]', '', readable)
        readable = re.sub(r'\s+', ' ', readable)
        
        return readable.strip()
    
    def _extract_variables(
        self,
        equation: str,
        context: Optional[str]
    ) -> List[VariableExplanation]:
        """Extract and explain variables from the equation."""
        variables = []
        found = set()
        
        # Find Greek letters
        for latex, symbol in self.SYMBOL_MAP.items():
            if latex in equation and symbol not in found:
                desc = self.COMMON_VARIABLES.get(symbol, "parameter")
                variables.append(VariableExplanation(
                    symbol=symbol,
                    latex=latex,
                    description=desc
                ))
                found.add(symbol)
        
        # Find Latin letters (single letters)
        latin_matches = re.findall(r'(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z])', equation)
        for letter in latin_matches:
            if letter not in found and letter not in ['d', 'e', 'i', 'f', 'g']:
                desc = self.COMMON_VARIABLES.get(letter, "variable")
                variables.append(VariableExplanation(
                    symbol=letter,
                    latex=letter,
                    description=desc
                ))
                found.add(letter)
        
        return variables[:10]  # Limit to 10 most important
    
    def _explain_local(
        self,
        equation: str,
        eq_type: str,
        readable: str,
        variables: List[VariableExplanation]
    ) -> EquationExplanation:
        """Generate explanation without LLM."""
        # Type-specific explanations
        meaning_templates = {
            'loss': "This is a loss function that measures the error between predictions and actual values.",
            'gradient': "This computes the gradient (rate of change) of a function with respect to its parameters.",
            'probability': "This represents a probability distribution or conditional probability.",
            'expectation': "This calculates the expected value (average) over a probability distribution.",
            'softmax': "This applies the softmax function to convert values into probabilities that sum to 1.",
            'attention': "This computes attention weights to determine how much focus to give to different parts of the input.",
            'norm': "This calculates a norm (magnitude) of a vector or matrix.",
            'definition': "This defines a relationship or function between variables.",
            'equation': "This establishes equality between two mathematical expressions.",
            'inequality': "This describes a constraint or bound on values.",
            'expression': "This is a mathematical expression involving the given variables."
        }
        
        importance_templates = {
            'loss': "Loss functions are crucial for training ML models - they define what the model optimizes for.",
            'gradient': "Gradients enable backpropagation, allowing the model to learn by adjusting parameters.",
            'probability': "Probability formulations help the model reason about uncertainty and make predictions.",
            'expectation': "Expected values help in optimization and understanding average model behavior.",
            'softmax': "Softmax is fundamental for classification tasks, converting logits to class probabilities.",
            'attention': "Attention mechanisms allow models to focus on relevant parts of input, key for transformers.",
            'norm': "Norms help measure and control the magnitude of values, important for regularization."
        }
        
        meaning = meaning_templates.get(eq_type, meaning_templates['expression'])
        importance = importance_templates.get(eq_type, "This equation contributes to the mathematical foundation of the method.")
        
        return EquationExplanation(
            readable=readable,
            meaning=meaning,
            variables=variables,
            importance=importance,
            equation_type=eq_type
        )
    
    async def _explain_with_llm(
        self,
        equation: str,
        context: Optional[str],
        eq_type: str,
        readable: str,
        variables: List[VariableExplanation]
    ) -> EquationExplanation:
        """Generate explanation using LLM."""
        config = get_llm_config()
        
        var_list = ", ".join([f"{v.symbol} ({v.description})" for v in variables])
        
        prompt = f"""Explain this mathematical equation from a research paper:

LaTeX: {equation}
Readable form: {readable}
Equation type: {eq_type}
Variables: {var_list}
{f"Context: {context[:500]}" if context else ""}

Please provide:
1. A clear explanation of what this equation REPRESENTS (2-3 sentences)
2. Why this equation MATTERS in the context of ML/research (1-2 sentences)

Keep explanations concise and accurate. Format as:
MEANING: [explanation]
IMPORTANCE: [why it matters]"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                            {"role": "system", "content": "You are an expert ML researcher explaining equations. Be accurate and concise."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 400
                    }
                )
                response.raise_for_status()
                
                result = response.json()['choices'][0]['message']['content']
                
                # Parse response
                meaning = ""
                importance = ""
                
                if "MEANING:" in result:
                    meaning = result.split("MEANING:")[1].split("IMPORTANCE:")[0].strip()
                if "IMPORTANCE:" in result:
                    importance = result.split("IMPORTANCE:")[1].strip()
                
                return EquationExplanation(
                    readable=readable,
                    meaning=meaning or self._explain_local(equation, eq_type, readable, variables).meaning,
                    variables=variables,
                    importance=importance or "This equation is part of the paper's mathematical framework.",
                    equation_type=eq_type
                )
                
        except Exception as e:
            logger.error(f"LLM equation explanation failed: {e}")
            return self._explain_local(equation, eq_type, readable, variables)


# Singleton instance
_equation_service = None

def get_equation_service() -> EquationExplainerService:
    """Get the singleton equation explainer service instance."""
    global _equation_service
    if _equation_service is None:
        _equation_service = EquationExplainerService()
    return _equation_service
