"""
AI Research Paper Helper - Summarization Service
Multi-level summarization for research papers using LLM APIs.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass
import httpx

from config import settings, get_llm_config
from ml.text_processor import get_text_processor

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Contains all summary levels."""
    tldr: List[str]           # 5-6 bullet points
    technical: str            # For ML researchers
    beginner: str             # Plain English
    section_summaries: dict   # Per-section summaries


class SummarizerService:
    """Service for generating multi-level summaries using external LLM APIs."""
    
    def __init__(self):
        self.text_processor = get_text_processor()
    
    async def summarize(
        self,
        content: str,
        abstract: Optional[str] = None,
        title: Optional[str] = None,
        page_type: str = "arxiv"
    ) -> SummaryResult:
        """
        Generate multi-level summaries for a document.
        
        Returns:
            SummaryResult with TL;DR bullets, technical summary, and beginner explanation
        
        Raises:
            ValueError: If no LLM API is configured
        """
        llm_config = get_llm_config()
        if not llm_config:
            raise ValueError(
                "No LLM API configured. Please set AIHELPER_OPENROUTER_API_KEY or "
                "AIHELPER_GROQ_API_KEY in your environment or .env file."
            )
        
        # Clean and prepare text
        clean_content = self.text_processor.clean_text(content)
        clean_abstract = self.text_processor.clean_text(abstract) if abstract else ""
        
        # Prioritize abstract for initial understanding
        source_text = clean_abstract or self.text_processor.truncate_to_limit(clean_content, 1000)
        
        return await self._summarize_with_llm(source_text, clean_content, title)
    
    async def _summarize_with_llm(
        self,
        abstract_text: str,
        full_content: str,
        title: Optional[str]
    ) -> SummaryResult:
        """Generate summaries using external LLM API."""
        config = get_llm_config()
        
        # Prepare context
        context = f"Title: {title}\n\n" if title else ""
        context += f"Abstract/Introduction:\n{abstract_text}\n\n"
        context += f"Content excerpt:\n{self.text_processor.truncate_to_limit(full_content, 1500)}"
        
        prompt = f"""Analyze this research paper and provide:

1. TL;DR (exactly 5-6 bullet points, each starting with "•")
2. Technical Summary (2-3 paragraphs for ML researchers, use proper terminology)
3. Beginner-Friendly Explanation (2-3 paragraphs in plain English, no jargon)

Paper content:
{context}

Format your response EXACTLY as:
## TL;DR
• bullet 1
• bullet 2
...

## Technical Summary
[technical summary here]

## Beginner-Friendly
[beginner explanation here]"""

        try:
            response = await self._call_llm(prompt, config)
            return self._parse_llm_response(response)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            raise RuntimeError(f"Summarization failed: {e}") from e
    
    async def _call_llm(self, prompt: str, config: dict) -> str:
        """Call external LLM API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            if config['provider'] == 'openrouter':
                headers["HTTP-Referer"] = "https://ai-research-helper.local"
                headers["X-Title"] = "AI Research Paper Helper"
            
            payload = {
                "model": config['model'],
                "messages": [
                    {"role": "system", "content": "You are an expert ML researcher helping to summarize research papers. Be accurate and preserve technical correctness."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            response = await client.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data['choices'][0]['message']['content']
    
    def _parse_llm_response(self, response: str) -> SummaryResult:
        """Parse structured LLM response into SummaryResult."""
        tldr = []
        technical = ""
        beginner = ""
        
        # Extract TL;DR bullets
        if "## TL;DR" in response:
            tldr_section = response.split("## TL;DR")[1]
            if "## Technical" in tldr_section:
                tldr_section = tldr_section.split("## Technical")[0]
            
            for line in tldr_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    tldr.append(line.lstrip("•-* "))
        
        # Extract technical summary
        if "## Technical Summary" in response:
            tech_section = response.split("## Technical Summary")[1]
            if "## Beginner" in tech_section:
                tech_section = tech_section.split("## Beginner")[0]
            technical = tech_section.strip()
        
        # Extract beginner explanation
        if "## Beginner" in response:
            beginner_section = response.split("## Beginner")[1]
            beginner = beginner_section.strip()
        
        # Fallbacks
        if not tldr:
            tldr = ["Summary generation failed. Please try again."]
        if not technical:
            technical = response[:500] if response else "Technical summary unavailable."
        if not beginner:
            beginner = self._simplify_text(technical)
        
        return SummaryResult(
            tldr=tldr[:6],  # Max 6 bullets
            technical=technical,
            beginner=beginner,
            section_summaries={}
        )
    
    def _extract_bullets(self, text: str) -> List[str]:
        """Extract key points as bullet points from summary text."""
        sentences = self.text_processor.extract_sentences(text)
        
        # Take first 5-6 sentences as bullets
        bullets = []
        for sentence in sentences[:6]:
            if len(sentence) > 20:  # Minimum meaningful length
                bullets.append(sentence)
        
        if not bullets:
            bullets = [text[:200]]
        
        return bullets
    
    def _simplify_text(self, text: str) -> str:
        """Create a simplified version of text for non-experts."""
        # Basic simplification: remove complex terms patterns
        simplified = text
        
        # Replace common technical phrases with simpler ones
        replacements = {
            "state-of-the-art": "best current",
            "novel": "new",
            "propose": "suggest",
            "demonstrate": "show",
            "utilize": "use",
            "methodology": "method",
            "aforementioned": "mentioned earlier",
            "subsequently": "then",
            "leveraging": "using"
        }
        
        for term, replacement in replacements.items():
            simplified = simplified.replace(term, replacement)
            simplified = simplified.replace(term.capitalize(), replacement.capitalize())
        
        return simplified


# Singleton instance
_summarizer_service = None

def get_summarizer_service() -> SummarizerService:
    """Get the singleton summarizer service instance."""
    global _summarizer_service
    if _summarizer_service is None:
        _summarizer_service = SummarizerService()
    return _summarizer_service
