"""
AI Research Paper Helper - Text Processor
Handles text cleaning, section segmentation, and chunking.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    id: str
    text: str
    section: Optional[str]
    start_idx: int
    end_idx: int
    token_count: int


@dataclass
class Section:
    """Represents a document section."""
    title: str
    content: str
    section_type: str  # 'abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'other'
    level: int


class TextProcessor:
    """Handles all text processing operations."""
    
    # Section title patterns for academic papers
    SECTION_PATTERNS = {
        'abstract': r'^abstract\s*$',
        'introduction': r'^(1\.?\s*)?introduction\s*$',
        'related_work': r'^(2\.?\s*)?(related\s+work|background|literature\s+review)\s*$',
        'methods': r'^(3\.?\s*)?(method(s|ology)?|approach|model)\s*$',
        'experiments': r'^(4\.?\s*)?(experiment(s)?|evaluation|results)\s*$',
        'results': r'^(5\.?\s*)?(result(s)?|finding(s)?)\s*$',
        'discussion': r'^(6\.?\s*)?discussion\s*$',
        'conclusion': r'^(7\.?\s*)?(conclusion(s)?|summary)\s*$',
        'references': r'^references?\s*$',
        'appendix': r'^appendix\s*'
    }
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()
    
    def extract_sections(self, content: str, sections_data: List[Dict] = None) -> List[Section]:
        """Extract sections from document content."""
        if sections_data:
            # Use pre-extracted section data
            return [
                Section(
                    title=s.get('title', ''),
                    content=self.clean_text(s.get('content', '')),
                    section_type=self._classify_section(s.get('title', '')),
                    level=s.get('level', 2)
                )
                for s in sections_data
            ]
        
        # Try to parse sections from raw content
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a section header
            section_type = self._classify_section(line)
            if section_type != 'other' and len(line) < 100:
                # Save previous section
                if current_section is not None:
                    sections.append(Section(
                        title=current_section,
                        content=self.clean_text(' '.join(current_content)),
                        section_type=self._classify_section(current_section),
                        level=2
                    ))
                
                current_section = line
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section is not None:
            sections.append(Section(
                title=current_section,
                content=self.clean_text(' '.join(current_content)),
                section_type=self._classify_section(current_section),
                level=2
            ))
        
        return sections
    
    def _classify_section(self, title: str) -> str:
        """Classify a section based on its title."""
        title_lower = title.lower().strip()
        
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.match(pattern, title_lower, re.IGNORECASE):
                return section_type
        
        return 'other'
    
    def chunk_text(
        self,
        text: str,
        section: Optional[str] = None,
        chunk_id_prefix: str = "chunk"
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Uses word-based chunking to respect word boundaries.
        Overlap ensures context is preserved across chunks.
        """
        if not text:
            return []
        
        text = self.clean_text(text)
        words = text.split()
        
        if not words:
            return []
        
        chunks = []
        chunk_idx = 0
        word_idx = 0
        
        # Approximate tokens as words (rough estimate)
        words_per_chunk = self.chunk_size
        overlap_words = self.chunk_overlap
        
        while word_idx < len(words):
            # Get chunk words
            end_idx = min(word_idx + words_per_chunk, len(words))
            chunk_words = words[word_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions (approximate)
            start_char = len(' '.join(words[:word_idx])) + (1 if word_idx > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            chunks.append(TextChunk(
                id=f"{chunk_id_prefix}_{chunk_idx}",
                text=chunk_text,
                section=section,
                start_idx=start_char,
                end_idx=end_char,
                token_count=len(chunk_words)
            ))
            
            chunk_idx += 1
            word_idx = end_idx - overlap_words
            
            # Prevent infinite loop
            if word_idx >= len(words) or end_idx >= len(words):
                break
            if word_idx <= end_idx - words_per_chunk:
                word_idx = end_idx
        
        return chunks
    
    def chunk_document(
        self,
        content: str,
        abstract: Optional[str] = None,
        sections: List[Dict] = None
    ) -> List[TextChunk]:
        """
        Chunk an entire document, respecting section boundaries.
        
        Strategy:
        1. Abstract gets its own chunk(s)
        2. Each section is chunked separately
        3. If no sections, chunk the entire content
        """
        all_chunks = []
        chunk_counter = 0
        
        # Handle abstract separately
        if abstract:
            abstract_chunks = self.chunk_text(
                abstract,
                section="abstract",
                chunk_id_prefix=f"chunk_{chunk_counter}"
            )
            all_chunks.extend(abstract_chunks)
            chunk_counter += len(abstract_chunks)
        
        # Process sections
        if sections:
            parsed_sections = self.extract_sections(content, sections)
            
            for section in parsed_sections:
                if section.section_type == 'references':
                    continue  # Skip references
                
                section_chunks = self.chunk_text(
                    section.content,
                    section=section.title,
                    chunk_id_prefix=f"chunk_{chunk_counter}"
                )
                all_chunks.extend(section_chunks)
                chunk_counter += len(section_chunks)
        else:
            # Chunk entire content
            content_chunks = self.chunk_text(
                content,
                section=None,
                chunk_id_prefix=f"chunk_{chunk_counter}"
            )
            all_chunks.extend(content_chunks)
        
        return all_chunks
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def truncate_to_limit(self, text: str, max_words: int = 500) -> str:
        """Truncate text to a maximum number of words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + '...'


# Singleton instance
_text_processor = None

def get_text_processor() -> TextProcessor:
    """Get the singleton text processor instance."""
    global _text_processor
    if _text_processor is None:
        _text_processor = TextProcessor()
    return _text_processor
