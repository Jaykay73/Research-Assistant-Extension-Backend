"""
AI Research Paper Helper - Keyword Extractor Service
Extracts key concepts, algorithms, datasets, and metrics from papers.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import optional NLP libraries
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    logger.warning("YAKE not available, using fallback keyword extraction")


@dataclass
class Concept:
    """Represents an extracted concept with explanation."""
    name: str
    category: str  # 'algorithm', 'dataset', 'metric', 'concept', 'method'
    description: str
    frequency: int


@dataclass
class KeypointsResult:
    """Contains all extracted keypoints."""
    contributions: List[str]
    datasets: List[str]
    metrics: List[str]
    concepts: List[Concept]
    algorithms: List[str]
    assumptions: List[str]


class KeywordExtractorService:
    """Service for extracting keywords and key concepts from papers."""
    
    # Common ML datasets
    KNOWN_DATASETS = {
        'imagenet', 'cifar-10', 'cifar-100', 'mnist', 'fashion-mnist',
        'coco', 'pascal voc', 'squad', 'glue', 'superglue',
        'wikitext', 'penn treebank', 'ptb', 'imdb', 'yelp',
        'librispeech', 'commonvoice', 'audioset',
        'celeba', 'lfw', 'vggface', 'ms-celeb',
        'kinetics', 'ucf101', 'hmdb51', 'youtube-8m',
        'babi', 'triviaqa', 'hotpotqa', 'naturalquestions',
        'wmt', 'iwslt', 'opus', 'europarl',
        'openwebtext', 'c4', 'the pile', 'laion',
        'shapenet', 'modelnet', 'scannet', 's3dis'
    }
    
    # Common ML metrics
    KNOWN_METRICS = {
        'accuracy', 'precision', 'recall', 'f1', 'f1-score', 'f-measure',
        'auc', 'roc-auc', 'auc-roc', 'auroc', 'auprc',
        'bleu', 'rouge', 'meteor', 'cider', 'spice',
        'perplexity', 'ppl', 'cross-entropy', 'nll',
        'mse', 'rmse', 'mae', 'mape',
        'iou', 'miou', 'dice', 'jaccard',
        'map', 'ap', 'ap50', 'ap75',
        'fid', 'inception score', 'is', 'lpips',
        'wer', 'cer', 'ter', 'mos',
        'hit@k', 'mrr', 'ndcg', 'map@k'
    }
    
    # Common ML algorithms/architectures
    KNOWN_ALGORITHMS = {
        'transformer', 'attention', 'self-attention', 'multi-head attention',
        'bert', 'gpt', 'gpt-2', 'gpt-3', 'gpt-4', 't5', 'bart',
        'vit', 'deit', 'swin', 'beit',
        'resnet', 'vgg', 'alexnet', 'inception', 'efficientnet',
        'lstm', 'gru', 'rnn', 'seq2seq',
        'gan', 'vae', 'ddpm', 'diffusion',
        'reinforcement learning', 'ppo', 'dqn', 'a3c', 'sac',
        'adam', 'sgd', 'adamw', 'rmsprop',
        'dropout', 'batch normalization', 'layer normalization',
        'cross-attention', 'masked language modeling', 'mlm',
        'contrastive learning', 'simclr', 'moco', 'clip',
        'yolo', 'faster r-cnn', 'mask r-cnn', 'detr'
    }
    
    # Contribution indicator phrases
    CONTRIBUTION_PATTERNS = [
        r'we propose',
        r'we introduce',
        r'we present',
        r'our (main |key )?contribution',
        r'our approach',
        r'our method',
        r'novel',
        r'first to',
        r'state-of-the-art',
        r'outperform',
        r'achieve .{0,30}improvement',
        r'improve .{0,30}over'
    ]
    
    def __init__(self):
        self.yake_extractor = None
        if YAKE_AVAILABLE:
            self.yake_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,  # max n-gram size
                dedupLim=0.8,
                top=30,
                features=None
            )
    
    async def extract(
        self,
        content: str,
        abstract: Optional[str] = None,
        title: Optional[str] = None
    ) -> KeypointsResult:
        """
        Extract key points from a research paper.
        
        Returns:
            KeypointsResult with contributions, datasets, metrics, concepts, etc.
        """
        # Combine relevant text
        full_text = ""
        if title:
            full_text += title + " "
        if abstract:
            full_text += abstract + " "
        full_text += content
        
        # Normalize text
        text_lower = full_text.lower()
        
        # Extract each category
        contributions = self._extract_contributions(full_text)
        datasets = self._extract_datasets(text_lower)
        metrics = self._extract_metrics(text_lower)
        algorithms = self._extract_algorithms(text_lower)
        concepts = self._extract_concepts(full_text)
        assumptions = self._extract_assumptions(full_text)
        
        return KeypointsResult(
            contributions=contributions,
            datasets=datasets,
            metrics=metrics,
            concepts=concepts,
            algorithms=algorithms,
            assumptions=assumptions
        )
    
    def _extract_contributions(self, text: str) -> List[str]:
        """Extract novel contributions from the text."""
        contributions = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            for pattern in self.CONTRIBUTION_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Clean and add
                    clean = sentence.strip()
                    if len(clean) > 30 and len(clean) < 500:
                        contributions.append(clean)
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for c in contributions:
            c_lower = c.lower()
            if c_lower not in seen:
                seen.add(c_lower)
                unique.append(c)
        
        return unique[:6]  # Top 6 contributions
    
    def _extract_datasets(self, text_lower: str) -> List[str]:
        """Extract mentioned datasets."""
        found = []
        
        for dataset in self.KNOWN_DATASETS:
            # Look for dataset name with word boundaries
            pattern = r'\b' + re.escape(dataset) + r'\b'
            if re.search(pattern, text_lower):
                found.append(dataset.upper() if len(dataset) <= 5 else dataset.title())
        
        return list(set(found))
    
    def _extract_metrics(self, text_lower: str) -> List[str]:
        """Extract mentioned evaluation metrics."""
        found = []
        
        for metric in self.KNOWN_METRICS:
            pattern = r'\b' + re.escape(metric) + r'\b'
            if re.search(pattern, text_lower):
                found.append(metric.upper() if len(metric) <= 4 else metric.title())
        
        return list(set(found))
    
    def _extract_algorithms(self, text_lower: str) -> List[str]:
        """Extract mentioned algorithms and architectures."""
        found = []
        
        for algo in self.KNOWN_ALGORITHMS:
            pattern = r'\b' + re.escape(algo) + r'\b'
            if re.search(pattern, text_lower):
                found.append(algo.title())
        
        return list(set(found))
    
    def _extract_concepts(self, text: str) -> List[Concept]:
        """Extract key concepts using YAKE or fallback."""
        if self.yake_extractor:
            return self._extract_with_yake(text)
        else:
            return self._extract_fallback(text)
    
    def _extract_with_yake(self, text: str) -> List[Concept]:
        """Extract concepts using YAKE keyword extractor."""
        keywords = self.yake_extractor.extract_keywords(text)
        
        concepts = []
        for keyword, score in keywords[:15]:  # Top 15
            # Skip if it's a known dataset, metric, or algorithm
            kw_lower = keyword.lower()
            if kw_lower in self.KNOWN_DATASETS or kw_lower in self.KNOWN_METRICS:
                continue
            if kw_lower in self.KNOWN_ALGORITHMS:
                continue
            
            # Categorize
            category = self._categorize_keyword(keyword, text)
            
            concepts.append(Concept(
                name=keyword,
                category=category,
                description="",  # Would need LLM for descriptions
                frequency=text.lower().count(kw_lower)
            ))
        
        return concepts[:10]
    
    def _extract_fallback(self, text: str) -> List[Concept]:
        """Simple frequency-based keyword extraction."""
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'were', 'being', 'this', 'that', 'with',
            'from', 'they', 'will', 'would', 'there', 'their', 'what',
            'about', 'which', 'when', 'make', 'like', 'into', 'just',
            'over', 'such', 'than', 'then', 'also', 'more', 'these',
            'some', 'them', 'each', 'other'
        }
        
        words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        freq = Counter(words)
        
        concepts = []
        for word, count in freq.most_common(20):
            if count >= 3:  # Minimum frequency
                concepts.append(Concept(
                    name=word.title(),
                    category='concept',
                    description="",
                    frequency=count
                ))
        
        return concepts[:10]
    
    def _categorize_keyword(self, keyword: str, text: str) -> str:
        """Categorize a keyword based on context."""
        kw_lower = keyword.lower()
        
        # Check patterns in surrounding text
        pattern = r'.{0,50}' + re.escape(kw_lower) + r'.{0,50}'
        matches = re.findall(pattern, text.lower())
        context = ' '.join(matches)
        
        if any(w in context for w in ['algorithm', 'method', 'approach', 'technique']):
            return 'method'
        elif any(w in context for w in ['loss', 'objective', 'function', 'equation']):
            return 'concept'
        elif any(w in context for w in ['layer', 'network', 'model', 'architecture']):
            return 'architecture'
        
        return 'concept'
    
    def _extract_assumptions(self, text: str) -> List[str]:
        """Extract assumptions made in the paper."""
        assumptions = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        assumption_patterns = [
            r'we assume',
            r'assuming that',
            r'under the assumption',
            r'it is assumed',
            r'we consider .{0,30} to be'
        ]
        
        for sentence in sentences:
            for pattern in assumption_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    clean = sentence.strip()
                    if len(clean) > 20 and len(clean) < 300:
                        assumptions.append(clean)
                    break
        
        return assumptions[:5]


# Singleton instance
_keyword_service = None

def get_keyword_service() -> KeywordExtractorService:
    """Get the singleton keyword extractor service instance."""
    global _keyword_service
    if _keyword_service is None:
        _keyword_service = KeywordExtractorService()
    return _keyword_service
