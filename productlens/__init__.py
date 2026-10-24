"""
ProductLens Sentiment Analysis Platform.

Production-grade Aspect-Based Sentiment Analysis (ABSA) system powered by a
Hierarchical Mixture of Experts (MoE) with domain-specialist DeBERTa-v3 backbones.
"""

__version__ = "2.0.0"

from productlens.schemas import (
    AspectMention,
    SentimentPrediction,
    AspectSentimentResult,
    ReviewAnalysisReport,
    BatchAnalysisReport,
)
from productlens.sentiment.router import MoERouter
from productlens.sentiment.pipeline import ProductLensMoEPipeline
from productlens.aspects.extractor import AspectExtractor
from productlens.engine import ProductLensEngine
from productlens.utils import find_aspect_span

__all__ = [
    "AspectMention",
    "SentimentPrediction",
    "AspectSentimentResult",
    "ReviewAnalysisReport",
    "BatchAnalysisReport",
    "MoERouter",
    "ProductLensMoEPipeline",
    "AspectExtractor",
    "ProductLensEngine",
    "find_aspect_span",
]
