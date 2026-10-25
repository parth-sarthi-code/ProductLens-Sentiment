"""
ProductLens Sentiment Analysis Platform.

Production-grade Aspect-Based Sentiment Analysis (ABSA) system powered by an
aspect-targeted, calibrated DeBERTa-v3 architecture.
"""

__version__ = "2.2.0"

from productlens.schemas import (
    AspectMention,
    SentimentPrediction,
    AspectSentimentResult,
    ReviewAnalysisReport,
    BatchAnalysisReport,
)
from productlens.sentiment.pipeline import ProductLensSentimentPipeline, ProductLensMoEPipeline
from productlens.aspects.extractor import AspectExtractor
from productlens.engine import ProductLensEngine
from productlens.utils import find_aspect_span

__all__ = [
    "AspectMention",
    "SentimentPrediction",
    "AspectSentimentResult",
    "ReviewAnalysisReport",
    "BatchAnalysisReport",
    "ProductLensSentimentPipeline",
    "ProductLensMoEPipeline",
    "AspectExtractor",
    "ProductLensEngine",
    "find_aspect_span",
]
