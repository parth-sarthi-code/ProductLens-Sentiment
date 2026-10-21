"""
ProductLens Sentiment Analysis Platform.

Production-grade Aspect-Based Sentiment Analysis (ABSA) system powered by a
Hierarchical Mixture of Experts (MoE) with domain-specialist RoBERTa backbones.
"""

__version__ = "1.0.0"

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
]
