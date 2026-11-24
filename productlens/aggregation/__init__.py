"""
ProductLens 3.0: Stage 4 Aggregation & Product Scorecard Engine.
"""

from productlens.aggregation.schemas import EvidenceSnippet, FeatureScorecard, ProductReport
from productlens.aggregation.normalizer import AspectNormalizer
from productlens.aggregation.scoring import FeatureScorer
from productlens.aggregation.evidence import EvidenceSelector
from productlens.aggregation.engine import Stage4AggregationEngine

__all__ = [
    "EvidenceSnippet",
    "FeatureScorecard",
    "ProductReport",
    "AspectNormalizer",
    "FeatureScorer",
    "EvidenceSelector",
    "Stage4AggregationEngine",
]
