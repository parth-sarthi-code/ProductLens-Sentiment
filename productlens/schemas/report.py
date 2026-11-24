"""
Report and scorecard schemas for ProductLens 3.0.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad


class FeatureScorecard(BaseModel):
    """Aggregated scorecard for a single canonical feature/category."""
    category: str
    aspect_cluster: str
    mention_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    mixed_count: int = 0

    sentiment_score: float = Field(
        default=0.0,
        description="(positive - negative) / total with mixed counted 0.5"
    )
    negativity_rate: float = Field(
        default=0.0,
        description="(negative + 0.5 * mixed) / total"
    )
    impact_rank_score: float = Field(
        default=0.0,
        description="mention_count * negativity_rate to surface high-volume defects"
    )

    top_positive_evidence: List[str] = Field(default_factory=list)
    top_negative_evidence: List[str] = Field(default_factory=list)
    top_mixed_evidence: List[str] = Field(default_factory=list)


class ReviewAnalysisReport(BaseModel):
    """Analysis result for a single review."""
    review_id: Optional[str] = None
    review_text: str
    vertical: VerticalDomain
    quads: List[AspectQuad] = Field(default_factory=list)
    overall_sentiment: Sentiment
    aspect_count: int = 0
    has_mixed: bool = False
    execution_time_ms: float = 0.0


class ProductReport(BaseModel):
    """Roll-up scorecard and actionable report for a product."""
    product_id: str
    vertical: VerticalDomain
    total_reviews: int = 0
    total_quads: int = 0
    overall_sentiment_score: float = 0.0
    features: List[FeatureScorecard] = Field(default_factory=list)
    top_defects: List[FeatureScorecard] = Field(default_factory=list)
    top_strengths: List[FeatureScorecard] = Field(default_factory=list)
    execution_time_sec: float = 0.0
