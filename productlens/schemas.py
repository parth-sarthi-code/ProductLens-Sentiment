from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AspectMention(BaseModel):
    """An extracted candidate aspect term with its character offsets."""
    aspect: str = Field(..., description="The aspect term (e.g. 'battery life', 'screen')")
    start_char: int = Field(..., description="Zero-indexed start character offset in the review text")
    end_char: int = Field(..., description="Zero-indexed end character offset in the review text")
    confidence: float = Field(default=1.0, description="Extraction confidence score (0.0 to 1.0)")


class SentimentPrediction(BaseModel):
    """Aspect-targeted sentiment classification result."""
    sentiment: str = Field(..., description="'positive', 'neutral', or 'negative'")
    confidence: float = Field(..., description="Softmax confidence score for the winning class")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities {positive, neutral, negative}")
    routed_expert: str = Field(..., description="Specialist model key ('tech', 'home', 'apparel', 'fallback')")
    routing_reason: str = Field(..., description="Rationale for expert selection")


class AspectSentimentResult(BaseModel):
    """Coupled aspect mention, self-extracted evidence clause, and predicted sentiment."""
    aspect: str
    start_char: int
    end_char: int
    evidence_snippet: str = Field(default="", description="Self-extracted sentence or clause providing evidence for this aspect")
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    routed_expert: str
    routing_reason: str


class ReviewAnalysisReport(BaseModel):
    """Complete end-to-end sentiment analysis report for a review."""
    review_text: str
    category: Optional[str] = None
    aspect_count: int
    results: List[AspectSentimentResult]
    summary_sentiment: str = Field(..., description="Aggregated sentiment across all aspects")
    execution_time_ms: float = 0.0


class BatchAnalysisReport(BaseModel):
    """Batch sentiment analysis report across multiple reviews."""
    total_reviews: int
    total_aspects: int
    reports: List[ReviewAnalysisReport]
    throughput_reviews_per_sec: float = 0.0
