"""
Gold Set Schema, Partitioning, and Hard Case definitions.
Supports English and Hinglish (Latin-script Hindi + English) reviews with negation, sarcasm, and implicit aspects.
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad


class GoldReview(BaseModel):
    """A curated, gold-standard review for benchmarking."""
    review_id: str
    product_id: str
    text: str
    vertical: VerticalDomain
    language: str = Field(default="en", description="'en' for English or 'hi-en' for Hinglish")
    is_hard_case: bool = False
    hard_case_type: Optional[str] = Field(
        default=None,
        description="negation, sarcasm, implicit_aspect, cross_clause_contrast, etc."
    )
    quads: List[AspectQuad] = Field(default_factory=list)


class GoldSetManifest(BaseModel):
    """Manifest summary for the gold set."""
    total_reviews: int
    english_reviews: int
    hinglish_reviews: int
    hard_cases: int
    total_quads: int
    products_count: int
