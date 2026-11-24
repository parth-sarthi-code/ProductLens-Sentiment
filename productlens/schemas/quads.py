"""
Quad definitions and compact serialization for ProductLens 3.0.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from productlens.schemas.enums import Sentiment, VerticalDomain, normalize_category


class RawCompactQuad(BaseModel):
    """
    Unvalidated quad extracted from compact format or LLM generation.
    Format: aspect|category|sentiment|confidence|evidence
    """
    aspect: Optional[str] = None
    category: str
    sentiment: Sentiment
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str

    @classmethod
    def from_compact_line(cls, line: str) -> "RawCompactQuad":
        parts = line.split("|", 4)
        if len(parts) < 5:
            raise ValueError(f"Malformed compact quad line: '{line}'. Expected 5 pipe-delimited fields.")

        raw_aspect = parts[0].strip()
        aspect = None if raw_aspect.lower() in {"", "none", "null"} else raw_aspect
        category = parts[1].strip()
        sentiment = Sentiment.from_str(parts[2].strip())
        try:
            confidence = float(parts[3].strip())
        except ValueError:
            confidence = 1.0
        evidence = parts[4].strip()

        return cls(
            aspect=aspect,
            category=category,
            sentiment=sentiment,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence
        )

    def to_compact_line(self) -> str:
        asp_str = self.aspect if self.aspect is not None else "null"
        return f"{asp_str}|{self.category}|{self.sentiment.value}|{self.confidence:.2f}|{self.evidence}"


class AspectQuad(BaseModel):
    """
    Fully validated ABSA quad with exact character offsets strictly verified in review text.
    """
    aspect: Optional[str] = Field(default=None, description="Aspect term or None for implicit aspects")
    category: str = Field(..., description="Canonical category (e.g. BATTERY, FABRIC_MATERIAL)")
    sentiment: Sentiment = Field(..., description="positive, negative, neutral, mixed")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    evidence: str = Field(..., description="Exact supporting evidence clause in source text")

    # Character offsets in the parent review text
    aspect_start: Optional[int] = Field(default=None, ge=0)
    aspect_end: Optional[int] = Field(default=None, ge=0)
    evidence_start: int = Field(..., ge=0)
    evidence_end: int = Field(..., ge=0)

    @property
    def is_implicit(self) -> bool:
        return self.aspect is None

    @model_validator(mode="after")
    def validate_offsets(self) -> "AspectQuad":
        if self.evidence_end <= self.evidence_start:
            raise ValueError(
                f"Invalid evidence offsets: start ({self.evidence_start}) >= end ({self.evidence_end})"
            )
        if self.aspect is not None:
            if self.aspect_start is None or self.aspect_end is None:
                raise ValueError("Explicit aspect requires non-null aspect_start and aspect_end offsets.")
            if self.aspect_end <= self.aspect_start:
                raise ValueError(
                    f"Invalid aspect offsets: start ({self.aspect_start}) >= end ({self.aspect_end})"
                )
        else:
            if self.aspect_start is not None or self.aspect_end is not None:
                raise ValueError("Implicit aspect (aspect=None) must have null aspect offsets.")
        return self

    def to_compact_line(self) -> str:
        asp_str = self.aspect if self.aspect is not None else "null"
        return f"{asp_str}|{self.category}|{self.sentiment.value}|{self.confidence:.2f}|{self.evidence}"
