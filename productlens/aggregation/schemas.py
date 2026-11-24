"""
Pydantic schemas for ProductLens 3.0 Stage 4 Aggregation & Product Scorecards.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class EvidenceSnippet(BaseModel):
    """Verbatim quote providing supporting evidence for a sentiment mention."""
    review_id: str
    evidence: str
    sentiment: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FeatureScorecard(BaseModel):
    """Aggregated metrics and evidence for a normalized product feature."""
    feature_name: str = Field(description="Canonical normalized feature label (e.g. 'OLED Display')")
    category: str = Field(description="Canonical category enum name (e.g. 'DISPLAY')")
    mention_count: int = Field(default=0, description="Total count of customer mentions")
    weighted_mentions: float = Field(default=0.0, description="Confidence-weighted total mentions")
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    mixed_count: int = 0
    raw_sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Raw confidence-weighted sentiment: (pos - neg) / total in [-1.0, 1.0]",
    )
    shrunk_sentiment_score: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Bayesian-shrunk sentiment score penalizing small sample sizes toward 0.0",
    )
    negativity_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of mentions with negative/mixed sentiment",
    )
    impact_score: float = Field(
        default=0.0,
        description="Defect impact score: mention_count * negativity_rate (high-volume issues rank highest)",
    )
    evidence: Dict[str, List[EvidenceSnippet]] = Field(
        default_factory=dict,
        description="2-3 deduplicated high-confidence quotes per sentiment polarity ('positive', 'negative', 'mixed')",
    )


class ProductReport(BaseModel):
    """Top-level executive product scorecard report."""
    product_id: str
    vertical: str
    total_reviews_analyzed: int
    total_aspect_mentions: int
    overall_sentiment_score: float = Field(
        description="Overall catalog/product sentiment index in [-1.0, 1.0]",
    )
    category_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Aggregated sentiment score per canonical category",
    )
    top_defects: List[FeatureScorecard] = Field(
        default_factory=list,
        description="Top defect features ranked by impact score",
    )
    top_strengths: List[FeatureScorecard] = Field(
        default_factory=list,
        description="Top strength features ranked by positive volume and score",
    )
    features: List[FeatureScorecard] = Field(
        default_factory=list,
        description="Complete list of all evaluated product features",
    )

    def to_markdown(self) -> str:
        """Generates a structured human-readable executive scorecard in markdown."""
        lines = [
            f"# 📊 Product Intelligence Report: {self.product_id}",
            f"> **Domain:** `{self.vertical}` | **Reviews Analyzed:** {self.total_reviews_analyzed} | **Total Feature Mentions:** {self.total_aspect_mentions}",
            f"> **Overall Sentiment Index:** `{self.overall_sentiment_score:+.2f}`",
            "",
            "## 🏷️ Category Overview",
            "| Category | Sentiment Index |",
            "| :--- | :---: |",
        ]
        for cat, score in sorted(self.category_scores.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| **{cat}** | `{score:+.2f}` |")

        lines.extend([
            "",
            "## 🚨 Top Product Pain Points (Ranked by Impact)",
            "| Feature | Category | Mentions | Negativity | Impact Score | Shrunk Score |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ])
        for f in self.top_defects[:5]:
            lines.append(
                f"| **{f.feature_name}** | `{f.category}` | {f.mention_count} | {f.negativity_rate*100:.1f}% | `{f.impact_score:.2f}` | `{f.shrunk_sentiment_score:+.2f}` |"
            )

        lines.extend([
            "",
            "## 🌟 Top Product Strengths",
            "| Feature | Category | Mentions | Positive Rate | Shrunk Score |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ])
        for f in self.top_strengths[:5]:
            pos_rate = (f.positive_count / f.mention_count * 100) if f.mention_count > 0 else 0.0
            lines.append(
                f"| **{f.feature_name}** | `{f.category}` | {f.mention_count} | {pos_rate:.1f}% | `{f.shrunk_sentiment_score:+.2f}` |"
            )

        lines.extend([
            "",
            "## 📝 Key Customer Evidence Quotes",
        ])
        for f in (self.top_defects[:3] + self.top_strengths[:2]):
            lines.append(f"### Feature: **{f.feature_name}** ({f.category})")
            if "negative" in f.evidence and f.evidence["negative"]:
                lines.append("- **Critical Feedback:**")
                for ev in f.evidence["negative"]:
                    lines.append(f'  - *"{ev.evidence}"* (conf: {ev.confidence:.2f}, rev: `{ev.review_id}`)')
            if "positive" in f.evidence and f.evidence["positive"]:
                lines.append("- **Positive Feedback:**")
                for ev in f.evidence["positive"]:
                    lines.append(f'  - *"{ev.evidence}"* (conf: {ev.confidence:.2f}, rev: `{ev.review_id}`)')
            lines.append("")

        return "\n".join(lines)
