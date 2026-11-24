"""
Stage 4 Aggregation & Product Scorecard Engine.
Orchestrates normalization, confidence-weighted scoring, impact ranking,
and evidence selection to generate executive product intelligence reports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad
from productlens.aggregation.schemas import FeatureScorecard, ProductReport
from productlens.aggregation.normalizer import AspectNormalizer
from productlens.aggregation.scoring import FeatureScorer
from productlens.aggregation.evidence import EvidenceSelector


class Stage4AggregationEngine:
    """Orchestrates feature extraction, normalization, Bayesian scoring, and report generation."""

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        shrinkage_prior_strength: float = 3.0,
        max_quotes_per_sentiment: int = 3,
        use_embeddings: bool = True,
    ):
        self.normalizer = AspectNormalizer(
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings,
        )
        self.scorer = FeatureScorer(shrinkage_prior_strength=shrinkage_prior_strength)
        self.evidence_selector = EvidenceSelector(max_quotes_per_sentiment=max_quotes_per_sentiment)

    def aggregate(
        self,
        product_id: str,
        quad_records: List[Tuple[AspectQuad, str]],  # (AspectQuad, review_id)
        vertical: str = "electronics_tech",
        total_reviews: Optional[int] = None,
    ) -> ProductReport:
        """
        Builds a comprehensive ProductReport from a list of validated aspect quads.
        """
        if total_reviews is None:
            unique_rev_ids = {rev_id for _, rev_id in quad_records}
            total_reviews = len(unique_rev_ids)

        if not quad_records:
            return ProductReport(
                product_id=product_id,
                vertical=vertical,
                total_reviews_analyzed=total_reviews,
                total_aspect_mentions=0,
                overall_sentiment_score=0.0,
                category_scores={},
                top_defects=[],
                top_strengths=[],
                features=[],
            )

        # 1. Group by category and cluster into canonical feature names
        grouped = self.normalizer.group_and_normalize(quad_records)

        # 2. Score each feature cluster and select evidence
        feature_scorecards: List[FeatureScorecard] = []
        category_weights: Dict[str, float] = defaultdict(float)
        category_pos: Dict[str, float] = defaultdict(float)
        category_neg: Dict[str, float] = defaultdict(float)

        for category, feature_clusters in grouped.items():
            for feature_name, quads in feature_clusters.items():
                metrics = self.scorer.score_feature(quads)
                evidence = self.evidence_selector.select_evidence(quads)

                card = FeatureScorecard(
                    feature_name=feature_name,
                    category=category,
                    mention_count=metrics["mention_count"],
                    weighted_mentions=metrics["weighted_mentions"],
                    positive_count=metrics["positive_count"],
                    negative_count=metrics["negative_count"],
                    neutral_count=metrics["neutral_count"],
                    mixed_count=metrics["mixed_count"],
                    raw_sentiment_score=metrics["raw_sentiment_score"],
                    shrunk_sentiment_score=metrics["shrunk_sentiment_score"],
                    negativity_rate=metrics["negativity_rate"],
                    impact_score=metrics["impact_score"],
                    evidence=evidence,
                )
                feature_scorecards.append(card)

                # Accumulate category totals
                for q, _ in quads:
                    conf = float(q.confidence) if q.confidence is not None else 1.0
                    category_weights[category] += conf
                    if q.sentiment == Sentiment.POSITIVE:
                        category_pos[category] += conf
                    elif q.sentiment == Sentiment.NEGATIVE:
                        category_neg[category] += conf

        # 3. Compute per-category overall scores
        category_scores: Dict[str, float] = {}
        total_w_pos = sum(category_pos.values())
        total_w_neg = sum(category_neg.values())
        total_weight = sum(category_weights.values())

        for cat, w_total in category_weights.items():
            if w_total > 0:
                cat_score = (category_pos[cat] - category_neg[cat]) / w_total
                category_scores[cat] = round(float(cat_score), 4)

        overall_score = (
            (total_w_pos - total_w_neg) / total_weight if total_weight > 0 else 0.0
        )

        # 4. Rank Top Defects (impact_score descending, requiring at least 1 negative mention)
        defects = [f for f in feature_scorecards if f.negative_count > 0 or f.mixed_count > 0]
        defects.sort(key=lambda f: (f.impact_score, f.mention_count), reverse=True)

        # 5. Rank Top Strengths (shrunk_sentiment_score descending, requiring positive mentions)
        strengths = [f for f in feature_scorecards if f.positive_count > 0 and f.shrunk_sentiment_score > 0]
        strengths.sort(key=lambda f: (f.shrunk_sentiment_score, f.mention_count), reverse=True)

        return ProductReport(
            product_id=product_id,
            vertical=vertical,
            total_reviews_analyzed=total_reviews,
            total_aspect_mentions=len(quad_records),
            overall_sentiment_score=round(float(overall_score), 4),
            category_scores=category_scores,
            top_defects=defects,
            top_strengths=strengths,
            features=feature_scorecards,
        )

    @classmethod
    def from_predictions_file(
        cls,
        jsonl_path: Union[str, Path],
        product_id: str = "BENCHMARK_CATALOG",
        vertical: str = "electronics_tech",
    ) -> ProductReport:
        """Loads validated quads from a predictions JSONL file and builds a ProductReport."""
        quad_records: List[Tuple[AspectQuad, str]] = []
        total_reviews = 0

        p = Path(jsonl_path)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_reviews += 1
                record = json.loads(line)
                rev_id = record.get("review_id", f"rev_{total_reviews:04d}")
                for q_dict in record.get("validated_quads", []):
                    quad = AspectQuad.model_validate(q_dict)
                    quad_records.append((quad, rev_id))

        engine = cls()
        return engine.aggregate(
            product_id=product_id,
            quad_records=quad_records,
            vertical=vertical,
            total_reviews=total_reviews,
        )
