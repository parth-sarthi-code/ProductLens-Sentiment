"""
Confidence-Weighted Scoring and Bayesian Shrinkage.
Computes stabilized, confidence-weighted sentiment scores and defect impact metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from productlens.schemas.enums import Sentiment
from productlens.schemas.quads import AspectQuad


class FeatureScorer:
    """Computes confidence-weighted sentiment and Bayesian shrinkage scores."""

    def __init__(self, shrinkage_prior_strength: float = 3.0):
        """
        Args:
            shrinkage_prior_strength (C): Pseudo-count weight pulling scores toward 0.0 (neutral).
        """
        self.C = shrinkage_prior_strength

    def score_feature(
        self,
        quad_tuples: List[Tuple[AspectQuad, str]],  # (quad, review_id)
    ) -> Dict[str, any]:
        """
        Calculates all sentiment metrics for a group of quads belonging to a feature.
        """
        if not quad_tuples:
            return {
                "mention_count": 0,
                "weighted_mentions": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "mixed_count": 0,
                "raw_sentiment_score": 0.0,
                "shrunk_sentiment_score": 0.0,
                "negativity_rate": 0.0,
                "impact_score": 0.0,
            }

        pos_count = 0
        neg_count = 0
        neu_count = 0
        mix_count = 0

        w_pos = 0.0
        w_neg = 0.0
        w_neu = 0.0
        w_mix = 0.0

        for q, _ in quad_tuples:
            conf = float(q.confidence) if q.confidence is not None else 1.0
            sent = q.sentiment

            if sent == Sentiment.POSITIVE:
                pos_count += 1
                w_pos += conf
            elif sent == Sentiment.NEGATIVE:
                neg_count += 1
                w_neg += conf
            elif sent == Sentiment.NEUTRAL:
                neu_count += 1
                w_neu += conf
            elif sent == Sentiment.MIXED:
                mix_count += 1
                w_mix += conf

        mention_count = len(quad_tuples)
        weighted_total = w_pos + w_neg + w_neu + w_mix

        if weighted_total <= 0:
            raw_score = 0.0
            shrunk_score = 0.0
            neg_rate = 0.0
            impact = 0.0
        else:
            # Raw confidence-weighted score in [-1.0, 1.0]
            raw_score = (w_pos - w_neg) / weighted_total
            # Bayesian shrinkage toward 0.0 (neutral prior)
            shrunk_score = (w_pos - w_neg) / (weighted_total + self.C)
            # Negativity rate (including half-weight for mixed sentiment)
            neg_rate = (w_neg + 0.5 * w_mix) / weighted_total
            # Impact score: high volume + high negativity = maximum impact
            impact = w_neg + 0.5 * w_mix

        return {
            "mention_count": mention_count,
            "weighted_mentions": round(weighted_total, 4),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "neutral_count": neu_count,
            "mixed_count": mix_count,
            "raw_sentiment_score": round(float(raw_score), 4),
            "shrunk_sentiment_score": round(float(shrunk_score), 4),
            "negativity_rate": round(float(neg_rate), 4),
            "impact_score": round(float(impact), 4),
        }
