"""
Unit tests for ProductLens 3.0 Stage 4 Aggregation & Product Scorecard Engine.
Tests aspect normalization, Bayesian shrinkage, impact ranking, evidence deduplication,
and end-to-end ProductReport generation.
"""

import math
from pathlib import Path
import pytest

from productlens.schemas.enums import Sentiment, ElectronicsCategory, ApparelCategory
from productlens.schemas.quads import AspectQuad
from productlens.aggregation.schemas import EvidenceSnippet, FeatureScorecard, ProductReport
from productlens.aggregation.scoring import FeatureScorer
from productlens.aggregation.normalizer import AspectNormalizer
from productlens.aggregation.evidence import EvidenceSelector, compute_string_jaccard
from productlens.aggregation.engine import Stage4AggregationEngine


def make_quad(
    aspect: str | None,
    category: str,
    sentiment: Sentiment,
    evidence: str = "sample evidence text",
    confidence: float = 1.0,
) -> AspectQuad:
    """Helper to construct valid AspectQuads for testing."""
    if aspect is not None:
        return AspectQuad(
            aspect=aspect,
            category=category,
            sentiment=sentiment,
            evidence=evidence,
            confidence=confidence,
            aspect_start=0,
            aspect_end=len(aspect),
            evidence_start=0,
            evidence_end=len(evidence),
        )
    else:
        return AspectQuad(
            aspect=None,
            category=category,
            sentiment=sentiment,
            evidence=evidence,
            confidence=confidence,
            evidence_start=0,
            evidence_end=len(evidence),
        )


# ============================================================================
# 1. FeatureScorer & Bayesian Shrinkage Tests
# ============================================================================

def test_feature_scorer_empty():
    scorer = FeatureScorer(shrinkage_prior_strength=3.0)
    res = scorer.score_feature([])
    assert res["mention_count"] == 0
    assert res["raw_sentiment_score"] == 0.0
    assert res["shrunk_sentiment_score"] == 0.0
    assert res["impact_score"] == 0.0


def test_feature_scorer_bayesian_shrinkage():
    scorer = FeatureScorer(shrinkage_prior_strength=3.0)

    # 1 mention, positive, confidence=1.0
    q1 = make_quad("screen", "DISPLAY", Sentiment.POSITIVE, "screen is bright", 1.0)
    res1 = scorer.score_feature([(q1, "rev_1")])
    assert res1["mention_count"] == 1
    assert res1["raw_sentiment_score"] == 1.0
    # Shrunk: 1.0 / (1.0 + 3.0) = 0.25
    assert math.isclose(res1["shrunk_sentiment_score"], 0.25, abs_tol=1e-4)
    assert res1["impact_score"] == 0.0

    # 100 mentions, all positive, confidence=1.0
    quads_100 = [(q1, f"rev_{i}") for i in range(100)]
    res100 = scorer.score_feature(quads_100)
    assert res100["mention_count"] == 100
    assert res100["raw_sentiment_score"] == 1.0
    # Shrunk: 100.0 / (100.0 + 3.0) = 0.9709
    assert math.isclose(res100["shrunk_sentiment_score"], 100 / 103, abs_tol=1e-3)


def test_feature_scorer_impact_and_negativity():
    scorer = FeatureScorer(shrinkage_prior_strength=3.0)

    # 10 negative mentions, 2 mixed mentions, 8 positive mentions
    quads = []
    for i in range(10):
        quads.append((make_quad("battery", "BATTERY", Sentiment.NEGATIVE, "dies in 2 hours", 1.0), f"rev_neg_{i}"))
    for i in range(2):
        quads.append((make_quad("battery", "BATTERY", Sentiment.MIXED, "okay battery but charges slow", 1.0), f"rev_mix_{i}"))
    for i in range(8):
        quads.append((make_quad("battery", "BATTERY", Sentiment.POSITIVE, "decent battery life", 1.0), f"rev_pos_{i}"))

    res = scorer.score_feature(quads)
    assert res["mention_count"] == 20
    assert res["negative_count"] == 10
    assert res["mixed_count"] == 2
    assert res["positive_count"] == 8

    # Negativity: (10 + 0.5 * 2) / 20 = 11 / 20 = 0.55
    assert math.isclose(res["negativity_rate"], 0.55, abs_tol=1e-4)
    # Impact: 10 + 0.5 * 2 = 11.0
    assert math.isclose(res["impact_score"], 11.0, abs_tol=1e-4)
    # Raw sentiment: (8 - 10) / 20 = -0.10
    assert math.isclose(res["raw_sentiment_score"], -0.10, abs_tol=1e-4)
    # Shrunk sentiment: (8 - 10) / (20 + 3) = -2 / 23 = -0.0870
    assert math.isclose(res["shrunk_sentiment_score"], -2 / 23, abs_tol=1e-3)


def test_feature_scorer_confidence_weighting():
    scorer = FeatureScorer(shrinkage_prior_strength=3.0)

    # 1 positive with confidence 0.5, 1 negative with confidence 1.0
    q_pos = make_quad("keyboard", "HARDWARE", Sentiment.POSITIVE, "nice keys", 0.5)
    q_neg = make_quad("keyboard", "HARDWARE", Sentiment.NEGATIVE, "sticky spacebar", 1.0)
    res = scorer.score_feature([(q_pos, "rev1"), (q_neg, "rev2")])
    assert res["mention_count"] == 2
    assert math.isclose(res["weighted_mentions"], 1.5, abs_tol=1e-4)
    # Raw: (0.5 - 1.0) / 1.5 = -0.3333
    assert math.isclose(res["raw_sentiment_score"], -0.3333, abs_tol=1e-3)


# ============================================================================
# 2. AspectNormalizer Tests
# ============================================================================

def test_aspect_normalizer_canonical_naming():
    assert AspectNormalizer.format_canonical_name("oled display", "DISPLAY") == "OLED Display"
    assert AspectNormalizer.format_canonical_name("usb-c port", "CONNECTIVITY") == "USB-C Port"
    assert AspectNormalizer.format_canonical_name("battery life", "BATTERY") == "Battery Life"
    assert AspectNormalizer.format_canonical_name(None, "SOUND") == "Overall Sound"


def test_aspect_normalizer_clustering_tfidf():
    # Force TF-IDF fallback to verify deterministic string clustering
    normalizer = AspectNormalizer(similarity_threshold=0.5, use_embeddings=False)

    q1 = make_quad("sound quality", "SOUND", Sentiment.POSITIVE, "great sound quality")
    q2 = make_quad("sound", "SOUND", Sentiment.POSITIVE, "clear sound")
    q3 = make_quad("audio sound", "SOUND", Sentiment.POSITIVE, "amazing audio sound")
    q4 = make_quad("battery duration", "BATTERY", Sentiment.POSITIVE, "long battery duration")

    quad_records = [
        (q1, "rev1"),
        (q2, "rev2"),
        (q3, "rev3"),
        (q4, "rev4"),
    ]

    grouped = normalizer.group_and_normalize(quad_records)
    # Verify categories are segregated
    assert "SOUND" in grouped
    assert "BATTERY" in grouped
    assert len(grouped["BATTERY"]) == 1


def test_aspect_normalizer_clustering_embeddings():
    # Test with sentence-transformers if available
    normalizer = AspectNormalizer(similarity_threshold=0.6, use_embeddings=True)
    q1 = make_quad("screen brightness", "DISPLAY", Sentiment.POSITIVE, "very bright screen")
    q2 = make_quad("display luminosity", "DISPLAY", Sentiment.POSITIVE, "luminosity is high")

    quad_records = [(q1, "rev1"), (q2, "rev2")]
    grouped = normalizer.group_and_normalize(quad_records)
    assert "DISPLAY" in grouped
    total_quads = sum(len(quads) for quads in grouped["DISPLAY"].values())
    assert total_quads == 2


# ============================================================================
# 3. EvidenceSelector Tests
# ============================================================================

def test_evidence_selector_deduplication():
    selector = EvidenceSelector(max_quotes_per_sentiment=3, similarity_threshold=0.6)

    # 3 duplicate or near-duplicate quotes, 1 unique quote
    quads = [
        (make_quad("screen", "DISPLAY", Sentiment.POSITIVE, "The screen is extremely bright and crisp", 0.9), "rev1"),
        (make_quad("screen", "DISPLAY", Sentiment.POSITIVE, "screen is extremely bright", 0.8), "rev2"),  # substring
        (make_quad("screen", "DISPLAY", Sentiment.POSITIVE, "The screen is very bright and crisp", 0.7), "rev3"),  # high Jaccard
        (make_quad("screen", "DISPLAY", Sentiment.POSITIVE, "Viewing angles are wide and colors are accurate", 0.95), "rev4"),  # unique
    ]

    selected = selector.select_evidence(quads)
    assert "positive" in selected
    pos_quotes = selected["positive"]
    # Only 2 distinct quotes should be kept (the duplicate substring/Jaccard dropped)
    assert len(pos_quotes) == 2
    # Verify top confidence quote is retained
    evidences = [s.evidence for s in pos_quotes]
    assert "Viewing angles are wide and colors are accurate" in evidences
    assert "The screen is extremely bright and crisp" in evidences


def test_string_jaccard():
    sim = compute_string_jaccard("great sound quality", "great sound quality indeed")
    assert sim > 0.6
    diff = compute_string_jaccard("great sound quality", "terrible battery life")
    assert diff == 0.0


# ============================================================================
# 4. Stage4AggregationEngine & ProductReport End-to-End Tests
# ============================================================================

def test_stage4_aggregation_engine_end_to_end():
    engine = Stage4AggregationEngine(use_embeddings=False)

    quads = [
        (make_quad("battery life", "BATTERY", Sentiment.NEGATIVE, "battery drains fast", 0.9), "rev1"),
        (make_quad("battery", "BATTERY", Sentiment.NEGATIVE, "battery dies quickly", 0.85), "rev2"),
        (make_quad("build quality", "BUILD_QUALITY", Sentiment.POSITIVE, "solid aluminum chassis", 0.95), "rev3"),
        (make_quad("sound", "SOUND", Sentiment.POSITIVE, "clear audio sound", 0.9), "rev4"),
    ]

    report = engine.aggregate(
        product_id="B00TEST123",
        quad_records=quads,
        vertical="electronics_tech",
        total_reviews=4,
    )

    assert isinstance(report, ProductReport)
    assert report.product_id == "B00TEST123"
    assert report.total_reviews_analyzed == 4
    assert report.total_aspect_mentions == 4
    assert len(report.features) > 0

    # Top defect should be Battery related
    assert len(report.top_defects) > 0
    top_defect = report.top_defects[0]
    assert "BATTERY" in top_defect.category
    assert top_defect.impact_score > 0.0
    assert top_defect.negative_count >= 1

    # Check Markdown formatting
    md = report.to_markdown()
    assert "# 📊 Product Intelligence Report: B00TEST123" in md
    assert "Category Overview" in md
    assert "Top Product Pain Points" in md
    assert "Top Product Strengths" in md


def test_stage4_from_predictions_file():
    gold_pred_path = Path("outputs/gold_eval/teacher_gold_predictions.jsonl")
    if not gold_pred_path.exists():
        pytest.skip("outputs/gold_eval/teacher_gold_predictions.jsonl not found")

    engine = Stage4AggregationEngine(use_embeddings=False)
    report = engine.from_predictions_file(
        jsonl_path=gold_pred_path,
        product_id="Amazon_Electronics_Gold_Set",
        vertical="electronics_tech",
    )

    assert report.total_reviews_analyzed == 350
    assert report.total_aspect_mentions == 616
    assert len(report.features) > 0
    assert len(report.category_scores) > 0

    # Ensure Pydantic schema validation succeeds
    json_str = report.model_dump_json()
    assert len(json_str) > 100
    loaded = ProductReport.model_validate_json(json_str)
    assert loaded.product_id == report.product_id
