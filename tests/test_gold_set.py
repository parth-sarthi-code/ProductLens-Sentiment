"""
Unit tests for Gold Set and Stage 1 Evaluation Harness.
"""

import pytest
from productlens.gold.manager import GoldSetManager
from productlens.evaluation.runner import Stage1Evaluator
from productlens.evaluation.metrics import (
    compute_explicit_aspect_spans_f1,
    compute_implicit_category_f1,
    compute_sentiment_macro_f1,
    compute_expected_calibration_error,
    compute_parse_and_offset_rates,
)
from productlens.schemas.enums import Sentiment
from productlens.schemas.quads import AspectQuad


def test_gold_set_manifest():
    reviews = GoldSetManager.load_reviews()
    manifest = GoldSetManager.get_manifest(reviews)
    assert manifest.total_reviews >= 300
    assert manifest.english_reviews >= 200
    assert manifest.hinglish_reviews >= 50
    assert manifest.hard_cases > 0
    assert manifest.total_quads == 550


def test_gold_offsets_are_100_percent_valid():
    reviews = GoldSetManager.load_reviews()
    for r in reviews:
        for q in r.quads:
            # Evidence match
            assert r.text[q.evidence_start : q.evidence_end] == q.evidence
            # Aspect match (if not implicit)
            if q.aspect is not None:
                assert r.text[q.aspect_start : q.aspect_end] == q.aspect


def test_metrics_explicit_aspect_f1_multiset():
    q1 = AspectQuad(
        aspect="screen",
        category="DISPLAY",
        sentiment=Sentiment.POSITIVE,
        confidence=1.0,
        evidence="good screen",
        aspect_start=5,
        aspect_end=11,
        evidence_start=0,
        evidence_end=11,
    )
    # Duplicate aspect in same review
    gold = [[q1, q1]]
    pred = [[q1, q1]]
    res = compute_explicit_aspect_spans_f1(gold, pred)
    assert res["aspect_f1"] == 1.0
    assert res["explicit_matched_count"] == 2


def test_metrics_implicit_category_f1():
    q_imp = AspectQuad(
        aspect=None,
        category="VALUE",
        sentiment=Sentiment.POSITIVE,
        confidence=1.0,
        evidence="great price",
        evidence_start=0,
        evidence_end=11,
    )
    gold = [[q_imp]]
    pred = [[q_imp]]
    res = compute_implicit_category_f1(gold, pred)
    assert res["implicit_category_f1"] == 1.0
    assert res["implicit_matched_count"] == 1


def test_metrics_ece():
    confidences = [0.9, 0.8, 0.7, 0.6]
    correctness = [1, 1, 1, 0]
    ece = compute_expected_calibration_error(confidences, correctness)
    assert 0.0 <= ece <= 1.0


def test_evaluator_runner_full_gold_reconciliation():
    reviews = GoldSetManager.load_reviews()
    evaluator = Stage1Evaluator(reviews)
    results = evaluator.evaluate_predictions(lambda r: r.quads)

    ov = results["overall"]
    assert ov["aspect_f1"] == 1.0
    assert ov["implicit_category_f1"] == 1.0
    assert ov["sentiment_macro_f1"] == 1.0
    assert ov["explicit_matched_count"] == 539
    assert ov["implicit_matched_count"] == 11
    assert ov["total_matched_quads"] == 550
    assert results["total_matched_quads"] == 550
