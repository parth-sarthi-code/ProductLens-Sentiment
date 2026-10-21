import pytest
from productlens.engine import ProductLensEngine


def test_engine_analyze_review():
    engine = ProductLensEngine(lazy_load=True)
    report = engine.analyze_review(
        text="The display on this monitor is razor sharp, but the remote control is unresponsive.",
        category="Electronics",
    )

    assert report.aspect_count >= 1
    assert len(report.results) >= 1
    assert report.summary_sentiment in ["positive", "negative", "mixed", "neutral"]
    assert report.execution_time_ms > 0.0


def test_engine_analyze_batch():
    engine = ProductLensEngine(lazy_load=True)
    reviews = [
        {"text": "The laptop keyboard is wonderful.", "category": "Computers"},
        {"text": "The t-shirt fit is awful and too tight.", "category": "Clothing"},
    ]
    batch_report = engine.analyze_batch(reviews)

    assert batch_report.total_reviews == 2
    assert len(batch_report.reports) == 2
    assert batch_report.total_aspects >= 2
