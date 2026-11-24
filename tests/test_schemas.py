"""
Unit tests for ProductLens 3.0 schemas and serializations.
"""

import pytest
from pydantic import ValidationError
from productlens.schemas.enums import (
    Sentiment,
    VerticalDomain,
    ElectronicsCategory,
    ApparelCategory,
    normalize_category,
)
from productlens.schemas.quads import AspectQuad, RawCompactQuad


def test_sentiment_enum():
    assert Sentiment.from_str("positive") == Sentiment.POSITIVE
    assert Sentiment.from_str("NEGATIVE") == Sentiment.NEGATIVE
    assert Sentiment.from_str("Neutral") == Sentiment.NEUTRAL
    assert Sentiment.from_str("mixed") == Sentiment.MIXED

    with pytest.raises(ValueError):
        Sentiment.from_str("unknown")


def test_category_taxonomies_and_normalization():
    assert normalize_category("display", VerticalDomain.ELECTRONICS_TECH) == "DISPLAY"
    assert normalize_category("battery", VerticalDomain.ELECTRONICS_TECH) == "BATTERY"
    assert normalize_category("random_feature", VerticalDomain.ELECTRONICS_TECH) == "OTHER"

    assert normalize_category("fabric material", VerticalDomain.APPAREL_FASHION) == "FABRIC_MATERIAL"
    assert normalize_category("fit_sizing", VerticalDomain.APPAREL_FASHION) == "FIT_SIZING"
    assert normalize_category("unknown_style", VerticalDomain.APPAREL_FASHION) == "OTHER"


def test_raw_compact_quad_parse():
    line = "battery life|BATTERY|negative|0.95|battery drains in under 3 hours"
    quad = RawCompactQuad.from_compact_line(line)
    assert quad.aspect == "battery life"
    assert quad.category == "BATTERY"
    assert quad.sentiment == Sentiment.NEGATIVE
    assert quad.confidence == 0.95
    assert quad.evidence == "battery drains in under 3 hours"

    # Implicit aspect parsing
    implicit_line = "null|VALUE|positive|0.88|worth every penny"
    implicit_quad = RawCompactQuad.from_compact_line(implicit_line)
    assert implicit_quad.aspect is None
    assert implicit_quad.sentiment == Sentiment.POSITIVE


def test_aspect_quad_validation():
    # Valid explicit aspect quad
    q = AspectQuad(
        aspect="screen",
        category="DISPLAY",
        sentiment=Sentiment.POSITIVE,
        confidence=0.99,
        evidence="the screen is great",
        aspect_start=4,
        aspect_end=10,
        evidence_start=0,
        evidence_end=19,
    )
    assert not q.is_implicit
    assert q.aspect == "screen"

    # Invalid aspect offsets (end <= start)
    with pytest.raises(ValidationError):
        AspectQuad(
            aspect="screen",
            category="DISPLAY",
            sentiment=Sentiment.POSITIVE,
            confidence=0.99,
            evidence="the screen is great",
            aspect_start=10,
            aspect_end=4,
            evidence_start=0,
            evidence_end=19,
        )

    # Valid implicit aspect quad
    impl = AspectQuad(
        aspect=None,
        category="VALUE",
        sentiment=Sentiment.POSITIVE,
        confidence=0.92,
        evidence="great price point",
        evidence_start=5,
        evidence_end=22,
    )
    assert impl.is_implicit
    assert impl.aspect is None
