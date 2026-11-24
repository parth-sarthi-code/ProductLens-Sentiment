"""
Unit tests for deterministic substring validator.
"""

import pytest
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import RawCompactQuad
from productlens.validator.substring import SubstringValidator, find_all_occurrences
from productlens.validator.compact_parser import CompactFormatParser


def test_find_all_occurrences():
    text = "The screen is sharp and the display screen looks clean."
    spans = find_all_occurrences(text, "screen")
    assert len(spans) == 2
    assert spans[0] == (4, 10)
    assert spans[1] == (36, 42)


def test_substring_validator_explicit_aspect():
    text = "The display is vibrant, but the battery drains in under 3 hours."
    raw = RawCompactQuad(
        aspect="display",
        category="DISPLAY",
        sentiment=Sentiment.POSITIVE,
        confidence=0.95,
        evidence="The display is vibrant",
    )

    quad, err = SubstringValidator.validate_quad(text, raw, VerticalDomain.ELECTRONICS_TECH)
    assert err is None
    assert quad is not None
    assert quad.aspect == "display"
    assert text[quad.aspect_start:quad.aspect_end] == "display"
    assert text[quad.evidence_start:quad.evidence_end] == "The display is vibrant"
    assert quad.aspect_start >= quad.evidence_start
    assert quad.aspect_end <= quad.evidence_end


def test_substring_validator_implicit_aspect():
    text = "Overall it was completely overpriced for what it provides."
    raw = RawCompactQuad(
        aspect=None,
        category="VALUE",
        sentiment=Sentiment.NEGATIVE,
        confidence=0.90,
        evidence="completely overpriced for what it provides",
    )

    quad, err = SubstringValidator.validate_quad(text, raw, VerticalDomain.ELECTRONICS_TECH)
    assert err is None
    assert quad is not None
    assert quad.is_implicit
    assert quad.aspect is None
    assert text[quad.evidence_start:quad.evidence_end] == "completely overpriced for what it provides"


def test_substring_validator_rejection_on_hallucination():
    text = "The sound quality is crisp and clear."
    # Evidence hallucination
    raw = RawCompactQuad(
        aspect="sound",
        category="SOUND",
        sentiment=Sentiment.POSITIVE,
        confidence=0.8,
        evidence="the audio reproduction was stellar",  # Hallucinated / not in review!
    )
    quad, err = SubstringValidator.validate_quad(text, raw, VerticalDomain.ELECTRONICS_TECH)
    assert quad is None
    assert "not found in review text" in err


def test_compact_format_parser():
    text = "The keyboard feels great and key travel is perfect. However, battery life is poor."
    output_block = """
keyboard|BUILD_QUALITY|positive|0.98|The keyboard feels great
key travel|BUILD_QUALITY|positive|0.92|key travel is perfect
battery life|BATTERY|negative|0.95|However, battery life is poor
"""
    quads, total, parsed = CompactFormatParser.parse_and_validate_review(
        text, output_block, VerticalDomain.ELECTRONICS_TECH
    )
    assert total == 3
    assert parsed == 3
    assert len(quads) == 3
    assert quads[0].aspect == "keyboard"
    assert quads[1].aspect == "key travel"
    assert quads[2].aspect == "battery life"
    assert quads[2].sentiment == Sentiment.NEGATIVE
