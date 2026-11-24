"""
Unit tests for Stage 3 dataset preparation and compact quad serialization.
Tests zero-leakage constraints, roundtrip parsing, and dataset integrity.
"""

import json
import pytest
from pathlib import Path
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import RawCompactQuad
from productlens.gold.manager import GoldSetManager
from scripts.prepare_stage3_data import (
    get_forbidden_leakage_sets,
    build_grouped_dataset,
    SYSTEM_PROMPT,
)


def test_raw_compact_quad_roundtrip_explicit():
    """Verifies that explicit aspect quad serializes and deserializes with 100% fidelity."""
    original_line = "OLED screen|DISPLAY|positive|0.98|The OLED screen is bright"
    quad = RawCompactQuad.from_compact_line(original_line)
    assert quad.aspect == "OLED screen"
    assert quad.category == "DISPLAY"
    assert quad.sentiment == Sentiment.POSITIVE
    assert quad.confidence == 0.98
    assert quad.evidence == "The OLED screen is bright"

    # Roundtrip check
    serialized = quad.to_compact_line()
    assert serialized == original_line

    quad_reparsed = RawCompactQuad.from_compact_line(serialized)
    assert quad_reparsed == quad


def test_raw_compact_quad_roundtrip_implicit():
    """Verifies that implicit aspect quad (aspect=null) serializes and deserializes accurately."""
    original_line = "null|VALUE|positive|0.95|Paisa vasool mobile hai bhai"
    quad = RawCompactQuad.from_compact_line(original_line)
    assert quad.aspect is None
    assert quad.category == "VALUE"
    assert quad.sentiment == Sentiment.POSITIVE
    assert quad.confidence == 0.95
    assert quad.evidence == "Paisa vasool mobile hai bhai"

    # Roundtrip check
    serialized = quad.to_compact_line()
    assert serialized == original_line

    quad_reparsed = RawCompactQuad.from_compact_line(serialized)
    assert quad_reparsed == quad


def test_raw_compact_quad_handles_pipes_in_evidence():
    """Verifies that evidence strings containing pipes '|' are preserved without truncation."""
    line = "fabric|FABRIC_MATERIAL|positive|0.90|soft fabric | great quality"
    quad = RawCompactQuad.from_compact_line(line)
    assert quad.aspect == "fabric"
    assert quad.category == "FABRIC_MATERIAL"
    assert quad.evidence == "soft fabric | great quality"


def test_zero_leakage_detector():
    """Verifies that get_forbidden_leakage_sets correctly identifies Gold Set texts."""
    forbidden_texts, forbidden_pids = get_forbidden_leakage_sets()
    assert len(forbidden_texts) > 350
    assert len(forbidden_pids) > 0

    gold_reviews = GoldSetManager.load_reviews()
    for gr in gold_reviews:
        assert gr.text.strip().lower() in forbidden_texts

    # Test that build_grouped_dataset raises AssertionError if a leaked review is injected
    leaked_record = [
        {
            "text": gold_reviews[0].text,
            "vertical": "electronics_tech",
            "aspect": "screen",
            "category": "DISPLAY",
            "sentiment": "positive",
            "confidence": 1.0,
            "evidence": "sample",
            "source": "test",
        }
    ]
    with pytest.raises(AssertionError, match="Zero-leakage violation"):
        build_grouped_dataset(leaked_record, forbidden_texts, forbidden_pids)


def test_build_grouped_dataset_deduplication():
    """Verifies that multiple mentions of identical quads are deduplicated."""
    records = [
        {
            "text": "Great battery life and clear display.",
            "vertical": "electronics_tech",
            "aspect": "battery life",
            "category": "BATTERY",
            "sentiment": "positive",
            "confidence": 0.95,
            "evidence": "Great battery life",
            "source": "test",
        },
        # Duplicate quad
        {
            "text": "Great battery life and clear display.",
            "vertical": "electronics_tech",
            "aspect": "battery life",
            "category": "BATTERY",
            "sentiment": "positive",
            "confidence": 0.90,
            "evidence": "Great battery life",
            "source": "test",
        },
        # Distinct quad
        {
            "text": "Great battery life and clear display.",
            "vertical": "electronics_tech",
            "aspect": "display",
            "category": "DISPLAY",
            "sentiment": "positive",
            "confidence": 0.98,
            "evidence": "clear display",
            "source": "test",
        },
    ]

    dataset = build_grouped_dataset(records, forbidden_texts=set(), forbidden_pids=set())
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["num_quads"] == 2
    lines = sample["compact_target"].split("\n")
    assert len(lines) == 2
