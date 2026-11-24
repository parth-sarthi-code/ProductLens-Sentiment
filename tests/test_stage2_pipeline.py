"""
Unit tests for Stage 2 Kaggle Teacher & Consensus Pipeline components.
Tests schema generation, dual-pass consensus logic, and checkpoint manager.
"""

import json
import pytest
from pathlib import Path
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad
from scripts.stage2_teacher_kaggle import (
    TeacherAspectQuad,
    TeacherExtractionOutput,
    get_structured_output_schema_json,
    DualPassConsensusEngine,
    CheckpointManager,
    build_argument_parser,
)


def test_teacher_schema_generation():
    schema_str = get_structured_output_schema_json()
    schema = json.loads(schema_str)
    assert "properties" in schema
    assert "quads" in schema["properties"]

    # Verify quads item definition
    quad_def = schema["$defs"]["TeacherAspectQuad"]
    assert "category" in quad_def["properties"]
    assert "sentiment" in quad_def["properties"]
    assert "enum" in quad_def["properties"]["sentiment"]
    assert "positive" in quad_def["properties"]["sentiment"]["enum"]
    assert "mixed" in quad_def["properties"]["sentiment"]["enum"]


def test_dual_pass_consensus_agreement():
    text = "The OLED display is stunning and vibrant, but battery drains in under 3 hours."

    pass1_quads = [
        TeacherAspectQuad(
            aspect="OLED display",
            category="DISPLAY",
            sentiment="positive",
            confidence=0.96,
            evidence="The OLED display is stunning and vibrant",
        ),
        TeacherAspectQuad(
            aspect="battery",
            category="BATTERY",
            sentiment="negative",
            confidence=0.90,
            evidence="battery drains in under 3 hours",
        ),
    ]

    pass2_quads = [
        # Agrees on OLED display with slight confidence difference
        TeacherAspectQuad(
            aspect="OLED display",
            category="DISPLAY",
            sentiment="positive",
            confidence=0.94,
            evidence="The OLED display is stunning and vibrant",
        ),
        # Disagrees on battery sentiment (neutral vs negative)
        TeacherAspectQuad(
            aspect="battery",
            category="BATTERY",
            sentiment="neutral",
            confidence=0.85,
            evidence="battery drains in under 3 hours",
        ),
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_quads,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )

    # Only OLED display should survive consensus
    assert len(consensus_quads) == 1
    kept = consensus_quads[0]
    assert kept.aspect == "OLED display"
    assert kept.category == "DISPLAY"
    assert kept.sentiment == Sentiment.POSITIVE
    assert kept.confidence == round((0.96 + 0.94) / 2.0, 4)
    assert stats["consensus_kept"] == 1


def test_dual_pass_consensus_filters_hallucinated_evidence():
    text = "The keyboard is responsive and typing feels great."

    pass1_quads = [
        TeacherAspectQuad(
            aspect="keyboard",
            category="BUILD_QUALITY",
            sentiment="positive",
            confidence=0.95,
            evidence="The keyboard is responsive and typing feels great",
        )
    ]

    pass2_quads = [
        # Hallucinated evidence not in text!
        TeacherAspectQuad(
            aspect="keyboard",
            category="BUILD_QUALITY",
            sentiment="positive",
            confidence=0.95,
            evidence="the keys have fantastic tactile travel",
        )
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_quads,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )

    # Pass 2 is rejected by SubstringValidator, so consensus fails
    assert len(consensus_quads) == 0
    assert stats["pass1_valid"] == 1
    assert stats["pass2_valid"] == 0
    assert stats["consensus_kept"] == 0


def test_checkpoint_manager_roundtrip(tmp_path):
    records = [
        {
            "review_id": "rev_001",
            "product_id": "prod_1",
            "text": "sample text",
            "aspect": "screen",
            "category": "DISPLAY",
            "sentiment": "positive",
            "confidence": 0.95,
            "evidence": "sample",
            "aspect_start": 0,
            "aspect_end": 6,
            "evidence_start": 0,
            "evidence_end": 6,
            "vertical": "electronics_tech",
        }
    ]

    # Save batch 1
    saved_file = CheckpointManager.save_batch_parquet(tmp_path, 1, records)
    assert saved_file.exists()

    # Verify processed review IDs
    processed = CheckpointManager.get_processed_review_ids(tmp_path)
    assert "rev_001" in processed
    assert len(processed) == 1

    # Check next batch index
    next_idx = CheckpointManager.get_next_batch_index(tmp_path)
    assert next_idx == 2


def test_dual_pass_consensus_iou_overlapping_retained():
    """Verifies that overlapping aspect spans with IoU >= 0.5 and matching category/sentiment are kept."""
    text = "The battery life is decent, lasting around 6 hours on a single charge."
    # 'battery life': span [4, 16] (len 12)
    # 'battery':      span [4, 11] (len 7)
    # intersection: 7, union: 12, IoU = 7/12 = 0.5833 >= 0.5

    pass1_quads = [
        TeacherAspectQuad(
            aspect="battery life",
            category="BATTERY",
            sentiment="positive",
            confidence=0.92,
            evidence="The battery life is decent",
        )
    ]
    pass2_quads = [
        TeacherAspectQuad(
            aspect="battery",
            category="BATTERY",
            sentiment="positive",
            confidence=0.86,
            evidence="The battery life is decent",
        )
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_quads,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )

    assert len(consensus_quads) == 1
    kept = consensus_quads[0]
    # Pass 1 span and evidence must be preserved
    assert kept.aspect == "battery life"
    assert kept.aspect_start == 4
    assert kept.aspect_end == 16
    assert kept.evidence == "The battery life is decent"
    assert kept.category == "BATTERY"
    assert kept.sentiment == Sentiment.POSITIVE
    # Confidence averaged across passes
    assert kept.confidence == round((0.92 + 0.86) / 2.0, 4)
    assert stats["consensus_kept"] == 1


def test_dual_pass_consensus_iou_low_overlap_rejected():
    """Verifies that aspect spans with IoU < 0.5 are discarded."""
    text = "The super high resolution display is gorgeous and vibrant."
    # 'display':                         span [26, 33] (len 7)
    # 'super high resolution display':   span [4, 33]  (len 29)
    # intersection: 7, union: 29, IoU = 7/29 = 0.241 < 0.5

    pass1_quads = [
        TeacherAspectQuad(
            aspect="super high resolution display",
            category="DISPLAY",
            sentiment="positive",
            confidence=0.90,
            evidence="super high resolution display is gorgeous and vibrant",
        )
    ]
    pass2_quads = [
        TeacherAspectQuad(
            aspect="display",
            category="DISPLAY",
            sentiment="positive",
            confidence=0.95,
            evidence="display is gorgeous and vibrant",
        )
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_quads,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )

    assert len(consensus_quads) == 0
    assert stats["consensus_kept"] == 0


def test_dual_pass_consensus_mismatched_sentiment_or_category_rejected():
    """Verifies that spans with IoU >= 0.5 are discarded if category or sentiment mismatch."""
    text = "The camera performs well in daylight."

    pass1_quads = [
        TeacherAspectQuad(
            aspect="camera",
            category="PERFORMANCE",
            sentiment="positive",
            confidence=0.90,
            evidence="The camera performs well in daylight",
        )
    ]
    # Mismatched category (CONNECTIVITY vs PERFORMANCE)
    pass2_cat_mismatch = [
        TeacherAspectQuad(
            aspect="camera",
            category="CONNECTIVITY",
            sentiment="positive",
            confidence=0.90,
            evidence="The camera performs well in daylight",
        )
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_cat_mismatch,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )
    assert len(consensus_quads) == 0

    # Mismatched sentiment (negative vs positive)
    pass2_sent_mismatch = [
        TeacherAspectQuad(
            aspect="camera",
            category="PERFORMANCE",
            sentiment="negative",
            confidence=0.90,
            evidence="The camera performs well in daylight",
        )
    ]
    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_sent_mismatch,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )
    assert len(consensus_quads) == 0


def test_dual_pass_consensus_implicit_aspects():
    """Verifies that implicit aspects (aspect: null) match strictly on category and sentiment."""
    text = "Paisa vasool mobile hai bhai, definitely worth every penny."

    pass1_quads = [
        TeacherAspectQuad(
            aspect=None,
            category="VALUE",
            sentiment="positive",
            confidence=0.95,
            evidence="Paisa vasool mobile hai bhai, definitely worth every penny",
        )
    ]
    pass2_quads = [
        TeacherAspectQuad(
            aspect=None,
            category="VALUE",
            sentiment="positive",
            confidence=0.85,
            evidence="worth every penny",
        )
    ]

    consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
        text=text,
        pass1_quads=pass1_quads,
        pass2_quads=pass2_quads,
        vertical=VerticalDomain.ELECTRONICS_TECH,
    )

    assert len(consensus_quads) == 1
    kept = consensus_quads[0]
    assert kept.aspect is None
    assert kept.category == "VALUE"
    assert kept.sentiment == Sentiment.POSITIVE
    assert kept.confidence == round((0.95 + 0.85) / 2.0, 4)
    assert stats["consensus_kept"] == 1


def test_checkpoint_manager_worker_prefixed_files(tmp_path):
    """Verifies that CheckpointManager discovers IDs from worker-prefixed parquet files."""
    rec_gpu0 = [
        {
            "review_id": "rev_gpu0_001",
            "product_id": "prod_1",
            "text": "sample text 1",
            "aspect": "screen",
            "category": "DISPLAY",
            "sentiment": "positive",
            "confidence": 0.95,
            "evidence": "sample",
            "aspect_start": 0,
            "aspect_end": 6,
            "evidence_start": 0,
            "evidence_end": 6,
            "vertical": "electronics_tech",
        }
    ]
    rec_gpu1 = [
        {
            "review_id": "rev_gpu1_001",
            "product_id": "prod_2",
            "text": "sample text 2",
            "aspect": "fabric",
            "category": "FABRIC_MATERIAL",
            "sentiment": "positive",
            "confidence": 0.90,
            "evidence": "sample",
            "aspect_start": 0,
            "aspect_end": 6,
            "evidence_start": 0,
            "evidence_end": 6,
            "vertical": "apparel_fashion",
        }
    ]

    f0 = CheckpointManager.save_batch_parquet(tmp_path, 1, rec_gpu0, prefix="gpu0_batch")
    f1 = CheckpointManager.save_batch_parquet(tmp_path, 1, rec_gpu1, prefix="gpu1_batch")

    assert f0.name == "gpu0_batch_0001.parquet"
    assert f1.name == "gpu1_batch_0001.parquet"

    processed = CheckpointManager.get_processed_review_ids(tmp_path)
    assert "rev_gpu0_001" in processed
    assert "rev_gpu1_001" in processed
    assert len(processed) == 2

    assert CheckpointManager.get_next_batch_index(tmp_path, prefix="gpu0_batch") == 2
    assert CheckpointManager.get_next_batch_index(tmp_path, prefix="gpu1_batch") == 2


def test_cli_argument_defaults():
    parser = build_argument_parser()
    args = parser.parse_args([])
    assert args.model == "Qwen/Qwen3.5-9B"
    assert args.tp == 1
    assert args.dtype == "float16"
    assert args.quantization == "bitsandbytes"
    assert args.language_model_only is True
    assert args.mode == "gold_eval"
