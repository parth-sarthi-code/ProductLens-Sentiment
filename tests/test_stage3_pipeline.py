"""
Unit tests for ProductLens 3.0 Stage 3 Training and Pipeline components.
Verifies completion-only loss masking, dynamic collator padding, and data integrity.
"""

import json
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from scripts.stage3_train_lfm3_local import ABSACompletionDataset, CompletionDataCollator
from productlens.schemas.quads import RawCompactQuad
from productlens.validator.compact_parser import CompactFormatParser
from productlens.aggregation.engine import Stage4AggregationEngine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_DATA_PATH = PROJECT_ROOT / "data" / "stage3" / "stage3_val.jsonl"


@pytest.fixture(scope="module")
def tokenizer():
    # Use cached LiquidAI/LFM3-1.7B tokenizer
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def test_absa_completion_dataset_masking(tokenizer):
    """Verifies that prompt tokens have label=-100 and assistant tokens have valid token IDs."""
    dataset = ABSACompletionDataset(VAL_DATA_PATH, tokenizer=tokenizer, max_seq_length=512)
    assert len(dataset) > 0, "Dataset should contain validation samples"

    sample = dataset[0]
    input_ids = sample["input_ids"]
    labels = sample["labels"]
    attention_mask = sample["attention_mask"]

    assert len(input_ids) == len(labels) == len(attention_mask)

    # Check that there is at least one -100 (masked prompt) and at least one non -100 (target)
    masked_indices = (labels == -100).nonzero(as_tuple=True)[0]
    unmasked_indices = (labels != -100).nonzero(as_tuple=True)[0]

    assert len(masked_indices) > 0, "Prompt tokens must be masked with -100"
    assert len(unmasked_indices) > 0, "Target tokens must not be masked"

    # Verify that all masked tokens appear BEFORE unmasked tokens (completion-only)
    assert masked_indices[-1].item() < unmasked_indices[0].item(), (
        "All prompt tokens must precede target completion tokens"
    )

    # Decode unmasked tokens and verify they contain the compact target
    unmasked_tokens = labels[unmasked_indices].tolist()
    decoded_target = tokenizer.decode(unmasked_tokens)
    assert "|" in decoded_target or "NONE" in decoded_target, (
        f"Target text should be compact quad format or NONE, got: {decoded_target}"
    )


def test_completion_data_collator_padding(tokenizer):
    """Verifies dynamic batch padding for input_ids, attention_mask, and labels."""
    dataset = ABSACompletionDataset(VAL_DATA_PATH, tokenizer=tokenizer, max_seq_length=512)
    collator = CompletionDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    sample1 = dataset[0]
    sample2 = dataset[1]

    batch = collator([sample1, sample2])

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    assert input_ids.shape == attention_mask.shape == labels.shape
    assert input_ids.shape[0] == 2
    # Verify padded to multiple of 8
    assert input_ids.shape[1] % 8 == 0

    # Verify pad tokens
    pad_id = tokenizer.pad_token_id
    for b_idx in range(2):
        non_pad_count = attention_mask[b_idx].sum().item()
        if non_pad_count < input_ids.shape[1]:
            # Padded region in input_ids should equal pad_id
            assert (input_ids[b_idx, non_pad_count:] == pad_id).all()
            # Padded region in labels should equal -100
            assert (labels[b_idx, non_pad_count:] == -100).all()
            # Padded region in attention_mask should equal 0
            assert (attention_mask[b_idx, non_pad_count:] == 0).all()


def test_stage4_aggregation_from_quads():
    """Verifies that Stage 4 Aggregation engine runs and produces valid scorecards."""
    text = "The display is crystal clear and bright, but the battery life is terrible."
    compact_output = (
        "display|DISPLAY|positive|0.95|The display is crystal clear and bright\n"
        "battery life|BATTERY|negative|0.90|the battery life is terrible"
    )
    validated_quads, total, valid = CompactFormatParser.parse_and_validate_review(
        text=text,
        compact_output=compact_output,
        vertical="electronics_tech",
    )
    assert len(validated_quads) == 2

    engine = Stage4AggregationEngine(use_embeddings=False)
    report = engine.aggregate(
        product_id="TEST_PROD",
        quad_records=[(q, "rev_001") for q in validated_quads],
        vertical="electronics_tech",
        total_reviews=1,
    )
    assert report.total_reviews_analyzed == 1
    assert report.total_aspect_mentions == 2
    assert len(report.features) == 2
    md = report.to_markdown()
    assert "TEST_PROD" in md
    assert "Top Product Pain Points" in md
