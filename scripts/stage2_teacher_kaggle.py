#!/usr/bin/env python3
"""
ProductLens 3.0: Stage 2 Kaggle Teacher & Dual-Pass Consensus Pipeline
Offline high-volume silver labeling and gold evaluation using vLLM on Kaggle T4 GPUs.

Key Capabilities:
1. vLLM with modern StructuredOutputsParams(json=...) for strict Pydantic JSON decoding.
2. 4-bit quantization (bitsandbytes), float16 dtype, language_model_only=True, enable_thinking=False.
3. Dual-pass consensus (Pass 1 @ temp 0.0, Pass 2 @ temp 0.2) + Stage 1 deterministic validator.
4. Resumable chunked parquet checkpointing (survives Kaggle session timeouts).
5. Two modes:
   - gold_eval: upper-bound benchmark against the 350-review Gold Set.
   - silver_label: high-volume dual-pass consensus labeling for Amazon Reviews 2023.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

# Avoid FlashInfer sampler issues on Tesla T4
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pydantic import BaseModel, Field

from productlens.schemas.enums import (
    ApparelCategory,
    ElectronicsCategory,
    Sentiment,
    VerticalDomain,
    normalize_category,
)
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator
from productlens.gold.schema import GoldReview
from productlens.gold.manager import GoldSetManager
from productlens.evaluation.runner import Stage1Evaluator


# ==============================================================================
# 1. Pydantic Schemas for Constrained Structured Decoding
# ==============================================================================

CanonicalCategory = Literal[
    "DISPLAY", "BATTERY", "SOUND", "PERFORMANCE", "BUILD_QUALITY",
    "CONNECTIVITY", "SOFTWARE", "DELIVERY_INSTALLATION", "VALUE",
    "FABRIC_MATERIAL", "FIT_SIZING", "COLOR_PRINT", "STITCHING_DURABILITY",
    "COMFORT", "DELIVERY", "OTHER",
]

SentimentType = Literal["positive", "negative", "neutral", "mixed"]


class TeacherAspectQuad(BaseModel):
    """Aspect quad extracted by the teacher model."""
    aspect: Optional[str] = Field(
        default=None,
        description="Explicit product aspect term mentioned in the review, or null for implicit aspects"
    )
    category: CanonicalCategory = Field(
        ...,
        description="Canonical feature taxonomy category"
    )
    sentiment: SentimentType = Field(
        ...,
        description="Sentiment polarity: positive, negative, neutral, or mixed"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0 - 1.0]"
    )
    evidence: str = Field(
        ...,
        description="Exact substring quote from the review text providing supporting evidence"
    )


class TeacherExtractionOutput(BaseModel):
    """Top-level structured JSON wrapper for vLLM decoding."""
    quads: List[TeacherAspectQuad] = Field(
        default_factory=list,
        description="List of extracted aspect-sentiment quads"
    )


def get_structured_output_schema_json() -> str:
    """Returns the strict JSON schema string for vLLM StructuredOutputsParams."""
    return json.dumps(TeacherExtractionOutput.model_json_schema())


# ==============================================================================
# 2. Dual-Pass Prompt Templates (Pass 1 & Pass 2 Paraphrases)
# ==============================================================================

PASS1_SYSTEM_PROMPT = """You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews.
Your task is to analyze customer feedback and extract fine-grained aspect sentiment quads.

Strict Rules:
1. 'aspect': MUST BE an exact verbatim substring noun/entity copied directly from the review text (e.g., "battery life", "fabric quality", "picture clarity", "bass").
   CRITICAL: NEVER set 'aspect' to a category enum name (DO NOT output "DISPLAY", "BATTERY", "SOUND", "FABRIC_MATERIAL", "FIT_SIZING", "VALUE", etc. as the aspect).
2. If an aspect is implicit (no specific noun term is named, e.g., "worth every penny", "paisa vasool", "terrible purchase", "too expensive"), set 'aspect' to null.
3. Map every aspect to the canonical category:
   - Electronics: DISPLAY, BATTERY, SOUND, PERFORMANCE, BUILD_QUALITY, CONNECTIVITY, SOFTWARE, DELIVERY_INSTALLATION, VALUE, OTHER.
   - Apparel: FABRIC_MATERIAL, FIT_SIZING, COLOR_PRINT, STITCHING_DURABILITY, COMFORT, DELIVERY, VALUE, OTHER.
4. Classify sentiment strictly into: "positive", "negative", "neutral", or "mixed".
5. Evidence MUST be an exact verbatim substring from the review text containing the opinion or rationale. Do not paraphrase or add punctuation.
6. Provide calibrated confidence between 0.0 and 1.0."""

PASS2_SYSTEM_PROMPT = """You are a precise review intelligence analyst for tech and retail products.
Dissect the provided user review into feature-level sentiment quads:

Guidelines:
1. 'aspect': Must be the concrete noun or phrase extracted verbatim from the review text (e.g., "screen", "keyboard", "stitching").
   CRITICAL: NEVER use uppercase taxonomy categories (like "DISPLAY", "PERFORMANCE", "FABRIC_MATERIAL", "VALUE") in the 'aspect' field. For implied aspects with no concrete noun in text, set 'aspect' to null.
2. Assign the canonical category that best matches the product vertical (e.g., DISPLAY, BATTERY, FABRIC_MATERIAL, FIT_SIZING, VALUE).
3. Grade sentiment polarity as "positive", "negative", "neutral", or "mixed".
4. The evidence field MUST be a verbatim quote copied directly from the review text without any modification.
5. Assign a confidence score from 0.0 to 1.0."""

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    # Example 1: English Tech
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\n"
            "Review Text:\n"
            '"""The OLED screen is incredibly bright and colors pop, but battery life drops quickly under heavy load."""\n\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "OLED screen",
                    "category": "DISPLAY",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "The OLED screen is incredibly bright and colors pop",
                },
                {
                    "aspect": "battery life",
                    "category": "BATTERY",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "battery life drops quickly under heavy load",
                },
            ]
        }),
    },
    # Example 2: English Apparel
    {
        "role": "user",
        "content": (
            "Product Domain: apparel_fashion\n"
            "Review Text:\n"
            '"""The cotton fabric feels soft against the skin, but stitching came undone along the collar seam after two washes."""\n\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "cotton fabric",
                    "category": "FABRIC_MATERIAL",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "The cotton fabric feels soft against the skin",
                },
                {
                    "aspect": "stitching",
                    "category": "STITCHING_DURABILITY",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "stitching came undone along the collar seam after two washes",
                },
            ]
        }),
    },
    # Example 3: Hinglish Contrast
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\n"
            "Review Text:\n"
            '"""Earphones ka bass dhamakedar hai lekin mic quality calls ke liye bilkul acchi nahi hai."""\n\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": "bass",
                    "category": "SOUND",
                    "sentiment": "positive",
                    "confidence": 0.98,
                    "evidence": "Earphones ka bass dhamakedar hai",
                },
                {
                    "aspect": "mic quality",
                    "category": "SOUND",
                    "sentiment": "negative",
                    "confidence": 0.95,
                    "evidence": "mic quality calls ke liye bilkul acchi nahi hai",
                },
            ]
        }),
    },
    # Example 4: Implicit aspect (aspect: null)
    {
        "role": "user",
        "content": (
            "Product Domain: electronics_tech\n"
            "Review Text:\n"
            '"""Paisa vasool mobile hai bhai, definitely worth every penny."""\n\n'
            "Extract all aspect quads as JSON adhering to the schema."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "quads": [
                {
                    "aspect": None,
                    "category": "VALUE",
                    "sentiment": "positive",
                    "confidence": 0.95,
                    "evidence": "Paisa vasool mobile hai bhai, definitely worth every penny",
                },
            ]
        }),
    },
]


def build_chat_messages(
    system_prompt: str,
    review_text: str,
    vertical: VerticalDomain | str,
    include_few_shot: bool = True,
) -> List[Dict[str, str]]:
    """Builds chat messages array for Qwen chat template."""
    vert_str = vertical.value if isinstance(vertical, VerticalDomain) else str(vertical)
    messages = [{"role": "system", "content": system_prompt}]
    if include_few_shot:
        messages.extend(FEW_SHOT_EXAMPLES)
    user_prompt = f"Product Domain: {vert_str}\nReview Text:\n\"\"\"{review_text}\"\"\"\n\nExtract all aspect quads as JSON adhering to the schema."
    messages.append({"role": "user", "content": user_prompt})
    return messages


# ==============================================================================
# 3. Consensus Engine & Substring Verification
# ==============================================================================

def compute_char_iou(span1: Tuple[Optional[int], Optional[int]], span2: Tuple[Optional[int], Optional[int]]) -> float:
    """Computes character-span Intersection over Union (IoU) between two (start, end) tuples."""
    s1, e1 = span1
    s2, e2 = span2
    if s1 is None or e1 is None or s2 is None or e2 is None:
        return 0.0
    inter_start = max(s1, s2)
    inter_end = min(e1, e2)
    intersection = max(0, inter_end - inter_start)
    len1 = max(0, e1 - s1)
    len2 = max(0, e2 - s2)
    union = len1 + len2 - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


class DualPassConsensusEngine:
    """Validates and merges Pass 1 and Pass 2 predictions through consensus."""

    @staticmethod
    def parse_teacher_json(raw_json_str: str) -> List[TeacherAspectQuad]:
        """Safely parses teacher raw JSON response into TeacherAspectQuad items."""
        cleaned = raw_json_str.strip()
        if not cleaned:
            return []
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "quads" in data:
                return [TeacherAspectQuad.model_validate(q) for q in data["quads"]]
            elif isinstance(data, list):
                return [TeacherAspectQuad.model_validate(q) for q in data]
            return []
        except Exception:
            return []

    @classmethod
    def validate_teacher_quads(
        cls,
        text: str,
        teacher_quads: List[TeacherAspectQuad],
        vertical: VerticalDomain | str,
    ) -> List[AspectQuad]:
        """Passes candidate quads through the deterministic Stage 1 substring validator."""
        validated: List[AspectQuad] = []
        for tq in teacher_quads:
            raw = RawCompactQuad(
                aspect=tq.aspect,
                category=tq.category,
                sentiment=Sentiment(tq.sentiment),
                confidence=tq.confidence,
                evidence=tq.evidence,
            )
            quad, err = SubstringValidator.validate_quad(text, raw, vertical)
            if quad is not None:
                validated.append(quad)
        return validated

    @classmethod
    def compute_consensus(
        cls,
        text: str,
        pass1_quads: List[TeacherAspectQuad],
        pass2_quads: List[TeacherAspectQuad],
        vertical: VerticalDomain | str,
        min_aspect_iou: float = 0.5,
    ) -> Tuple[List[AspectQuad], Dict[str, int]]:
        """
        Validates both passes through deterministic validator and retains quads where both passes agree:
        1. Explicit aspects: character-span IoU >= min_aspect_iou (0.5) AND exact match on (category, sentiment).
           Keeps the Pass 1 (temp=0.0) span/evidence and averages confidence scores.
        2. Implicit aspects (aspect: null): exact match on (category, sentiment) and averages confidence scores.
        """
        val1 = cls.validate_teacher_quads(text, pass1_quads, vertical)
        val2 = cls.validate_teacher_quads(text, pass2_quads, vertical)

        stats = {
            "pass1_raw": len(pass1_quads),
            "pass2_raw": len(pass2_quads),
            "pass1_valid": len(val1),
            "pass2_valid": len(val2),
            "consensus_kept": 0,
        }

        consensus_quads: List[AspectQuad] = []
        pool2 = list(val2)

        for q1 in val1:
            cat1_key = normalize_category(q1.category, vertical)
            sent1_key = q1.sentiment.value

            best_idx: Optional[int] = None
            best_iou: float = -1.0

            for idx, q2 in enumerate(pool2):
                cat2_key = normalize_category(q2.category, vertical)
                sent2_key = q2.sentiment.value

                if cat1_key != cat2_key or sent1_key != sent2_key:
                    continue

                if q1.aspect is None and q2.aspect is None:
                    best_idx = idx
                    best_iou = 1.0
                    break
                elif q1.aspect is not None and q2.aspect is not None:
                    if (
                        q1.aspect_start is not None
                        and q1.aspect_end is not None
                        and q2.aspect_start is not None
                        and q2.aspect_end is not None
                    ):
                        iou = compute_char_iou(
                            (q1.aspect_start, q1.aspect_end),
                            (q2.aspect_start, q2.aspect_end),
                        )
                        if iou >= min_aspect_iou and iou > best_iou:
                            best_iou = iou
                            best_idx = idx

            if best_idx is not None:
                q2 = pool2.pop(best_idx)
                avg_conf = round(float((q1.confidence + q2.confidence) / 2.0), 4)

                consensus_quad = AspectQuad(
                    aspect=q1.aspect,
                    category=cat1_key,
                    sentiment=q1.sentiment,
                    confidence=avg_conf,
                    evidence=q1.evidence,
                    aspect_start=q1.aspect_start,
                    aspect_end=q1.aspect_end,
                    evidence_start=q1.evidence_start,
                    evidence_end=q1.evidence_end,
                )
                consensus_quads.append(consensus_quad)

        stats["consensus_kept"] = len(consensus_quads)
        return consensus_quads, stats


# ==============================================================================
# 4. Checkpoint & Resumability Manager
# ==============================================================================

class CheckpointManager:
    """Handles resumable chunked parquet outputs so Kaggle timeouts never lose work."""

    @staticmethod
    def get_processed_review_ids(output_dir: Path | str) -> Set[str]:
        out_path = Path(output_dir)
        processed: Set[str] = set()
        if not out_path.exists():
            return processed

        parquet_files = sorted(out_path.glob("*.parquet"))
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf, columns=["review_id"])
                processed.update(df["review_id"].astype(str).tolist())
            except Exception as e:
                print(f"⚠️ Warning: Could not read {pf.name}: {e}")
        return processed

    @staticmethod
    def get_next_batch_index(output_dir: Path | str, prefix: str = "batch") -> int:
        out_path = Path(output_dir)
        if not out_path.exists():
            return 1
        parquet_files = list(out_path.glob(f"{prefix}_*.parquet"))
        if not parquet_files:
            return 1
        indices = []
        for pf in parquet_files:
            try:
                idx = int(pf.stem.split("_")[-1])
                indices.append(idx)
            except ValueError:
                pass
        return max(indices) + 1 if indices else 1

    @staticmethod
    def save_batch_parquet(
        output_dir: Path | str,
        batch_idx: int,
        records: List[Dict[str, Any]],
        prefix: str = "batch",
    ) -> Path:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"{prefix}_{batch_idx:04d}.parquet"

        if records:
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(columns=[
                "review_id", "product_id", "text", "aspect", "category",
                "sentiment", "confidence", "evidence", "aspect_start",
                "aspect_end", "evidence_start", "evidence_end", "vertical",
            ])

        df.to_parquet(file_path, index=False)
        return file_path


# ==============================================================================
# 5. vLLM Engine Wrapper
# ==============================================================================

class VLLMTeacherEngine:
    """Initializes and runs vLLM with StructuredOutputsParams and chat template."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",
        quantization: Optional[str] = "bitsandbytes",
        max_model_len: int = 4096,
        max_num_seqs: int = 64,
        gpu_memory_utilization: float = 0.90,
        language_model_only: bool = True,
    ):
        try:
            from vllm import LLM, SamplingParams
            try:
                from vllm.sampling_params import StructuredOutputsParams
            except ImportError:
                from vllm.model_executor.guided_decoding.guided_fields import StructuredOutputsParams
        except ImportError as e:
            raise ImportError(
                "vLLM is required to run inference on Kaggle GPUs. Install with: pip install vllm"
            ) from e

        self.SamplingParams = SamplingParams
        self.StructuredOutputsParams = StructuredOutputsParams
        schema_json = get_structured_output_schema_json()

        print(f"🚀 Initializing vLLM Teacher: {model_name}")
        print(f"   • Tensor Parallel Size: {tensor_parallel_size}")
        print(f"   • Quantization:         {quantization}")
        print(f"   • Dtype:                {dtype}")
        print(f"   • Max Model Len:        {max_model_len}")
        print(f"   • Max Num Seqs:         {max_num_seqs}")
        print(f"   • Language Model Only:  {language_model_only}")

        llm_init_kwargs: Dict[str, Any] = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        if quantization and quantization.lower() != "none":
            llm_init_kwargs["quantization"] = quantization
            # Support load_format="bitsandbytes" for older vLLM wheels
            if quantization.lower() == "bitsandbytes":
                llm_init_kwargs["load_format"] = "bitsandbytes"

        # language_model_only is a top-level LLM argument in vLLM EngineArgs / MultiModalConfig
        if language_model_only:
            llm_init_kwargs["language_model_only"] = True

        try:
            self.llm = LLM(**llm_init_kwargs)
        except TypeError as e:
            err_msg = str(e)
            if "language_model_only" in err_msg:
                llm_init_kwargs.pop("language_model_only", None)
            if "load_format" in err_msg:
                llm_init_kwargs.pop("load_format", None)
            self.llm = LLM(**llm_init_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.schema_json = schema_json

    def format_prompts(
        self,
        reviews: List[Tuple[str, str]],  # (review_text, vertical)
        system_prompt: str,
    ) -> List[str]:
        """Applies chat template with enable_thinking=False."""
        formatted: List[str] = []
        for text, vert in reviews:
            messages = build_chat_messages(system_prompt, text, vert)
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            formatted.append(prompt_str)
        return formatted

    def generate_structured_batch(
        self,
        formatted_prompts: List[str],
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> List[str]:
        """Runs batched inference with modern StructuredOutputsParams."""
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            structured_outputs=self.StructuredOutputsParams(json=self.schema_json),
        )
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]


# ==============================================================================
# 6. Execution Modes: Gold Eval & Silver Labeling
# ==============================================================================

def run_gold_eval_mode(args, teacher_engine: Optional[VLLMTeacherEngine] = None):
    """Evaluates teacher model on the 350-review Gold Set and prints Stage 1 report."""
    print("\n" + "=" * 72)
    print("📋 Stage 2: Running Teacher Evaluation on Stage 1 Gold Set")
    print("=" * 72)

    reviews = GoldSetManager.load_reviews()
    if args.limit and args.limit > 0:
        reviews = reviews[:args.limit]
    print(f"Loaded {len(reviews)} gold reviews for evaluation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "teacher_gold_predictions.jsonl"

    if teacher_engine is None:
        teacher_engine = VLLMTeacherEngine(
            model_name=args.model,
            tensor_parallel_size=args.tp,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=args.language_model_only,
        )

    t0 = time.perf_counter()
    prompt_pairs = [(r.text, r.vertical.value) for r in reviews]
    formatted = teacher_engine.format_prompts(prompt_pairs, PASS1_SYSTEM_PROMPT)

    print(f"Generating structured predictions for {len(formatted)} reviews...")
    raw_outputs = teacher_engine.generate_structured_batch(formatted, temperature=0.0)
    inference_time = time.perf_counter() - t0

    # Validate predictions through Stage 1 deterministic validator
    predicted_quads_by_id: Dict[str, List[AspectQuad]] = {}
    with open(predictions_file, "w", encoding="utf-8") as out_f:
        for review, raw_json in zip(reviews, raw_outputs):
            parsed = DualPassConsensusEngine.parse_teacher_json(raw_json)
            validated = DualPassConsensusEngine.validate_teacher_quads(
                text=review.text,
                teacher_quads=parsed,
                vertical=review.vertical,
            )
            predicted_quads_by_id[review.review_id] = validated

            record = {
                "review_id": review.review_id,
                "text": review.text,
                "raw_generated": raw_json,
                "validated_quads": [q.model_dump() for q in validated],
            }
            out_f.write(json.dumps(record) + "\n")

    print(f"✅ Teacher predictions saved to {predictions_file}")

    # Run Stage 1 evaluation harness
    evaluator = Stage1Evaluator(reviews)
    results = evaluator.evaluate_predictions(lambda r: predicted_quads_by_id.get(r.review_id, []))

    ov = results["overall"]
    print("\n" + "=" * 72)
    print("📊 Teacher Performance on Stage 1 Gold Set")
    print("=" * 72)
    print(f"Model:                         {args.model}")
    print(f"Total Reviews:                 {results['total_reviews']}")
    print(f"Total Matched Quads:           {results['total_matched_quads']}")
    print(f"Explicit Matched Count:        {ov['explicit_matched_count']} / {ov['total_gold_explicit']}")
    print(f"Implicit Matched Count:        {ov['implicit_matched_count']} / {ov['total_gold_implicit']}")
    print(f"Explicit Aspect F1 (Strict):   {ov['aspect_f1'] * 100:.2f}% (P: {ov['aspect_precision']*100:.2f}%, R: {ov['aspect_recall']*100:.2f}%)")
    print(f"Explicit Aspect F1 (IoU>=0.5): {ov.get('overlap_aspect_f1', 0.0) * 100:.2f}% (P: {ov.get('overlap_aspect_precision', 0.0)*100:.2f}%, R: {ov.get('overlap_aspect_recall', 0.0)*100:.2f}%)")
    print(f"Implicit Category F1:          {ov['implicit_category_f1'] * 100:.2f}% (P: {ov['implicit_category_precision']*100:.2f}%, R: {ov['implicit_category_recall']*100:.2f}%)")
    print(f"Sentiment Macro-F1:            {ov['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Expected Calib Error (ECE):    {ov['expected_calibration_error']:.4f}")
    print(f"--- Subsets ---")
    print(f"English Aspect F1:             {results['english']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {results['english'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {results['english']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Hinglish Aspect F1:            {results['hinglish']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {results['hinglish'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {results['hinglish']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Hard Cases Aspect F1:          {results['hard_cases']['aspect_f1'] * 100:.2f}% (IoU>=0.5: {results['hard_cases'].get('overlap_aspect_f1', 0.0)*100:.2f}%) | Sent F1: {results['hard_cases']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Throughput (sec / 500 revs):   {(inference_time / len(reviews)) * 500:.2f}s")
    print("=" * 72)


def run_silver_label_mode(args, teacher_engine: Optional[VLLMTeacherEngine] = None):
    """High-volume dual-pass consensus labeling on Amazon Reviews 2023 with resumability."""
    print("\n" + "=" * 72)
    print("🏷️  Stage 2: High-Volume Dual-Pass Silver Dataset Labeling")
    print("=" * 72)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source dataset
    input_path = Path(args.input_data)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Amazon 2023 dataset not found: {input_path}")

    print(f"Loading candidate source reviews from {input_path}...")
    if input_path.suffix == ".parquet":
        df_source = pd.read_parquet(input_path)
    else:
        df_source = pd.read_json(input_path, lines=True)

    if args.limit and args.limit > 0:
        df_source = df_source.head(args.limit)

    total_candidates = len(df_source)
    print(f"Total candidate reviews in dataset: {total_candidates}")

    # Standardize column names
    if "review_id" not in df_source.columns:
        df_source["review_id"] = [f"amz_{i:07d}" for i in range(len(df_source))]
    if "product_id" not in df_source.columns:
        df_source["product_id"] = df_source.get("parent_asin", "UNKNOWN_PROD")
    if "vertical" not in df_source.columns:
        df_source["vertical"] = args.vertical

    # 2. Checkpoint & Resumability: filter already processed
    processed_ids = CheckpointManager.get_processed_review_ids(output_dir)
    if processed_ids:
        print(f"🔄 Checkpoint detected: Found {len(processed_ids)} already processed review_ids.")
        df_source = df_source[~df_source["review_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
        print(f"Remaining unprocessed reviews: {len(df_source)}")

    if len(df_source) == 0:
        print("🎉 All reviews in dataset have already been processed! Nothing to do.")
        return

    # 3. Initialize Teacher Engine
    if teacher_engine is None:
        teacher_engine = VLLMTeacherEngine(
            model_name=args.model,
            tensor_parallel_size=args.tp,
            dtype=args.dtype,
            quantization=args.quantization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            language_model_only=args.language_model_only,
        )

    # 4. Process in batches
    batch_idx = CheckpointManager.get_next_batch_index(output_dir)
    batch_size = args.batch_size
    num_batches = (len(df_source) + batch_size - 1) // batch_size

    print(f"Beginning batch inference across {num_batches} batches (batch_size={batch_size})...")

    total_raw_p1 = 0
    total_raw_p2 = 0
    total_valid = 0
    total_consensus_kept = 0

    for b in range(num_batches):
        b_start = b * batch_size
        b_end = min((b + 1) * batch_size, len(df_source))
        chunk = df_source.iloc[b_start:b_end]

        t_batch_start = time.perf_counter()
        review_tuples = [(str(row["text"]), str(row["vertical"])) for _, row in chunk.iterrows()]

        # Pass 1: Temp 0.0
        prompts_p1 = teacher_engine.format_prompts(review_tuples, PASS1_SYSTEM_PROMPT)
        outputs_p1 = teacher_engine.generate_structured_batch(prompts_p1, temperature=args.temperature_pass1)

        # Pass 2: Temp 0.2
        prompts_p2 = teacher_engine.format_prompts(review_tuples, PASS2_SYSTEM_PROMPT)
        outputs_p2 = teacher_engine.generate_structured_batch(prompts_p2, temperature=args.temperature_pass2)

        # Consensus & Substring Validation
        batch_records: List[Dict[str, Any]] = []
        batch_raw_p1 = 0
        batch_raw_p2 = 0
        batch_valid = 0
        batch_consensus = 0

        for (_, row), raw1, raw2 in zip(chunk.iterrows(), outputs_p1, outputs_p2):
            p1_quads = DualPassConsensusEngine.parse_teacher_json(raw1)
            p2_quads = DualPassConsensusEngine.parse_teacher_json(raw2)

            consensus_quads, stats = DualPassConsensusEngine.compute_consensus(
                text=str(row["text"]),
                pass1_quads=p1_quads,
                pass2_quads=p2_quads,
                vertical=str(row["vertical"]),
            )

            batch_raw_p1 += stats["pass1_raw"]
            batch_raw_p2 += stats["pass2_raw"]
            batch_valid += (stats["pass1_valid"] + stats["pass2_valid"])
            batch_consensus += stats["consensus_kept"]

            for q in consensus_quads:
                batch_records.append({
                    "review_id": str(row["review_id"]),
                    "product_id": str(row["product_id"]),
                    "text": str(row["text"]),
                    "aspect": q.aspect,
                    "category": q.category,
                    "sentiment": q.sentiment.value,
                    "confidence": q.confidence,
                    "evidence": q.evidence,
                    "aspect_start": q.aspect_start,
                    "aspect_end": q.aspect_end,
                    "evidence_start": q.evidence_start,
                    "evidence_end": q.evidence_end,
                    "vertical": str(row["vertical"]),
                })

        # Save Checkpoint Parquet
        saved_file = CheckpointManager.save_batch_parquet(output_dir, batch_idx, batch_records)
        elapsed_batch = time.perf_counter() - t_batch_start
        revs_per_sec = len(chunk) / elapsed_batch if elapsed_batch > 0 else 0.0

        total_raw_p1 += batch_raw_p1
        total_raw_p2 += batch_raw_p2
        total_valid += batch_valid
        total_consensus_kept += batch_consensus

        pass_rate = (batch_valid / max(1, batch_raw_p1 + batch_raw_p2)) * 100
        keep_rate = (batch_consensus / max(1, batch_valid // 2)) * 100 if batch_valid > 0 else 0.0

        print(
            f"📦 Batch {batch_idx:04d} [{b+1}/{num_batches}]: "
            f"{len(chunk)} reviews in {elapsed_batch:.1f}s ({revs_per_sec:.1f} rev/s) | "
            f"Consensus kept: {batch_consensus} quads | "
            f"Val Pass Rate: {pass_rate:.1f}% | "
            f"Keep Rate: {keep_rate:.1f}% -> {saved_file.name}"
        )

        batch_idx += 1

    print("\n" + "=" * 72)
    print("🎉 Silver Labeling Run Completed!")
    print(f"Total Raw Quads (Pass 1 + Pass 2): {total_raw_p1 + total_raw_p2}")
    print(f"Total Substring-Valid Quads:        {total_valid}")
    print(f"Total Agreed Consensus Quads Kept:  {total_consensus_kept}")
    print(f"Output files stored in:             {output_dir}")
    print("=" * 72)


# ==============================================================================
# 7. CLI Entry Point
# ==============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ProductLens 3.0: Stage 2 Kaggle Teacher & Consensus Silver Labeling"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gold_eval", "silver_label"],
        default="gold_eval",
        help="Operating mode: 'gold_eval' (eval on 350 Gold Set) or 'silver_label' (dual-pass consensus)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face model ID (e.g. Qwen/Qwen3.5-9B or Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (set to 2 for Kaggle dual-T4 GPUs)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weights dtype (float16 required on T4)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="bitsandbytes",
        help="Quantization method: bitsandbytes (4-bit), awq, gptq, or none",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model context length (capped to 2048 to conserve KV-cache VRAM on T4)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help="Maximum concurrent sequences for vLLM batching (capped to 64 to fit Mamba cache on T4)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM allocator",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        default=True,
        help="Skip vision encoder weights to free up KV-cache VRAM",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="/home/parth/Projects/ProductLens-2.0/data/refined_amazon_absa.parquet",
        help="Path to Amazon Reviews input dataset for silver labeling (e.g. refined_amazon_absa.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/parth/Projects/ProjectLens-3.0/data/stage2_teacher_output",
        help="Directory to save gold eval reports or chunked silver parquets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for checkpointed silver labeling (500–1000 recommended)",
    )
    parser.add_argument(
        "--vertical",
        type=str,
        default="electronics_tech",
        choices=["electronics_tech", "apparel_fashion"],
        help="Default vertical domain for input reviews",
    )
    parser.add_argument(
        "--temperature-pass1",
        type=float,
        default=0.0,
        help="Sampling temperature for Pass 1",
    )
    parser.add_argument(
        "--temperature-pass2",
        type=float,
        default=0.2,
        help="Sampling temperature for Pass 2 (slight variation for consensus)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of reviews to process (for testing)",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.mode == "gold_eval":
        run_gold_eval_mode(args)
    elif args.mode == "silver_label":
        run_silver_label_mode(args)


if __name__ == "__main__":
    main()
