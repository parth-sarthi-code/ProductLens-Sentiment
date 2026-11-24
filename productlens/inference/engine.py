"""
ProductLens 3.0 — High-Precision Review Analysis Engine.

Preserves sentence/clause level attention focus for compact LLMs (LFM3-1.7B),
eliminating context dilution and guaranteeing 100% tight verbatim evidence extraction.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad
from productlens.validator.compact_parser import CompactFormatParser
from productlens.validator.substring import SubstringValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "stage3_lfm3_merged"

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews. "
    "Analyze the customer review and extract fine-grained aspect sentiment quads adhering strictly to the compact format: "
    "aspect|category|sentiment|confidence|evidence. Use 'null' for implicit aspects. Output 'NONE' if no aspects are present."
)

APPAREL_KEYWORDS = {
    "fabric", "cotton", "shirt", "t-shirt", "tshirt", "jeans", "dress", "pant", "pants",
    "cloth", "clothes", "clothing", "sizing", "size", "fit", "fitting", "stitch", "stitching",
    "neckline", "sleeve", "collar", "wash", "shrink", "shrank", "shrinkage", "color bleed",
    "fade", "faded", "drape", "wear", "wearing", "polyester", "silk", "wool", "kurta", "saree"
}

# Electronics category noun fallbacks for ambiguous terms
ELECTRONICS_CATEGORY_FALLBACKS = {
    "speaker": "SOUND",
    "speakers": "SOUND",
    "sound": "SOUND",
    "audio": "SOUND",
    "volume": "SOUND",
    "mic": "SOUND",
    "microphone": "SOUND",
    "gyro": "PERFORMANCE",
    "gyro delay": "PERFORMANCE",
    "fps": "PERFORMANCE",
    "lag": "PERFORMANCE",
    "speed": "PERFORMANCE",
    "gaming": "PERFORMANCE",
    "gaming performance": "PERFORMANCE",
    "battery": "BATTERY",
    "charger": "BATTERY",
    "charging": "BATTERY",
    "display": "DISPLAY",
    "screen": "DISPLAY",
    "panel": "DISPLAY",
    "brightness": "DISPLAY",
}


class ValidatedAspectResult(BaseModel):
    """Clean API output representing a validated aspect-sentiment quad."""
    aspect: Optional[str]
    category: str
    sentiment: str
    confidence: float
    evidence: str
    aspect_start: Optional[int] = None
    aspect_end: Optional[int] = None
    evidence_start: int
    evidence_end: int


class ReviewAnalysisResult(BaseModel):
    """Complete structured response for a single customer review."""
    review_text: str
    domain: str
    domain_label: str
    latency_ms: float
    total_aspects: int
    net_sentiment_score: float = Field(
        ..., description="Normalized score from -1.0 (purely negative) to +1.0 (purely positive)"
    )
    sentiment_counts: Dict[str, int]
    category_breakdown: Dict[str, Dict[str, int]]
    aspects: List[ValidatedAspectResult]


class ReviewAnalysisEngine:
    """
    Production-grade inference engine with sentence segmentation,
    fine-tuned LFM3-1.7B execution, and deterministic substring validation.
    """

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.model_path = model_path or MODEL_PATH
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Loads tokenizer and weights into GPU memory."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {self.model_path} does not exist.")

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else "cpu",
            trust_remote_code=True,
        )
        self.model.eval()
        load_time = time.perf_counter() - t0
        print(f"[ReviewAnalysisEngine] Model loaded in {load_time:.2f}s on {self.device}.")

    @classmethod
    def detect_vertical(cls, text: str) -> str:
        """Determines product vertical from text keywords."""
        lower = text.lower()
        if any(k in lower for k in APPAREL_KEYWORDS):
            return "apparel_fashion"
        return "electronics_tech"

    @classmethod
    def segment_text(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text into natural clause/sentence segments while tracking exact
        character offsets in the original full string.
        """
        segments = []
        raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
        cursor = 0

        for line in raw_lines:
            # If line is long (>120 chars) and contains sentence-ending punctuation, split into sentences
            if len(line) > 120 and re.search(r"[.!?]\s+[A-Za-z]", line):
                clauses = re.split(r"(?<=[.!?])\s+", line)
            else:
                clauses = [line]

            for c in clauses:
                clean_clause = c.strip()
                if not clean_clause:
                    continue
                idx = text.find(clean_clause, cursor)
                if idx == -1:
                    idx = text.find(clean_clause)
                if idx != -1:
                    segments.append((clean_clause, idx, idx + len(clean_clause)))
                    cursor = idx + len(clean_clause)

        # Fallback if no segments extracted
        if not segments and text.strip():
            clean = text.strip()
            idx = text.find(clean)
            segments.append((clean, max(0, idx), max(0, idx) + len(clean)))

        return segments

    def _infer_segment(self, segment_text: str, vertical: str) -> Tuple[List[AspectQuad], str]:
        """Runs model on an individual sentence segment."""
        user_prompt = (
            f"Product Domain: {vertical}\nReview Text:\n\"\"\"{segment_text}\"\"\"\n\n"
            "Extract all aspect quads adhering to the compact format: aspect|category|sentiment|confidence|evidence"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse and validate with SubstringValidator on this segment
        validated_quads, _, _ = CompactFormatParser.parse_and_validate_review(
            text=segment_text,
            compact_output=raw_output,
            vertical=vertical,
        )
        return validated_quads, raw_output

    def analyze(self, text: str, vertical_override: Optional[str] = None) -> ReviewAnalysisResult:
        """
        Analyzes a full review by segmenting it into focused clauses,
        performing per-segment extraction, and projecting verified character spans.
        """
        t0 = time.perf_counter()
        vertical = vertical_override or self.detect_vertical(text)
        domain_label = "Electronics & Tech" if vertical == "electronics_tech" else "Apparel & Fashion"

        segments = self.segment_text(text)
        all_aspects: List[ValidatedAspectResult] = []
        seen_keys = set()

        for seg_text, seg_start, seg_end in segments:
            seg_quads, _ = self._infer_segment(seg_text, vertical)

            for q in seg_quads:
                # Project character offsets from segment to full text
                full_asp_start = (q.aspect_start + seg_start) if q.aspect_start is not None else None
                full_asp_end = (q.aspect_end + seg_start) if q.aspect_end is not None else None
                full_evi_start = q.evidence_start + seg_start
                full_evi_end = q.evidence_end + seg_start

                # Verify containment and self-heal offsets against full original text
                if full_asp_start is not None and full_asp_end is not None:
                    if text[full_asp_start:full_asp_end] != q.aspect:
                        idx = text.find(q.aspect, seg_start)
                        if idx != -1:
                            full_asp_start, full_asp_end = idx, idx + len(q.aspect)
                if text[full_evi_start:full_evi_end] != q.evidence:
                    idx = text.find(q.evidence, seg_start)
                    if idx != -1:
                        full_evi_start, full_evi_end = idx, idx + len(q.evidence)

                # Check category fallback for OTHER
                category = q.category
                if category == "OTHER" and q.aspect:
                    asp_lower = q.aspect.lower()
                    if asp_lower in ELECTRONICS_CATEGORY_FALLBACKS:
                        category = ELECTRONICS_CATEGORY_FALLBACKS[asp_lower]

                # Deduplicate identical aspect + evidence spans
                key = (q.aspect, category, q.sentiment.value, full_evi_start, full_evi_end)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                all_aspects.append(
                    ValidatedAspectResult(
                        aspect=q.aspect,
                        category=category,
                        sentiment=q.sentiment.value,
                        confidence=q.confidence,
                        evidence=q.evidence,
                        aspect_start=full_asp_start,
                        aspect_end=full_asp_end,
                        evidence_start=full_evi_start,
                        evidence_end=full_evi_end,
                    )
                )

        total_elapsed = time.perf_counter() - t0

        # Calculate sentiment counts and category breakdown
        counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
        cat_breakdown: Dict[str, Dict[str, int]] = {}

        for a in all_aspects:
            counts[a.sentiment] = counts.get(a.sentiment, 0) + 1
            if a.category not in cat_breakdown:
                cat_breakdown[a.category] = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
            cat_breakdown[a.category][a.sentiment] += 1

        total = len(all_aspects)
        if total > 0:
            net_sentiment = (counts["positive"] - counts["negative"]) / total
        else:
            net_sentiment = 0.0

        return ReviewAnalysisResult(
            review_text=text,
            domain=vertical,
            domain_label=domain_label,
            latency_ms=round(total_elapsed * 1000, 1),
            total_aspects=total,
            net_sentiment_score=round(net_sentiment, 3),
            sentiment_counts=counts,
            category_breakdown=cat_breakdown,
            aspects=all_aspects,
        )
