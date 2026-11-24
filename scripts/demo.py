#!/usr/bin/env python3
"""
ProductLens 3.0 — End-to-End Local Demo Pipeline (scripts/demo.py).

Demonstrates the entire ProductLens 3.0 system running on the RTX 4050:
1. Input: Loads real product review streams (Electronics & Apparel, including Hinglish and sarcasm).
2. Extraction: Runs local LiquidAI/LFM3-1.7B student to emit compact quads.
3. Validation: Enforces deterministic SubstringValidator checks (100% exact substring character offsets).
4. Aggregation: Normalizes features, computes Bayesian-shrunk sentiment scores, defect impact ranking,
   and selects deduplicated verbatim evidence snippets.
5. Output: Generates executive Markdown & JSON ProductReport scorecards.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad
from productlens.validator.compact_parser import CompactFormatParser
from productlens.aggregation.engine import Stage4AggregationEngine
from productlens.aggregation.schemas import ProductReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("productlens_demo")

# Built-in diverse demonstration reviews
DEMO_ELECTRONICS_REVIEWS = [
    {
        "review_id": "elec_001",
        "text": "The OLED screen is absolutely gorgeous with deep blacks, but the battery life barely lasts 4 hours on medium brightness.",
    },
    {
        "review_id": "elec_002",
        "text": "Trackpad is smooth and responsive. Build quality feels premium aluminum, though the thermal fan gets loud under load.",
    },
    {
        "review_id": "elec_003",
        "text": "Keyboard travel to bohot achha hai lekin speaker sound quality bilkul bakwaas hai.",  # Hinglish
    },
    {
        "review_id": "elec_004",
        "text": "Oh wonderful, the charger stopped working completely after just one week! Great engineering.",  # Sarcasm / defect
    },
    {
        "review_id": "elec_005",
        "text": "Ridiculously overpriced for what it delivers, though customer service replaced the missing cable overnight.",
    },
]

DEMO_APPAREL_REVIEWS = [
    {
        "review_id": "app_001",
        "text": "The organic cotton fabric feels super soft and breathable, but the stitching around the neckline started fraying after the first wash.",
    },
    {
        "review_id": "app_002",
        "text": "Color bilkul same hai as pictured in the catalog, but sizing run thoda small around the chest.",  # Hinglish
    },
    {
        "review_id": "app_003",
        "text": "Perfect slim fit drape! Wore it to an evening party and received multiple compliments.",
    },
    {
        "review_id": "app_004",
        "text": "Shrank two sizes in cold water wash. Terrible shrinkage and color bleed on white collars.",
    },
]

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews. "
    "Analyze the customer review and extract fine-grained aspect sentiment quads adhering strictly to the compact format: "
    "aspect|category|sentiment|confidence|evidence. Use 'null' for implicit aspects. Output 'NONE' if no aspects are present."
)


def load_demo_model(
    model_path: Path,
    adapter_path: Optional[Path] = None,
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """Loads model with graceful fallback to base or CPU if needed."""
    logger.info("Initializing model from %s ...", model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = "auto" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if adapter_path and adapter_path.exists():
        logger.info("Loading base model with adapter from %s ...", adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        logger.info("Loading model directly...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


def extract_quads_for_review(
    model: Any,
    tokenizer: Any,
    text: str,
    vertical: str,
) -> Tuple[List[AspectQuad], str]:
    """Generates compact response and runs SubstringValidator."""
    user_content = (
        f"Product Domain: {vertical}\nReview Text:\n\"\"\"{text}\"\"\"\n\n"
        "Extract all aspect quads adhering to the compact format: aspect|category|sentiment|confidence|evidence"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Deterministic SubstringValidator
    validated_quads, _, _ = CompactFormatParser.parse_and_validate_review(
        text=text,
        compact_output=response_text,
        vertical=vertical,
    )
    return validated_quads, response_text


def run_pipeline(
    model: Any,
    tokenizer: Any,
    product_id: str,
    vertical: str,
    reviews: List[Dict[str, str]],
    output_dir: Path,
) -> ProductReport:
    """Chains Extraction -> Substring Validation -> Stage 4 Scorecard Generation."""
    print("\n" + "=" * 76)
    print(f"🚀 RUNNING PRODUCTLENS 3.0 END-TO-END PIPELINE: {product_id} ({vertical})")
    print("=" * 76)

    quad_records: List[Tuple[AspectQuad, str]] = []
    total_valid_quads = 0

    t_start = time.perf_counter()

    for idx, r in enumerate(reviews, 1):
        rev_id = r["review_id"]
        text = r["text"]
        print(f"\n[Review {idx}/{len(reviews)}] ID: {rev_id}")
        print(f"  Input Text: \"{text[:90]}{'...' if len(text) > 90 else ''}\"")

        quads, raw_output = extract_quads_for_review(model, tokenizer, text, vertical)
        print(f"  Raw Output:\n    {raw_output.replace(chr(10), chr(10) + '    ')}")
        print(f"  Validated Quads ({len(quads)} passed SubstringValidator):")
        for q in quads:
            asp_name = q.aspect if q.aspect else "[IMPLICIT]"
            print(f"    • {asp_name} [{q.category}] -> {q.sentiment.value.upper()} (conf: {q.confidence:.2f})")
            quad_records.append((q, rev_id))
            total_valid_quads += 1

    inference_time = time.perf_counter() - t_start
    print(f"\n⚡ Extraction & Validation completed in {inference_time:.2f}s ({len(reviews)/inference_time:.1f} rev/s)")

    # Stage 4 Aggregation Engine
    print("\n⚙️ Aggregating into Stage 4 Executive Product Scorecard...")
    engine = Stage4AggregationEngine(use_embeddings=False)
    report = engine.aggregate(
        product_id=product_id,
        quad_records=quad_records,
        vertical=vertical,
        total_reviews=len(reviews),
    )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{product_id}_scorecard.json"
    md_path = output_dir / f"{product_id}_scorecard.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    md_content = report.to_markdown()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"💾 Report JSON saved to: {json_path}")
    print(f"📄 Report Markdown saved to: {md_path}")
    print("\n" + md_content + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="ProductLens 3.0 End-to-End Demo")
    parser.add_argument("--model-path", type=str, default="outputs/stage3_lfm3_merged")
    parser.add_argument("--adapter-path", type=str, default="outputs/stage3_lfm3_lora")
    parser.add_argument("--output-dir", type=str, default="outputs/reports")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_p = PROJECT_ROOT / args.model_path
    adapter_p = PROJECT_ROOT / args.adapter_path

    if not model_p.exists() and not adapter_p.exists():
        logger.warning("Merged model %s not found. Using base LiquidAI/LFM3-1.7B...", model_p)
        model_p = Path("LiquidAI/LFM3-1.7B")

    model, tokenizer = load_demo_model(
        model_path=model_p,
        adapter_path=adapter_p if not (PROJECT_ROOT / args.model_path).exists() else None,
        device=args.device,
    )

    out_dir = PROJECT_ROOT / args.output_dir

    # Demo Run 1: Electronics Product (UltraBook Pro X1)
    run_pipeline(
        model=model,
        tokenizer=tokenizer,
        product_id="PROD_ELEC_X1_ULTRABOOK",
        vertical="electronics_tech",
        reviews=DEMO_ELECTRONICS_REVIEWS,
        output_dir=out_dir,
    )

    # Demo Run 2: Apparel Product (Classic Fit Cotton Tee)
    run_pipeline(
        model=model,
        tokenizer=tokenizer,
        product_id="PROD_FASH_T01_COTTON_TEE",
        vertical="apparel_fashion",
        reviews=DEMO_APPAREL_REVIEWS,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
