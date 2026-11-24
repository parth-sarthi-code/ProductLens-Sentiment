#!/usr/bin/env python3
"""
ProductLens 3.0: Stage 3 Training Data Builder
Assembles training data from 2-vertical human-labeled benchmarks + downsampled MAMS + Stage 2b silver smoke quads.
Strictly enforces ZERO evaluation leakage against Gold Set and all test benchmarks.
Formats outputs into compact quad target format:
  aspect|category|sentiment|confidence|evidence
(using 'null' for implicit aspects and 'NONE' for reviews with no aspects).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from transformers import AutoTokenizer

from productlens.schemas.enums import Sentiment, VerticalDomain, normalize_category
from productlens.schemas.quads import RawCompactQuad, AspectQuad
from productlens.benchmarks.loader import BenchmarkLoader
from productlens.benchmarks.adapter import BenchmarkAdapter
from productlens.gold.manager import GoldSetManager

SYSTEM_PROMPT = (
    "You are an expert Aspect-Based Sentiment Analysis (ABSA) system specialized in e-commerce customer reviews. "
    "Analyze the customer review and extract fine-grained aspect sentiment quads adhering strictly to the compact format: "
    "aspect|category|sentiment|confidence|evidence. Use 'null' for implicit aspects. Output 'NONE' if no aspects are present."
)


def get_forbidden_leakage_sets() -> Tuple[Set[str], Set[str]]:
    """
    Collects all forbidden texts and product IDs from evaluation sets:
    - 350-review Gold Set
    - Laptop14 test split
    - Television test split
    - T-Shirt test split
    - ARTS Laptop test split
    """
    forbidden_texts: Set[str] = set()
    forbidden_pids: Set[str] = set()

    # 1. Gold Set
    gold_reviews = GoldSetManager.load_reviews()
    for gr in gold_reviews:
        forbidden_texts.add(gr.text.strip().lower())
        if gr.product_id:
            forbidden_pids.add(gr.product_id.strip().lower())

    # 2. Benchmark test sets
    test_datasets = ["tech_laptop", "home_television", "apparel_tshirt", "arts_laptop"]
    for ds_name in test_datasets:
        try:
            df_test = BenchmarkLoader.load_dataset(ds_name, "test")
            for t in df_test["text"].dropna():
                clean_t = str(t).strip().lower()
                if clean_t:
                    forbidden_texts.add(clean_t)
            if "product_id" in df_test.columns:
                for pid in df_test["product_id"].dropna():
                    forbidden_pids.add(str(pid).strip().lower())
        except Exception as e:
            print(f"⚠️ Warning loading test set {ds_name}: {e}")

    return forbidden_texts, forbidden_pids


def load_and_adapt_benchmarks(
    forbidden_texts: Set[str],
    forbidden_pids: Set[str],
    mams_sample_size: int = 1800,
) -> List[Dict]:
    """
    Loads training splits from Laptop14, Television, T-Shirt, and stratified MAMS,
    filtering out any review matching forbidden evaluation sets.
    """
    raw_records: List[Dict] = []
    skipped_leakage = 0

    sources = [
        ("tech_laptop", VerticalDomain.ELECTRONICS_TECH),
        ("home_television", VerticalDomain.ELECTRONICS_TECH),
        ("apparel_tshirt", VerticalDomain.APPAREL_FASHION),
    ]

    for ds_name, vertical in sources:
        df = BenchmarkLoader.load_dataset(ds_name, "train")
        print(f"Loading {ds_name} train ({len(df)} rows)...")
        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            clean_text = text.lower()
            pid = str(row.get("product_id", "")).strip().lower()

            if clean_text in forbidden_texts or (pid and pid in forbidden_pids):
                skipped_leakage += 1
                continue

            quad, err = BenchmarkAdapter.row_to_quad(row.to_dict(), vertical)
            if quad is not None:
                raw_records.append({
                    "text": text,
                    "vertical": vertical.value,
                    "aspect": quad.aspect,
                    "category": quad.category,
                    "sentiment": quad.sentiment.value,
                    "confidence": quad.confidence,
                    "evidence": quad.evidence,
                    "source": ds_name,
                })

    # Stratified downsample of MAMS train
    print(f"Loading multi_aspect_mams train (downsampling to {mams_sample_size})...")
    df_mams = BenchmarkLoader.load_dataset("multi_aspect_mams", "train")
    # Stratify across sentiments
    sample_per_sent = mams_sample_size // 3
    sampled_dfs = []
    for s in ["positive", "negative", "neutral"]:
        subset = df_mams[df_mams["sentiment"].str.lower() == s]
        n = min(len(subset), sample_per_sent)
        sampled_dfs.append(subset.sample(n=n, random_state=42))
    df_mams_sampled = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)

    for _, row in df_mams_sampled.iterrows():
        text = str(row.get("text", "")).strip()
        clean_text = text.lower()
        if clean_text in forbidden_texts:
            skipped_leakage += 1
            continue

        quad, err = BenchmarkAdapter.row_to_quad(row.to_dict(), VerticalDomain.ELECTRONICS_TECH)
        if quad is not None:
            raw_records.append({
                "text": text,
                "vertical": VerticalDomain.ELECTRONICS_TECH.value,
                "aspect": quad.aspect,
                "category": quad.category,
                "sentiment": quad.sentiment.value,
                "confidence": quad.confidence,
                "evidence": quad.evidence,
                "source": "multi_aspect_mams",
            })

    print(f"Loaded {len(raw_records)} quad rows from benchmarks (filtered {skipped_leakage} leakage rows).")
    return raw_records


def load_silver_smoke_data(
    silver_dir: Path | str,
    forbidden_texts: Set[str],
    forbidden_pids: Set[str],
) -> List[Dict]:
    """Loads existing Stage 2b consensus silver smoke quads from parquet files."""
    silver_path = Path(silver_dir)
    records: List[Dict] = []
    if not silver_path.exists():
        print(f"⚠️ Silver output path not found: {silver_path}")
        return records

    parquet_files = sorted(silver_path.glob("*.parquet"))
    print(f"Loading Stage 2b silver consensus files from {silver_path} ({len(parquet_files)} files)...")

    skipped_leakage = 0
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                text = str(row.get("text", "")).strip()
                clean_t = text.lower()
                pid = str(row.get("product_id", "")).strip().lower()

                if clean_t in forbidden_texts or (pid and pid in forbidden_pids):
                    skipped_leakage += 1
                    continue

                raw_asp = row.get("aspect")
                aspect = str(raw_asp).strip() if pd.notna(raw_asp) and raw_asp is not None else None
                vert = str(row.get("vertical", "electronics_tech"))
                cat = normalize_category(str(row.get("category", "OTHER")), vert)
                sent = Sentiment.from_str(str(row.get("sentiment", "neutral")))
                conf = float(row.get("confidence", 1.0))
                evidence = str(row.get("evidence", text)).strip()

                records.append({
                    "text": text,
                    "vertical": vert,
                    "aspect": aspect,
                    "category": cat,
                    "sentiment": sent.value,
                    "confidence": conf,
                    "evidence": evidence,
                    "source": "silver_smoke_consensus",
                })
        except Exception as e:
            print(f"Error reading {pf.name}: {e}")

    print(f"Loaded {len(records)} silver consensus quad rows (filtered {skipped_leakage} leakage rows).")
    return records


def build_grouped_dataset(
    all_quad_records: List[Dict],
    forbidden_texts: Set[str],
    forbidden_pids: Set[str],
) -> List[Dict]:
    """
    Groups quad records by review text and vertical, formatting assistant target into compact lines:
      aspect|category|sentiment|confidence|evidence
    """
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], List[RawCompactQuad]] = defaultdict(list)

    for r in all_quad_records:
        text = r["text"]
        vertical = r["vertical"]
        compact = RawCompactQuad(
            aspect=r["aspect"],
            category=r["category"],
            sentiment=Sentiment(r["sentiment"]),
            confidence=round(float(r["confidence"]), 2),
            evidence=r["evidence"],
        )
        grouped[(text, vertical)].append(compact)

    dataset_samples: List[Dict] = []
    idx = 1
    for (text, vertical), quads in grouped.items():
        # Zero-leakage sanity check
        assert text.strip().lower() not in forbidden_texts, f"Zero-leakage violation: {text[:50]}"

        # Deduplicate quads by (aspect, category, sentiment)
        unique_quads: List[RawCompactQuad] = []
        seen = set()
        for q in quads:
            key = (q.aspect.lower() if q.aspect else None, q.category, q.sentiment.value)
            if key not in seen:
                seen.add(key)
                unique_quads.append(q)

        if unique_quads:
            compact_lines = "\n".join(q.to_compact_line() for q in unique_quads)
        else:
            compact_lines = "NONE"

        user_content = (
            f"Product Domain: {vertical}\n"
            f"Review Text:\n\"\"\"{text}\"\"\"\n\n"
            "Extract all aspect quads adhering to the compact format: aspect|category|sentiment|confidence|evidence"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": compact_lines},
        ]

        sample = {
            "review_id": f"stage3_sample_{idx:06d}",
            "vertical": vertical,
            "text": text,
            "num_quads": len(unique_quads),
            "compact_target": compact_lines,
            "messages": messages,
        }
        dataset_samples.append(sample)
        idx += 1

    return dataset_samples


def main():
    print("=" * 72)
    print("📦 ProductLens 3.0: Stage 3 Training Data Builder")
    print("=" * 72)

    output_dir = PROJECT_ROOT / "data" / "stage3"
    output_dir.mkdir(parents=True, exist_ok=True)

    silver_dir = PROJECT_ROOT / "outputs" / "silver_label" / "stage2_silver_output"

    # 1. Collect Forbidden Sets
    print("\n🔍 Collecting forbidden evaluation leakage sets...")
    forbidden_texts, forbidden_pids = get_forbidden_leakage_sets()
    print(f"Total forbidden evaluation texts: {len(forbidden_texts)}")
    print(f"Total forbidden Gold product IDs: {len(forbidden_pids)}")

    # 2. Load Benchmarks & Silver Quads
    print("\n📚 Loading source datasets...")
    benchmark_records = load_and_adapt_benchmarks(forbidden_texts, forbidden_pids, mams_sample_size=1800)
    silver_records = load_silver_smoke_data(silver_dir, forbidden_texts, forbidden_pids)

    all_records = benchmark_records + silver_records
    print(f"\nTotal combined quad rows: {len(all_records)}")

    # 3. Group by Review Text
    print("\n🔗 Grouping quads by review text and formatting compact targets...")
    dataset = build_grouped_dataset(all_records, forbidden_texts, forbidden_pids)
    print(f"Total unique review samples: {len(dataset)}")

    # Strict Zero-Leakage Assertion
    print("\n🛡️ Verifying Zero-Leakage Assertion...")
    for item in dataset:
        clean_text = item["text"].strip().lower()
        if clean_text in forbidden_texts:
            raise AssertionError(f"CRITICAL: Leakage detected for text: {item['text'][:80]}")
    print("✅ ZERO-LEAKAGE ASSERTION PASSED: 0 test/gold reviews present in training set.")

    # 4. Apply Tokenizer Chat Template
    tokenizer_name = "LiquidAI/LFM3-1.7B"
    print(f"\n⚙️ Formatting chat template for {tokenizer_name} (enable_thinking=False)...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    for item in dataset:
        formatted_prompt = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        item["formatted_text"] = formatted_prompt

    # 5. Split Train (90%) and Val (10%)
    df_full = pd.DataFrame(dataset)
    df_shuffled = df_full.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_size = int(len(df_shuffled) * 0.10)
    df_val = df_shuffled.iloc[:val_size].reset_index(drop=True)
    df_train = df_shuffled.iloc[val_size:].reset_index(drop=True)

    print(f"\n📊 Dataset Split Summary:")
    print(f"   • Train samples: {len(df_train)} (90%)")
    print(f"   • Val samples:   {len(df_val)} (10%)")
    print(f"   • Total:         {len(df_shuffled)}")

    # Breakdown by vertical
    print("\n   Train Verticals:")
    for v, c in df_train["vertical"].value_counts().items():
        print(f"     - {v}: {c} reviews")

    # 6. Save JSONL and Parquet
    train_jsonl = output_dir / "stage3_train.jsonl"
    val_jsonl = output_dir / "stage3_val.jsonl"
    train_parquet = output_dir / "stage3_train.parquet"
    val_parquet = output_dir / "stage3_val.parquet"

    df_train.to_json(train_jsonl, orient="records", lines=True)
    df_val.to_json(val_jsonl, orient="records", lines=True)
    df_train.to_parquet(train_parquet, index=False)
    df_val.to_parquet(val_parquet, index=False)

    print(f"\n💾 Saved training data to:")
    print(f"   • {train_jsonl}")
    print(f"   • {val_jsonl}")
    print(f"   • {train_parquet}")
    print(f"   • {val_parquet}")

    # Summary JSON
    summary = {
        "total_reviews": int(len(df_shuffled)),
        "train_reviews": int(len(df_train)),
        "val_reviews": int(len(df_val)),
        "total_quads": int(df_full["num_quads"].sum()),
        "train_verticals": {str(k): int(v) for k, v in df_train["vertical"].value_counts().items()},
        "val_verticals": {str(k): int(v) for k, v in df_val["vertical"].value_counts().items()},
        "forbidden_texts_checked": int(len(forbidden_texts)),
        "zero_leakage_verified": True,
    }
    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
