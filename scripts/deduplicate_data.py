#!/usr/bin/env python3
"""
Data Deduplication Script.

Removes:
  1. Exact (text, aspect) duplicates within each train.parquet.
  2. Rows in train.parquet whose (text, aspect) also appear in test.parquet
     (train/test leakage).

Run this ONCE before any training run:
  python scripts/deduplicate_data.py

Prints a per-domain summary of rows removed.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DOMAINS = {
    "tech_laptop":      BASE_DIR / "data/benchmarks/tech_laptop",
    "home_television":  BASE_DIR / "data/benchmarks/home_television",
    "apparel_tshirt":   BASE_DIR / "data/benchmarks/apparel_tshirt",
}


def deduplicate_domain(data_dir: Path, domain: str) -> None:
    train_path = data_dir / "train.parquet"
    test_path  = data_dir / "test.parquet"

    if not train_path.exists():
        print(f"[{domain}] SKIP — train.parquet not found at {train_path}")
        return

    train_df = pd.read_parquet(train_path)
    original_len = len(train_df)

    # 1. Remove intra-train duplicates (keep first occurrence)
    train_df = train_df.drop_duplicates(subset=["text", "aspect"], keep="first")
    after_intra_dedup = len(train_df)

    # 2. Remove train rows that appear in test (train/test leakage)
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        test_keys = set(zip(test_df["text"], test_df["aspect"]))
        mask_leakage = train_df.apply(
            lambda row: (row["text"], row["aspect"]) in test_keys, axis=1
        )
        train_df = train_df[~mask_leakage].reset_index(drop=True)
        after_leakage_removal = len(train_df)
    else:
        after_leakage_removal = after_intra_dedup
        print(f"[{domain}] WARNING — test.parquet not found, skipping leakage removal.")

    intra_removed   = original_len - after_intra_dedup
    leakage_removed = after_intra_dedup - after_leakage_removal
    total_removed   = original_len - after_leakage_removal

    print(
        f"[{domain}] "
        f"original={original_len}  "
        f"intra_dups_removed={intra_removed}  "
        f"leakage_removed={leakage_removed}  "
        f"final={after_leakage_removal}  "
        f"(total removed: {total_removed})"
    )

    train_df.to_parquet(train_path, index=False)
    print(f"[{domain}] Saved cleaned train.parquet → {train_path}")


def main():
    print("=== ProductLens Data Deduplication ===\n")
    for domain, data_dir in DOMAINS.items():
        deduplicate_domain(data_dir, domain)
    print("\nDone. Re-run training and evaluation to get updated benchmark numbers.")


if __name__ == "__main__":
    main()
