#!/usr/bin/env python3
"""
CLI Script to train either a Generic Foundation Model or a Category Specialist Model.

Usage:
  python scripts/train.py --domain generic    # Trains Generic Foundation on multi_aspect_mams
  python scripts/train.py --domain tech       # Trains Tech Specialist on tech_laptop
  python scripts/train.py --domain home       # Trains Home Specialist on home_television
  python scripts/train.py --domain apparel    # Trains Apparel Specialist on apparel_tshirt
"""

import argparse
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("productlens.train")

from productlens.constants import LABEL2ID
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset
from productlens.training.loss import compute_class_weights
from productlens.training.trainer import AspectSentimentTrainer


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

DATA_MAP = {
    "generic": "data/benchmarks/multi_aspect_mams",
    "tech": "data/benchmarks/tech_laptop",
    "home": "data/benchmarks/home_television",
    "apparel": "data/benchmarks/apparel_tshirt",
    "unified": "data/benchmarks/unified_ecommerce",
}

OUTPUT_MAP = {
    "generic": "models/category_experts/foundation_mams",
    "tech": "models/category_experts/expert_tech",
    "home": "models/category_experts/expert_home",
    "apparel": "models/category_experts/expert_apparel",
    "unified": "models/unified_backbone",
}


def main():
    parser = argparse.ArgumentParser(description="Train ProductLens ABSA Models")
    parser.add_argument("--domain", choices=["generic", "tech", "home", "apparel", "unified"], required=True)
    parser.add_argument("--backbone", default="yangheng/deberta-v3-base-absa-v1.1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=150)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None, help="Optional custom output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick flow verification")
    parser.add_argument("--class_weight_strategy", choices=["none", "inverse_freq", "effective_num"], default="none", help="Class imbalance handling strategy")
    parser.add_argument("--beta", type=float, default=0.999, help="Beta parameter for effective_num class weighting")
    parser.add_argument("--w_macro", type=float, default=0.7, help="Weight for Macro F1 in composite checkpoint selection")
    parser.add_argument("--w_neutral", type=float, default=0.3, help="Weight for Neutral F1 in composite checkpoint selection")
    parser.add_argument("--lambda_ordinal", type=float, default=0.15, help="Weight for Ordinal Wasserstein loss")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor for CrossEntropyLoss")
    parser.add_argument("--no_calibration", action="store_true", help="Disable post-hoc temperature scaling")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(DATA_MAP[args.domain])
    out_dir = Path(args.output_dir) if args.output_dir else Path(OUTPUT_MAP[args.domain])

    print(f"=== Training {args.domain.upper()} Model ===")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {out_dir}")
    print(f"Class Weight Strategy: {args.class_weight_strategy}")
    print(f"Checkpoint Selection: w_macro={args.w_macro}, w_neutral={args.w_neutral}")
    print(f"Post-hoc Calibration: {not args.no_calibration}")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_path = data_dir / "val.parquet"
    if val_path.exists():
        val_df = pd.read_parquet(val_path)
    else:
        # Seeded, stratified, group-aware split (groups on 'text' to prevent
        # multi-aspect reviews from appearing on both sides of the split).
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=args.seed)
        train_idx, val_idx = next(
            splitter.split(train_df, train_df["sentiment"], groups=train_df["text"])
        )
        val_df   = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx].reset_index(drop=True)

    if args.max_samples:
        train_df = train_df.iloc[:args.max_samples].reset_index(drop=True)
        val_limit = max(10, min(len(val_df), args.max_samples // 4))
        val_df = val_df.iloc[:val_limit].reset_index(drop=True)
        print(f"Verification Mode: subsampled to {len(train_df)} train / {len(val_df)} val samples.")

    # Calculate class weights if enabled
    train_labels = [LABEL2ID.get(str(s).lower(), 1) for s in train_df["sentiment"]]
    class_weights = compute_class_weights(train_labels, strategy=args.class_weight_strategy, beta=args.beta)
    if class_weights is not None:
        print(f"Computed Class Weights: {class_weights.tolist()}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = AspectSentimentDataset(train_df, tokenizer)
    val_ds = AspectSentimentDataset(val_df, tokenizer)

    model = AspectTargetedSentimentModel(backbone_name=args.backbone, num_labels=3, aspect_boost=3.5)
    trainer = AspectSentimentTrainer(
        model=model,
        tokenizer=tokenizer,
        class_weights=class_weights,
        w_macro=args.w_macro,
        w_neutral=args.w_neutral,
        calibrate=not args.no_calibration,
        lambda_ordinal=args.lambda_ordinal,
        label_smoothing=args.label_smoothing,
    )

    results = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_every_steps=args.eval_steps,
        patience=args.patience,
        output_dir=out_dir,
    )

    print("Training finished!")
    print(results)


if __name__ == "__main__":
    main()
