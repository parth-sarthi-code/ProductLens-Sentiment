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
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer

from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset
from productlens.training.trainer import AspectSentimentTrainer

DATA_MAP = {
    "generic": "data/benchmarks/multi_aspect_mams",
    "tech": "data/benchmarks/tech_laptop",
    "home": "data/benchmarks/home_television",
    "apparel": "data/benchmarks/apparel_tshirt",
}

OUTPUT_MAP = {
    "generic": "models/category_experts/foundation_mams",
    "tech": "models/category_experts/expert_tech",
    "home": "models/category_experts/expert_home",
    "apparel": "models/category_experts/expert_apparel",
}


def main():
    parser = argparse.ArgumentParser(description="Train ProductLens ABSA Models")
    parser.add_argument("--domain", choices=["generic", "tech", "home", "apparel"], required=True)
    parser.add_argument("--backbone", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=150)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(DATA_MAP[args.domain])
    out_dir = Path(OUTPUT_MAP[args.domain])

    print(f"=== Training {args.domain.upper()} Model ===")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {out_dir}")

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_path = data_dir / "val.parquet"
    if val_path.exists():
        val_df = pd.read_parquet(val_path)
    else:
        # Split train set for validation if no separate val set
        val_df = train_df.iloc[: min(len(train_df) // 10, 100)]
        train_df = train_df.iloc[len(val_df) :]

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    train_ds = AspectSentimentDataset(train_df, tokenizer)
    val_ds = AspectSentimentDataset(val_df, tokenizer)

    model = AspectTargetedSentimentModel(backbone_name=args.backbone, num_labels=3, aspect_boost=3.5)
    trainer = AspectSentimentTrainer(model=model, tokenizer=tokenizer)

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
