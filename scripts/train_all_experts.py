#!/usr/bin/env python3
"""
Master Model Generation Script for ProductLens.
Reads configs/training_configs.yaml and trains any or all domain specialist models
as well as the generic foundation model from scratch.

Usage:
  python scripts/train_all_experts.py --model tech       # Trains Tech Specialist
  python scripts/train_all_experts.py --model home       # Trains Home Specialist
  python scripts/train_all_experts.py --model apparel    # Trains Apparel Specialist
  python scripts/train_all_experts.py --model generic    # Trains Generic Foundation Model
  python scripts/train_all_experts.py --all              # Trains the entire suite sequentially
"""

import argparse
import random
from pathlib import Path
import numpy as np
import yaml
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset
from productlens.training.trainer import AspectSentimentTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs/training_configs.yaml"


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model_from_config(model_key: str, config: dict, global_defaults: dict):
    m_cfg = config["models"][model_key]
    print(f"\n{'='*70}")
    print(f"GENERATING MODEL: {model_key.upper()} ({m_cfg['domain']})")
    print(f"{'='*70}")

    train_path = BASE_DIR / m_cfg["train_data"]
    output_ckpt = BASE_DIR / m_cfg["output_checkpoint"]
    output_dir = output_ckpt.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)

    val_path_str = m_cfg.get("val_data")
    if val_path_str and (BASE_DIR / val_path_str).exists():
        val_df = pd.read_parquet(BASE_DIR / val_path_str)
    else:
        # Seeded, stratified, group-aware split (groups on 'text' to prevent
        # multi-aspect reviews from appearing on both sides of the split).
        seed = global_defaults.get("seed", 42)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
        train_idx, val_idx = next(
            splitter.split(train_df, train_df["sentiment"], groups=train_df["text"])
        )
        val_df   = train_df.iloc[val_idx].reset_index(drop=True)
        train_df = train_df.iloc[train_idx].reset_index(drop=True)

    print(f"Loaded Train Samples: {len(train_df)} | Validation Samples: {len(val_df)}")

    backbone = global_defaults.get("backbone", DEFAULT_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    train_ds = AspectSentimentDataset(train_df, tokenizer, max_length=global_defaults.get("max_length", 128))
    val_ds = AspectSentimentDataset(val_df, tokenizer, max_length=global_defaults.get("max_length", 128))

    model = AspectTargetedSentimentModel(
        backbone_name=backbone,
        num_labels=3,
        aspect_boost=global_defaults.get("aspect_boost", 3.5),
    )

    trainer = AspectSentimentTrainer(
        model=model,
        tokenizer=tokenizer,
        backbone_lr=global_defaults.get("backbone_lr", 1.5e-5),
        head_lr=global_defaults.get("head_lr", 6.0e-5),
        weight_decay=global_defaults.get("weight_decay", 0.01),
        lambda_scl=global_defaults.get("lambda_scl", 0.10),
        scl_temp=global_defaults.get("scl_temperature", 0.07),
    )

    results = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=global_defaults.get("batch_size", 16),
        epochs=m_cfg.get("epochs", 5),
        eval_every_steps=m_cfg.get("eval_every_steps", 100),
        patience=m_cfg.get("patience", 4),
        output_dir=output_dir,
    )

    print(f"Successfully generated checkpoint: {output_ckpt}")
    print(f"Best Validation Macro F1: {results['best_val_macro_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="ProductLens Model Suite Generator")
    parser.add_argument(
        "--model",
        choices=["generic_foundation", "expert_tech", "expert_home", "expert_apparel", "generic", "tech", "home", "apparel"],
        help="Specify which model to generate",
    )
    parser.add_argument("--all", action="store_true", help="Generate all 4 models sequentially")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("global_defaults", {})

    alias_map = {
        "generic": "generic_foundation",
        "tech": "expert_tech",
        "home": "expert_home",
        "apparel": "expert_apparel",
    }

    if args.all:
        for m_key in ["generic_foundation", "expert_tech", "expert_home", "expert_apparel"]:
            train_model_from_config(m_key, config, defaults)
    elif args.model:
        m_key = alias_map.get(args.model, args.model)
        train_model_from_config(m_key, config, defaults)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
