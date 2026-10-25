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

from productlens.constants import DEFAULT_BACKBONE, LABEL2ID
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset
from productlens.training.loss import compute_class_weights
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

    class_weight_strat = m_cfg.get("class_weight_strategy", global_defaults.get("class_weight_strategy", "none"))
    beta = float(m_cfg.get("effective_num_beta", global_defaults.get("effective_num_beta", 0.999)))
    w_macro = float(m_cfg.get("w_macro", global_defaults.get("w_macro", 0.7)))
    w_neutral = float(m_cfg.get("w_neutral", global_defaults.get("w_neutral", 0.3)))
    calibrate = bool(m_cfg.get("calibrate", global_defaults.get("calibrate", True)))
    lambda_ordinal = float(m_cfg.get("lambda_ordinal", global_defaults.get("lambda_ordinal", 0.15)))
    label_smoothing = float(m_cfg.get("label_smoothing", global_defaults.get("label_smoothing", 0.05)))

    train_labels = [LABEL2ID.get(str(s).lower(), 1) for s in train_df["sentiment"]]
    class_weights = compute_class_weights(train_labels, strategy=class_weight_strat, beta=beta)
    if class_weights is not None:
        print(f"Class Weights ({class_weight_strat}): {class_weights.tolist()}")

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
        class_weights=class_weights,
        w_macro=w_macro,
        w_neutral=w_neutral,
        calibrate=calibrate,
        lambda_ordinal=lambda_ordinal,
        label_smoothing=label_smoothing,
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
    print(f"Selection Reason: {results.get('selection_reason', 'N/A')}")
    print(f"Best Validation Macro F1: {results['best_val_macro_f1']:.4f} | Neutral F1: {results.get('best_val_neutral_f1', 0.0):.4f} | Composite Score: {results.get('best_composite_score', 0.0):.4f}")
    if "temperature" in results:
        print(f"Post-Hoc Calibration: T={results['temperature']:.4f} (ECE: {results.get('val_ece_before', 0.0):.4f} -> {results.get('val_ece_after', 0.0):.4f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="ProductLens Model Suite Generator")
    parser.add_argument(
        "--model",
        choices=["expert_tech", "expert_home", "expert_apparel", "tech", "home", "apparel"],
        help="Specify which domain specialist to generate",
    )
    parser.add_argument("--all", action="store_true", help="Generate all 3 domain specialists sequentially")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("global_defaults", {})

    alias_map = {
        "tech": "expert_tech",
        "home": "expert_home",
        "apparel": "expert_apparel",
    }

    if args.all:
        for m_key in ["expert_tech", "expert_home", "expert_apparel"]:
            train_model_from_config(m_key, config, defaults)
    elif args.model:
        m_key = alias_map.get(args.model, args.model)
        train_model_from_config(m_key, config, defaults)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
