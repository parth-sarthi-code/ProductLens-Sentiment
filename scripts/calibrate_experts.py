#!/usr/bin/env python3
"""
Post-hoc Temperature Calibration Script for ProductLens ABSA Experts.
Computes optimal temperature scaling parameter T on validation sets
and saves calibration.json alongside each expert checkpoint.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE, DEFAULT_CHECKPOINT_PATHS
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset
from productlens.sentiment.calibration import TemperatureScaler, compute_calibration_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERTS = [
    {
        "key": "tech",
        "ckpt": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
        "train_data": BASE_DIR / "data/benchmarks/tech_laptop/train.parquet",
    },
    {
        "key": "home",
        "ckpt": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
        "train_data": BASE_DIR / "data/benchmarks/home_television/train.parquet",
    },
    {
        "key": "apparel",
        "ckpt": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
        "train_data": BASE_DIR / "data/benchmarks/apparel_tshirt/train.parquet",
    },
]


def calibrate_expert(expert_info: dict, tokenizer):
    ckpt_path = expert_info["ckpt"]
    if not ckpt_path.exists():
        print(f"Skipping {ckpt_path} (not found)")
        return

    print(f"\nCalibrating {expert_info['key']} at {ckpt_path}...")

    if "val_data" in expert_info and expert_info["val_data"].exists():
        val_df = pd.read_parquet(expert_info["val_data"])
    else:
        train_df = pd.read_parquet(expert_info["train_data"])
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
        _, val_idx = next(splitter.split(train_df, train_df["sentiment"], groups=train_df["text"]))
        val_df = train_df.iloc[val_idx].reset_index(drop=True)

    val_ds = AspectSentimentDataset(val_df, tokenizer)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AspectTargetedSentimentModel(backbone_name=DEFAULT_BACKBONE, num_labels=3)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask = batch["aspect_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask, aspect_mask)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)

    # Uncalibrated metrics
    uncal_probs = torch.softmax(torch.tensor(logits_arr), dim=-1).numpy()
    uncal_metrics = compute_calibration_metrics(uncal_probs, labels_arr)

    # Fit temperature
    scaler = TemperatureScaler()
    opt_temp = scaler.fit(logits_arr, labels_arr)

    # Calibrated metrics
    cal_probs = torch.softmax(torch.tensor(logits_arr) / opt_temp, dim=-1).numpy()
    cal_metrics = compute_calibration_metrics(cal_probs, labels_arr)

    out_file = ckpt_path.parent / "calibration.json"
    scaler.save(
        out_file,
        metadata={
            "domain": expert_info["key"],
            "val_samples": len(labels_arr),
            "uncalibrated": uncal_metrics,
            "calibrated": cal_metrics,
        },
    )

    print(f"  -> T = {opt_temp:.4f}")
    print(f"  -> Val ECE: {uncal_metrics['ece']*100:.2f}% -> {cal_metrics['ece']*100:.2f}%")
    print(f"  -> Val NLL: {uncal_metrics['nll']:.4f} -> {cal_metrics['nll']:.4f}")
    print(f"  -> Val Brier: {uncal_metrics['brier']:.4f} -> {cal_metrics['brier']:.4f}")
    print(f"  Saved to {out_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)
    for exp in EXPERTS:
        calibrate_expert(exp, tokenizer)


if __name__ == "__main__":
    main()
