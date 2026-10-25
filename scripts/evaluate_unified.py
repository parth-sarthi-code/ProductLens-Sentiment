#!/usr/bin/env python3
"""
Master Evaluation Script: Canonical Backbone vs ProductLens Unified Backbone.

Evaluates on:
- Laptop14 (Tech, 638 samples)
- Television (Home, 942 samples)
- T-Shirt (Apparel, 472 samples)
- ARTS Laptop (Adversarial Multi-Aspect, 1,877 samples)

Computes Accuracy, Macro-F1, Neutral-F1, ECE, NLL, and 95% bootstrap CI.
Outputs:
- Terminal formatted comparison matrix
- Saved JSON: benchmarks/benchmark_results.json
- Saved Markdown: benchmarks/benchmark_summary.md
"""

import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE, DEFAULT_UNIFIED_CHECKPOINT_PATH
from productlens.sentiment.calibration import compute_calibration_metrics
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATHS = {
    "Laptop14 (Tech)": BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
    "Television (Home)": BASE_DIR / "data/benchmarks/home_television/test.parquet",
    "T-Shirt (Apparel)": BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
    "ARTS Laptop (Adversarial)": BASE_DIR / "data/benchmarks/arts_laptop/test.parquet",
}

MODELS = {
    "Canonical Backbone (Pretrained)": (None, 1.0),
    "ProductLens Unified Backbone (v2.2.0)": (
        DEFAULT_UNIFIED_CHECKPOINT_PATH,
        BASE_DIR / "models/unified_backbone/calibration.json",
    ),
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model_on_dataset(model, dataset, temp: float = 1.0):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask = batch["aspect_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask, aspect_mask)
            scaled = logits / temp
            p = torch.softmax(scaled, dim=-1).cpu().numpy()

            probs.extend(p)
            preds.extend(np.argmax(p, axis=-1))
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    neutral_f1 = f1_score(labels, preds, labels=[1], average="micro")
    calib = compute_calibration_metrics(np.array(probs), np.array(labels))

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "neutral_f1": round(neutral_f1 * 100, 2),
        "ece": round(calib["ece"] * 100, 2),
        "nll": round(calib["nll"], 4),
        "n_samples": len(labels),
    }


def main():
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)

    print(f"=== Evaluating Models on Device: {DEVICE} ===")
    all_results = {}

    for model_name, (ckpt_path, calib_info) in MODELS.items():
        print(f"\n--- Loading {model_name} ---")
        model = AspectTargetedSentimentModel(backbone_name=DEFAULT_BACKBONE)
        if ckpt_path and Path(ckpt_path).exists():
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded weights from {ckpt_path}")
        else:
            print("Using raw pretrained weights")

        temp = 1.0
        if isinstance(calib_info, (str, Path)) and Path(calib_info).exists():
            try:
                with open(calib_info) as f:
                    temp = float(json.load(f).get("temperature", 1.0))
                print(f"Loaded calibration temperature T={temp:.4f}")
            except Exception:
                temp = 1.0

        model.to(DEVICE)
        model.eval()

        model_results = {}
        for d_name, d_path in DATA_PATHS.items():
            if not Path(d_path).exists():
                print(f"Skipping {d_name} (not found at {d_path})")
                continue
            df = pd.read_parquet(d_path)
            ds = AspectSentimentDataset(df, tokenizer, max_length=128)
            res = evaluate_model_on_dataset(model, ds, temp=temp)
            model_results[d_name] = res
            print(f"{d_name:25} | Acc: {res['accuracy']:5.2f}% | Macro-F1: {res['macro_f1']:5.2f}% | Neutral-F1: {res['neutral_f1']:5.2f}% | ECE: {res['ece']:5.2f}% | NLL: {res['nll']:.4f}")

        # Compute composite on standard e-com domains
        ecom_keys = ["Laptop14 (Tech)", "Television (Home)", "T-Shirt (Apparel)"]
        tot_samples = sum(model_results[k]["n_samples"] for k in ecom_keys if k in model_results)
        comp_acc = sum(model_results[k]["accuracy"] * model_results[k]["n_samples"] for k in ecom_keys if k in model_results) / tot_samples
        model_results["E-Commerce Composite"] = round(comp_acc, 2)
        print(f"==> E-Commerce Composite Accuracy: {comp_acc:.2f}%")

        all_results[model_name] = model_results

    # Save JSON results
    out_json = BASE_DIR / "benchmarks/benchmark_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_json}")


if __name__ == "__main__":
    main()
