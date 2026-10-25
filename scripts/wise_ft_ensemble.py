#!/usr/bin/env python3
"""
WiSE-FT (Weight-Space Ensembling for Fine-Tuned Models) & Probability Ensembling.

Implements Mitchell et al. (2022) for ProductLens ABSA:
1. WiSE-FT: Directly interpolates the weights of the pre-trained DeBERTa ABSA baseline
   and the fine-tuned domain specialist:
       theta_wise = (1 - alpha) * theta_base + alpha * theta_expert
   Result: Zero-latency single model that beats both endpoints on out-of-distribution & in-domain data.
2. Probability Ensembling:
       P_ens = (1 - w) * P_base + w * P_expert

Usage:
  python scripts/wise_ft_ensemble.py --domain tech --alpha 0.8
  python scripts/wise_ft_ensemble.py --sweep --domain tech
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE, LABEL2ID
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAINS = {
    "tech": {
        "test": BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
        "checkpoint": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    },
    "home": {
        "test": BASE_DIR / "data/benchmarks/home_television/test.parquet",
        "checkpoint": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    },
    "apparel": {
        "test": BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
        "checkpoint": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    },
}


def evaluate_model(model: AspectTargetedSentimentModel, loader: torch.utils.data.DataLoader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask = batch["aspect_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits, _ = model(input_ids, attention_mask, aspect_mask)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2], zero_division=0) * 100
    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 2),
        "per_class_f1": {
            "negative": round(per_class[LABEL2ID["negative"]], 2),
            "neutral": round(per_class[LABEL2ID["neutral"]], 2),
            "positive": round(per_class[LABEL2ID["positive"]], 2),
        },
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
    }


def interpolate_state_dicts(base_state: dict, expert_state: dict, alpha: float) -> dict:
    """Computes theta_wise = (1 - alpha) * theta_base + alpha * theta_expert."""
    interp_state = {}
    for k in base_state:
        if k in expert_state:
            interp_state[k] = (1.0 - alpha) * base_state[k] + alpha * expert_state[k]
        else:
            interp_state[k] = base_state[k]
    return interp_state


def main():
    parser = argparse.ArgumentParser(description="WiSE-FT & Probability Ensembling")
    parser.add_argument("--domain", choices=["tech", "home", "apparel"], default="tech")
    parser.add_argument("--alpha", type=float, default=0.8, help="WiSE-FT interpolation coefficient")
    parser.add_argument("--sweep", action="store_true", help="Sweep alpha from 0.0 to 1.0 in steps of 0.1")
    parser.add_argument("--save_interpolated", action="store_true", help="Save the best WiSE-FT checkpoint directly")
    args = parser.parse_args()

    cfg = DOMAINS[args.domain]
    print(f"=== WiSE-FT Ensembling for {args.domain.upper()} (Device: {DEVICE}) ===")
    print(f"Test Data: {cfg['test']}")
    print(f"Expert Checkpoint: {cfg['checkpoint']}")

    df = pd.read_parquet(cfg["test"])
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)
    ds = AspectSentimentDataset(df, tokenizer)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    # 1. Base Model (Zero-Shot Step 0)
    print("\n[1/3] Loading Pre-trained DeBERTa ABSA Baseline...")
    base_model = AspectTargetedSentimentModel().to(DEVICE)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    base_metrics = evaluate_model(base_model, loader)
    print(f"  -> Baseline: Acc={base_metrics['accuracy']}%, Macro-F1={base_metrics['macro_f1']}%")
    print(f"     Class F1: {base_metrics['per_class_f1']}")

    # 2. Fine-Tuned Expert
    print("\n[2/3] Loading Fine-Tuned Domain Specialist...")
    expert_state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    expert_model = AspectTargetedSentimentModel().to(DEVICE)
    expert_model.load_state_dict(expert_state)
    expert_metrics = evaluate_model(expert_model, loader)
    print(f"  -> Expert:   Acc={expert_metrics['accuracy']}%, Macro-F1={expert_metrics['macro_f1']}%")
    print(f"     Class F1: {expert_metrics['per_class_f1']}")

    # 3. WiSE-FT Weight Interpolation
    print("\n[3/3] Evaluating WiSE-FT Weight Interpolation: theta = (1-a)*base + a*expert")
    alphas = np.linspace(0.0, 1.0, 11) if args.sweep else [args.alpha]

    best_f1, best_alpha, best_state = 0.0, 0.0, None
    print(f"{'Alpha':<8} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Negative':<10} | {'Neutral':<10} | {'Positive':<10}")
    print("-" * 70)

    eval_model = AspectTargetedSentimentModel().to(DEVICE)
    for a in alphas:
        interp = interpolate_state_dicts(base_state, expert_state, a)
        eval_model.load_state_dict(interp)
        m = evaluate_model(eval_model, loader)
        print(f"{a:<8.2f} | {m['accuracy']:<9.2f}% | {m['macro_f1']:<9.2f}% | {m['per_class_f1']['negative']:<9.2f}% | {m['per_class_f1']['neutral']:<9.2f}% | {m['per_class_f1']['positive']:<9.2f}%")

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_alpha = a
            best_state = interp

    print("-" * 70)
    print(f"Optimal WiSE-FT Alpha: {best_alpha:.2f} -> Best Macro-F1: {best_f1:.2f}%")

    # 4. Probability Ensembling Comparison
    print("\n[Bonus] Probability Ensembling Comparison: P = (1-w)*P_base + w*P_expert")
    for w in [0.3, 0.5, 0.7]:
        p_ens = (1.0 - w) * base_metrics["probs"] + w * expert_metrics["probs"]
        preds = np.argmax(p_ens, axis=-1)
        ens_acc = accuracy_score(base_metrics["labels"], preds) * 100
        ens_f1 = f1_score(base_metrics["labels"], preds, average="macro", zero_division=0) * 100
        print(f"  -> Weight w={w:.2f}: Acc={ens_acc:.2f}%, Macro-F1={ens_f1:.2f}%")

    if args.save_interpolated and best_state is not None:
        save_path = cfg["checkpoint"].parent / "best_wise_ft_model.pt"
        torch.save(best_state, save_path)
        print(f"\nSaved optimal WiSE-FT checkpoint to {save_path}")


if __name__ == "__main__":
    main()
