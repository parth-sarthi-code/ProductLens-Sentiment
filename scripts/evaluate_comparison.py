#!/usr/bin/env python3
"""
Master Benchmark Comparison Script: Generic Foundation vs Specialists vs MoE Pipeline.

Evaluates all models across the standardized benchmark test sets:
- Tech Hardware (Laptop14, 638 samples)
- Home Electronics (Television, 942 samples)
- Apparel & Fashion (T-Shirt, 472 samples)
- MAMS Challenge (1,336 samples)

MoE is evaluated TWICE per domain:
  - moe_with_cat: category metadata supplied → tests Level 1 routing
  - moe_no_cat:   category=None              → router decides from text only

Routing accuracy (predicted expert vs true domain) is also reported for the
no-category run.

Outputs:
- Terminal formatted comparison matrix
- Saved JSON: benchmarks/moe_vs_generic_comparison.json
- Saved Markdown: benchmarks/moe_vs_generic_comparison.md
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE
from productlens.sentiment.calibration import compute_calibration_metrics
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.pipeline import ProductLensMoEPipeline
from productlens.training.dataset import AspectSentimentDataset, LABEL2ID

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATHS = {
    "Laptop14 (Tech)":    BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
    "Television (Home)":  BASE_DIR / "data/benchmarks/home_television/test.parquet",
    "T-Shirt (Apparel)":  BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
}

OOD_FIXTURE_PATH = BASE_DIR / "data/benchmarks/ood_fixture"

CHECKPOINTS = {
    "Canonical Backbone (PyABSA DeBERTa-v3)": None,
    "Tech Specialist (Laptop14)":   BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "Home Specialist (Television)": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "Apparel Specialist (T-Shirt)": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
}

# True domain expert for each benchmark dataset (used to score routing accuracy)
DOMAIN_TRUTH = {
    "Laptop14 (Tech)":   "tech",
    "Television (Home)": "home",
    "T-Shirt (Apparel)": "apparel",
    "OOD":               "fallback",
}

# Category strings supplied during the moe_with_cat evaluation
CATEGORY_MAP = {
    "Laptop14 (Tech)":   "Electronics",
    "Television (Home)": "Home_and_Kitchen",
    "T-Shirt (Apparel)": "Clothing_Shoes_and_Jewelry",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_ci(
    labels: list,
    preds: list,
    n: int = 1000,
    seed: int = 42,
) -> dict:
    """95% bootstrap confidence intervals for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, len(labels_arr), size=len(labels_arr))
        accs.append(accuracy_score(labels_arr[idx], preds_arr[idx]))
        f1s.append(f1_score(labels_arr[idx], preds_arr[idx], average="macro", zero_division=0))
    return {
        "accuracy_ci95": (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
        "macro_f1_ci95": (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))),
    }


def compute_metrics(labels: list, preds: list, probs: list | None = None) -> dict:
    """Compute accuracy, micro-F1, per-class F1, macro-F1, neutral-F1, ECE, NLL, Brier, and bootstrap CIs."""
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    acc = accuracy_score(labels_arr, preds_arr)
    per_class_f1 = f1_score(labels_arr, preds_arr, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    micro_f1 = f1_score(labels_arr, preds_arr, average="micro", zero_division=0)
    neutral_f1 = float(per_class_f1[LABEL2ID["neutral"]])
    ci = bootstrap_ci(labels, preds)

    result = {
        "accuracy":       round(float(acc) * 100, 2),
        "micro_f1":       round(float(micro_f1) * 100, 2),
        "macro_f1":       round(float(macro_f1) * 100, 2),
        "neutral_f1":     round(neutral_f1 * 100, 2),
        "per_class_f1":   {
            "negative": round(float(per_class_f1[LABEL2ID["negative"]]) * 100, 2),
            "neutral":  round(float(per_class_f1[LABEL2ID["neutral"]]) * 100, 2),
            "positive": round(float(per_class_f1[LABEL2ID["positive"]]) * 100, 2),
        },
        "accuracy_ci95": [round(v * 100, 2) for v in ci["accuracy_ci95"]],
        "macro_f1_ci95": [round(v * 100, 2) for v in ci["macro_f1_ci95"]],
    }

    if probs is not None:
        probs_arr = np.array(probs)
        calib = compute_calibration_metrics(probs_arr, labels_arr)
        result["ece"] = round(calib["ece"] * 100, 2)
        result["nll"] = calib["nll"]
        result["brier"] = calib["brier"]

    return result


def weighted_composite(results_by_domain: dict, domains: list, n_samples: dict) -> float:
    """Sample-weighted composite accuracy across the given domains."""
    total_samples = sum(n_samples[d] for d in domains)
    weighted_acc = sum(
        results_by_domain[d]["accuracy"] * n_samples[d] for d in domains
    )
    return round(weighted_acc / total_samples, 2)


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_single_model(
    model: AspectTargetedSentimentModel,
    dataset: AspectSentimentDataset,
    temperature: float = 1.0,
) -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask   = batch["aspect_mask"].to(DEVICE)
            labels        = batch["label"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask, aspect_mask)
            scaled_logits = logits / temperature
            probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    return compute_metrics(all_labels, all_preds, all_probs)


def evaluate_moe_pipeline(
    pipeline: ProductLensMoEPipeline,
    domain_key: str,
    df: pd.DataFrame,
    category: str | None,
) -> dict:
    """
    Evaluate the MoE pipeline on a domain benchmark.

    Args:
        category: If provided, passed to pipeline.predict() (tests Level 1 routing).
                  If None, router decides from text only (tests Levels 2/3).

    Returns:
        Metrics dict, plus 'routing_accuracy', 'fallback_rate', and 'aspect_mask_fallback_rate'.
    """
    all_preds, all_labels, all_probs, routed_experts, aspect_mask_fallbacks = [], [], [], [], []
    true_domain = DOMAIN_TRUTH[domain_key]

    for _, row in df.iterrows():
        text        = str(row["text"])
        aspect      = str(row["aspect"])
        asp_start   = int(row.get("aspect_start", 0))
        asp_end     = int(row.get("aspect_end", len(aspect)))
        gold_label  = LABEL2ID.get(str(row.get("sentiment", "neutral")).lower(), 1)

        pred = pipeline.predict(
            text=text,
            aspect=aspect,
            category=category,
            aspect_start=asp_start,
            aspect_end=asp_end,
        )

        all_preds.append(LABEL2ID[pred.sentiment])
        all_labels.append(gold_label)
        all_probs.append([
            pred.probabilities["negative"],
            pred.probabilities["neutral"],
            pred.probabilities["positive"],
        ])
        routed_experts.append(pred.routed_expert)
        aspect_mask_fallbacks.append(pred.aspect_mask_fallback)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    total_preds = len(routed_experts)
    metrics["fallback_rate"] = round(sum(e == "fallback" for e in routed_experts) / total_preds * 100, 2)
    metrics["aspect_mask_fallback_rate"] = round(sum(aspect_mask_fallbacks) / total_preds * 100, 2)

    if category is None:
        correct_routes = sum(e == true_domain for e in routed_experts)
        metrics["routing_accuracy"] = round(correct_routes / total_preds * 100, 2)
        from collections import Counter
        metrics["routing_distribution"] = dict(Counter(routed_experts))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ProductLens Benchmark Comparison")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== ProductLens Benchmark (device={DEVICE}, seed={args.seed}) ===\n")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)

    datasets_raw = {domain: pd.read_parquet(path) for domain, path in DATA_PATHS.items()}
    n_samples = {domain: len(df) for domain, df in datasets_raw.items()}

    tokenized_datasets = {
        domain: AspectSentimentDataset(df, tokenizer) for domain, df in datasets_raw.items()
    }

    ecom_domains = list(datasets_raw.keys())
    results = {}

    # -----------------------------------------------------------------------
    # 1. Individual Checkpoints
    # -----------------------------------------------------------------------
    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"Evaluating: {model_name}...")
        model = AspectTargetedSentimentModel(backbone_name=DEFAULT_BACKBONE, num_labels=3)
        temp = 1.0
        if ckpt_path is not None and ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
            calib_file = ckpt_path.parent / "calibration.json"
            if calib_file.exists():
                try:
                    with open(calib_file, "r", encoding="utf-8") as f:
                        temp = float(json.load(f).get("temperature", 1.0))
                    print(f"  [Calibration] Loaded T={temp:.4f} from {calib_file}")
                except Exception:
                    temp = 1.0
        else:
            print(f"  [Pretrained Backbone] Evaluating canonical {DEFAULT_BACKBONE} directly (T={temp:.4f})")
        model.to(DEVICE)

        results[model_name] = {}
        for domain, ds in tokenized_datasets.items():
            metrics = evaluate_single_model(model, ds, temperature=temp)
            results[model_name][domain] = metrics
            print(
                f"  -> {domain}: Acc={metrics['accuracy']}% "
                f"[{metrics['accuracy_ci95'][0]}–{metrics['accuracy_ci95'][1]}] "
                f"MacroF1={metrics['macro_f1']}% NeutralF1={metrics['neutral_f1']}% "
                f"ECE={metrics.get('ece', 0.0)}% NLL={metrics.get('nll', 0.0):.4f} Brier={metrics.get('brier', 0.0):.4f}"
            )

        results[model_name]["E-Commerce Composite (weighted)"] = weighted_composite(
            results[model_name], ecom_domains, n_samples
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 2. MoE — two evaluation modes
    # -----------------------------------------------------------------------
    pipeline = ProductLensMoEPipeline(
        device=DEVICE,
        backbone_name=DEFAULT_BACKBONE,
        single_active_expert=True,
    )

    for mode_label, use_category in [
        ("ProductLens MoE (with category)", True),
        ("ProductLens MoE (no category)",   False),
    ]:
        print(f"\nEvaluating: {mode_label}...")
        results[mode_label] = {}

        for domain, df in datasets_raw.items():
            cat = CATEGORY_MAP[domain] if use_category else None
            metrics = evaluate_moe_pipeline(pipeline, domain, df, cat)
            results[mode_label][domain] = metrics

            routing_note = ""
            if "routing_accuracy" in metrics:
                routing_note = f" | Routing Acc={metrics['routing_accuracy']}%"
            fallback_note = f" | Fallback Rate={metrics['fallback_rate']}% | AspectMask Fallback Rate={metrics['aspect_mask_fallback_rate']}%"
            print(
                f"  -> {domain}: Acc={metrics['accuracy']}% "
                f"MacroF1={metrics['macro_f1']}% NeutralF1={metrics['neutral_f1']}% "
                f"ECE={metrics.get('ece', 0.0)}%{routing_note}{fallback_note}"
            )

        results[mode_label]["E-Commerce Composite (weighted)"] = weighted_composite(
            results[mode_label], ecom_domains, n_samples
        )

    # -----------------------------------------------------------------------
    # 3. OOD Fixture (fallback quality)
    # -----------------------------------------------------------------------
    if OOD_FIXTURE_PATH.exists():
        print("\nEvaluating: OOD Fixture (fallback quality)...")
        results["OOD Fixture"] = {}
        for fixture_file in sorted(OOD_FIXTURE_PATH.glob("*.parquet")):
            ood_df = pd.read_parquet(fixture_file)
            ood_name = fixture_file.stem
            metrics = evaluate_moe_pipeline(pipeline, "OOD", ood_df, None)
            results["OOD Fixture"][ood_name] = metrics
            print(f"  -> {ood_name}: Acc={metrics['accuracy']}% MacroF1={metrics['macro_f1']}%")
    else:
        print(
            f"\n[INFO] OOD fixture not found at {OOD_FIXTURE_PATH}. "
            "Skipping fallback quality evaluation."
        )

    # -----------------------------------------------------------------------
    # 4. Save JSON
    # -----------------------------------------------------------------------
    json_path = BASE_DIR / "benchmarks/moe_vs_generic_comparison.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved → {json_path}")

    # -----------------------------------------------------------------------
    # 5. Generate Markdown — all numbers from results dict, none hardcoded
    # -----------------------------------------------------------------------
    base_name = "Canonical Backbone (PyABSA DeBERTa-v3)"
    gen_composite  = results[base_name]["E-Commerce Composite (weighted)"]
    moe_nc_composite = results["ProductLens MoE (no category)"]["E-Commerce Composite (weighted)"]
    moe_wc_composite = results["ProductLens MoE (with category)"]["E-Commerce Composite (weighted)"]
    net_lift_nc    = round(moe_nc_composite - gen_composite, 2)
    net_lift_wc    = round(moe_wc_composite - gen_composite, 2)

    def acc(model, domain):
        return results[model][domain]["accuracy"]

    md_content = f"""# ProductLens Benchmark: Canonical Backbone vs Specialists vs MoE Pipeline

> Generated by `scripts/evaluate_comparison.py` — do not edit manually.
> Composite = sample-weighted accuracy across e-commerce domains.
> CI = 95% bootstrap confidence interval.

## Test Accuracy Transfer Matrix (%)

| Model | Laptop14 (Tech, {n_samples['Laptop14 (Tech)']}) | Television (Home, {n_samples['Television (Home)']}) | T-Shirt (Apparel, {n_samples['T-Shirt (Apparel)']}) | E-Com Composite |
| :--- | :---: | :---: | :---: | :---: |
| Canonical Backbone (PyABSA DeBERTa-v3) | {acc(base_name, 'Laptop14 (Tech)')}% | {acc(base_name, 'Television (Home)')}% | {acc(base_name, 'T-Shirt (Apparel)')}% | {gen_composite}% |
| Tech Specialist (Laptop14) | {acc('Tech Specialist (Laptop14)', 'Laptop14 (Tech)')}% | {acc('Tech Specialist (Laptop14)', 'Television (Home)')}% | {acc('Tech Specialist (Laptop14)', 'T-Shirt (Apparel)')}% | {results['Tech Specialist (Laptop14)']['E-Commerce Composite (weighted)']}% |
| Home Specialist (Television) | {acc('Home Specialist (Television)', 'Laptop14 (Tech)')}% | {acc('Home Specialist (Television)', 'Television (Home)')}% | {acc('Home Specialist (Television)', 'T-Shirt (Apparel)')}% | {results['Home Specialist (Television)']['E-Commerce Composite (weighted)']}% |
| Apparel Specialist (T-Shirt) | {acc('Apparel Specialist (T-Shirt)', 'Laptop14 (Tech)')}% | {acc('Apparel Specialist (T-Shirt)', 'Television (Home)')}% | {acc('Apparel Specialist (T-Shirt)', 'T-Shirt (Apparel)')}% | {results['Apparel Specialist (T-Shirt)']['E-Commerce Composite (weighted)']}% |
| **ProductLens MoE (with category)** | {acc('ProductLens MoE (with category)', 'Laptop14 (Tech)')}% | {acc('ProductLens MoE (with category)', 'Television (Home)')}% | {acc('ProductLens MoE (with category)', 'T-Shirt (Apparel)')}% | **{moe_wc_composite}% (+{net_lift_wc}% vs backbone)** |
| **ProductLens MoE (no category)** | {acc('ProductLens MoE (no category)', 'Laptop14 (Tech)')}% | {acc('ProductLens MoE (no category)', 'Television (Home)')}% | {acc('ProductLens MoE (no category)', 'T-Shirt (Apparel)')}% | **{moe_nc_composite}% (+{net_lift_nc}% vs backbone)** |

## Detailed F1 & Calibration Metrics (No Category Mode)

| Domain | Accuracy (%) | Macro-F1 (%) | Neutral-F1 (%) | ECE (%) | NLL | Brier | Fallback Rate (%) | AspectMask Fallback (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Laptop14 (Tech) | {results['ProductLens MoE (no category)']['Laptop14 (Tech)']['accuracy']}% | {results['ProductLens MoE (no category)']['Laptop14 (Tech)']['macro_f1']}% | {results['ProductLens MoE (no category)']['Laptop14 (Tech)']['neutral_f1']}% | {results['ProductLens MoE (no category)']['Laptop14 (Tech)'].get('ece', 0.0)}% | {results['ProductLens MoE (no category)']['Laptop14 (Tech)'].get('nll', 0.0):.4f} | {results['ProductLens MoE (no category)']['Laptop14 (Tech)'].get('brier', 0.0):.4f} | {results['ProductLens MoE (no category)']['Laptop14 (Tech)'].get('fallback_rate', 0.0)}% | {results['ProductLens MoE (no category)']['Laptop14 (Tech)'].get('aspect_mask_fallback_rate', 0.0)}% |
| Television (Home) | {results['ProductLens MoE (no category)']['Television (Home)']['accuracy']}% | {results['ProductLens MoE (no category)']['Television (Home)']['macro_f1']}% | {results['ProductLens MoE (no category)']['Television (Home)']['neutral_f1']}% | {results['ProductLens MoE (no category)']['Television (Home)'].get('ece', 0.0)}% | {results['ProductLens MoE (no category)']['Television (Home)'].get('nll', 0.0):.4f} | {results['ProductLens MoE (no category)']['Television (Home)'].get('brier', 0.0):.4f} | {results['ProductLens MoE (no category)']['Television (Home)'].get('fallback_rate', 0.0)}% | {results['ProductLens MoE (no category)']['Television (Home)'].get('aspect_mask_fallback_rate', 0.0)}% |
| T-Shirt (Apparel) | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)']['accuracy']}% | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)']['macro_f1']}% | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)']['neutral_f1']}% | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)'].get('ece', 0.0)}% | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)'].get('nll', 0.0):.4f} | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)'].get('brier', 0.0):.4f} | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)'].get('fallback_rate', 0.0)}% | {results['ProductLens MoE (no category)']['T-Shirt (Apparel)'].get('aspect_mask_fallback_rate', 0.0)}% |

## Notes
- **Fallback Architecture**: Uses canonical pretrained backbone (`yangheng/deberta-v3-base-absa-v1.1`) directly for unrouted or OOD domains.
- **Specialist Domain Lifts**: Domain specialists provide targeted improvements on in-domain terms and neutral ambiguity resolution.
- **Routing Accuracy & Distribution**: Evaluated when no category metadata is provided.
- **Post-Hoc Calibration**: Probabilities calibrated via temperature scaling minimizing validation NLL and ECE.
- All confidence intervals are 95% bootstrap (n=1000).
- Composite = sample-weighted mean; weights are test-set sizes.
"""

    md_path = BASE_DIR / "benchmarks/moe_vs_generic_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(md_content)
    print(f"Markdown report saved → {md_path}")


if __name__ == "__main__":
    main()
