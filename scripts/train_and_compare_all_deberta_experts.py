#!/usr/bin/env python3
"""
Full Suite Training & Benchmark: RoBERTa vs DeBERTa-v3 Specialists & MoE.

1. Trains DeBERTa-v3 ABSA Specialists on deduplicated datasets:
   - Tech Hardware (Laptop14)
   - Home Electronics (Television)
   - Apparel & Fashion (T-Shirt)
   (MAMS Foundation was trained in the previous step)

2. Evaluates Head-to-Head:
   - RoBERTa Specialist vs DeBERTa Specialist across all held-out test sets
   - Full 4x4 Transfer Matrix for both suites
   - RoBERTa MoE Pipeline vs DeBERTa MoE Pipeline (with & without category)
   - Sample-weighted E-Commerce Composite comparisons

Outputs:
- benchmarks/roberta_vs_deberta_experts_comparison.json
- benchmarks/roberta_vs_deberta_experts_comparison.md
"""

import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.pipeline import ProductLensMoEPipeline
from productlens.training.dataset import AspectSentimentDataset, LABEL2ID
from productlens.training.trainer import AspectSentimentTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_NAME = "yangheng/deberta-v3-base-absa-v1.1"

# Datasets
DATA_PATHS = {
    "Laptop14 (Tech)":   BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
    "Television (Home)": BASE_DIR / "data/benchmarks/home_television/test.parquet",
    "T-Shirt (Apparel)": BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
    "MAMS Challenge":    BASE_DIR / "data/benchmarks/multi_aspect_mams/test.parquet",
}

TRAIN_CONFIGS = {
    "expert_tech": {
        "domain": "Laptop14 (Tech)",
        "train_data": BASE_DIR / "data/benchmarks/tech_laptop/train.parquet",
        "output_dir": BASE_DIR / "models/category_experts_deberta/expert_tech",
        "epochs": 3,
        "eval_every_steps": 50,
        "patience": 4,
    },
    "expert_home": {
        "domain": "Television (Home)",
        "train_data": BASE_DIR / "data/benchmarks/home_television/train.parquet",
        "output_dir": BASE_DIR / "models/category_experts_deberta/expert_home",
        "epochs": 3,
        "eval_every_steps": 50,
        "patience": 4,
    },
    "expert_apparel": {
        "domain": "T-Shirt (Apparel)",
        "train_data": BASE_DIR / "data/benchmarks/apparel_tshirt/train.parquet",
        "output_dir": BASE_DIR / "models/category_experts_deberta/expert_apparel",
        "epochs": 3,
        "eval_every_steps": 50,
        "patience": 4,
    },
}

ROBERTA_CHECKPOINTS = {
    "Generic Foundation (MAMS)":    BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
    "Tech Specialist (Laptop14)":   BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "Home Specialist (Television)": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "Apparel Specialist (T-Shirt)": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
}

ROBERTA_PIPELINE_CKPTS = {
    "tech":     BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "home":     BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "apparel":  BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    "beauty":   BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
    "fallback": BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
}

DEBERTA_PIPELINE_CKPTS = {
    "tech":     BASE_DIR / "models/category_experts_deberta/expert_tech/best_model.pt",
    "home":     BASE_DIR / "models/category_experts_deberta/expert_home/best_model.pt",
    "apparel":  BASE_DIR / "models/category_experts_deberta/expert_apparel/best_model.pt",
    "beauty":   BASE_DIR / "models/comparison/deberta_v3_mams/best_model.pt",
    "fallback": BASE_DIR / "models/comparison/deberta_v3_mams/best_model.pt",
}

DEBERTA_CHECKPOINTS = {
    "Generic Foundation (MAMS)":    BASE_DIR / "models/comparison/deberta_v3_mams/best_model.pt",
    "Tech Specialist (Laptop14)":   BASE_DIR / "models/category_experts_deberta/expert_tech/best_model.pt",
    "Home Specialist (Television)": BASE_DIR / "models/category_experts_deberta/expert_home/best_model.pt",
    "Apparel Specialist (T-Shirt)": BASE_DIR / "models/category_experts_deberta/expert_apparel/best_model.pt",
}

CATEGORY_MAP = {
    "Laptop14 (Tech)":   "Electronics",
    "Television (Home)": "Home_and_Kitchen",
    "T-Shirt (Apparel)": "Clothing_Shoes_and_Jewelry",
    "MAMS Challenge":    None,
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_ci(labels: list, preds: list, n: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    accs, f1s = [], []
    for _ in range(n):
        idx = rng.integers(0, len(labels_arr), size=len(labels_arr))
        accs.append(accuracy_score(labels_arr[idx], preds_arr[idx]))
        f1s.append(f1_score(labels_arr[idx], preds_arr[idx], average="macro", zero_division=0))
    return {
        "accuracy_ci95": [round(float(np.percentile(accs, 2.5)) * 100, 2), round(float(np.percentile(accs, 97.5)) * 100, 2)],
        "macro_f1_ci95": [round(float(np.percentile(f1s, 2.5)) * 100, 2), round(float(np.percentile(f1s, 97.5)) * 100, 2)],
    }


def compute_metrics(labels: list, preds: list) -> dict:
    acc = accuracy_score(labels, preds)
    per_class_f1 = f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    ci = bootstrap_ci(labels, preds)
    return {
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "per_class_f1": {
            "positive": round(float(per_class_f1[0]) * 100, 2),
            "neutral":  round(float(per_class_f1[1]) * 100, 2),
            "negative": round(float(per_class_f1[2]) * 100, 2),
        },
        "accuracy_ci95": ci["accuracy_ci95"],
        "macro_f1_ci95": ci["macro_f1_ci95"],
    }


def evaluate_single_model(model: AspectTargetedSentimentModel, dataset: AspectSentimentDataset) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask   = batch["aspect_mask"].to(DEVICE)
            labels        = batch["label"].to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits, _ = model(input_ids, attention_mask, aspect_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_preds)


def evaluate_moe(pipeline: ProductLensMoEPipeline, raw_df: pd.DataFrame, category: str | None) -> dict:
    all_preds, all_labels, routed_experts = [], [], []
    for _, row in raw_df.iterrows():
        text      = str(row["text"])
        aspect    = str(row["aspect"])
        asp_start = int(row.get("aspect_start", 0))
        asp_end   = int(row.get("aspect_end", len(aspect)))
        gold      = LABEL2ID.get(str(row.get("sentiment", "neutral")).lower(), 1)

        pred = pipeline.predict(
            text=text,
            aspect=aspect,
            category=category,
            aspect_start=asp_start,
            aspect_end=asp_end,
        )
        all_preds.append(LABEL2ID[pred.sentiment])
        all_labels.append(gold)
        routed_experts.append(pred.routed_expert)

    metrics = compute_metrics(all_labels, all_preds)
    return metrics


def train_deberta_expert(expert_key: str, seed: int = 42) -> Path:
    cfg = TRAIN_CONFIGS[expert_key]
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best_model.pt"

    print(f"\n{'='*70}")
    print(f"TRAINING DEBERTA-V3 SPECIALIST: {expert_key.upper()} ({cfg['domain']})")
    print(f"{'='*70}")

    train_df = pd.read_parquet(cfg["train_data"])

    # Seeded GroupShuffleSplit on 'text'
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, val_idx = next(splitter.split(train_df, train_df["sentiment"], groups=train_df["text"]))
    val_df   = train_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_df.iloc[train_idx].reset_index(drop=True)

    print(f"  Train Samples: {len(train_df)} | Validation Samples: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_NAME)
    train_ds = AspectSentimentDataset(train_df, tokenizer, max_length=128)
    val_ds   = AspectSentimentDataset(val_df, tokenizer, max_length=128)

    model = AspectTargetedSentimentModel(
        backbone_name=DEBERTA_NAME,
        num_labels=3,
        aspect_boost=3.5,
    )

    trainer = AspectSentimentTrainer(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        backbone_lr=1.5e-5,
        head_lr=6.0e-5,
        weight_decay=0.01,
        lambda_scl=0.10,
        scl_temp=0.07,
    )

    res = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=8,
        epochs=cfg["epochs"],
        eval_every_steps=cfg["eval_every_steps"],
        patience=cfg["patience"],
        output_dir=out_dir,
    )
    print(f"Saved: {best_ckpt} | Best Val Macro F1: {res['best_val_macro_f1']:.4f}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train & Compare DeBERTa-v3 Specialists vs RoBERTa")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoints exist")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== ProductLens Specialists: RoBERTa vs DeBERTa-v3 Suite Comparison (seed={args.seed}) ===\n")

    # 1. Train DeBERTa specialists if needed
    for key in ["expert_tech", "expert_home", "expert_apparel"]:
        ckpt = TRAIN_CONFIGS[key]["output_dir"] / "best_model.pt"
        if not (args.skip_train and ckpt.exists()):
            train_deberta_expert(key, seed=args.seed)
        else:
            print(f"Using existing checkpoint: {ckpt}")

    # Ensure MAMS foundation checkpoint exists
    mams_deberta = BASE_DIR / "models/comparison/deberta_v3_mams/best_model.pt"
    assert mams_deberta.exists(), f"MAMS DeBERTa checkpoint missing at {mams_deberta}!"

    # 2. Prepare Datasets
    raw_dfs = {k: pd.read_parquet(p) for k, p in DATA_PATHS.items()}
    n_samples = {k: len(df) for k, df in raw_dfs.items()}
    ecom_domains = [d for d in DATA_PATHS if d != "MAMS Challenge"]
    total_ecom_samples = sum(n_samples[d] for d in ecom_domains)

    roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_NAME)

    roberta_datasets = {k: AspectSentimentDataset(df, roberta_tok) for k, df in raw_dfs.items()}
    deberta_datasets = {k: AspectSentimentDataset(df, deberta_tok) for k, df in raw_dfs.items()}

    results = {
        "metadata": {
            "device": DEVICE,
            "seed": args.seed,
            "roberta_backbone": "roberta-base",
            "deberta_backbone": DEBERTA_NAME,
            "test_sample_counts": n_samples,
        },
        "roberta_suite": {},
        "deberta_suite": {},
        "moe_roberta": {},
        "moe_deberta": {},
    }

    # 3. Evaluate Individual Specialists (RoBERTa vs DeBERTa)
    print("\n" + "="*70)
    print("EVALUATING ROBERTA SPECIALIST SUITE")
    print("="*70)
    for model_name, ckpt_path in ROBERTA_CHECKPOINTS.items():
        print(f"Evaluating: {model_name}...")
        model = AspectTargetedSentimentModel("roberta-base", num_labels=3)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)

        results["roberta_suite"][model_name] = {}
        for dom, ds in roberta_datasets.items():
            m = evaluate_single_model(model, ds)
            results["roberta_suite"][model_name][dom] = m
            print(f"  -> {dom}: Acc={m['accuracy']}% [{m['accuracy_ci95'][0]}–{m['accuracy_ci95'][1]}] MacroF1={m['macro_f1']}%")

        # Composite
        weighted_acc = sum(results["roberta_suite"][model_name][d]["accuracy"] * n_samples[d] for d in ecom_domains) / total_ecom_samples
        results["roberta_suite"][model_name]["E-Commerce Composite"] = round(weighted_acc, 2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("EVALUATING DEBERTA-V3 SPECIALIST SUITE")
    print("="*70)
    for model_name, ckpt_path in DEBERTA_CHECKPOINTS.items():
        print(f"Evaluating: {model_name}...")
        model = AspectTargetedSentimentModel(DEBERTA_NAME, num_labels=3)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)

        results["deberta_suite"][model_name] = {}
        for dom, ds in deberta_datasets.items():
            m = evaluate_single_model(model, ds)
            results["deberta_suite"][model_name][dom] = m
            print(f"  -> {dom}: Acc={m['accuracy']}% [{m['accuracy_ci95'][0]}–{m['accuracy_ci95'][1]}] MacroF1={m['macro_f1']}%")

        # Composite
        weighted_acc = sum(results["deberta_suite"][model_name][d]["accuracy"] * n_samples[d] for d in ecom_domains) / total_ecom_samples
        results["deberta_suite"][model_name]["E-Commerce Composite"] = round(weighted_acc, 2)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Evaluate MoE Pipelines (RoBERTa vs DeBERTa)
    print("\n" + "="*70)
    print("EVALUATING ROBERTA MOE PIPELINE")
    print("="*70)
    pipe_roberta = ProductLensMoEPipeline(checkpoint_paths=ROBERTA_PIPELINE_CKPTS, backbone_name="roberta-base", device=DEVICE, single_active_expert=True)
    for mode_name, use_cat in [("with_category", True), ("no_category", False)]:
        results["moe_roberta"][mode_name] = {}
        print(f"Mode: {mode_name}...")
        for dom, df in raw_dfs.items():
            cat = CATEGORY_MAP[dom] if use_cat else None
            m = evaluate_moe(pipe_roberta, df, cat)
            results["moe_roberta"][mode_name][dom] = m
            print(f"  -> {dom}: Acc={m['accuracy']}% MacroF1={m['macro_f1']}%")

        weighted_acc = sum(results["moe_roberta"][mode_name][d]["accuracy"] * n_samples[d] for d in ecom_domains) / total_ecom_samples
        results["moe_roberta"][mode_name]["E-Commerce Composite"] = round(weighted_acc, 2)

    pipe_roberta.unload_all()
    del pipe_roberta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "="*70)
    print("EVALUATING DEBERTA-V3 MOE PIPELINE")
    print("="*70)
    pipe_deberta = ProductLensMoEPipeline(checkpoint_paths=DEBERTA_PIPELINE_CKPTS, backbone_name=DEBERTA_NAME, device=DEVICE, single_active_expert=True)
    for mode_name, use_cat in [("with_category", True), ("no_category", False)]:
        results["moe_deberta"][mode_name] = {}
        print(f"Mode: {mode_name}...")
        for dom, df in raw_dfs.items():
            cat = CATEGORY_MAP[dom] if use_cat else None
            m = evaluate_moe(pipe_deberta, df, cat)
            results["moe_deberta"][mode_name][dom] = m
            print(f"  -> {dom}: Acc={m['accuracy']}% MacroF1={m['macro_f1']}%")

        weighted_acc = sum(results["moe_deberta"][mode_name][d]["accuracy"] * n_samples[d] for d in ecom_domains) / total_ecom_samples
        results["moe_deberta"][mode_name]["E-Commerce Composite"] = round(weighted_acc, 2)

    pipe_deberta.unload_all()
    del pipe_deberta
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Save JSON
    json_path = BASE_DIR / "benchmarks/roberta_vs_deberta_experts_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 6. Generate Markdown Report
    def r_acc(m_key, d_key):
        return results["roberta_suite"][m_key][d_key]["accuracy"]

    def d_acc(m_key, d_key):
        return results["deberta_suite"][m_key][d_key]["accuracy"]

    def diff_str(val_deb, val_rob):
        diff = round(val_deb - val_rob, 2)
        return f"+{diff}%" if diff > 0 else f"{diff}%"

    md_lines = [
        "# Head-to-Head Benchmark: RoBERTa-base Specialists vs DeBERTa-v3 Specialists",
        "",
        "> Comparing domain-specialist fine-tuning between RoBERTa-base and DeBERTa-v3 ABSA across held-out test benchmarks.",
        "> All specialist models use ProductLens Aspect-Targeted Attention Pooling (Boost=3.5) + Supervised Contrastive Loss (SCL).",
        "> Composite = sample-weighted test accuracy across held-out e-commerce domains (2,052 samples).",
        "",
        "## 1. Direct Specialist In-Domain Head-to-Head",
        "",
        "| Domain Vertical | Held-Out Test Set | RoBERTa Specialist | DeBERTa-v3 Specialist | Delta (DeBERTa vs RoBERTa) |",
        "| :--- | :--- | :---: | :---: | :---: |",
        f"| **Tech Hardware** | Laptop14 ({n_samples['Laptop14 (Tech)']}) | {r_acc('Tech Specialist (Laptop14)', 'Laptop14 (Tech)')}% | **{d_acc('Tech Specialist (Laptop14)', 'Laptop14 (Tech)')}%** | **{diff_str(d_acc('Tech Specialist (Laptop14)', 'Laptop14 (Tech)'), r_acc('Tech Specialist (Laptop14)', 'Laptop14 (Tech)'))}** |",
        f"| **Home Electronics** | Television ({n_samples['Television (Home)']}) | {r_acc('Home Specialist (Television)', 'Television (Home)')}% | **{d_acc('Home Specialist (Television)', 'Television (Home)')}%** | **{diff_str(d_acc('Home Specialist (Television)', 'Television (Home)'), r_acc('Home Specialist (Television)', 'Television (Home)'))}** |",
        f"| **Apparel & Fashion** | T-Shirt ({n_samples['T-Shirt (Apparel)']}) | {r_acc('Apparel Specialist (T-Shirt)', 'T-Shirt (Apparel)')}% | **{d_acc('Apparel Specialist (T-Shirt)', 'T-Shirt (Apparel)')}%** | **{diff_str(d_acc('Apparel Specialist (T-Shirt)', 'T-Shirt (Apparel)'), r_acc('Apparel Specialist (T-Shirt)', 'T-Shirt (Apparel)'))}** |",
        f"| **Universal Fallback** | MAMS Challenge ({n_samples['MAMS Challenge']}) | {r_acc('Generic Foundation (MAMS)', 'MAMS Challenge')}% | **{d_acc('Generic Foundation (MAMS)', 'MAMS Challenge')}%** | **{diff_str(d_acc('Generic Foundation (MAMS)', 'MAMS Challenge'), r_acc('Generic Foundation (MAMS)', 'MAMS Challenge'))}** |",
        "",
        "## 2. MoE Pipeline System Comparison",
        "",
        "| MoE System Configuration | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite (Weighted) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **RoBERTa MoE (No Category)** | {results['moe_roberta']['no_category']['Laptop14 (Tech)']['accuracy']}% | {results['moe_roberta']['no_category']['Television (Home)']['accuracy']}% | {results['moe_roberta']['no_category']['T-Shirt (Apparel)']['accuracy']}% | {results['moe_roberta']['no_category']['MAMS Challenge']['accuracy']}% | **{results['moe_roberta']['no_category']['E-Commerce Composite']}%** |",
        f"| **DeBERTa MoE (No Category)** | {results['moe_deberta']['no_category']['Laptop14 (Tech)']['accuracy']}% | {results['moe_deberta']['no_category']['Television (Home)']['accuracy']}% | {results['moe_deberta']['no_category']['T-Shirt (Apparel)']['accuracy']}% | {results['moe_deberta']['no_category']['MAMS Challenge']['accuracy']}% | **{results['moe_deberta']['no_category']['E-Commerce Composite']}%** |",
        f"| **RoBERTa MoE (With Category)** | {results['moe_roberta']['with_category']['Laptop14 (Tech)']['accuracy']}% | {results['moe_roberta']['with_category']['Television (Home)']['accuracy']}% | {results['moe_roberta']['with_category']['T-Shirt (Apparel)']['accuracy']}% | {results['moe_roberta']['with_category']['MAMS Challenge']['accuracy']}% | **{results['moe_roberta']['with_category']['E-Commerce Composite']}%** |",
        f"| **DeBERTa MoE (With Category)** | {results['moe_deberta']['with_category']['Laptop14 (Tech)']['accuracy']}% | {results['moe_deberta']['with_category']['Television (Home)']['accuracy']}% | {results['moe_deberta']['with_category']['T-Shirt (Apparel)']['accuracy']}% | {results['moe_deberta']['with_category']['MAMS Challenge']['accuracy']}% | **{results['moe_deberta']['with_category']['E-Commerce Composite']}%** |",
        "",
        "## 3. Full Transfer Matrix: DeBERTa-v3 Specialists",
        "",
        "| Specialist Model | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name in DEBERTA_CHECKPOINTS.keys():
        row_accs = [f"{d_acc(model_name, dom)}%" for dom in DATA_PATHS.keys()]
        comp = f"{results['deberta_suite'][model_name]['E-Commerce Composite']}%"
        md_lines.append(f"| **{model_name}** | {' | '.join(row_accs)} | {comp} |")

    md_path = BASE_DIR / "benchmarks/roberta_vs_deberta_experts_comparison.md"
    md_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + md_content)
    print(f"\nFull results saved to:\n  - JSON: {json_path}\n  - MD:   {md_path}")


if __name__ == "__main__":
    main()
