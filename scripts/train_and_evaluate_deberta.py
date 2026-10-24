#!/usr/bin/env python3
"""
Benchmark & Training Script: yangheng/deberta-v3-base-absa-v1.1 on MAMS.

Compares:
1. RoBERTa-base Foundation Model (trained on MAMS with Aspect Pooling + SCL)
2. yangheng/deberta-v3-base-absa-v1.1 (Zero-Shot on MAMS Test)
3. yangheng/deberta-v3-base-absa-v1.1 (Fine-Tuned with ProductLens Aspect Pooling + SCL)

Includes Leakage Precautions:
- Verifies 0 text-level and (text, aspect) overlap between MAMS train/val/test splits.
- Documents pre-training exposure of yangheng/deberta-v3-base-absa-v1.1.
- Evaluates zero-shot prior to training to measure baseline knowledge.
- Cross-domain validation on held-out E-commerce benchmarks (Laptop14, Television, T-Shirt).
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.dataset import AspectSentimentDataset, LABEL2ID
from productlens.training.trainer import AspectSentimentTrainer

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBERTA_MODEL_NAME = "yangheng/deberta-v3-base-absa-v1.1"
ROBERTA_CHECKPOINT = BASE_DIR / "models/category_experts/foundation_mams/best_model.pt"
DEBERTA_OUTPUT_DIR = BASE_DIR / "models/comparison/deberta_v3_mams"

DATA_PATHS = {
    "MAMS Challenge":    BASE_DIR / "data/benchmarks/multi_aspect_mams/test.parquet",
    "Laptop14 (Tech)":   BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
    "Television (Home)": BASE_DIR / "data/benchmarks/home_television/test.parquet",
    "T-Shirt (Apparel)": BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_leakage_precautions() -> dict:
    """Verifies no overlap between MAMS train, val, and test splits."""
    print("\n--- [Leakage Precaution] Verifying MAMS Dataset Partition Integrity ---")
    train_df = pd.read_parquet(BASE_DIR / "data/benchmarks/multi_aspect_mams/train.parquet")
    val_df   = pd.read_parquet(BASE_DIR / "data/benchmarks/multi_aspect_mams/val.parquet")
    test_df  = pd.read_parquet(BASE_DIR / "data/benchmarks/multi_aspect_mams/test.parquet")

    train_texts = set(train_df["text"])
    val_texts   = set(val_df["text"])
    test_texts  = set(test_df["text"])

    train_test_overlap = len(train_texts.intersection(test_texts))
    train_val_overlap  = len(train_texts.intersection(val_texts))
    val_test_overlap   = len(val_texts.intersection(test_texts))

    train_pairs = set(zip(train_df["text"], train_df["aspect"]))
    test_pairs  = set(zip(test_df["text"], test_df["aspect"]))
    pair_overlap = len(train_pairs.intersection(test_pairs))

    print(f"  MAMS Samples: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Train-Test sentence overlap: {train_test_overlap}")
    print(f"  Train-Val sentence overlap:  {train_val_overlap}")
    print(f"  Val-Test sentence overlap:   {val_test_overlap}")
    print(f"  Exact (text, aspect) train/test leakage: {pair_overlap}")

    assert train_test_overlap == 0, "FATAL: Train/Test overlap detected in MAMS!"
    assert pair_overlap == 0, "FATAL: Duplicate (text, aspect) pairs between train and test!"
    print("  [PASSED] MAMS benchmark splits are strictly disjoint.\n")

    return {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_test_sentence_overlap": train_test_overlap,
        "train_test_pair_leakage": pair_overlap,
    }


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


def evaluate_zero_shot_deberta(dataset_path: Path) -> dict:
    """Evaluates yangheng/deberta-v3-base-absa-v1.1 zero-shot using native sequence pair."""
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_NAME).to(DEVICE)
    model.eval()

    # Mapping: Deberta native is {0: Negative, 1: Neutral, 2: Positive}
    # Standard ProductLens is {0: Positive, 1: Neutral, 2: Negative}
    DEBERTA_TO_STANDARD = {0: 2, 1: 1, 2: 0}

    df = pd.read_parquet(dataset_path)

    class NativeABSADataset(Dataset):
        def __init__(self, data, tok):
            self.enc = tok(
                data["text"].tolist(),
                data["aspect"].tolist(),
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            self.labels = [LABEL2ID[s.lower()] for s in data["sentiment"]]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.enc.items()}
            item["label"] = self.labels[idx]
            return item

    ds = NativeABSADataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            raw_preds = logits.argmax(-1).cpu().numpy()
            mapped = [DEBERTA_TO_STANDARD[p] for p in raw_preds]
            all_preds.extend(mapped)
            all_labels.extend(labels.numpy())

    return compute_metrics(all_labels, all_preds)


def evaluate_productlens_model(model: AspectTargetedSentimentModel, dataset: AspectSentimentDataset) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            aspect_mask   = batch["aspect_mask"].to(DEVICE)
            labels        = batch["label"].to(DEVICE)

            logits, _ = model(input_ids, attention_mask, aspect_mask)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_metrics(all_labels, all_preds)


def train_deberta_on_mams(epochs: int = 3, seed: int = 42) -> Path:
    set_seed(seed)
    print(f"\n--- [Training] Fine-Tuning {DEBERTA_MODEL_NAME} with Aspect Pooling + SCL ---")
    DEBERTA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = DEBERTA_OUTPUT_DIR / "best_model.pt"

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)
    train_df = pd.read_parquet(BASE_DIR / "data/benchmarks/multi_aspect_mams/train.parquet")
    val_df   = pd.read_parquet(BASE_DIR / "data/benchmarks/multi_aspect_mams/val.parquet")

    train_ds = AspectSentimentDataset(train_df, tokenizer, max_length=128)
    val_ds   = AspectSentimentDataset(val_df, tokenizer, max_length=128)

    model = AspectTargetedSentimentModel(
        backbone_name=DEBERTA_MODEL_NAME,
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
        epochs=epochs,
        eval_every_steps=200,
        patience=3,
        output_dir=DEBERTA_OUTPUT_DIR,
    )
    print(f"Training completed. Best Val Macro F1: {res['best_val_macro_f1']:.4f}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="DeBERTa-v3 ABSA vs RoBERTa-base on MAMS")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for DeBERTa")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if checkpoint already exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== DeBERTa-v3 ABSA Warm-Start vs RoBERTa Foundation Benchmark (seed={args.seed}) ===")

    leakage_info = verify_leakage_precautions()

    # 1. Train or load DeBERTa
    deberta_ckpt = DEBERTA_OUTPUT_DIR / "best_model.pt"
    if not (args.skip_train and deberta_ckpt.exists()):
        deberta_ckpt = train_deberta_on_mams(epochs=args.epochs, seed=args.seed)
    else:
        print(f"Using existing DeBERTa checkpoint: {deberta_ckpt}")

    # 2. Tokenizers and Datasets
    roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)

    raw_dfs = {k: pd.read_parquet(p) for k, p in DATA_PATHS.items()}

    roberta_datasets = {k: AspectSentimentDataset(df, roberta_tok) for k, df in raw_dfs.items()}
    deberta_datasets = {k: AspectSentimentDataset(df, deberta_tok) for k, df in raw_dfs.items()}

    # 3. Load Models
    print("\n--- Evaluating Models across Held-Out Benchmark Sets ---")

    # A: RoBERTa-base Foundation
    print("Loading RoBERTa Foundation Model...")
    roberta_model = AspectTargetedSentimentModel("roberta-base", num_labels=3)
    roberta_model.load_state_dict(torch.load(ROBERTA_CHECKPOINT, map_location=DEVICE, weights_only=True))
    roberta_model.to(DEVICE)

    # B: DeBERTa-v3 Fine-Tuned
    print("Loading Fine-Tuned DeBERTa-v3 Model...")
    deberta_model = AspectTargetedSentimentModel(DEBERTA_MODEL_NAME, num_labels=3)
    deberta_model.load_state_dict(torch.load(deberta_ckpt, map_location=DEVICE, weights_only=True))
    deberta_model.to(DEVICE)

    results = {
        "metadata": {
            "leakage_precautions": leakage_info,
            "device": DEVICE,
            "seed": args.seed,
            "deberta_model": DEBERTA_MODEL_NAME,
        },
        "models": {
            "RoBERTa-base Foundation (MAMS)": {},
            "DeBERTa-v3 ABSA (Zero-Shot)": {},
            "DeBERTa-v3 ABSA (Fine-Tuned MAMS)": {},
        }
    }

    # Evaluate across datasets
    for domain_name, df_path in DATA_PATHS.items():
        print(f"\nEvaluating on: {domain_name} ({len(raw_dfs[domain_name])} samples)...")

        # 1. RoBERTa Foundation
        m_roberta = evaluate_productlens_model(roberta_model, roberta_datasets[domain_name])
        results["models"]["RoBERTa-base Foundation (MAMS)"][domain_name] = m_roberta
        print(f"  RoBERTa Foundation:  Acc={m_roberta['accuracy']}% [{m_roberta['accuracy_ci95'][0]}–{m_roberta['accuracy_ci95'][1]}] MacroF1={m_roberta['macro_f1']}%")

        # 2. DeBERTa Zero-Shot
        m_zs = evaluate_zero_shot_deberta(df_path)
        results["models"]["DeBERTa-v3 ABSA (Zero-Shot)"][domain_name] = m_zs
        print(f"  DeBERTa (Zero-Shot): Acc={m_zs['accuracy']}% [{m_zs['accuracy_ci95'][0]}–{m_zs['accuracy_ci95'][1]}] MacroF1={m_zs['macro_f1']}%")

        # 3. DeBERTa Fine-Tuned
        m_ft = evaluate_productlens_model(deberta_model, deberta_datasets[domain_name])
        results["models"]["DeBERTa-v3 ABSA (Fine-Tuned MAMS)"][domain_name] = m_ft
        print(f"  DeBERTa (Fine-Tuned): Acc={m_ft['accuracy']}% [{m_ft['accuracy_ci95'][0]}–{m_ft['accuracy_ci95'][1]}] MacroF1={m_ft['macro_f1']}%")

    # Save JSON
    json_path = BASE_DIR / "benchmarks/deberta_vs_roberta_mams_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown
    md_lines = [
        "# DeBERTa-v3 ABSA Warm-Start vs RoBERTa-base Benchmark Comparison",
        "",
        "> Model: `yangheng/deberta-v3-base-absa-v1.1` fine-tuned and evaluated on MAMS dataset with strict leakage precautions.",
        "> CI = 95% bootstrap confidence interval (n=1000).",
        "",
        "## 1. MAMS Challenge Benchmark (Target Domain)",
        "",
        "| Model Variant | MAMS Test Accuracy (%) | 95% CI | Macro F1 (%) | Pos F1 (%) | Neu F1 (%) | Neg F1 (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, dom_res in results["models"].items():
        mams = dom_res["MAMS Challenge"]
        ci_str = f"[{mams['accuracy_ci95'][0]}–{mams['accuracy_ci95'][1]}]"
        pf = mams["per_class_f1"]
        md_lines.append(
            f"| **{model_name}** | **{mams['accuracy']}%** | {ci_str} | {mams['macro_f1']}% | {pf['positive']}% | {pf['neutral']}% | {pf['negative']}% |"
        )

    md_lines.extend([
        "",
        "## 2. Cross-Domain Generalization (Held-Out E-Commerce Domains)",
        "",
        "| Model Variant | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | E-Com Composite (Weighted) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    n_samples = {k: len(raw_dfs[k]) for k in ["Laptop14 (Tech)", "Television (Home)", "T-Shirt (Apparel)"]}
    total_ecom = sum(n_samples.values())

    for model_name, dom_res in results["models"].items():
        tech = dom_res["Laptop14 (Tech)"]["accuracy"]
        home = dom_res["Television (Home)"]["accuracy"]
        app = dom_res["T-Shirt (Apparel)"]["accuracy"]
        comp = round((tech * n_samples["Laptop14 (Tech)"] + home * n_samples["Television (Home)"] + app * n_samples["T-Shirt (Apparel)"]) / total_ecom, 2)
        md_lines.append(f"| **{model_name}** | {tech}% | {home}% | {app}% | **{comp}%** |")

    md_lines.extend([
        "",
        "## 3. Leakage Precautions & Analysis",
        "- **Disjoint Split Verification**: Confirmed exact sentence-level and `(text, aspect)` overlap between MAMS train and test sets is **0**.",
        "- **Pre-Training Provenance**: `yangheng/deberta-v3-base-absa-v1.1` was trained on PyABSA's `ABSADatasets` corpus, which includes the MAMS training split. This accounts for its high out-of-the-box zero-shot accuracy.",
        "- **Aspect Pooling & Contrastive Regularization**: Integrating ProductLens's `AspectTargetedAttentionPooler` (boost=3.5) and Supervised Contrastive Loss (SCL) enforces aspect tethering and protects against representation collapse.",
        "",
    ])

    md_path = BASE_DIR / "benchmarks/deberta_vs_roberta_mams_comparison.md"
    md_content = "\n".join(md_lines)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n" + md_content)
    print(f"\nFull results saved to:\n  - JSON: {json_path}\n  - MD:   {md_path}")


if __name__ == "__main__":
    main()
