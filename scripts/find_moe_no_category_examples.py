#!/usr/bin/env python3
"""
Find concrete examples where DeBERTa MoE (No Category) gets the correct sentiment
while RoBERTa MoE (No Category) fails.
"""

import json
from pathlib import Path
import pandas as pd
import torch

from productlens.sentiment.pipeline import ProductLensMoEPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROBERTA_CKPTS = {
    "tech": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "home": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "apparel": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    "fallback": BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
}

DEBERTA_CKPTS = {
    "tech": BASE_DIR / "models/category_experts_deberta/expert_tech/best_model.pt",
    "home": BASE_DIR / "models/category_experts_deberta/expert_home/best_model.pt",
    "apparel": BASE_DIR / "models/category_experts_deberta/expert_apparel/best_model.pt",
    "fallback": BASE_DIR / "models/comparison/deberta_v3_mams/best_model.pt",
}

DATA_PATHS = {
    "Laptop14 (Tech)": BASE_DIR / "data/benchmarks/tech_laptop/test.parquet",
    "Television (Home)": BASE_DIR / "data/benchmarks/home_television/test.parquet",
    "T-Shirt (Apparel)": BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet",
}


def main():
    collected_examples = []

    for domain_name, data_path in DATA_PATHS.items():
        print(f"\n==========================================")
        print(f"Scanning domain: {domain_name}")
        print(f"==========================================")
        df = pd.read_parquet(data_path)

        # 1. Collect RoBERTa predictions
        print("Running RoBERTa MoE (No Category)...")
        pipe_roberta = ProductLensMoEPipeline(
            checkpoint_paths=ROBERTA_CKPTS,
            backbone_name="roberta-base",
            device=DEVICE,
            single_active_expert=True,
        )
        roberta_preds = []
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            asp_start = int(row.get("aspect_start", 0))
            asp_end = int(row.get("aspect_end", len(aspect)))
            pred = pipe_roberta.predict(
                text=text,
                aspect=aspect,
                category=None,
                aspect_start=asp_start,
                aspect_end=asp_end,
            )
            roberta_preds.append(pred)

        pipe_roberta.unload_all()
        del pipe_roberta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Collect DeBERTa predictions
        print("Running DeBERTa-v3 MoE (No Category)...")
        pipe_deberta = ProductLensMoEPipeline(
            checkpoint_paths=DEBERTA_CKPTS,
            backbone_name="yangheng/deberta-v3-base-absa-v1.1",
            device=DEVICE,
            single_active_expert=True,
        )
        deberta_preds = []
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            asp_start = int(row.get("aspect_start", 0))
            asp_end = int(row.get("aspect_end", len(aspect)))
            pred = pipe_deberta.predict(
                text=text,
                aspect=aspect,
                category=None,
                aspect_start=asp_start,
                aspect_end=asp_end,
            )
            deberta_preds.append(pred)

        pipe_deberta.unload_all()
        del pipe_deberta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Compare and select high-contrast cases
        domain_contrast = 0
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            aspect = str(row["aspect"]).strip()
            gold = str(row.get("sentiment", "")).lower()
            pred_r = roberta_preds[idx]
            pred_d = deberta_preds[idx]

            if pred_d.sentiment == gold and pred_r.sentiment != gold:
                domain_contrast += 1
                collected_examples.append({
                    "domain": domain_name,
                    "text": text,
                    "aspect": aspect,
                    "gold": gold,
                    "roberta": {
                        "sentiment": pred_r.sentiment,
                        "confidence": round(pred_r.confidence, 4),
                        "routed_expert": pred_r.routed_expert,
                        "probabilities": {k: round(v, 4) for k, v in pred_r.probabilities.items()},
                    },
                    "deberta": {
                        "sentiment": pred_d.sentiment,
                        "confidence": round(pred_d.confidence, 4),
                        "routed_expert": pred_d.routed_expert,
                        "probabilities": {k: round(v, 4) for k, v in pred_d.probabilities.items()},
                    },
                })

        print(f"  -> Found {domain_contrast} contrast examples in {domain_name}")

    print(f"\nTotal contrast examples found across all domains: {len(collected_examples)}")

    out_file = BASE_DIR / "benchmarks/no_category_contrast_examples.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(collected_examples, f, indent=2)
    print(f"Saved full collected list to {out_file}")


if __name__ == "__main__":
    main()
