# ProductLens: Complete Model Generation Recipe & Codebase Map

This document confirms that **the ProductLens unified sentiment model is fully reproducible from source code** in this repository. There are no black-box or unaccounted-for `.pt` files.

---

## 1. Codebase Architecture Map

| Pipeline Component | Source File Location | Purpose & Mechanics |
| :--- | :--- | :--- |
| **Neural Backbone & Attention Pooling** | [`productlens/sentiment/model.py`](../productlens/sentiment/model.py) | DeBERTa-v3 ABSA encoder (`yangheng/deberta-v3-base-absa-v1.1`), `AspectTargetedAttentionPooler` with Query-Key Cross-Attention ($\beta=3.5$) and Tripartite Context Fusion. |
| **Ordinal Wasserstein & SCL Loss** | [`productlens/training/loss.py`](../productlens/training/loss.py) | `JointABSALoss` combining class-weighted cross-entropy with 1D Earth Mover's Distance ($\lambda_{\text{ordinal}}=0.15$), label smoothing ($\epsilon=0.05$), and contrastive SCL ($\tau=0.07$, $\lambda=0.10$). |
| **Context Windowing & Tokenization** | [`productlens/training/dataset.py`](../productlens/training/dataset.py) | Dynamic aspect-centered windowing preserving clauses $>128$ tokens, with exact subword binary masks for target aspect spans. |
| **End-to-End Trainer** | [`productlens/training/trainer.py`](../productlens/training/trainer.py) | Optimization loop with mixed precision (FP16), differential learning rates, composite neutral-aware checkpointing, live console progress, and early stopping. |
| **Post-Hoc Calibration** | [`productlens/sentiment/calibration.py`](../productlens/sentiment/calibration.py) | TemperatureScaler fitting validation logits to minimize Negative Log-Likelihood and Expected Calibration Error. |
| **Hyperparameter Configurations** | [`configs/training_configs.yaml`](../configs/training_configs.yaml) | YAML manifest detailing dataset paths, learning rates, epochs, and early-stopping parameters. |
| **Dataset Preparation Script** | [`scripts/prepare_unified_dataset.py`](../scripts/prepare_unified_dataset.py) | Consolidates multi-domain e-commerce reviews and downloads the official ARTS adversarial benchmark. |
| **Unified Training Script** | [`scripts/train.py`](../scripts/train.py) | Executable CLI to train the unified model with live evaluation output. |
| **Master Benchmark Evaluator** | [`scripts/evaluate_unified.py`](../scripts/evaluate_unified.py) | Evaluates models across held-out e-commerce domains and ARTS adversarial test sets. |

---

## 2. Production Model Generation Recipe

### Unified E-Commerce Sentiment Model (`unified_backbone`)
* **Objective**: Universal, high-accuracy e-commerce aspect sentiment classification with neutral-aware calibration.
* **Consolidated Training Data**: [`data/benchmarks/unified_ecommerce/train.parquet`](../data/benchmarks/unified_ecommerce/train.parquet) (**7,838 samples** across Tech, Home, and Apparel).
* **Consolidated Test Data**: [`data/benchmarks/unified_ecommerce/test.parquet`](../data/benchmarks/unified_ecommerce/test.parquet) (**2,052 samples**).
* **Adversarial Benchmark**: [`data/benchmarks/arts_laptop/test.parquet`](../data/benchmarks/arts_laptop/test.parquet) (**1,877 samples**).

### Step 1: Consolidate Training Data
```bash
python scripts/prepare_unified_dataset.py
```

### Step 2: Train Unified Model
```bash
python scripts/train.py --domain unified --batch_size 16 --epochs 10 --eval_steps 150 --patience 6
```

### Step 3: Evaluate on Held-Out Benchmark Suites
```bash
python scripts/evaluate_unified.py
```

* **Output Artifact**: `models/unified_backbone/best_model.pt` (712 MB) & `models/unified_backbone/calibration.json`
* **Benchmark Result**: **89.37% E-Commerce Composite Accuracy** (83.54% Tech Acc, 66.90% Neutral F1, 30.4% lower NLL).
