# ProductLens: Complete Model Generation Recipes & Codebase Map

This document confirms that **every single model in the ProductLens MoE platform is fully reproducible from source code** in this repository. There are no black-box or unaccounted-for `.pt` files.

---

## 1. Codebase Architecture Map

| Pipeline Component | Source File Location | Purpose & Mechanics |
| :--- | :--- | :--- |
| **Neural Backbone & Attention Pooling** | [`productlens/sentiment/model.py`](../productlens/sentiment/model.py) | RoBERTa encoder, `AspectTargetedAttentionPooler` ($\beta=3.5$), 128-dim projection head, and classification MLP. |
| **Supervised Contrastive Loss** | [`productlens/training/loss.py`](../productlens/training/loss.py) | `JointABSALoss` combining class-weighted cross-entropy with temperature-scaled SupCon ($\tau=0.07$, $\lambda=0.10$). |
| **Subword Span Alignment** | [`productlens/training/dataset.py`](../productlens/training/dataset.py) | Tokenizes review text and creates exact subword binary masks for target aspect spans. |
| **End-to-End Trainer** | [`productlens/training/trainer.py`](../productlens/training/trainer.py) | Optimization loop with differential learning rates, linear warmup, sub-epoch validation, and early stopping. |
| **Hyperparameter Configurations** | [`configs/training_configs.yaml`](../configs/training_configs.yaml) | YAML manifest detailing dataset paths, learning rates, epochs, and early-stopping parameters for every model. |
| **Master Generator Script** | [`scripts/train_all_experts.py`](../scripts/train_all_experts.py) | Executable CLI to train any individual model or the entire 4-model suite sequentially from scratch. |
| **Automated Benchmark Evaluator** | [`scripts/evaluate_comparison.py`](../scripts/evaluate_comparison.py) | Runs all models across 3,388 test samples and generates the transfer matrix comparison. |

---

## 2. Model Generation Recipes

### Recipe A: Generic Foundation Model (`foundation_mams`)
* **Objective**: Universal aspect sentiment handling; serves as the MoE universal fallback.
* **Training Data**: [`data/benchmarks/multi_aspect_mams/train.parquet`](../data/benchmarks/multi_aspect_mams/train.parquet) (11,186 multi-aspect challenge samples).
* **Validation Data**: [`data/benchmarks/multi_aspect_mams/val.parquet`](../data/benchmarks/multi_aspect_mams/val.parquet) (1,332 samples).
* **Generation Command**:
  ```bash
  python scripts/train_all_experts.py --model generic
  ```
* **Output Artifact**: `models/category_experts/foundation_mams/best_model.pt`
* **Benchmark Result**: **84.21%** on MAMS Challenge.

---

### Recipe B: Tech Hardware Specialist (`expert_tech`)
* **Objective**: Maximum precision on laptop, computer, and hardware reviews.
* **Training Data**: [`data/benchmarks/tech_laptop/train.parquet`](../data/benchmarks/tech_laptop/train.parquet) (SemEval Laptop14 dataset).
* **Generation Command**:
  ```bash
  python scripts/train_all_experts.py --model tech
  ```
* **Output Artifact**: `models/category_experts/expert_tech/best_model.pt`
* **Benchmark Result**: **84.17%** on Laptop14 (+12.38% over Generic Foundation).

---

### Recipe C: Home Electronics Specialist (`expert_home`)
* **Objective**: Maximum precision on TV, audio, and appliance reviews.
* **Training Data**: [`data/benchmarks/home_television/train.parquet`](../data/benchmarks/home_television/train.parquet) (Television benchmark dataset).
* **Generation Command**:
  ```bash
  python scripts/train_all_experts.py --model home
  ```
* **Output Artifact**: `models/category_experts/expert_home/best_model.pt`
* **Benchmark Result**: **89.81%** on Television (+10.72% over Generic Foundation).

---

### Recipe D: Apparel & Fashion Specialist (`expert_apparel`)
* **Objective**: Maximum precision on clothing, fit, and fabric reviews.
* **Training Data**: [`data/benchmarks/apparel_tshirt/train.parquet`](../data/benchmarks/apparel_tshirt/train.parquet) (T-Shirt benchmark dataset).
* **Generation Command**:
  ```bash
  python scripts/train_all_experts.py --model apparel
  ```
* **Output Artifact**: `models/category_experts/expert_apparel/best_model.pt`
* **Benchmark Result**: **94.07%** on T-Shirt (+6.57% over Generic Foundation).

---

## 3. How to Retrain the Entire Suite in One Command

To reproduce all 4 checkpoints completely from scratch using the GPU:
```bash
python scripts/train_all_experts.py --all
```
This will sequentially train:
1. `generic_foundation` $\rightarrow$ `models/category_experts/foundation_mams/best_model.pt`
2. `expert_tech` $\rightarrow$ `models/category_experts/expert_tech/best_model.pt`
3. `expert_home` $\rightarrow$ `models/category_experts/expert_home/best_model.pt`
4. `expert_apparel` $\rightarrow$ `models/category_experts/expert_apparel/best_model.pt`
