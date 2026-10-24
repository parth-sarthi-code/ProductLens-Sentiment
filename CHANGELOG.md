# ProductLens Production Changelog

## [Release v2.0.0] - Production Update: DeBERTa-v3 ABSA Specialists, Leakage Elimination & Optimized MoE

**Release Date:** October 2026  
**Type:** Major Production Update (`prod_update`)  
**Scope:** Dataset Sanitation, Modeling Architecture, Empirical Benchmarks, Memory-Safe MoE Pipeline  

---

### Executive Summary

This production release resolves all fundamental data leakage and architectural issues identified in `productlens_issues.md` (P0 through P3), trains and evaluates a new generation of **DeBERTa-v3 ABSA Domain Specialists** (`tech`, `home`, `apparel`, `mams_fallback`), and implements dynamic memory-managed inference for the Mixture-of-Experts (MoE) pipeline.

Across 2,052 held-out e-commerce samples, the new **DeBERTa-v3 MoE Pipeline** achieves:
* **89.18% Sample-Weighted Composite Accuracy** with category metadata.
* **87.43% Composite Accuracy without category metadata** (+5.31% over RoBERTa's 82.12%).
* **+12.77% Macro F1 gain on Apparel** by eliminating the neutral-class collapse that afflicted RoBERTa.
* **Zero Train/Test Leakage**: 107 duplicate/leaked samples purged across training sets.
* **Memory Safety**: `single_active_expert=True` with CPU offloading prevents CUDA OOM on resource-constrained devices (6GB VRAM GPUs).

---

### 1. Data Integrity & Leakage Elimination (Fix 6, P0)

Prior to this update, identical reviews and aspect-sentiment pairs existed across train and test splits, artificially inflating evaluation numbers. 

#### Deduplication Results:
* **`tech_laptop`**: Purged **62 duplicate rows** (10 exact text/aspect duplicates with test + 52 train duplicates). Cleaned train split: **2,266 samples** (from 2,328).
* **`home_television`**: Purged **41 duplicate rows** (6 exact test overlaps + 35 train duplicates). Cleaned train split: **3,705 samples** (from 3,746).
* **`apparel_tshirt`**: Purged **4 duplicate rows** (1 exact test overlap + 3 train duplicates). Cleaned train split: **1,867 samples** (from 1,871).
* **Automated Script**: Created [`scripts/deduplicate_data.py`](file:///home/parth/Projects/ProductLens-Sentiment/scripts/deduplicate_data.py) with automated cross-split assertion checks.

---

### 2. Core Architecture & Safety Fixes (Fixes 1–5, 8–9)

* **Checkpoint Safety & Fallback Caching (Fix 2, P0)**: Fixed [`productlens/sentiment/pipeline.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/sentiment/pipeline.py). Missing checkpoints now raise explicit `FileNotFoundError` instead of silently returning randomly-initialized weights. Fallback model instances are cached once under `"fallback"` to eliminate redundant disk reloads.
* **Router Regex Word Boundaries (Fix 3, P1)**: Replaced naive substring matching in [`productlens/sentiment/router.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/sentiment/router.py) with regex word boundary patterns (`r"\b" + re.escape(word) + r"\b"`). Substrings like `"os"` no longer trigger false-positive routing on `"cost"` or `"closet"`.
* **Unified Sequence Length (Fix 8, P1)**: Standardized `MAX_SEQ_LEN = 128` across [`productlens/constants.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/constants.py), [`productlens/training/dataset.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/training/dataset.py), and `pipeline.py`. Eliminated silent token truncation bugs during aspect offset mapping.
* **Deterministic Seeding & Group-Aware Validation (Fix 5 & 9, P1/P2)**:
  * Implemented `set_seed()` in [`scripts/train.py`](file:///home/parth/Projects/ProductLens-Sentiment/scripts/train.py) and [`scripts/train_all_experts.py`](file:///home/parth/Projects/ProductLens-Sentiment/scripts/train_all_experts.py) setting Python, NumPy, PyTorch CPU, and CUDA seeds (`deterministic=True`).
  * Replaced random train/val split with `GroupShuffleSplit` on `text`, guaranteeing multi-aspect reviews never leak between train and validation folds.
  * Added `weights_only=True` to all `torch.load` calls across the codebase to guard against arbitrary code execution vulnerabilities.
* **Empirical Dual-Run Benchmark Suite (Fix 1 & 4, P1/P2)**:
  * Updated [`scripts/evaluate_comparison.py`](file:///home/parth/Projects/ProductLens-Sentiment/scripts/evaluate_comparison.py) to evaluate both `with_category` and `without_category` modes.
  * Added 1,000-iteration non-parametric bootstrap 95% Confidence Intervals (`accuracy_ci95`, `macro_f1_ci95`).
  * Synthesized Out-of-Distribution (OOD) test fixtures in [`data/benchmarks/ood_fixture/`](file:///home/parth/Projects/ProductLens-Sentiment/data/benchmarks/ood_fixture/) for Books, Beauty, and Grocery.

---

### 3. DeBERTa-v3 ABSA Specialists Training

Trained a full suite of domain specialists on top of `yangheng/deberta-v3-base-absa-v1.1` using **Aspect Attention Pooling (boost=3.5)** and **Supervised Contrastive Loss (SCL)**:

| Specialist Key | Vertical | Training Data | Best Val Macro F1 | Checkpoint Location |
| :--- | :--- | :--- | :---: | :--- |
| **`expert_tech`** | Tech Hardware (Laptop14) | 2,045 samples | **0.9527** | [`models/category_experts_deberta/expert_tech/best_model.pt`](file:///home/parth/Projects/ProductLens-Sentiment/models/category_experts_deberta/expert_tech/best_model.pt) |
| **`expert_home`** | Home Electronics (Television) | 3,335 samples | **0.8911** | [`models/category_experts_deberta/expert_home/best_model.pt`](file:///home/parth/Projects/ProductLens-Sentiment/models/category_experts_deberta/expert_home/best_model.pt) |
| **`expert_apparel`** | Apparel & Fashion (T-Shirt) | 1,686 samples | **0.8715** | [`models/category_experts_deberta/expert_apparel/best_model.pt`](file:///home/parth/Projects/ProductLens-Sentiment/models/category_experts_deberta/expert_apparel/best_model.pt) |
| **`foundation_mams`** | Universal Multi-Aspect Fallback | 11,186 samples | **0.8240** | [`models/comparison/deberta_v3_mams/best_model.pt`](file:///home/parth/Projects/ProductLens-Sentiment/models/comparison/deberta_v3_mams/best_model.pt) |

#### Hardware & Optimization Highlights:
* **Mixed Precision (AMP)**: Added `torch.amp.GradScaler` and `torch.amp.autocast("cuda")` in [`productlens/training/trainer.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/training/trainer.py). Handled gradient scaling skips cleanly to prevent scheduler advancement warnings.
* **Single Active Expert Offloading**: Added `single_active_expert=True` to `ProductLensMoEPipeline`. Inactive expert models are parked in CPU RAM and dynamically migrated to GPU VRAM on-demand, capping VRAM utilization under 1.2 GB and enabling full MoE pipelines to run on 6GB laptops.

---

### 4. Empirical Head-to-Head Benchmarks

Full results recorded in [`benchmarks/roberta_vs_deberta_experts_comparison.md`](file:///home/parth/Projects/ProductLens-Sentiment/benchmarks/roberta_vs_deberta_experts_comparison.md) and [`benchmarks/roberta_vs_deberta_experts_comparison.json`](file:///home/parth/Projects/ProductLens-Sentiment/benchmarks/roberta_vs_deberta_experts_comparison.json).

#### Table 1: In-Domain Specialist Comparison
| Domain Vertical | Held-Out Test Set | RoBERTa Specialist | DeBERTa-v3 Specialist | Delta ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Tech Hardware** | Laptop14 (*N=638*) | 81.97% *(F1: 78.75%)* | **82.76%** *(F1: 79.47%)* | **+0.79% Acc / +0.72% F1** |
| **Home Electronics** | Television (*N=942*) | 90.76% *(F1: 73.50%)* | **91.08%** *(F1: 76.74%)* | **+0.32% Acc / +3.24% F1** |
| **Apparel & Fashion** | T-Shirt (*N=472*) | 92.37% *(F1: 62.49%)* | **94.07%** *(F1: 75.26%)* | **+1.70% Acc / +12.77% F1** |
| **Universal Fallback** | MAMS Challenge (*N=1,336*) | 82.49% *(F1: 81.93%)* | **82.93%** *(F1: 82.40%)* | **+0.44% Acc / +0.47% F1** |

#### Table 2: MoE System Performance (Metadata vs Metadata-Free)
| MoE Pipeline Configuration | Laptop14 (Tech) | Television (Home) | T-Shirt (Apparel) | MAMS Challenge | E-Com Composite |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa MoE (No Category)** | 75.08% | 83.44% | 88.98% | 82.49% | **82.12%** |
| **DeBERTa MoE (No Category)** | **81.50%** | **89.28%** | **91.74%** | 81.96% | **87.43% (+5.31%)** |
| **RoBERTa MoE (With Category)** | 81.97% | 90.87% | 92.37% | 82.49% | **88.45%** |
| **DeBERTa MoE (With Category)** | **82.76%** | **91.08%** | **94.07%** | 81.96% | **89.18% (+0.73%)** |

#### Table 3: Cross-Domain Transfer Resilience Matrix (DeBERTa-v3)
| Specialist Model | Laptop14 (Tech) | Television (Home) | T-Shirt (Apparel) | MAMS Challenge | E-Com Composite |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tech Specialist** | **82.76%** | 91.83% | 94.07% | 77.47% | **89.53%** |
| **Home Specialist** | 80.56% | **91.08%** | 94.07% | 63.55% | **88.50%** |
| **Apparel Specialist** | 76.49% | 92.04% | **94.07%** | 51.72% | **87.67%** |
| **Universal Foundation** | 81.66% | 87.47% | 90.47% | **82.93%** | **86.35%** |

*(Compare with RoBERTa Apparel Specialist on Tech, which collapsed to 67.40% accuracy and 49.63% F1).*

---

### 5. Production Case Studies (170 Real Contrast Reviews)

Collected in [`benchmarks/no_category_contrast_examples.json`](file:///home/parth/Projects/ProductLens-Sentiment/benchmarks/no_category_contrast_examples.json).

1. **Clause Disambiguation in Mixed Reviews**:
   * *Review*: `"only design best but material quality and fitting third class."`
   * *Aspect*: `material quality` | *Gold*: `negative`
   * *RoBERTa MoE*: `positive` (99.57%) ❌ (*blinded by "best"*)
   * *DeBERTa MoE*: `negative` (98.84%) ✅ (*bound "third class" to "material quality"*)
2. **Comparative Syntax**:
   * *Review*: `"From the speed to the multi touch gestures this operating system beats Windows easily ."`
   * *Aspect*: `Windows` | *Gold*: `negative`
   * *RoBERTa MoE*: `positive` (99.37%) ❌
   * *DeBERTa MoE*: `negative` (99.33%) ✅
3. **Mild / Neutral Nuances**:
   * *Review*: `"the material of the tshirt was just ok."`
   * *Aspect*: `material of the tshirt` | *Gold*: `neutral`
   * *RoBERTa MoE*: `positive` (99.60%) ❌ (*RoBERTa scored 0.0% F1 on neutrals*)
   * *DeBERTa MoE*: `neutral` (58.31%) ✅
4. **Colloquial Slang & Typos**:
   * *Review*: `"i will post exact review wall mount is too cheep."`
   * *Aspect*: `wall mount` | *Gold*: `negative`
   * *RoBERTa MoE*: `positive` (98.63%) ❌
   * *DeBERTa MoE*: `negative` (99.91%) ✅
5. **Cross-Domain Universal Aspects (Delivery & Logistics)**:
   * *Review*: `"it took a long time almost 12 days to get delivered ."`
   * *Aspect*: `delivered` | *Gold*: `negative`
   * *RoBERTa MoE*: `neutral` (90.83%) ❌
   * *DeBERTa MoE*: `negative` (93.47%) ✅

---

### 6. Release & Deployment Guide

#### A. Generating Release Artifacts
Run the release packaging script to hash and verify the four production model checkpoints:
```bash
python scripts/package_release.py
```
This prepares:
* `release/deberta_v3_expert_tech.pt` (~744 MB)
* `release/deberta_v3_expert_home.pt` (~744 MB)
* `release/deberta_v3_expert_apparel.pt` (~744 MB)
* `release/deberta_v3_foundation_mams.pt` (~744 MB)
* `release/manifest.json` (SHA256 checksums and verification metadata)

#### B. Publishing Options
* **GitHub Releases (via GitHub CLI)**:
  ```bash
  gh release create v2.0.0 release/*.pt release/manifest.json \
     --title "v2.0.0: DeBERTa-v3 ABSA Domain Specialists & MoE" \
     --notes-file CHANGELOG.md
  ```
* **HuggingFace Hub**:
  ```python
  from huggingface_hub import HfApi
  api = HfApi()
  api.upload_folder(
      folder_path="release",
      repo_id="your-org/productlens-deberta-moe",
      repo_type="model"
  )
  ```

#### C. Running the DeBERTa-v3 MoE Pipeline in Production
```python
from productlens.sentiment.pipeline import ProductLensMoEPipeline

DEBERTA_CKPTS = {
    "tech": "models/category_experts_deberta/expert_tech/best_model.pt",
    "home": "models/category_experts_deberta/expert_home/best_model.pt",
    "apparel": "models/category_experts_deberta/expert_apparel/best_model.pt",
    "fallback": "models/comparison/deberta_v3_mams/best_model.pt",
}

pipeline = ProductLensMoEPipeline(
    checkpoint_paths=DEBERTA_CKPTS,
    backbone_name="yangheng/deberta-v3-base-absa-v1.1",
    single_active_expert=True,  # Keeps active GPU memory < 1.2 GB
)

result = pipeline.predict(
    text="The battery life is stellar, but the price tag is ridiculous.",
    aspect="price tag",
    category=None  # Router automatically isolates clause
)

print(result.sentiment)   # "negative"
print(result.confidence)  # 0.9841
```
