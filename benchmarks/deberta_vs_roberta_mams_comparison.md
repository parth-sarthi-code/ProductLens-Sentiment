# DeBERTa-v3 ABSA Warm-Start vs RoBERTa-base Benchmark Comparison

> Model: `yangheng/deberta-v3-base-absa-v1.1` fine-tuned and evaluated on MAMS dataset with strict leakage precautions.
> CI = 95% bootstrap confidence interval (n=1000).

## 1. MAMS Challenge Benchmark (Target Domain)

| Model Variant | MAMS Test Accuracy (%) | 95% CI | Macro F1 (%) | Pos F1 (%) | Neu F1 (%) | Neg F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa-base Foundation (MAMS)** | **83.83%** | [81.88–85.78] | 83.21% | 83.68% | 86.12% | 79.82% |
| **DeBERTa-v3 ABSA (Zero-Shot)** | **83.38%** | [81.29–85.33] | 82.99% | 82.04% | 85.3% | 81.63% |
| **DeBERTa-v3 ABSA (Fine-Tuned MAMS)** | **82.93%** | [80.76–84.96] | 82.4% | 82.08% | 85.07% | 80.06% |

## 2. Cross-Domain Generalization (Held-Out E-Commerce Domains)

| Model Variant | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | E-Com Composite (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **RoBERTa-base Foundation (MAMS)** | 73.04% | 78.56% | 88.14% | **79.05%** |
| **DeBERTa-v3 ABSA (Zero-Shot)** | 81.82% | 91.83% | 94.49% | **89.33%** |
| **DeBERTa-v3 ABSA (Fine-Tuned MAMS)** | 81.66% | 87.47% | 90.47% | **86.35%** |

## 3. Leakage Precautions & Analysis
- **Disjoint Split Verification**: Confirmed exact sentence-level and `(text, aspect)` overlap between MAMS train and test sets is **0**.
- **Pre-Training Provenance**: `yangheng/deberta-v3-base-absa-v1.1` was trained on PyABSA's `ABSADatasets` corpus, which includes the MAMS training split. This accounts for its high out-of-the-box zero-shot accuracy.
- **Aspect Pooling & Contrastive Regularization**: Integrating ProductLens's `AspectTargetedAttentionPooler` (boost=3.5) and Supervised Contrastive Loss (SCL) enforces aspect tethering and protects against representation collapse.
