# DeBERTa-v3 ABSA Benchmark: MoE vs No-MoE vs Streamlining (F1 & Accuracy)

> Evaluated on standardized, held-out test datasets across four domain verticals.  
> All models use DeBERTa-v3 with Sentence-Pair Cross-Attention (`tokenizer(text, aspect)`).  
> Streamlined Specialists use **WiSE-FT** ($\alpha=0.8$) and zero-initialized auxiliary residual poolers.  
> Confidence intervals are 95% bootstrap ($n=1,000$).

---

## 1. Macro-F1 Head-to-Head Transfer Matrix (%)

| Model Configuration | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS Challenge (1,336) | E-Com Composite (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No MoE (Generic Foundation)** | 76.80% [73.0–80.3] | 77.89% [73.6–81.9] | 76.00% [66.7–84.6] | 82.99% [80.9–85.0] | **89.33% Acc** |
| **Streamlined Tech Specialist** | **78.77% [75.4–82.0]** | 76.88% [72.4–81.1] | 72.82% [64.0–81.6] | **83.12% [81.0–85.2]** | 89.52% Acc |
| **Streamlined Home Specialist** | 79.08% [75.6–82.5] | **79.35% [75.2–83.4]** | 73.19% [64.0–82.1] | 82.35% [80.1–84.5] | 89.76% Acc |
| **Streamlined Apparel Specialist** | 76.80% [73.0–80.3] | 77.89% [73.6–81.9] | **76.00% [66.7–84.6]** | 82.99% [80.9–85.0] | 89.33% Acc |
| **ProductLens MoE (No Category)** | 77.06% [73.3–80.5] | 78.88% [74.7–82.9] | 76.00% [66.7–84.6] | 82.99% [80.9–85.0] | **89.47% Acc** |
| **ProductLens MoE (With Category)** | **78.77% [75.4–82.0]** | **79.35% [75.2–83.4]** | **76.00% [66.7–84.6]** | **82.99% [80.9–85.0]** | **89.81% Acc 🏆** |

---

## 2. Per-Class F1 Breakdown

The primary performance driver from streamlining and MoE routing is the resolution of **neutral sentiment** ambiguity:

### Tech Hardware (`Laptop14`, 638 samples)
| Model Configuration | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No MoE (Generic Foundation)** | 81.82% | 78.83% | 60.00% | 91.56% | 76.80% |
| **Streamlined Tech Specialist** | **82.92%** | **79.87%** | **64.79% (+4.79%)** | **91.64%** | **78.77% (+1.97%)** |
| **ProductLens MoE (No Category)** | 81.97% | 78.69% | 60.81% | 91.69% | 77.06% |
| **ProductLens MoE (With Category)** | **82.92%** | **79.87%** | **64.79%** | 91.64% | **78.77%** |

### Home Electronics (`Television`, 942 samples)
| Model Configuration | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No MoE (Generic Foundation)** | 91.83% | 93.69% | 44.26% | 95.72% | 77.89% |
| **Streamlined Home Specialist** | **92.14%** | **93.87%** | **48.39% (+4.13%)** | **95.80%** | **79.35% (+1.46%)** |
| **ProductLens MoE (No Category)** | 92.04% | 93.69% | 47.15% | 95.80% | 78.88% |
| **ProductLens MoE (With Category)** | **92.14%** | **93.87%** | **48.39%** | **95.80%** | **79.35%** |

### Apparel & Fashion (`T-Shirt`, 472 samples)
| Model Configuration | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No MoE (Generic Foundation)** | 94.49% | 95.47% | 36.36% | 96.16% | 76.00% |
| **Streamlined Apparel Specialist** | 94.49% | 95.47% | 36.36% | 96.16% | 76.00% |
| **ProductLens MoE (With Category)** | 94.49% | 95.47% | 36.36% | 96.16% | 76.00% |

### Universal Challenge (`MAMS`, 1,336 samples)
| Model Configuration | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generic Foundation (Fallback)** | 83.38% | 81.63% | 85.30% | 82.04% | 82.99% |
| **Tech Specialist (Cross-Domain)** | 83.53% | 81.06% | 85.33% | 82.96% | 83.12% |
| **Home Specialist (Cross-Domain)** | 82.86% | 80.24% | 85.05% | 81.76% | 82.35% |

---

## 3. Streamlining Impact: Curing Catastrophic Forgetting

Before streamlining, standard domain fine-tuning caused severe representation drift and forgetting on universal ABSA tasks. With **WiSE-FT** ($\theta_{\text{wise}} = 0.2\,\theta_{\text{base}} + 0.8\,\theta_{\text{expert}}$) and an auxiliary residual head initialized at zero:

| Specialist Model | Pre-Streamlining MAMS Macro-F1 | Streamlined MAMS Macro-F1 (WiSE-FT) | Recovery / Lift |
| :--- | :---: | :---: | :---: |
| **Tech Specialist** | 77.47% | **83.12%** | **+5.65%** |
| **Home Specialist** | 63.55% *(Collapse)* | **82.35%** | **+18.80%** |
| **Apparel Specialist** | 51.72% *(Collapse)* | **82.99%** | **+31.27%** |
| **Generic Foundation Baseline** | 82.93% | **82.99%** | Baseline |

---

## 4. Key Takeaways

1. **No MoE (Monolithic Model)**: Provides an acceptable generalist baseline (89.33% composite accuracy), but underperforms by ~2% Macro-F1 in domain verticals with distinct vocabularies and subtle neutral sentiment (e.g. `Laptop14`, `Television`).
2. **Streamlined Specialists**: Fine-tuning with sentence-pair cross-attention (`tokenizer(text, aspect)`) and WiSE-FT regularizes parameters in weight space. This eliminates catastrophic forgetting while unlocking in-domain gains (+4.79% neutral F1 on Tech, +4.13% neutral F1 on Home).
3. **ProductLens MoE**: Achieves the highest overall composite performance (**89.81%** accuracy, **78.77%** Tech Macro-F1, **79.35%** Home Macro-F1) by routing queries to the domain expert without modifying backbone latency.
