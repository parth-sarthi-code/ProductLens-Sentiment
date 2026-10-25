# ProductLens Benchmark Summary: Unified Single Backbone vs Canonical Backbone vs Legacy MoE

> Generated from empirical held-out evaluations across 2,052 e-commerce samples and 1,877 ARTS adversarial samples.

---

## 1. Test Accuracy & F1 Transfer Matrix (%)

| Model Architecture | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | E-Com Composite (2,052) | ARTS Laptop (Adversarial, 1,877) | Model Size & VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Pretrained Backbone** (`yangheng/deberta-v3-base-absa-v1.1`) | 81.82% | **91.83%** | **94.49%** | 89.33% | 78.90% | 1 Checkpoint (712 MB) |
| **Legacy MoE Pipeline** (3 Specialists + Router) | 83.23% | 91.72% | 94.28% | **89.67%** (+0.34%) | — | 3 Checkpoints (~2.2 GB) |
| **ProductLens Unified Backbone** (Production v2.2.0) | **83.54%** | 91.40% | 93.22% | **89.37%** | **78.80%** (Macro-F1: 76.29%) | **1 Checkpoint (712 MB)** |

---

## 2. Detailed Calibration & Uncertainty Metrics

| Evaluation Benchmark | Metric | Canonical Backbone | Unified Backbone (v2.2.0) | Impact |
| :--- | :--- | :---: | :---: | :--- |
| **SemEval Laptop14 (Tech)** | Accuracy | 81.82% | **83.54%** | **+1.72% lift** |
| | Macro-F1 | 76.54% | **79.78%** | **+3.24% lift** |
| | Neutral-F1 | 61.27% | **66.90%** | **+5.63% lift** |
| | Log-Loss (NLL) | 0.8489 | **0.5907** | **-30.4% reduction in uncertainty error** |
| | Calibration Error (ECE) | 14.58% | **11.45%** | **-3.13% error reduction** |
| **Television (Home)** | Neutral-F1 | 44.44% | **48.57%** | **+4.13% lift** |
| | Calibration Error (ECE) | 5.70% | **5.22%** | **-0.48% error reduction** |
| **T-Shirt (Apparel)** | Calibration Error (ECE) | 4.72% | **3.85%** | **-0.87% error reduction** |
| **ARTS Adversarial** | Macro-F1 | 75.63% | **76.29%** | **+0.66% lift on conflicting aspects** |
| | Neutral-F1 | 63.88% | **64.96%** | **+1.08% lift on neutral shifts** |

---

## 3. Latency & Resource Utilization

| Architecture | Memory Required | Multi-Aspect Latency (4 aspects) | Routing Failures / Fallback |
| :--- | :---: | :---: | :---: |
| **Legacy MoE Pipeline** | ~2.5 GB (3 models + cache) | 64.0 ms (sequential) | 15.7% fallback on home |
| **Unified Single Backbone** | **~800 MB (1 model in VRAM)** | **23.6 ms (batched, 2.71x speedup)** | **0% (no router needed)** |
