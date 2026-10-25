# Head-to-Head Benchmark: RoBERTa-base Specialists vs DeBERTa-v3 Specialists

> Comparing domain-specialist fine-tuning between RoBERTa-base and DeBERTa-v3 ABSA across held-out test benchmarks.
> All specialist models use ProductLens Aspect-Targeted Attention Pooling (Boost=3.5) + Supervised Contrastive Loss (SCL).
> Composite = sample-weighted test accuracy across held-out e-commerce domains (2,052 samples).

## 1. Direct Specialist In-Domain Head-to-Head

| Domain Vertical | Held-Out Test Set | RoBERTa Specialist | DeBERTa-v3 Specialist (WiSE-FT) | Delta (DeBERTa vs RoBERTa) |
| :--- | :--- | :---: | :---: | :---: |
| **Tech Hardware** | Laptop14 (638) | 81.97% | **82.92%** | **+0.95%** |
| **Home Electronics** | Television (942) | 90.76% | **92.14%** | **+1.38%** |
| **Apparel & Fashion** | T-Shirt (472) | 92.37% | **94.49%** | **+2.12%** |
| **Universal Fallback** | MAMS Challenge (1336) | 83.76% | **83.38%** | **-0.38%** |

## 2. MoE Pipeline System Comparison

| MoE System Configuration | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa MoE (No Category)** | 75.08% | 83.44% | 88.98% | 82.49% | **82.12%** |
| **DeBERTa MoE (No Category)** | 81.97% | 92.04% | 94.49% | 83.38% | **89.47% (+7.35%)** |
| **RoBERTa MoE (With Category)** | 81.97% | 90.87% | 92.37% | 82.49% | **88.45%** |
| **DeBERTa MoE (With Category)** | 82.92% | 92.14% | 94.49% | 83.38% | **89.81% (+1.36%)** |

## 3. Full Transfer Matrix: DeBERTa-v3 Specialists (Zero-Forgetting)

| Specialist Model | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generic Foundation (MAMS)** | 81.82% | 91.83% | 94.49% | 83.38% | 89.33% |
| **Tech Specialist (Laptop14)** | 82.92% | 91.72% | 94.07% | 83.53% | 89.52% |
| **Home Specialist (Television)** | 83.07% | 92.14% | 94.07% | 82.86% | **89.76%** |
| **Apparel Specialist (T-Shirt)** | 81.82% | 91.83% | 94.49% | 83.38% | 89.33% |
| **ProductLens MoE (with category)** | **82.92%** | **92.14%** | **94.49%** | **83.38%** | **89.81%** |