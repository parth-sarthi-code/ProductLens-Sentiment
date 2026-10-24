# Head-to-Head Benchmark: RoBERTa-base Specialists vs DeBERTa-v3 Specialists

> Comparing domain-specialist fine-tuning between RoBERTa-base and DeBERTa-v3 ABSA across held-out test benchmarks.
> All specialist models use ProductLens Aspect-Targeted Attention Pooling (Boost=3.5) + Supervised Contrastive Loss (SCL).
> Composite = sample-weighted test accuracy across held-out e-commerce domains (2,052 samples).

## 1. Direct Specialist In-Domain Head-to-Head

| Domain Vertical | Held-Out Test Set | RoBERTa Specialist | DeBERTa-v3 Specialist | Delta (DeBERTa vs RoBERTa) |
| :--- | :--- | :---: | :---: | :---: |
| **Tech Hardware** | Laptop14 (638) | 81.97% | **82.76%** | **+0.79%** |
| **Home Electronics** | Television (942) | 90.76% | **91.08%** | **+0.32%** |
| **Apparel & Fashion** | T-Shirt (472) | 92.37% | **94.07%** | **+1.7%** |
| **Universal Fallback** | MAMS Challenge (1336) | 83.76% | **82.93%** | **-0.83%** |

## 2. MoE Pipeline System Comparison

| MoE System Configuration | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RoBERTa MoE (No Category)** | 75.08% | 83.44% | 88.98% | 82.49% | **82.12%** |
| **DeBERTa MoE (No Category)** | 81.5% | 89.28% | 91.74% | 81.96% | **87.43%** |
| **RoBERTa MoE (With Category)** | 81.97% | 90.87% | 92.37% | 82.49% | **88.45%** |
| **DeBERTa MoE (With Category)** | 82.76% | 91.08% | 94.07% | 81.96% | **89.18%** |

## 3. Full Transfer Matrix: DeBERTa-v3 Specialists

| Specialist Model | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS (1,336) | E-Com Composite |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generic Foundation (MAMS)** | 81.66% | 87.47% | 90.47% | 82.93% | 86.35% |
| **Tech Specialist (Laptop14)** | 82.76% | 91.83% | 94.07% | 77.47% | 89.53% |
| **Home Specialist (Television)** | 80.56% | 91.08% | 94.07% | 63.55% | 88.5% |
| **Apparel Specialist (T-Shirt)** | 76.49% | 92.04% | 94.07% | 51.72% | 87.67% |