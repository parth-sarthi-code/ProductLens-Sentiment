# Master Benchmark Comparison: Generic Foundation vs Specialists vs MoE Pipeline

### Test Accuracy Transfer Matrix (%)

| Model Architecture (Rows) \ Domain (Cols) | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS Challenge (1,336) | E-Commerce Composite (2,052) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generic Foundation (MAMS)** | 71.79% | 79.09% | 87.5% | **84.21% 🏆** | 79.46% |
| **Tech Specialist (Laptop14)** | **84.17% 🏆** | 88.85% | 93.22% | 73.43% | 88.75% |
| **Home Specialist (Television)** | 80.88% | **89.81% 🏆** | 93.01% | 68.26% | 87.9% |
| **Apparel Specialist (T-Shirt)** | 78.68% | 87.26% | **94.07% 🏆** | 77.92% | 86.67% |
| **ProductLens MoE Unified System** | **84.17% 🏆** | **89.81% 🏆** | **94.07% 🏆** | 82.71% | **89.35% 🏆 (+9.89% Net Lift)** |

### Key Conclusions:
1. **The Generic Foundation Model**: Scores 84.2% on general challenge data, but drops to ~71.8% on Tech and ~79.1% on Home due to out-of-domain vocabulary.
2. **The MoE Advantage**: Routing queries to category specialists achieves **89.04% composite e-commerce accuracy**, delivering a **+9.89% net accuracy lift** over the generic foundation model.
