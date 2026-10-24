# ProductLens Category Experts & Foundation Model Checkpoints

This directory houses the trained PyTorch state dictionaries for the **Hierarchical Mixture of Experts (MoE)** suite.

### Directory Structure
```
models/category_experts/
├── expert_tech/
│   └── best_model.pt          # Tech Hardware Specialist (Laptop14: Val Macro F1: 0.9527, 710 MB)
├── expert_home/
│   └── best_model.pt          # Home Electronics Specialist (Television: Val Macro F1: 0.8911, 710 MB)
├── expert_apparel/
│   └── best_model.pt          # Apparel & Fashion Specialist (T-Shirt: Val Macro F1: 0.8715, 710 MB)
└── foundation_mams/
    └── best_model.pt          # Universal MAMS Foundation Fallback (Challenge: Val Macro F1: 0.8240, 710 MB)
```

### Model Architecture
* **Backbone Encoder**: `yangheng/deberta-v3-base-absa-v1.1` (DeBERTa-v3 ABSA pre-trained warm start, 768 hidden units)
* **Pooling**: `AspectTargetedAttentionPooler` (Aspect Boost $\beta = 3.5$)
* **Head**: 128-dim Projection Head + 2-layer GELU Classification MLP (3 sentiment classes: `positive`, `neutral`, `negative`)
* **Training Objective**: Joint Cross-Entropy + Supervised Contrastive Loss (SCL, $\tau=0.07$, $\lambda_{\text{SCL}}=0.10$)

### Retraining
To train any of these models from scratch:
```bash
python scripts/train.py --domain generic    # Trains foundation_mams
python scripts/train.py --domain tech       # Trains expert_tech
python scripts/train.py --domain home       # Trains expert_home
python scripts/train.py --domain apparel    # Trains expert_apparel
```
