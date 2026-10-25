# ProductLens Unified Sentiment Model Checkpoints

This directory houses the unified calibrated PyTorch model weights for ProductLens.

### Directory Structure
```
models/
└── unified_backbone/
    ├── best_model.pt          # Unified Calibrated DeBERTa-v3 Model (712 MB)
    └── calibration.json       # Temperature Scaling Parameters (T = 0.8999)
```

### Model Architecture
* **Backbone Encoder**: `yangheng/deberta-v3-base-absa-v1.1` (DeBERTa-v3 ABSA pre-trained representation, 768 hidden units)
* **Pooling**: `AspectTargetedAttentionPooler` with Query-Key Cross-Attention ($Q_{\text{aspect}} \to K_{\text{review}}$) and Tripartite Context Fusion
* **Regularization**: 1D Earth Mover's Distance (`OrdinalWassersteinLoss`, $\lambda = 0.15$) and Label Smoothing ($\epsilon = 0.05$)
* **Post-Hoc Calibration**: Temperature Scaling fitted on validation set ($T = 0.8999$) minimizing NLL and ECE.

### Retraining
To train the unified model from scratch:
```bash
python scripts/prepare_unified_dataset.py
python scripts/train.py --domain unified --batch_size 16 --epochs 10 --eval_steps 150 --patience 6
```
