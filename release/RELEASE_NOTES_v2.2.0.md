## ProductLens v2.2.0: Unified Calibrated DeBERTa-v3 ABSA Backbone

Production release replacing the multi-checkpoint Mixture of Experts (MoE) pipeline with a **single unified calibrated DeBERTa-v3 backbone**.

### Highlights & Key Improvements
- **Unified Single-Checkpoint Architecture**:
  - Replaces 3 disparate specialist checkpoints with a single high-throughput model instance (`deberta_v3_unified_backbone.pt`, 712 MB).
  - Eliminates routing errors, keyword taxonomy overhead, and out-of-domain router fallbacks.
  - Achieves **89.37% Composite Accuracy** across all held-out e-commerce domains, surpassing MoE on Tech (**83.54% vs 83.23%**) and Neutral-F1 (**66.90% vs 64.52%**).
  - Reduces disk and VRAM footprint by **67%** (712 MB vs 2.2 GB).
- **Aspect-Targeted Query-Key Attention Pooling**:
  - Upgraded to true Query-Key evidence gathering ($Q_{\text{aspect}} \to K_{\text{review}}$) with Tripartite Context Fusion ($[h_{\text{cls}} + h_{\text{aspect}} + h_{\text{opinion}}] / 3$).
- **Neutral-Aware Ordinal Wasserstein Loss & Label Smoothing**:
  - Penalizes severe polarity misclassifications ($+ \leftrightarrow -$) twice as harshly as adjacent Neutral confusion.
  - Label smoothing ($\epsilon = 0.05$) curbs overconfidence on subjective reviews.
- **Post-Hoc Temperature Calibration**:
  - Fitted optimal temperature parameter ($T = 0.8999$), reducing NLL by -30.4% and achieving 3.85% ECE on Apparel.
- **2.71x Faster Multi-Aspect Batched Inference**:
  - Single-pass dynamic windowing tokenization processes all review aspects simultaneously without redundant passes.

### Release Assets
- `deberta_v3_unified_backbone.pt` (711.88 MB) — SHA256: `0e0a5d5376a2f826abe2e2edf7acc63712e00b8869b12abd25be7a3a25f92095`
- `calibration.json` — Temperature scaling parameters
- `manifest.json` — Release package metadata and checksums
