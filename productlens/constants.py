"""
Shared constants for ProductLens-Sentiment.

Centralised here so that training (dataset.py) and inference (pipeline.py)
always use the same values, preventing silent train/inference mismatches.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Maximum number of subword tokens for text encoding.
# Used in both AspectSentimentDataset (training) and ProductLensMoEPipeline (inference).
MAX_SEQ_LEN: int = 128

# Canonical pretrained backbone for aspect-targeted sentiment analysis
DEFAULT_BACKBONE: str = "yangheng/deberta-v3-base-absa-v1.1"

# Default paths to canonical DeBERTa-v3 domain specialists and foundation fallback
DEFAULT_CHECKPOINT_PATHS: dict[str, Path] = {
    "tech": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "home": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "apparel": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    "fallback": BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
}

# Canonical label ↔ id mappings shared by training (dataset.py) and inference (pipeline.py).
# Defined once here to prevent silent train/inference label-order mismatches.
LABEL2ID: dict[str, int] = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL: dict[int, str] = {0: "positive", 1: "neutral", 2: "negative"}
