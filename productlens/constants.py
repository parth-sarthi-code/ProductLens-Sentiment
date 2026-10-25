"""
Shared constants for ProductLens-Sentiment.

Centralised here so that training (dataset.py) and inference (pipeline.py)
always use the same values, preventing silent train/inference mismatches.
"""

from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

# Maximum number of subword tokens for text encoding.
# Used in both AspectSentimentDataset (training) and ProductLensMoEPipeline (inference).
MAX_SEQ_LEN: int = 128

# Canonical pretrained backbone for aspect-targeted sentiment analysis
DEFAULT_BACKBONE: str = "yangheng/deberta-v3-base-absa-v1.1"

# Default paths to canonical DeBERTa-v3 domain specialists and fallback
# Fallback uses the canonical pretrained backbone directly (None = pure pretrained model)
DEFAULT_CHECKPOINT_PATHS: dict[str, Optional[Path]] = {
    "tech": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "home": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "apparel": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    "fallback": None,
}

# Canonical label ↔ id mappings shared by training (dataset.py) and inference (pipeline.py).
# Aligned directly with pretrained DeBERTa-v3 ABSA head: 0 -> negative, 1 -> neutral, 2 -> positive.
LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
