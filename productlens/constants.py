"""
Shared constants for ProductLens-Sentiment.

Centralised here so that training (dataset.py) and inference (pipeline.py)
always use the same values, preventing silent train/inference mismatches.
"""

from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

# Maximum number of subword tokens for text encoding.
# Used in both AspectSentimentDataset (training) and ProductLensSentimentPipeline (inference).
MAX_SEQ_LEN: int = 128

# Canonical pretrained backbone for aspect-targeted sentiment analysis
DEFAULT_BACKBONE: str = "yangheng/deberta-v3-base-absa-v1.1"

# Default path to the unified calibrated DeBERTa-v3 checkpoint
DEFAULT_UNIFIED_CHECKPOINT_PATH: Path = BASE_DIR / "models/unified_backbone/best_model.pt"

# Legacy checkpoint paths mapping (retained for backward-compatible script references)
DEFAULT_CHECKPOINT_PATHS: dict[str, Optional[Path]] = {
    "unified": DEFAULT_UNIFIED_CHECKPOINT_PATH,
    "fallback": None,
}

# Canonical label <-> id mappings shared by training (dataset.py) and inference (pipeline.py).
# Aligned directly with pretrained DeBERTa-v3 ABSA head: 0 -> negative, 1 -> neutral, 2 -> positive.
LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
