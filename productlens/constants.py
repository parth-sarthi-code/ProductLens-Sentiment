"""
Shared constants for ProductLens-Sentiment.

Centralised here so that training (dataset.py) and inference (pipeline.py)
always use the same values, preventing silent train/inference mismatches.
"""

# Maximum number of subword tokens for text encoding.
# Used in both AspectSentimentDataset (training) and ProductLensMoEPipeline (inference).
# Changing this value requires re-training all expert models.
MAX_SEQ_LEN: int = 128
