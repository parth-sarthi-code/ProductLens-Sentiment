from __future__ import annotations

import logging
from typing import Any, Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


class AspectSentimentDataset(Dataset):
    """
    Aspect-targeted tokenized dataset.
    Converts (review_text, aspect, aspect_span, sentiment_label) into
    input_ids, attention_mask, aspect_mask, and class label tensors.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: Any, max_length: int = 128):
        self.samples = []
        for _, row in df.iterrows():
            text = str(row["text"])
            aspect = str(row["aspect"])
            asp_start = int(row.get("aspect_start", 0))
            asp_end = int(row.get("aspect_end", len(aspect)))
            sentiment = str(row.get("sentiment", "neutral")).lower()
            label = LABEL2ID.get(sentiment, 1)

            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            offset_mapping = encoding["offset_mapping"].squeeze(0)

            # Construct binary aspect mask over subword tokens
            aspect_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
            found_aspect_token = False
            for tok_idx, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:
                    continue
                if start < asp_end and end > asp_start:
                    aspect_mask[tok_idx] = 1.0
                    found_aspect_token = True

            # Fallback subword sequence alignment if character offsets didn't intersect
            if not found_aspect_token:
                asp_enc = tokenizer(aspect, add_special_tokens=False)["input_ids"]
                if len(asp_enc) > 0:
                    seq = input_ids.tolist()
                    asp_len = len(asp_enc)
                    for idx in range(len(seq) - asp_len + 1):
                        if seq[idx : idx + asp_len] == asp_enc:
                            for m in range(asp_len):
                                aspect_mask[idx + m] = 1.0
                            found_aspect_token = True
                            break

            if not found_aspect_token:
                aspect_mask[0] = 1.0

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "aspect_mask": aspect_mask,
                "label": torch.tensor(label, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
