"""
Unified Aspect-Based Sentiment Analysis (ABSA) Pipeline.
Evaluates customer reviews using a calibrated, aspect-targeted DeBERTa-v3 architecture
with aspect-to-context cross-attention pooling and aspect-centered windowing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from productlens.constants import (
    DEFAULT_BACKBONE,
    DEFAULT_UNIFIED_CHECKPOINT_PATH,
    ID2LABEL,
    LABEL2ID,
    MAX_SEQ_LEN,
)
from productlens.schemas import SentimentPrediction
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.utils import find_aspect_span

logger = logging.getLogger("productlens.sentiment")


def checkpoints_exist(
    checkpoint_paths: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
) -> bool:
    """Returns True if the unified checkpoint exists on disk or fallback is available."""
    if checkpoint_paths:
        return any(p is not None and Path(p).exists() for p in checkpoint_paths.values())
    return DEFAULT_UNIFIED_CHECKPOINT_PATH.exists()


class ProductLensSentimentPipeline:
    """
    Production-grade Aspect-Based Sentiment Analysis Pipeline.
    Loads a single unified calibrated DeBERTa-v3 model with aspect-targeted
    cross-attention pooling, delivering high-throughput inference without
    multi-model routing overhead.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path, dict]] = None,
        backbone_name: str = DEFAULT_BACKBONE,
        device: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,  # Gracefully accepts legacy MoE args like lazy_load, single_active_expert
    ):
        self.backbone_name = backbone_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # Resolve checkpoint path (handles legacy dictionary or direct Path)
        self.checkpoint_path: Optional[Path] = None
        if isinstance(checkpoint_path, dict):
            if "unified" in checkpoint_path and checkpoint_path["unified"]:
                self.checkpoint_path = Path(checkpoint_path["unified"])
            else:
                for k in ["tech", "fallback", "home", "apparel"]:
                    if k in checkpoint_path and checkpoint_path[k]:
                        self.checkpoint_path = Path(checkpoint_path[k])
                        break
        elif checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        elif DEFAULT_UNIFIED_CHECKPOINT_PATH.exists():
            self.checkpoint_path = DEFAULT_UNIFIED_CHECKPOINT_PATH

        # Initialize single unified aspect-targeted model
        self.model = AspectTargetedSentimentModel(
            backbone_name=self.backbone_name,
            num_labels=3,
            aspect_boost=3.5,
        )

        if self.checkpoint_path and self.checkpoint_path.exists():
            logger.info("Loading unified sentiment model weights from %s", self.checkpoint_path)
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            logger.info("Using canonical pretrained backbone %s directly", self.backbone_name)

        self.model.to(self.device)
        self.model.eval()

        # Load temperature calibration
        self.temperature: float = 1.0
        if temperature is not None:
            self.temperature = float(temperature)
        elif self.checkpoint_path:
            calib_file = self.checkpoint_path.parent / "calibration.json"
            if calib_file.exists():
                try:
                    with open(calib_file, "r") as f:
                        calib_data = json.load(f)
                    self.temperature = float(calib_data.get("temperature", 1.0))
                    logger.info("Loaded calibration temperature T=%.4f from %s", self.temperature, calib_file)
                except Exception as e:
                    logger.warning("Failed to load calibration from %s: %s", calib_file, e)

        logger.info(
            "Initialized ProductLens Sentiment Pipeline on device=%s (backbone=%s, calibrated=%s)",
            self.device,
            self.backbone_name,
            self.temperature != 1.0,
        )

    @property
    def experts(self) -> Dict[str, AspectTargetedSentimentModel]:
        """Backward-compatibility mapping for pipeline model inspection."""
        return {"unified": self.model}

    @property
    def checkpoint_paths(self) -> Dict[str, Optional[Path]]:
        """Backward-compatibility mapping for pipeline checkpoints."""
        return {"unified": self.checkpoint_path}

    def predict(
        self,
        text: str,
        aspect: str,
        category: Optional[str] = None,
        force_expert: Optional[str] = None,
        aspect_start: Optional[int] = None,
        aspect_end: Optional[int] = None,
    ) -> SentimentPrediction:
        """
        Runs aspect-targeted sentiment inference on a single (text, aspect) pair.
        """
        text = text.strip()
        aspect = aspect.strip()

        # Aspect span resolution
        fell_back_aspect_mask = False
        if aspect_start is not None and aspect_end is not None:
            s_char, e_char = aspect_start, aspect_end
        else:
            s_char, e_char = find_aspect_span(text, aspect)
            if aspect.lower() not in text.lower():
                fell_back_aspect_mask = True

        # Aspect-centered context windowing for long reviews
        proc_text = text
        max_char_len = MAX_SEQ_LEN * 4
        if len(text) > max_char_len and not fell_back_aspect_mask and s_char >= 0 and e_char > s_char:
            center = (s_char + e_char) // 2
            half = max_char_len // 2
            w_start = max(0, center - half)
            w_end = min(len(text), w_start + max_char_len)
            if w_end == len(text):
                w_start = max(0, w_end - max_char_len)
            proc_text = text[w_start:w_end]
            s_char -= w_start
            e_char -= w_start

        encoding = self.tokenizer(
            proc_text,
            aspect,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
        seq_ids = encoding.sequence_ids(0) if hasattr(encoding, "sequence_ids") else None

        aspect_mask = torch.zeros_like(attention_mask[0], dtype=torch.float32)

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            is_text_seq = (seq_ids[i] == 0) if seq_ids is not None else True
            if is_text_seq and not fell_back_aspect_mask:
                if not (tok_end <= s_char or tok_start >= e_char):
                    aspect_mask[i] = 1.0

        if aspect_mask.sum() == 0:
            aspect_mask[0] = 1.0
            fell_back_aspect_mask = True

        aspect_mask = aspect_mask.unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask, aspect_mask)
            scaled_logits = logits / self.temperature
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])

        prob_dict = {
            "positive": float(probs[LABEL2ID["positive"]]),
            "neutral": float(probs[LABEL2ID["neutral"]]),
            "negative": float(probs[LABEL2ID["negative"]]),
        }

        return SentimentPrediction(
            sentiment=pred_label,
            confidence=confidence,
            probabilities=prob_dict,
            routed_expert="unified",
            routing_reason="unified_calibrated_backbone",
            aspect_mask_fallback=fell_back_aspect_mask,
            calibrated=self.temperature != 1.0,
        )

    def predict_aspects(
        self,
        text: str,
        aspects: List[Union[str, Tuple[str, Optional[int], Optional[int]]]],
        category: Optional[str] = None,
        force_expert: Optional[str] = None,
    ) -> List[SentimentPrediction]:
        """
        High-throughput multi-aspect inference for a review.
        Performs batched tensor forward pass across all aspects in the review.
        """
        if not aspects:
            return []

        # Normalize aspects into tuples: (aspect_text, s_char, e_char)
        normalized_aspects: List[Tuple[str, int, int]] = []
        for item in aspects:
            if isinstance(item, tuple):
                asp_text = item[0]
                s = item[1] if len(item) > 1 and item[1] is not None else None
                e = item[2] if len(item) > 2 and item[2] is not None else None
            else:
                asp_text = str(item)
                s, e = None, None

            if s is None or e is None:
                s, e = find_aspect_span(text, asp_text)
            normalized_aspects.append((asp_text, s, e))

        # Build batched tensors
        batch_input_ids = []
        batch_attention_mask = []
        batch_aspect_mask = []
        fallback_flags = []

        max_char_len = MAX_SEQ_LEN * 4

        for asp_text, s_char, e_char in normalized_aspects:
            fell_back = (asp_text.lower() not in text.lower()) or (s_char == -1 or s_char is None or e_char is None)
            proc_text = text

            if len(text) > max_char_len and not fell_back and s_char >= 0 and e_char > s_char:
                center = (s_char + e_char) // 2
                half = max_char_len // 2
                w_start = max(0, center - half)
                w_end = min(len(text), w_start + max_char_len)
                if w_end == len(text):
                    w_start = max(0, w_end - max_char_len)
                proc_text = text[w_start:w_end]
                s_char -= w_start
                e_char -= w_start

            encoding = self.tokenizer(
                proc_text,
                asp_text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
            offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
            seq_ids = encoding.sequence_ids(0) if hasattr(encoding, "sequence_ids") else None

            asp_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end:
                    continue
                is_text_seq = (seq_ids[i] == 0) if seq_ids is not None else True
                if is_text_seq and not fell_back:
                    if not (tok_end <= s_char or tok_start >= e_char):
                        asp_mask[i] = 1.0

            if asp_mask.sum() == 0:
                asp_mask[0] = 1.0
                fell_back = True

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_aspect_mask.append(asp_mask)
            fallback_flags.append(fell_back)

        tensor_input_ids = torch.stack(batch_input_ids).to(self.device)
        tensor_attention_mask = torch.stack(batch_attention_mask).to(self.device)
        tensor_aspect_mask = torch.stack(batch_aspect_mask).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(tensor_input_ids, tensor_attention_mask, tensor_aspect_mask)
            scaled_logits = logits / self.temperature
            probs_matrix = F.softmax(scaled_logits, dim=-1).cpu().numpy()

        results: List[SentimentPrediction] = []
        for i, probs in enumerate(probs_matrix):
            pred_id = int(np.argmax(probs))
            pred_label = ID2LABEL[pred_id]
            confidence = float(probs[pred_id])
            prob_dict = {
                "positive": float(probs[LABEL2ID["positive"]]),
                "neutral": float(probs[LABEL2ID["neutral"]]),
                "negative": float(probs[LABEL2ID["negative"]]),
            }

            results.append(
                SentimentPrediction(
                    sentiment=pred_label,
                    confidence=confidence,
                    probabilities=prob_dict,
                    routed_expert="unified",
                    routing_reason="unified_calibrated_backbone",
                    aspect_mask_fallback=fallback_flags[i],
                    calibrated=self.temperature != 1.0,
                )
            )

        return results


# Backward compatibility alias
ProductLensMoEPipeline = ProductLensSentimentPipeline
