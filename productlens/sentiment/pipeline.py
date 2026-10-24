from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE, DEFAULT_CHECKPOINT_PATHS, MAX_SEQ_LEN
from productlens.schemas import SentimentPrediction
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.router import MoERouter

logger = logging.getLogger("productlens.sentiment")

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def checkpoints_exist(
    checkpoint_paths: Optional[Dict[str, Path]] = None,
) -> bool:
    """Returns True if at least the fallback checkpoint exists on disk."""
    paths = checkpoint_paths or DEFAULT_CHECKPOINT_PATHS
    fallback = paths.get("fallback")
    return fallback is not None and Path(fallback).exists()


class ProductLensMoEPipeline:
    """
    Unified Mixture of Experts (MoE) Inference Pipeline.
    Loads and caches domain specialists, dispatches queries via MoERouter,
    and returns aspect-conditioned sentiment predictions.
    """

    def __init__(
        self,
        checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
        backbone_name: str = DEFAULT_BACKBONE,
        device: Optional[str] = None,
        lazy_load: bool = True,
        single_active_expert: bool = False,
    ):
        self.backbone_name = backbone_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_paths = {k: Path(v) for k, v in (checkpoint_paths or DEFAULT_CHECKPOINT_PATHS).items()}
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.router = MoERouter()
        self.lazy_load = lazy_load
        self.single_active_expert = single_active_expert
        self.active_expert = None

        self.experts: Dict[str, AspectTargetedSentimentModel] = {}
        if not self.lazy_load:
            self._load_all_experts()

        logger.info("Initialized ProductLens MoE Pipeline on device=%s (backbone=%s)", self.device, self.backbone_name)

    def _get_expert(self, expert_key: str) -> Tuple[AspectTargetedSentimentModel, str]:
        """
        Loads and caches the requested domain expert.

        Returns:
            (model, resolved_key) — resolved_key is the expert that actually ran,
            which may differ from expert_key when a fallback is used.

        Raises:
            FileNotFoundError: If neither the requested checkpoint nor the fallback
                checkpoint exists on disk. Never runs with random/unloaded weights.
        """
        resolved_key = expert_key

        if resolved_key not in self.experts:
            ckpt_path = self.checkpoint_paths.get(resolved_key)

            if not ckpt_path or not ckpt_path.exists():
                logger.warning(
                    "Checkpoint for '%s' not found at %s. Attempting fallback expert.",
                    expert_key,
                    ckpt_path,
                )
                resolved_key = "fallback"
                ckpt_path = self.checkpoint_paths.get("fallback")

                if not ckpt_path or not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint for expert '{expert_key}' not found, and the fallback "
                        f"checkpoint is also missing at '{ckpt_path}'. "
                        "Download model weights before running inference. "
                        "See README.md § 'Download Weights' for instructions."
                    )

            logger.info("Loading expert '%s' from %s...", resolved_key, ckpt_path)
            model = AspectTargetedSentimentModel(backbone_name=self.backbone_name, num_labels=3, aspect_boost=3.5)
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            if self.single_active_expert:
                if self.active_expert is not None and self.active_expert in self.experts:
                    self.experts[self.active_expert].to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                model.to(self.device)
                self.active_expert = resolved_key
            else:
                model.to(self.device)

            # Cache under resolved_key so subsequent calls for the same expert
            # (or fallback) do not reload expert weights from disk.
            self.experts[resolved_key] = model
        else:
            if self.single_active_expert and self.active_expert != resolved_key:
                if self.active_expert is not None and self.active_expert in self.experts:
                    self.experts[self.active_expert].to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.experts[resolved_key].to(self.device)
                self.active_expert = resolved_key

        return self.experts[resolved_key], resolved_key

    def unload_all(self):
        """Unloads all expert weights to CPU and empties CUDA cache."""
        for model in self.experts.values():
            model.to("cpu")
        self.experts.clear()
        self.active_expert = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_all_experts(self):
        for key in self.checkpoint_paths.keys():
            self._get_expert(key)

    def predict(
        self,
        text: str,
        aspect: str,
        category: Optional[str] = None,
        aspect_start: Optional[int] = None,
        aspect_end: Optional[int] = None,
        force_expert: Optional[str] = None,
    ) -> SentimentPrediction:
        """
        Predicts sentiment for a target aspect in a review using the routed expert.
        """
        if force_expert:
            routed_key = force_expert
            reason = f"forced:{force_expert}"
        else:
            routed_key, reason = self.router.route(category=category, text=text, aspect=aspect)

        model, resolved_key = self._get_expert(routed_key)

        if resolved_key != routed_key:
            logger.warning(
                "Router selected '%s' but it ran under fallback expert '%s'.",
                routed_key,
                resolved_key,
            )

        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"][0].cpu().numpy()

        aspect_mask = torch.zeros_like(attention_mask[0], dtype=torch.float32)

        if aspect_start is not None and aspect_end is not None:
            s_char, e_char = aspect_start, aspect_end
        else:
            match = re.search(r"\b" + re.escape(aspect) + r"\b", text, re.IGNORECASE)
            if match:
                s_char, e_char = match.start(), match.end()
            else:
                idx = text.lower().find(aspect.lower())
                s_char = idx if idx != -1 else 0
                e_char = s_char + len(aspect) if idx != -1 else len(text)

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            if not (tok_end <= s_char or tok_start >= e_char):
                aspect_mask[i] = 1.0

        if aspect_mask.sum() == 0:
            logger.warning(
                "Aspect '%s' not found in token offsets (possibly truncated beyond max_length=%d). "
                "Falling back to CLS token as aspect anchor.",
                aspect,
                MAX_SEQ_LEN,
            )
            aspect_mask[0] = 1.0

        aspect_mask = aspect_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask, aspect_mask)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])

        prob_dict = {
            "positive": float(probs[0]),
            "neutral": float(probs[1]),
            "negative": float(probs[2]),
        }

        return SentimentPrediction(
            sentiment=pred_label,
            confidence=confidence,
            probabilities=prob_dict,
            routed_expert=resolved_key,   # reflects the expert that actually ran
            routing_reason=reason,
        )
