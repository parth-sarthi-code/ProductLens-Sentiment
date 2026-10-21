from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from productlens.schemas import SentimentPrediction
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.router import MoERouter

logger = logging.getLogger("productlens.sentiment")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT_PATHS = {
    "tech": BASE_DIR / "models/category_experts/expert_tech/best_model.pt",
    "home": BASE_DIR / "models/category_experts/expert_home/best_model.pt",
    "apparel": BASE_DIR / "models/category_experts/expert_apparel/best_model.pt",
    "beauty": BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
    "fallback": BASE_DIR / "models/category_experts/foundation_mams/best_model.pt",
}


class ProductLensMoEPipeline:
    """
    Unified Mixture of Experts (MoE) Inference Pipeline.
    Loads and caches domain specialists, dispatches queries via MoERouter,
    and returns aspect-conditioned sentiment predictions.
    """

    def __init__(
        self,
        checkpoint_paths: Optional[Dict[str, Union[str, Path]]] = None,
        backbone_name: str = "roberta-base",
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.backbone_name = backbone_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_paths = {k: Path(v) for k, v in (checkpoint_paths or DEFAULT_CHECKPOINT_PATHS).items()}
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.router = MoERouter()
        self.lazy_load = lazy_load

        self.experts: Dict[str, AspectTargetedSentimentModel] = {}
        if not self.lazy_load:
            self._load_all_experts()

        logger.info("Initialized ProductLens MoE Pipeline on device=%s (backbone=%s)", self.device, self.backbone_name)

    def _get_expert(self, expert_key: str) -> AspectTargetedSentimentModel:
        """Loads and caches the requested domain expert."""
        if expert_key not in self.experts:
            ckpt_path = self.checkpoint_paths.get(expert_key)
            if not ckpt_path or not ckpt_path.exists():
                logger.warning("Checkpoint for '%s' not found at %s. Falling back to 'fallback' expert.", expert_key, ckpt_path)
                expert_key = "fallback"
                ckpt_path = self.checkpoint_paths.get("fallback")

            logger.info("Loading expert '%s' from %s...", expert_key, ckpt_path)
            model = AspectTargetedSentimentModel(backbone_name=self.backbone_name, num_labels=3, aspect_boost=3.5)
            if ckpt_path and ckpt_path.exists():
                state_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.experts[expert_key] = model

        return self.experts[expert_key]

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
            expert_key = force_expert
            reason = f"forced:{force_expert}"
        else:
            expert_key, reason = self.router.route(category=category, text=text, aspect=aspect)

        model = self._get_expert(expert_key)

        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
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
            routed_expert=expert_key,
            routing_reason=reason,
        )
