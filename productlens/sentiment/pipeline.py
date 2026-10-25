import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE, DEFAULT_CHECKPOINT_PATHS, ID2LABEL, LABEL2ID, MAX_SEQ_LEN
from productlens.schemas import SentimentPrediction
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.router import LearnedGatingNetwork, MoERouter
from productlens.utils import find_aspect_span

logger = logging.getLogger("productlens.sentiment")


def checkpoints_exist(
    checkpoint_paths: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
) -> bool:
    """Returns True if at least one specialist checkpoint exists on disk, or fallback is enabled."""
    paths = checkpoint_paths or DEFAULT_CHECKPOINT_PATHS
    return any(p is not None and Path(p).exists() for p in paths.values())


class ProductLensMoEPipeline:
    """
    Unified Mixture of Experts (MoE) Inference Pipeline.
    Loads and caches domain specialists, dispatches queries via MoERouter,
    and returns aspect-conditioned sentiment predictions.
    """

    def __init__(
        self,
        checkpoint_paths: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
        backbone_name: str = DEFAULT_BACKBONE,
        device: Optional[str] = None,
        lazy_load: bool = True,
        single_active_expert: bool = False,
        router_checkpoint: Optional[Union[str, Path]] = None,
    ):
        self.backbone_name = backbone_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_paths = {
            k: Path(v) for k, v in (checkpoint_paths or DEFAULT_CHECKPOINT_PATHS).items()
            if v is not None
        }
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.temperatures: Dict[str, float] = {}

        learned_gate = None
        r_path = Path(router_checkpoint) if router_checkpoint else Path("models/learned_router.pt")
        if r_path.exists():
            try:
                gate = LearnedGatingNetwork(hidden_size=768, num_experts=4)
                gate.load_state_dict(torch.load(r_path, map_location=self.device, weights_only=True))
                gate.to(self.device)
                gate.eval()
                learned_gate = gate
                logger.info("Loaded Learned Gating Router from %s", r_path)
            except Exception as e:
                logger.warning("Could not load learned gating network from %s: %s", r_path, e)

        self.router = MoERouter(learned_gate=learned_gate)
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
                if resolved_key != "fallback":
                    logger.warning(
                        "Checkpoint for '%s' not found at %s. Attempting fallback expert.",
                        expert_key,
                        ckpt_path,
                    )
                    resolved_key = "fallback"
                    ckpt_path = self.checkpoint_paths.get("fallback")

            if resolved_key == "fallback" and ckpt_path is None:
                logger.info(
                    "Using canonical pretrained backbone '%s' for fallback expert.",
                    self.backbone_name,
                )
                model = AspectTargetedSentimentModel(
                    backbone_name=self.backbone_name,
                    num_labels=3,
                    aspect_boost=3.5,
                )
                model.eval()
                self.temperatures[resolved_key] = 1.0
            else:
                if not ckpt_path or not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint for expert '{expert_key}' not found, and fallback "
                        f"checkpoint is also missing at '{ckpt_path}'. "
                        "Download model weights before running inference. "
                        "See README.md § 'Download Weights' for instructions."
                    )

                logger.info("Loading expert '%s' from %s...", resolved_key, ckpt_path)
                model = AspectTargetedSentimentModel(backbone_name=self.backbone_name, num_labels=3, aspect_boost=3.5)
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()

                # Load post-hoc calibration temperature if present
                calib_file = ckpt_path.parent / "calibration.json"
                if not calib_file.exists():
                    calib_file = ckpt_path.with_name(ckpt_path.stem + ".calibration.json")
                if not calib_file.exists():
                    calib_file = ckpt_path.with_suffix(".calibration.json")

                if calib_file.exists():
                    try:
                        with open(calib_file, "r", encoding="utf-8") as f:
                            cal_data = json.load(f)
                        temp = float(cal_data.get("temperature", 1.0))
                        self.temperatures[resolved_key] = temp
                        logger.info("Loaded calibration temperature T=%.4f for '%s' from %s", temp, resolved_key, calib_file)
                    except Exception as e:
                        logger.warning("Error reading calibration from %s (%s). Using T=1.0.", calib_file, e)
                        self.temperatures[resolved_key] = 1.0
                else:
                    self.temperatures[resolved_key] = 1.0

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
        self.temperatures.clear()
        self.active_expert = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_all_experts(self):
        for key in list(self.checkpoint_paths.keys()) + ["fallback"]:
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
            embedding = None
            if category is None and self.router.learned_gate is not None:
                emb_key = self.active_expert or "fallback"
                try:
                    emb_model, _ = self._get_expert(emb_key)
                    with torch.no_grad():
                        enc = self.tokenizer(text, aspect, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(self.device)
                        out = emb_model.base_model.deberta(enc["input_ids"], enc["attention_mask"])
                        embedding = emb_model.base_model.pooler(out[0])
                except Exception as e:
                    logger.debug("Learned gating embedding extraction failed, falling back to lexical: %s", e)

            routed_key, reason = self.router.route(category=category, text=text, aspect=aspect, embedding=embedding)

        model, resolved_key = self._get_expert(routed_key)

        if resolved_key != routed_key:
            logger.warning(
                "Router selected '%s' but it ran under fallback expert '%s'.",
                routed_key,
                resolved_key,
            )

        encoding = self.tokenizer(
            text,
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

        fell_back_aspect_mask = False
        if aspect_start is not None and aspect_end is not None:
            s_char, e_char = aspect_start, aspect_end
        else:
            s_char, e_char = find_aspect_span(text, aspect)
            if aspect.lower() not in text.lower():
                fell_back_aspect_mask = True

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end:
                continue
            is_text_segment = (seq_ids[i] == 0) if seq_ids is not None else True
            if not fell_back_aspect_mask and is_text_segment and not (tok_end <= s_char or tok_start >= e_char):
                aspect_mask[i] = 1.0

        # If not found in text segment, anchor on query aspect tokens (seq_id == 1)
        if aspect_mask.sum() == 0 and seq_ids is not None:
            for tok_idx, sid in enumerate(seq_ids):
                if sid == 1:
                    aspect_mask[tok_idx] = 1.0
            fell_back_aspect_mask = True

        if aspect_mask.sum() == 0:
            logger.warning(
                "Aspect '%s' not found in token offsets (possibly truncated beyond max_length=%d). "
                "Falling back to CLS token as aspect anchor.",
                aspect,
                MAX_SEQ_LEN,
            )
            aspect_mask[0] = 1.0
            fell_back_aspect_mask = True

        aspect_mask = aspect_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask, aspect_mask)
            temp = self.temperatures.get(resolved_key, 1.0)
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])

        prob_dict = {
            "negative": float(probs[LABEL2ID["negative"]]),
            "neutral": float(probs[LABEL2ID["neutral"]]),
            "positive": float(probs[LABEL2ID["positive"]]),
        }

        return SentimentPrediction(
            sentiment=pred_label,
            confidence=confidence,
            probabilities=prob_dict,
            routed_expert=resolved_key,   # reflects the expert that actually ran
            routing_reason=reason,
            aspect_mask_fallback=fell_back_aspect_mask,
            calibrated=(temp != 1.0),
        )

    def predict_aspects(
        self,
        text: str,
        aspects: List[Union[str, Tuple[str, Optional[int], Optional[int]]]],
        category: Optional[str] = None,
        force_expert: Optional[str] = None,
    ) -> List[SentimentPrediction]:
        """
        Optimized multi-aspect inference for a single review.
        Avoids repeated tokenization and runs batched inference per domain expert,
        significantly reducing redundant DeBERTa encoding passes.
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
                s_c, e_c = find_aspect_span(text, asp_text)
            else:
                s_c, e_c = s, e
            normalized_aspects.append((asp_text, s_c, e_c))

        num_aspects = len(normalized_aspects)

        # Single aspect fallback directly delegates to predict
        if num_aspects == 1:
            asp_text, s_c, e_c = normalized_aspects[0]
            pred = self.predict(
                text=text,
                aspect=asp_text,
                category=category,
                aspect_start=s_c,
                aspect_end=e_c,
                force_expert=force_expert,
            )
            return [pred]

        # Step 1: Routing
        # If category is None and learned router is active: extract text representation ONCE
        shared_embedding = None
        if not force_expert and category is None and self.router.learned_gate is not None:
            emb_key = self.active_expert or "fallback"
            try:
                emb_model, _ = self._get_expert(emb_key)
                with torch.no_grad():
                    enc = self.tokenizer(
                        text,
                        normalized_aspects[0][0],
                        truncation=True,
                        max_length=MAX_SEQ_LEN,
                        return_tensors="pt",
                    ).to(self.device)
                    out = emb_model.base_model.deberta(enc["input_ids"], enc["attention_mask"])
                    shared_embedding = emb_model.base_model.pooler(out[0])
            except Exception as e:
                logger.debug("Multi-aspect gating embedding extraction failed: %s", e)

        # Route all aspects and collect expert targets
        aspect_routes: List[Tuple[str, str]] = []
        for asp_text, _, _ in normalized_aspects:
            if force_expert:
                routed_key = force_expert
                reason = f"forced:{force_expert}"
            else:
                routed_key, reason = self.router.route(
                    category=category,
                    text=text,
                    aspect=asp_text,
                    embedding=shared_embedding,
                )
            aspect_routes.append((routed_key, reason))

        # Group aspect indices by resolved expert
        expert_groups: Dict[str, List[int]] = {}
        for idx, (routed_key, _) in enumerate(aspect_routes):
            _, resolved_key = self._get_expert(routed_key)
            expert_groups.setdefault(resolved_key, []).append(idx)

        predictions: List[Optional[SentimentPrediction]] = [None] * num_aspects

        # Step 2: Batched inference per expert
        for resolved_key, indices in expert_groups.items():
            model = self.experts[resolved_key]
            group_aspect_texts = [normalized_aspects[i][0] for i in indices]
            group_texts = [text] * len(indices)

            # Batched tokenization: tokenize all aspect-review pairs simultaneously
            encoding = self.tokenizer(
                group_texts,
                group_aspect_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            offset_mappings = encoding["offset_mapping"].cpu().numpy()

            batch_size, seq_len = input_ids.shape
            aspect_masks = torch.zeros((batch_size, seq_len), dtype=torch.float32)
            has_sequence_ids = hasattr(encoding, "sequence_ids")

            aspect_mask_fallbacks = []

            for b_idx, orig_idx in enumerate(indices):
                asp_text, s_char, e_char = normalized_aspects[orig_idx]
                offsets = offset_mappings[b_idx]
                seq_ids = encoding.sequence_ids(b_idx) if has_sequence_ids else None
                fallback_flag = (asp_text.lower() not in text.lower())

                mask = torch.zeros(seq_len, dtype=torch.float32)
                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start == tok_end:
                        continue
                    is_text_segment = (seq_ids[tok_idx] == 0) if seq_ids is not None else True
                    if not fallback_flag and is_text_segment and not (tok_end <= s_char or tok_start >= e_char):
                        mask[tok_idx] = 1.0

                if mask.sum() == 0 and seq_ids is not None:
                    for tok_idx, sid in enumerate(seq_ids):
                        if sid == 1:
                            mask[tok_idx] = 1.0
                    fallback_flag = True

                if mask.sum() == 0:
                    mask[0] = 1.0
                    fallback_flag = True

                aspect_masks[b_idx] = mask
                aspect_mask_fallbacks.append(fallback_flag)

            aspect_masks = aspect_masks.to(self.device)

            with torch.no_grad():
                logits, _ = model(input_ids, attention_mask, aspect_masks)
                temp = self.temperatures.get(resolved_key, 1.0)
                scaled_logits = logits / temp
                probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()

            for b_idx, orig_idx in enumerate(indices):
                sample_probs = probs[b_idx]
                pred_id = int(np.argmax(sample_probs))
                pred_label = ID2LABEL[pred_id]
                confidence = float(sample_probs[pred_id])
                routed_key, reason = aspect_routes[orig_idx]

                prob_dict = {
                    "negative": float(sample_probs[LABEL2ID["negative"]]),
                    "neutral": float(sample_probs[LABEL2ID["neutral"]]),
                    "positive": float(sample_probs[LABEL2ID["positive"]]),
                }

                predictions[orig_idx] = SentimentPrediction(
                    sentiment=pred_label,
                    confidence=confidence,
                    probabilities=prob_dict,
                    routed_expert=resolved_key,
                    routing_reason=reason,
                    aspect_mask_fallback=aspect_mask_fallbacks[b_idx],
                    calibrated=(temp != 1.0),
                )

        return [p for p in predictions if p is not None]
