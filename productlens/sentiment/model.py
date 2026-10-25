from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification

from productlens.constants import DEFAULT_BACKBONE


class AspectTargetedAttentionPooler(nn.Module):
    """
    Attention pooling mechanism conditioned on target aspect tokens.
    Applies an additive attention bias to subword positions overlapping
    with the target aspect span to guarantee aspect-tethered contextualization.
    """
    def __init__(self, hidden_size: int, aspect_boost: float = 3.5):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = 1.0 / (hidden_size ** 0.5)
        self.aspect_boost = aspect_boost

    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Compute aspect query vector by pooling token embeddings on aspect positions
        if aspect_mask is not None and aspect_mask.sum() > 0:
            asp_weights = aspect_mask.unsqueeze(-1)
            asp_sum = torch.clamp(asp_weights.sum(dim=1, keepdim=True), min=1.0)
            aspect_rep = (token_embeddings * asp_weights).sum(dim=1, keepdim=True) / asp_sum  # B x 1 x D
        else:
            aspect_rep = token_embeddings[:, 0:1, :]  # B x 1 x D (fallback to CLS)

        # 2. Aspect-to-Context Cross-Attention:
        # Aspect acts as query, all review tokens act as keys to gather opinion evidence
        aspect_q = self.query_proj(aspect_rep)       # B x 1 x D
        context_k = self.key_proj(token_embeddings)  # B x L x D

        cross_scores = torch.bmm(aspect_q, context_k.transpose(1, 2)) * self.scale  # B x 1 x L

        # Mask padding tokens
        pad_mask = (attention_mask == 0).unsqueeze(1)  # B x 1 x L
        cross_scores = cross_scores.masked_fill(pad_mask, -10000.0)

        # Additive aspect boost to anchor aspect context
        if aspect_mask is not None:
            cross_scores = cross_scores + (aspect_mask.unsqueeze(1) * self.aspect_boost)

        cross_weights = F.softmax(cross_scores, dim=-1)  # B x 1 x L
        opinion_evidence = torch.bmm(cross_weights, token_embeddings).squeeze(1)  # B x D

        # 3. Tripartite Fusion: CLS (global sentiment) + Aspect (target) + Opinion Evidence (gathered predicate)
        cls_rep = token_embeddings[:, 0, :]
        aspect_flat = aspect_rep.squeeze(1)
        final_rep = (cls_rep + aspect_flat + opinion_evidence) / 3.0

        return final_rep



class AspectTargetedSentimentModel(nn.Module):
    """
    Aspect-Targeted Sentiment Classifier retaining the pre-trained
    AutoModelForSequenceClassification backbone and classification head,
    augmented with an additive aspect-targeted attention pooling branch.
    Zero-initialized at step 0 so fine-tuning starts strictly from baseline.
    """
    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        num_labels: int = 3,
        aspect_boost: float = 3.5,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone_name)
        # 1. Retain pre-trained backbone, pooler, and classification head
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            backbone_name,
            config=self.config,
        )
        hidden_size = self.config.hidden_size

        # 2. Auxiliary aspect-targeted attention pooling branch
        self.pooler = AspectTargetedAttentionPooler(hidden_size, aspect_boost=aspect_boost)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_labels),
        )
        # Zero-initialize the auxiliary classifier so at step 0:
        # aspect_logits == 0 and logits == base_logits (zero-shot baseline perfectly preserved)
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    @property
    def encoder(self):
        return self.base_model.deberta

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Loads state_dict with seamless backward-compatibility for legacy
        checkpoints where encoder weights were prefixed with 'encoder.'.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "base_model.deberta.", 1)
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        has_base_head = any(k.startswith("base_model.classifier.") for k in new_state_dict)
        return super().load_state_dict(new_state_dict, strict=has_base_head and strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Forward through DeBERTa encoder
        encoder_outputs = self.base_model.deberta(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = encoder_outputs[0]

        # 2. Base model head branch (Yangheng's pre-trained classification)
        base_pooled = self.base_model.pooler(token_embeddings)
        base_logits = self.base_model.classifier(self.base_model.dropout(base_pooled))

        # 3. Auxiliary aspect-targeted pooler branch
        aspect_pooled = self.pooler(token_embeddings, attention_mask, aspect_mask)
        aspect_logits = self.classifier(aspect_pooled)

        # 4. Residual fusion: base_logits + aspect_logits
        logits = base_logits + aspect_logits

        features = None
        if return_features:
            features = F.normalize(self.proj_head(aspect_pooled), dim=-1)

        return logits, features
