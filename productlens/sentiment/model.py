from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


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
        queries = self.query_proj(token_embeddings)
        keys = self.key_proj(token_embeddings)
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale

        key_padding_mask = attention_mask.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
        attn_scores = attn_scores.masked_fill(key_padding_mask == 0, -10000.0)

        if aspect_mask is not None:
            asp_bias = aspect_mask.unsqueeze(1).expand(-1, token_embeddings.size(1), -1)
            attn_scores = attn_scores + (asp_bias * self.aspect_boost)

        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, token_embeddings)

        if aspect_mask is not None and aspect_mask.sum() > 0:
            asp_weights = aspect_mask.unsqueeze(-1)
            asp_sum = torch.clamp(asp_weights.sum(dim=1), min=1.0)
            aspect_rep = (token_embeddings * asp_weights).sum(dim=1) / asp_sum
            cls_rep = pooled[:, 0, :]
            final_rep = (cls_rep + aspect_rep) / 2.0
        else:
            final_rep = pooled[:, 0, :]

        return final_rep


from productlens.constants import DEFAULT_BACKBONE


class AspectTargetedSentimentModel(nn.Module):
    """
    Aspect-Targeted Sentiment Classifier with DeBERTa-v3 ABSA backbone,
    aspect-targeted attention pooling, projection head, and classification MLP.
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
        self.encoder = AutoModel.from_pretrained(backbone_name, config=self.config)
        hidden_size = self.config.hidden_size

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs[0]
        pooled = self.pooler(token_embeddings, attention_mask, aspect_mask)
        logits = self.classifier(pooled)

        features = None
        if return_features:
            features = F.normalize(self.proj_head(pooled), dim=-1)

        return logits, features
