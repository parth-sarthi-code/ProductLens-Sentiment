from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khslosla et al., 2020) with temperature scaling.
    Pulls representations of identical sentiment classes closer in the 128-dim
    projection manifold while pushing opposing sentiment representations apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        sim_matrix = sim_matrix - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0.0,
        )
        mask = mask * logits_mask

        exp_sim = torch.exp(sim_matrix) * logits_mask
        sum_exp = torch.clamp(exp_sim.sum(1, keepdim=True), min=1e-7)
        log_prob = sim_matrix - torch.log(sum_exp)

        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.clamp(mask.sum(1), min=1e-7)
        valid_pos = mask.sum(1) > 0
        if not valid_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        return -mean_log_prob_pos[valid_pos].mean()


class JointABSALoss(nn.Module):
    """
    Joint loss function combining Class-Weighted Cross Entropy and Supervised Contrastive Loss.
    Loss = CE_Loss + lambda_scl * SCL_Loss
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_scl: float = 0.10,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.scl_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.lambda_scl = lambda_scl

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ce = self.ce_loss(logits, labels)
        if features is not None and self.lambda_scl > 0:
            loss_scl = self.scl_loss(features, labels)
        else:
            loss_scl = torch.tensor(0.0, device=logits.device)

        total_loss = loss_ce + (self.lambda_scl * loss_scl)
        return total_loss, loss_ce, loss_scl
