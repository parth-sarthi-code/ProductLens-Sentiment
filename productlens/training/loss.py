from typing import Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Union[Sequence[int], torch.Tensor, np.ndarray],
    strategy: Optional[str] = "none",
    beta: float = 0.999,
    num_classes: int = 3,
) -> Optional[torch.Tensor]:
    """
    Computes class weights from train labels to mitigate class imbalance.

    Strategies:
      - 'none' (or None): Returns None (unweighted CE loss).
      - 'inverse_freq': Weight proportional to inverse class frequency:
            w_c = (N / (num_classes * N_c)), normalized so sum(w_c) == num_classes.
      - 'effective_num': Weight based on effective number of samples (Cui et al., CVPR 2019):
            E_n = (1 - beta^N_c) / (1 - beta)
            w_c = 1 / E_n, normalized so sum(w_c) == num_classes.
    """
    if strategy is None or str(strategy).lower() == "none":
        return None

    strat = str(strategy).lower()

    if isinstance(labels, torch.Tensor):
        labels_arr = labels.detach().cpu().numpy()
    elif isinstance(labels, np.ndarray):
        labels_arr = labels
    else:
        labels_arr = np.array(list(labels))

    counts = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        counts[c] = np.sum(labels_arr == c)

    # Guard against zero count
    counts = np.maximum(counts, 1.0)
    total_samples = float(np.sum(counts))

    if strat == "inverse_freq":
        weights = total_samples / (num_classes * counts)
    elif strat == "effective_num":
        effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
    else:
        raise ValueError(
            f"Unsupported class weight strategy: '{strategy}'. "
            "Supported strategies are: 'none', 'inverse_freq', 'effective_num'."
        )

    # Normalize weights so they sum to num_classes (mean weight is 1.0)
    weights = (weights / np.sum(weights)) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020) with temperature scaling.
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


class OrdinalWassersteinLoss(nn.Module):
    """
    1D Earth Mover's Distance (Wasserstein-1) Loss for Ordinal Sentiment Classes.
    Enforces natural sentiment geometry: Negative (0) < Neutral (1) < Positive (2).
    Penalizes extreme polar confusion (+ <-> -) twice as much as neutral confusion (+ <-> neu).
    """

    def __init__(self, num_classes: int = 3, r: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.r = r

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        # Cumulative probability distributions: C_k = sum_{j=0}^k p_j for k in [0, K-2]
        cum_probs = torch.cumsum(probs, dim=-1)[:, :-1]  # B x (K-1)

        # True cumulative distribution: C_k(y) = 1.0 if y <= k else 0.0
        batch_size = labels.size(0)
        k_indices = torch.arange(self.num_classes - 1, device=labels.device).unsqueeze(0).expand(batch_size, -1)
        cum_labels = (labels.unsqueeze(1) <= k_indices).float()  # B x (K-1)

        diff = torch.abs(cum_probs - cum_labels)
        if self.r != 1.0:
            diff = torch.pow(diff, self.r)

        return diff.sum(dim=-1).mean()


class JointABSALoss(nn.Module):
    """
    Joint loss function combining:
    1. Class-Weighted Cross Entropy (with optional label smoothing)
    2. Ordinal Wasserstein (Earth Mover's Distance) Loss
    3. Supervised Contrastive Loss (SCL)
    Loss = CE_Loss + lambda_ordinal * Ordinal_Loss + lambda_scl * SCL_Loss
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_scl: float = 0.10,
        lambda_ordinal: float = 0.15,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.ordinal_loss = OrdinalWassersteinLoss(num_classes=3)
        self.scl_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.lambda_scl = lambda_scl
        self.lambda_ordinal = lambda_ordinal

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_ce = self.ce_loss(logits, labels)

        if self.lambda_ordinal > 0:
            loss_ord = self.ordinal_loss(logits, labels)
        else:
            loss_ord = torch.tensor(0.0, device=logits.device)

        if features is not None and self.lambda_scl > 0:
            loss_scl = self.scl_loss(features, labels)
        else:
            loss_scl = torch.tensor(0.0, device=logits.device)

        total_loss = loss_ce + (self.lambda_ordinal * loss_ord) + (self.lambda_scl * loss_scl)
        return total_loss, loss_ce, loss_scl
