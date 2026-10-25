from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("productlens.sentiment.calibration")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Partitions predictions into M equally spaced confidence bins and computes
    the weighted average difference between confidence and accuracy.
    """
    if len(probs) == 0:
        return 0.0

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(in_bin):
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            avg_conf_in_bin = float(np.mean(confidences[in_bin]))
            prop_in_bin = float(np.mean(in_bin))
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece)


def compute_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    """
    Negative Log-Likelihood (NLL) / multiclass log loss.
    """
    if len(probs) == 0:
        return 0.0

    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    n = len(labels)
    # Pick the probability corresponding to the ground truth label
    log_likelihoods = np.log(clipped_probs[np.arange(n), labels])
    return float(-np.mean(log_likelihoods))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int = 3) -> float:
    """
    Multiclass Brier Score (mean squared error between predicted probabilities
    and one-hot encoded ground truth labels).
    """
    if len(probs) == 0:
        return 0.0

    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels] = 1.0

    # Mean squared error across all samples and classes
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return float(brier)


def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Computes ECE, NLL, and Brier score.
    Returns metrics formatted as floats rounded to 4 decimal places.
    """
    ece = compute_ece(probs, labels, n_bins=n_bins)
    nll = compute_nll(probs, labels)
    brier = compute_brier_score(probs, labels, num_classes=num_classes)

    return {
        "ece": round(ece, 4),
        "nll": round(nll, 4),
        "brier": round(brier, 4),
    }


class TemperatureScaler(nn.Module):
    """
    Post-hoc Temperature Scaling (Guo et al., 2017).
    A single scalar parameter T > 0 softens or sharpens logits:
    p_i = softmax(z_i / T)
    Optimized via L-BFGS to minimize validation NLL without changing argmax predictions.
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * float(init_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        return logits / temp

    def fit(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Fits temperature T on validation logits and labels using L-BFGS.
        """
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_tensor = logits.detach().cpu().float()

        if isinstance(labels, np.ndarray):
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            labels_tensor = labels.detach().cpu().long()

        device = torch.device("cpu")
        logits_tensor = logits_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
        self.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            with torch.amp.autocast("cpu", enabled=False):
                scaled = self.forward(logits_tensor)
                loss = criterion(scaled, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # Clamp temperature to safe operating bounds
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)

        opt_temp = float(self.temperature.item())
        logger.info("Optimized temperature scaling parameter: T = %.4f", opt_temp)
        return opt_temp

    def save(self, path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves temperature and calibration metadata to a JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": round(float(self.temperature.item()), 4),
        }
        if metadata:
            data.update(metadata)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved calibration config to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TemperatureScaler":
        """
        Loads temperature from a JSON file. If file does not exist, returns default (T=1.0).
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Calibration file %s not found. Using default temperature T=1.0.", path)
            return cls(init_temperature=1.0)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            temp = float(data.get("temperature", 1.0))
            scaler = cls(init_temperature=temp)
            logger.info("Loaded calibration temperature T=%.4f from %s", temp, path)
            return scaler
        except Exception as e:
            logger.warning("Failed to load calibration from %s (%s). Falling back to T=1.0.", path, e)
            return cls(init_temperature=1.0)
