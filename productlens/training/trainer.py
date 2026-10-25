from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from productlens.constants import LABEL2ID
from productlens.sentiment.calibration import TemperatureScaler, compute_calibration_metrics
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.loss import JointABSALoss
from productlens.training.dataset import AspectSentimentDataset

logger = logging.getLogger("productlens.training")


class AspectSentimentTrainer:
    """
    End-to-end trainer for Aspect-Targeted Sentiment Models.
    Supports differential learning rates (backbone vs head), linear warmup,
    class-weighted loss, composite neutral-aware checkpointing,
    post-hoc temperature calibration, and early stopping.
    """

    def __init__(
        self,
        model: AspectTargetedSentimentModel,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        backbone_lr: float = 5.0e-6,
        head_lr: float = 5.0e-5,
        weight_decay: float = 0.01,
        lambda_scl: float = 0.10,
        scl_temp: float = 0.07,
        class_weights: Optional[torch.Tensor] = None,
        w_macro: float = 0.7,
        w_neutral: float = 0.3,
        calibrate: bool = True,
        lambda_ordinal: float = 0.15,
        label_smoothing: float = 0.0,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.lambda_scl = lambda_scl
        self.lambda_ordinal = lambda_ordinal
        self.label_smoothing = label_smoothing
        self.scl_temp = scl_temp
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.w_macro = w_macro
        self.w_neutral = w_neutral
        self.calibrate = calibrate
        self.criterion = JointABSALoss(
            class_weights=self.class_weights,
            lambda_scl=lambda_scl,
            lambda_ordinal=lambda_ordinal,
            temperature=scl_temp,
            label_smoothing=label_smoothing,
        )

    def train(
        self,
        train_dataset: AspectSentimentDataset,
        val_dataset: AspectSentimentDataset,
        batch_size: int = 16,
        epochs: int = 5,
        eval_every_steps: int = 150,
        patience: int = 4,
        output_dir: Path | str = "models/checkpoints",
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)

        # Differential Learning Rates: gentle backbone LR (5e-6) + higher head LR (5e-5)
        optimizer_params = [
            {"params": self.model.base_model.deberta.parameters(), "lr": self.backbone_lr, "weight_decay": self.weight_decay},
            {"params": self.model.base_model.pooler.parameters(), "lr": self.backbone_lr, "weight_decay": self.weight_decay},
            {"params": self.model.base_model.classifier.parameters(), "lr": self.backbone_lr, "weight_decay": self.weight_decay},
            {"params": self.model.pooler.parameters(), "lr": self.head_lr, "weight_decay": self.weight_decay},
            {"params": self.model.proj_head.parameters(), "lr": self.head_lr, "weight_decay": self.weight_decay},
            {"params": self.model.classifier.parameters(), "lr": self.head_lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        best_model_path = output_dir / "best_model.pt"

        # Step-0 Evaluation: assess baseline before training steps
        logger.info("Running Step-0 baseline evaluation on validation set...")
        step0_metrics = self.evaluate(val_dataset, batch_size=batch_size)
        best_composite_score = step0_metrics["composite_score"]
        best_val_f1 = step0_metrics["macro_f1"]
        best_val_neutral_f1 = step0_metrics["neutral_f1"]
        best_val_acc = step0_metrics["accuracy"]
        selection_reason = "Step 0 baseline saved"
        logger.info(
            "Step 0 Baseline: Val Acc=%.4f, Val Macro F1=%.4f, Val Neutral F1=%.4f, Composite Score=%.4f (w_macro=%.2f, w_neutral=%.2f)",
            best_val_acc,
            best_val_f1,
            best_val_neutral_f1,
            best_composite_score,
            self.w_macro,
            self.w_neutral,
        )
        torch.save(self.model.state_dict(), best_model_path)

        patience_counter = 0
        global_step = 0

        logger.info("Starting training: total_samples=%d, batch_size=%d, epochs=%d, total_steps=%d", len(train_dataset), batch_size, epochs, total_steps)

        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))
        device_type = self.device.split(":")[0]  # handles "cuda:0", "mps", "cpu"

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                global_step += 1
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                aspect_mask = batch["aspect_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                    logits, features = self.model(input_ids, attention_mask, aspect_mask, return_features=True)
                    loss, _, _ = self.criterion(logits, features, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skip_lr_sched = (scale > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()

                running_loss += loss.item()

                if global_step % eval_every_steps == 0:
                    val_metrics = self.evaluate(val_dataset, batch_size=batch_size)
                    val_acc = val_metrics["accuracy"]
                    val_f1 = val_metrics["macro_f1"]
                    val_neutral_f1 = val_metrics["neutral_f1"]
                    composite_score = val_metrics["composite_score"]
                    avg_train_loss = running_loss / eval_every_steps
                    running_loss = 0.0
                    status_tag = ""

                    if composite_score > best_composite_score:
                        selection_reason = (
                            f"Step {global_step}: composite score improved from {best_composite_score:.4f} "
                            f"to {composite_score:.4f} (Macro F1: {val_f1:.4f}, Neutral F1: {val_neutral_f1:.4f})"
                        )
                        best_composite_score = composite_score
                        best_val_f1 = val_f1
                        best_val_neutral_f1 = val_neutral_f1
                        patience_counter = 0
                        torch.save(self.model.state_dict(), best_model_path)
                        status_tag = " [NEW BEST SAVED ★]"
                    else:
                        patience_counter += 1
                        status_tag = f" [Patience: {patience_counter}/{patience}]"

                    print(
                        f"--> Step {global_step:4d} | Epoch {global_step / len(train_loader):.2f}/{epochs} | "
                        f"Loss: {avg_train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                        f"Macro-F1: {val_f1*100:.2f}% | Neutral-F1: {val_neutral_f1*100:.2f}% | "
                        f"Composite: {composite_score:.4f}{status_tag}",
                        flush=True,
                    )

                    if patience_counter >= patience:
                        print(f"Early stopping triggered at step {global_step}. Best Composite: {best_composite_score:.4f}", flush=True)
                        break

            if patience_counter >= patience:
                break

        # Load best weights before returning
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))

        calibration_info: Dict[str, Any] = {}
        if self.calibrate:
            logger.info("Fitting post-hoc temperature scaling on validation set...")
            val_eval = self.evaluate(val_dataset, batch_size=batch_size, return_logits=True)
            scaler_mod = TemperatureScaler()
            val_logits = val_eval["logits"]
            val_labels = val_eval["labels"]

            uncal_probs = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
            uncal_metrics = compute_calibration_metrics(uncal_probs, val_labels)

            opt_temp = scaler_mod.fit(val_logits, val_labels)

            cal_probs = torch.softmax(torch.tensor(val_logits) / opt_temp, dim=-1).numpy()
            cal_metrics = compute_calibration_metrics(cal_probs, val_labels)

            calib_file = output_dir / "calibration.json"
            scaler_mod.save(
                calib_file,
                metadata={
                    "uncalibrated": uncal_metrics,
                    "calibrated": cal_metrics,
                    "selection_reason": selection_reason,
                },
            )
            calibration_info = {
                "temperature": opt_temp,
                "val_ece_before": uncal_metrics["ece"],
                "val_ece_after": cal_metrics["ece"],
                "calibration_file": str(calib_file),
            }
            logger.info(
                "Calibration complete: T=%.4f | Val ECE: %.4f -> %.4f | Val NLL: %.4f -> %.4f",
                opt_temp,
                uncal_metrics["ece"],
                cal_metrics["ece"],
                uncal_metrics["nll"],
                cal_metrics["nll"],
            )

        return {
            "best_val_macro_f1": best_val_f1,
            "best_val_neutral_f1": best_val_neutral_f1,
            "best_composite_score": best_composite_score,
            "selection_reason": selection_reason,
            "best_checkpoint": str(best_model_path),
            **calibration_info,
        }

    def evaluate(
        self,
        dataset: AspectSentimentDataset,
        batch_size: int = 32,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds, all_labels = [], []
        all_logits = []

        device_type = self.device.split(":")[0]  # handles "cuda:0", "mps", "cpu"
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                aspect_mask = batch["aspect_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                with torch.amp.autocast(device_type, enabled=(device_type == "cuda")):
                    logits, _ = self.model(input_ids, attention_mask, aspect_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                if return_logits:
                    all_logits.extend(logits.cpu().numpy())

        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)
        acc = float(accuracy_score(all_labels_arr, all_preds_arr))
        macro_f1 = float(f1_score(all_labels_arr, all_preds_arr, average="macro", zero_division=0))
        micro_f1 = float(f1_score(all_labels_arr, all_preds_arr, average="micro", zero_division=0))
        per_class_f1 = f1_score(all_labels_arr, all_preds_arr, average=None, labels=[0, 1, 2], zero_division=0)
        neutral_f1 = float(per_class_f1[LABEL2ID["neutral"]])
        composite_score = float(self.w_macro * macro_f1 + self.w_neutral * neutral_f1)

        result: Dict[str, Any] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "neutral_f1": neutral_f1,
            "composite_score": composite_score,
            "per_class_f1": {
                "negative": float(per_class_f1[LABEL2ID["negative"]]),
                "neutral": neutral_f1,
                "positive": float(per_class_f1[LABEL2ID["positive"]]),
            },
        }
        if return_logits:
            result["logits"] = np.array(all_logits)
            result["labels"] = all_labels_arr
        return result
