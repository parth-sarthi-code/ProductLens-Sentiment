from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score

from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.training.loss import JointABSALoss
from productlens.training.dataset import AspectSentimentDataset

logger = logging.getLogger("productlens.training")


class AspectSentimentTrainer:
    """
    End-to-end trainer for Aspect-Targeted Sentiment Models.
    Supports differential learning rates (backbone vs head), linear warmup,
    sub-epoch validation checkpointing, and early stopping.
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
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.backbone_lr = backbone_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.lambda_scl = lambda_scl
        self.scl_temp = scl_temp
        self.criterion = JointABSALoss(lambda_scl=lambda_scl, temperature=scl_temp)

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
        best_val_f1 = step0_metrics["macro_f1"]
        best_val_acc = step0_metrics["accuracy"]
        logger.info(
            "Step 0 Baseline: Val Acc=%.4f, Val Macro F1=%.4f",
            best_val_acc,
            best_val_f1,
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
                    logger.info(
                        "Step %d (Epoch %.2f): Val Acc=%.4f, Val Macro F1=%.4f",
                        global_step,
                        global_step / len(train_loader),
                        val_acc,
                        val_f1,
                    )

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        patience_counter = 0
                        torch.save(self.model.state_dict(), best_model_path)
                        logger.info("Saved new best model checkpoint to %s (Val F1: %.4f)", best_model_path, best_val_f1)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info("Early stopping triggered at step %d. Best Val F1: %.4f", global_step, best_val_f1)
                            break

            if patience_counter >= patience:
                break

        # Load best weights before returning
        if best_model_path.exists():
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))

        return {"best_val_macro_f1": best_val_f1, "best_checkpoint": str(best_model_path)}

    def evaluate(self, dataset: AspectSentimentDataset, batch_size: int = 32) -> Dict[str, float]:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds, all_labels = [], []

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

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return {"accuracy": float(acc), "macro_f1": float(macro_f1)}
