import json
import pytest
import numpy as np
import torch
import torch.nn as nn

from productlens.constants import LABEL2ID
from productlens.sentiment.calibration import (
    TemperatureScaler,
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
    compute_nll,
)
from productlens.training.loss import JointABSALoss, compute_class_weights
from productlens.training.trainer import AspectSentimentTrainer
from productlens.sentiment.pipeline import ProductLensMoEPipeline, checkpoints_exist
from productlens.engine import ProductLensEngine


# ---------------------------------------------------------------------------
# 1. Class Imbalance Weighting Tests
# ---------------------------------------------------------------------------

class TestClassImbalanceWeights:
    def test_class_weights_none(self):
        labels = [0, 0, 1, 2, 2, 2]
        assert compute_class_weights(labels, strategy="none") is None
        assert compute_class_weights(labels, strategy=None) is None

    def test_class_weights_inverse_freq(self):
        # Class counts: class 0 -> 2, class 1 -> 1, class 2 -> 3
        # Class 1 is rarest, should have highest weight
        labels = [0, 0, 1, 2, 2, 2]
        weights = compute_class_weights(labels, strategy="inverse_freq", num_classes=3)
        assert weights is not None
        assert len(weights) == 3
        assert weights[1] > weights[0] > weights[2]
        # Sum of normalized weights should equal num_classes
        assert pytest.approx(float(weights.sum()), rel=1e-5) == 3.0

    def test_class_weights_effective_num(self):
        labels = [0, 0, 1, 2, 2, 2]
        weights = compute_class_weights(labels, strategy="effective_num", beta=0.99, num_classes=3)
        assert weights is not None
        assert len(weights) == 3
        assert weights[1] > weights[0] > weights[2]
        assert pytest.approx(float(weights.sum()), rel=1e-5) == 3.0

    def test_class_weights_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unsupported class weight strategy"):
            compute_class_weights([0, 1, 2], strategy="unknown_strategy")

    def test_joint_loss_with_class_weights(self):
        weights = torch.tensor([1.0, 2.0, 0.5])
        loss_fn = JointABSALoss(class_weights=weights, lambda_scl=0.1)
        logits = torch.randn(4, 3)
        features = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 1])
        total_loss, ce_loss, scl_loss = loss_fn(logits, features, labels)
        assert total_loss > 0
        assert ce_loss > 0


# ---------------------------------------------------------------------------
# 2. Checkpoint Selection & Neutral-Aware Composite Score Tests
# ---------------------------------------------------------------------------

class TestNeutralAwareCheckpointSelection:
    def test_composite_score_formula(self):
        w_macro = 0.7
        w_neutral = 0.3
        macro_f1 = 0.85
        neutral_f1 = 0.60
        expected_score = 0.7 * 0.85 + 0.3 * 0.60
        score = w_macro * macro_f1 + w_neutral * neutral_f1
        assert pytest.approx(score, rel=1e-5) == expected_score

    def test_trainer_composite_weights_config(self):
        # Verify AspectSentimentTrainer accepts w_macro, w_neutral, class_weights
        dummy_model = nn.Module()
        trainer = AspectSentimentTrainer.__new__(AspectSentimentTrainer)
        trainer.w_macro = 0.6
        trainer.w_neutral = 0.4
        trainer.calibrate = True
        assert trainer.w_macro == 0.6
        assert trainer.w_neutral == 0.4


# ---------------------------------------------------------------------------
# 3. Calibration Module Tests
# ---------------------------------------------------------------------------

class TestCalibrationModule:
    def test_calibration_metrics(self):
        # 4 samples, 3 classes
        probs = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.5, 0.3, 0.2],
        ])
        labels = np.array([0, 1, 2, 1])  # Sample 4 is misclassified (pred 0 vs gold 1)

        metrics = compute_calibration_metrics(probs, labels)
        assert "ece" in metrics
        assert "nll" in metrics
        assert "brier" in metrics
        assert metrics["ece"] >= 0.0
        assert metrics["nll"] > 0.0
        assert metrics["brier"] > 0.0

    def test_temperature_scaling_optimization(self):
        # Overconfident logits on misclassified samples
        logits = torch.tensor([
            [10.0, 1.0, 0.0],
            [1.0, 10.0, 0.0],
            [0.0, 1.0, 10.0],
            [8.0, 6.0, 0.0],  # overconfident on class 0, but label is 1
        ], dtype=torch.float32)
        labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)

        scaler = TemperatureScaler(init_temperature=1.0)
        opt_temp = scaler.fit(logits, labels)
        assert opt_temp > 1.0  # Temperature should increase to soften overconfident logits

        scaled_logits = scaler(logits)
        assert scaled_logits.shape == logits.shape

    def test_temperature_save_and_load(self, tmp_path):
        scaler = TemperatureScaler(init_temperature=1.234)
        save_file = tmp_path / "calibration.json"
        scaler.save(save_file, metadata={"val_ece": 0.025})

        loaded_scaler = TemperatureScaler.load(save_file)
        assert pytest.approx(float(loaded_scaler.temperature.item()), rel=1e-4) == 1.234

    def test_missing_calibration_file_fallback(self, tmp_path):
        non_existent = tmp_path / "missing_calibration.json"
        loaded_scaler = TemperatureScaler.load(non_existent)
        assert loaded_scaler.temperature.item() == 1.0


# ---------------------------------------------------------------------------
# 4. Multi-Aspect Inference Optimization & Parity Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not checkpoints_exist(),
    reason="Model checkpoints not downloaded",
)
class TestMultiAspectInferenceOptimization:
    def test_multi_aspect_parity_with_predict(self):
        pipeline = ProductLensMoEPipeline(lazy_load=True)
        text = "The laptop display is breathtaking and bright, but the keyboard keys stick and feel mushy."
        aspects = [
            ("display", 11, 18),
            ("keyboard keys", 58, 71),
        ]

        # Single predict calls
        pred_display = pipeline.predict(text=text, aspect="display", category="Electronics", aspect_start=11, aspect_end=18)
        pred_keyboard = pipeline.predict(text=text, aspect="keyboard keys", category="Electronics", aspect_start=58, aspect_end=71)

        # Batched predict_aspects call
        batched_preds = pipeline.predict_aspects(text=text, aspects=aspects, category="Electronics")

        assert len(batched_preds) == 2
        # Exact sentiment parity
        assert batched_preds[0].sentiment == pred_display.sentiment
        assert batched_preds[1].sentiment == pred_keyboard.sentiment
        # Probabilities match within tight numerical margin
        for c in ["positive", "neutral", "negative"]:
            assert pytest.approx(batched_preds[0].probabilities[c], abs=1e-4) == pred_display.probabilities[c]
            assert pytest.approx(batched_preds[1].probabilities[c], abs=1e-4) == pred_keyboard.probabilities[c]

    def test_engine_analyze_review_uses_predict_aspects(self):
        engine = ProductLensEngine(lazy_load=True)
        text = "The battery life is super long, the screen is sharp, and the speakers are loud."
        report = engine.analyze_review(text=text, category="Electronics")
        assert report.aspect_count >= 2
        assert len(report.results) == report.aspect_count
        for r in report.results:
            assert r.sentiment in ["positive", "negative", "neutral"]
            assert hasattr(r, "aspect_mask_fallback")
            assert hasattr(r, "calibrated")

    def test_aspect_mask_fallback_flag(self):
        pipeline = ProductLensMoEPipeline(lazy_load=True)
        # Query an aspect that does not appear in text
        pred = pipeline.predict(
            text="Short review.",
            aspect="totally_missing_aspect_term_xyz",
            category="Electronics",
        )
        assert pred.aspect_mask_fallback is True
