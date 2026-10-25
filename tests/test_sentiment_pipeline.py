import pytest
from productlens.sentiment.pipeline import ProductLensSentimentPipeline, ProductLensMoEPipeline, checkpoints_exist
from productlens.schemas import SentimentPrediction


class TestSentimentPipeline:
    @classmethod
    def setup_class(cls):
        cls.pipeline = ProductLensSentimentPipeline()

    def test_pipeline_initialization(self):
        assert self.pipeline.model is not None
        assert self.pipeline.tokenizer is not None
        assert self.pipeline.temperature > 0.0

    def test_tech_review_positive_screen(self):
        pred = self.pipeline.predict(
            text="The screen resolution on this laptop is incredible, but the fan is loud.",
            aspect="screen",
        )
        assert pred.sentiment == "positive"
        assert pred.confidence > 0.60
        assert pred.routed_expert == "unified"
        assert pred.aspect_mask_fallback is False

    def test_apparel_review_negative_material(self):
        pred = self.pipeline.predict(
            text="The shirt shrank in the wash and the material is scratchy.",
            aspect="material",
        )
        assert pred.sentiment == "negative"
        assert pred.confidence > 0.60
        assert pred.routed_expert == "unified"

    def test_home_review_positive_sound(self):
        pred = self.pipeline.predict(
            text="The sound quality on this television is superb.",
            aspect="sound",
        )
        assert pred.sentiment == "positive"
        assert pred.confidence > 0.70

    def test_multi_aspect_batching_parity(self):
        text = "Display is gorgeous, but battery life is horrible and keyboard is okay."
        aspects = [
            ("Display", 0, 7),
            ("battery life", 24, 36),
            ("keyboard", 54, 62),
        ]
        results = self.pipeline.predict_aspects(text=text, aspects=aspects)
        assert len(results) == 3
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"
        assert results[2].sentiment == "neutral" or results[2].confidence > 0.4

    def test_aspect_mask_fallback(self):
        pred = self.pipeline.predict(
            text="Short review about something.",
            aspect="completely_nonexistent_aspect",
        )
        assert pred.aspect_mask_fallback is True

    def test_backward_compatibility_alias(self):
        # ProductLensMoEPipeline alias should work interchangeably
        legacy_pipeline = ProductLensMoEPipeline()
        pred = legacy_pipeline.predict(
            text="Great laptop screen.",
            aspect="screen",
            category="Computers",
        )
        assert pred.sentiment == "positive"
        assert isinstance(pred, SentimentPrediction)
