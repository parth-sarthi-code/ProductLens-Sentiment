"""
Gold Set Manager: Loads, validates, and partitions the curated gold benchmark set.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
from productlens.gold.schema import GoldReview, GoldSetManifest
from productlens.schemas.enums import VerticalDomain


DEFAULT_GOLD_PATH = Path(__file__).parent / "gold_dataset.jsonl"


class GoldSetManager:
    """Manages the curated gold benchmark dataset."""

    @staticmethod
    def load_reviews(path: Path | str = DEFAULT_GOLD_PATH) -> List[GoldReview]:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Gold set file does not exist at {filepath}")

        reviews: List[GoldReview] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    reviews.append(GoldReview.model_validate_json(line))
        return reviews

    @classmethod
    def get_manifest(cls, reviews: List[GoldReview]) -> GoldSetManifest:
        products = {r.product_id for r in reviews}
        english_count = sum(1 for r in reviews if r.language == "en")
        hinglish_count = sum(1 for r in reviews if r.language == "hi-en")
        hard_count = sum(1 for r in reviews if r.is_hard_case)
        quad_count = sum(len(r.quads) for r in reviews)

        return GoldSetManifest(
            total_reviews=len(reviews),
            english_reviews=english_count,
            hinglish_reviews=hinglish_count,
            hard_cases=hard_count,
            total_quads=quad_count,
            products_count=len(products),
        )

    @classmethod
    def split_by_language(cls, reviews: List[GoldReview]) -> Dict[str, List[GoldReview]]:
        return {
            "en": [r for r in reviews if r.language == "en"],
            "hi-en": [r for r in reviews if r.language == "hi-en"],
        }
