"""
Adapter to transform benchmark rows into canonical Stage 1 AspectQuads.
Ensures offset integrity across legacy benchmarks and ARTS adversarial datasets.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd
from productlens.schemas.enums import Sentiment, VerticalDomain, normalize_category
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator


class BenchmarkAdapter:
    """Transforms raw benchmark rows into validated Stage 1 AspectQuads."""

    # Heuristic mapping from legacy aspect to canonical category if not present
    CATEGORY_HEURISTICS = {
        # Electronics
        "screen": "DISPLAY",
        "display": "DISPLAY",
        "resolution": "DISPLAY",
        "panel": "DISPLAY",
        "battery": "BATTERY",
        "charger": "BATTERY",
        "battery life": "BATTERY",
        "sound": "SOUND",
        "speakers": "SOUND",
        "audio": "SOUND",
        "volume": "SOUND",
        "speed": "PERFORMANCE",
        "ram": "PERFORMANCE",
        "processor": "PERFORMANCE",
        "cpu": "PERFORMANCE",
        "keyboard": "BUILD_QUALITY",
        "trackpad": "BUILD_QUALITY",
        "hinge": "BUILD_QUALITY",
        "wifi": "CONNECTIVITY",
        "bluetooth": "CONNECTIVITY",
        "hdmi": "CONNECTIVITY",
        "ports": "CONNECTIVITY",
        "os": "SOFTWARE",
        "windows": "SOFTWARE",
        "delivery": "DELIVERY_INSTALLATION",
        "installation": "DELIVERY_INSTALLATION",
        "price": "VALUE",
        "cost": "VALUE",
        # Apparel
        "fabric": "FABRIC_MATERIAL",
        "material": "FABRIC_MATERIAL",
        "cotton": "FABRIC_MATERIAL",
        "fit": "FIT_SIZING",
        "size": "FIT_SIZING",
        "length": "FIT_SIZING",
        "color": "COLOR_PRINT",
        "print": "COLOR_PRINT",
        "stitching": "STITCHING_DURABILITY",
        "threads": "STITCHING_DURABILITY",
        "comfort": "COMFORT",
    }

    @classmethod
    def infer_category(cls, aspect: Optional[str], vertical: VerticalDomain) -> str:
        if aspect:
            asp_clean = aspect.lower().strip()
            for key, cat in cls.CATEGORY_HEURISTICS.items():
                if key in asp_clean:
                    return normalize_category(cat, vertical)
        return "OTHER"

    @classmethod
    def row_to_quad(
        cls,
        row: dict,
        vertical: VerticalDomain,
    ) -> Tuple[Optional[AspectQuad], Optional[str]]:
        """
        Converts a row dict from a benchmark (Laptop14, Television, T-Shirt, ARTS)
        into a canonical AspectQuad with verified substring offsets.
        """
        text = str(row.get("text", "")).strip()
        raw_aspect = row.get("aspect")
        aspect = str(raw_aspect).strip() if raw_aspect is not None and pd.notna(raw_aspect) else None
        raw_sent = str(row.get("sentiment", "neutral")).strip().lower()

        try:
            sentiment = Sentiment.from_str(raw_sent)
        except ValueError:
            sentiment = Sentiment.NEUTRAL

        category = cls.infer_category(aspect, vertical)

        # In benchmarks, evidence defaults to the full sentence or clause containing the aspect
        evidence = text

        raw_quad = RawCompactQuad(
            aspect=aspect,
            category=category,
            sentiment=sentiment,
            confidence=1.0,
            evidence=evidence,
        )

        return SubstringValidator.validate_quad(text, raw_quad, vertical)

    @classmethod
    def dataframe_to_quads(
        cls,
        df: pd.DataFrame,
        vertical: VerticalDomain,
    ) -> Tuple[List[Tuple[str, AspectQuad]], int, int]:
        """
        Converts an entire benchmark DataFrame into pairs of (text, AspectQuad).
        Returns: (quads, total_rows, valid_rows)
        """
        results: List[Tuple[str, AspectQuad]] = []
        total = len(df)
        valid = 0

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            quad, err = cls.row_to_quad(row_dict, vertical)
            if quad is not None:
                results.append((str(row_dict.get("text", "")), quad))
                valid += 1

        return results, total, valid
