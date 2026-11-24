"""
Enumerations for ProductLens 3.0: Sentiments, Verticals, and Canonical Categories.
"""

from enum import Enum
from typing import Set


class Sentiment(str, Enum):
    """Aspect-level sentiment polarity."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

    @classmethod
    def from_str(cls, value: str) -> "Sentiment":
        val = value.strip().lower()
        for member in cls:
            if member.value == val:
                return member
        raise ValueError(f"Invalid sentiment '{value}'. Allowed: {[m.value for m in cls]}")


class VerticalDomain(str, Enum):
    """Target e-commerce domains for ProductLens v1/v2."""
    ELECTRONICS_TECH = "electronics_tech"
    APPAREL_FASHION = "apparel_fashion"


class ElectronicsCategory(str, Enum):
    """Canonical feature taxonomy for Electronics & Tech."""
    DISPLAY = "DISPLAY"
    BATTERY = "BATTERY"
    SOUND = "SOUND"
    PERFORMANCE = "PERFORMANCE"
    BUILD_QUALITY = "BUILD_QUALITY"
    CONNECTIVITY = "CONNECTIVITY"
    SOFTWARE = "SOFTWARE"
    DELIVERY_INSTALLATION = "DELIVERY_INSTALLATION"
    VALUE = "VALUE"
    OTHER = "OTHER"

    @classmethod
    def values(cls) -> Set[str]:
        return {item.value for item in cls}


class ApparelCategory(str, Enum):
    """Canonical feature taxonomy for Apparel & Fashion."""
    FABRIC_MATERIAL = "FABRIC_MATERIAL"
    FIT_SIZING = "FIT_SIZING"
    COLOR_PRINT = "COLOR_PRINT"
    STITCHING_DURABILITY = "STITCHING_DURABILITY"
    COMFORT = "COMFORT"
    DELIVERY = "DELIVERY"
    VALUE = "VALUE"
    OTHER = "OTHER"

    @classmethod
    def values(cls) -> Set[str]:
        return {item.value for item in cls}


def normalize_category(category: str, vertical: VerticalDomain | str) -> str:
    """
    Validates and normalizes category against the vertical's canonical taxonomy.
    """
    cat_upper = category.strip().upper().replace(" ", "_")
    vert = VerticalDomain(vertical) if isinstance(vertical, str) else vertical

    if vert == VerticalDomain.ELECTRONICS_TECH:
        allowed = ElectronicsCategory.values()
    elif vert == VerticalDomain.APPAREL_FASHION:
        allowed = ApparelCategory.values()
    else:
        raise ValueError(f"Unsupported vertical domain: {vertical}")

    if cat_upper in allowed:
        return cat_upper
    return "OTHER"
