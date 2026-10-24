from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from productlens.schemas import AspectMention

_COMMON_PRODUCT_ASPECTS: Set[str] = {
    # Tech
    "battery", "battery life", "screen", "display", "keyboard", "trackpad",
    "touchpad", "sound", "speakers", "ports", "processor", "ram", "memory",
    "ssd", "storage", "camera", "webcam", "hinge", "performance", "speed",
    "graphics", "charger", "cord", "fan", "heat", "build quality",
    # Apparel
    "fabric", "material", "fit", "sizing", "size", "comfort", "stitching",
    "color", "collar", "sleeve", "seam", "zipper", "wash", "cut", "length",
    # Home & Kitchen
    "picture quality", "resolution", "remote", "hdmi", "volume", "suction",
    "controls", "design", "installation", "setup", "cleaning", "noise",
    # General
    "customer service", "shipping", "packaging", "delivery", "instructions",
}


class AspectExtractor:
    """
    Extracts high-precision aspect mentions from product review text
    using vocabulary-guided gazetteer scanning with boundary safety.
    """

    def __init__(
        self,
        known_aspects: Optional[Set[str]] = None,
        min_length: int = 2,
        max_length: int = 35,
    ):
        self.known_aspects = known_aspects or _COMMON_PRODUCT_ASPECTS
        self.min_length = min_length
        self.max_length = max_length

    def extract_aspects(self, text: str) -> List[AspectMention]:
        """
        Extracts candidate aspects with character spans from the review text.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        candidates: List[AspectMention] = []
        seen_spans: Set[Tuple[int, int]] = set()

        # Exact matches against domain gazetteer (high precision)
        lower_text = clean_text.lower()
        for aspect in sorted(self.known_aspects, key=len, reverse=True):
            for match in re.finditer(r"\b" + re.escape(aspect) + r"\b", lower_text):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    overlap = any(s < span[1] and span[0] < e for s, e in seen_spans)
                    if not overlap:
                        seen_spans.add(span)
                        candidates.append(
                            AspectMention(
                                aspect=clean_text[match.start() : match.end()],
                                start_char=match.start(),
                                end_char=match.end(),
                                confidence=0.95,
                            )
                        )

        # Sort candidates chronologically by start character
        candidates.sort(key=lambda x: x.start_char)
        return candidates
