from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

CATEGORY_TAXONOMY_MAP: Dict[str, List[str]] = {
    "tech": [
        "electronics", "computers", "laptops", "tech", "hardware",
        "cell_phones_and_accessories", "phones", "gadgets", "video_games"
    ],
    "home": [
        "home_and_kitchen", "home", "appliances", "television",
        "audio", "speakers", "furniture", "kitchen", "household"
    ],
    "apparel": [
        "clothing_shoes_and_jewelry", "clothing", "apparel", "shoes",
        "fashion", "t_shirts", "tshirt", "mens_clothing", "womens_clothing"
    ],
    "beauty": [
        "all_beauty", "beauty", "personal_care", "cosmetics", "skincare",
        "haircare", "fragrance"
    ],
}

# Keywords used for Level 2 lexical routing.
# Rules for inclusion:
#   - Minimum 4 characters to prevent substring false positives (e.g. "os", "ac", "tv" removed)
#   - Must be domain-unambiguous; removed: "os", "mac", "ram", "ports" (tech),
#     "fan", "air" (home — appear in non-home contexts), "fit" (apparel — too generic),
#     "tv" moved to taxonomy only (too short for word-boundary safety in aspect branch)
LEXICAL_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "tech": [
        "laptop", "screen", "keyboard", "battery", "processor",
        "cpu", "gpu", "memory", "hard drive", "ssd", "display",
        "usb", "windows", "trackpad", "touchpad", "resolution",
        "mouse", "charger", "cord", "software", "firmware",
        "bluetooth", "wifi", "speed", "storage", "graphics",
    ],
    "home": [
        "television", "remote", "hdmi", "speakers", "sound", "volume", "channels",
        "refrigerator", "fridge", "microwave", "oven", "stove", "dishwasher", "vacuum",
        "blender", "coffee", "toaster", "couch", "table", "chair",
    ],
    "apparel": [
        "shirt", "t-shirt", "fabric", "cotton", "material", "size", "sizing",
        "sleeve", "collar", "wash", "shrink", "color", "stitching", "seam", "pants",
        "jeans", "dress", "jacket", "shoes", "sneakers", "boots", "comfort", "wear",
    ],
    "beauty": [
        "lotion", "cream", "skin", "scent", "fragrance", "smell", "serum", "moisturizer",
        "lipstick", "mascara", "makeup", "shampoo", "conditioner", "face", "wrinkles",
    ],
}

# Minimum character length for a cleaned category string to be eligible for
# substring matching against synonyms at Level 1. Prevents short strings like
# "pc", "tv" from matching arbitrary synonyms.
_MIN_CAT_LEN_FOR_SUBSTRING = 4


class MoERouter:
    """
    Hierarchical Router for dispatching queries to domain experts or MAMS fallback.
    Level 1: Exact or substring metadata match on e-commerce department tag (O(1)).
    Level 2: Lexical keyword detection in aspect span and review text.
    Level 3: Universal fallback to MAMS Foundation Expert.
    """

    def __init__(
        self,
        taxonomy_map: Optional[Dict[str, List[str]]] = None,
        lexical_keywords: Optional[Dict[str, List[str]]] = None,
    ):
        self.taxonomy_map = taxonomy_map or CATEGORY_TAXONOMY_MAP
        self.lexical_keywords = lexical_keywords or LEXICAL_DOMAIN_KEYWORDS

    def route(
        self,
        category: Optional[str] = None,
        text: Optional[str] = None,
        aspect: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Determines the optimal expert key and returns rationale.

        Returns:
            (expert_key, routing_reason)
            expert_key in {'tech', 'home', 'apparel', 'beauty', 'fallback'}
        """
        # --- Level 1: Metadata Routing ---
        if category:
            cat_clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(category).lower().strip())
            for domain, synonyms in self.taxonomy_map.items():
                for syn in synonyms:
                    # Require minimum length before allowing substring match to avoid
                    # very short category strings matching unrelated synonyms.
                    if syn == cat_clean:
                        return domain, f"metadata_match:{domain}"
                    if (
                        len(cat_clean) >= _MIN_CAT_LEN_FOR_SUBSTRING
                        and len(syn) >= _MIN_CAT_LEN_FOR_SUBSTRING
                        and (syn in cat_clean or cat_clean in syn)
                    ):
                        return domain, f"metadata_match:{domain}"

        # --- Level 2: Lexical / Semantic Keyword Gating ---
        query_text = f"{aspect or ''} {text or ''}".lower()
        domain_scores = {d: 0 for d in self.taxonomy_map.keys()}

        for domain, keywords in self.lexical_keywords.items():
            for kw in keywords:
                # Use word-boundary regex for BOTH the aspect branch and the text branch
                # to prevent substring false positives (e.g. "os" inside "cost").
                kw_pattern = r"\b" + re.escape(kw) + r"\b"
                if aspect and re.search(kw_pattern, aspect.lower()):
                    domain_scores[domain] += 3
                elif re.search(kw_pattern, query_text):
                    domain_scores[domain] += 1

        best_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[best_domain] >= 2:
            return best_domain, f"lexical_match:{best_domain}(score={domain_scores[best_domain]})"

        # --- Level 3: Universal Fallback ---
        return "fallback", "mams_foundation_fallback"
