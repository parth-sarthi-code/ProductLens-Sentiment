"""
Deterministic Substring Locator and Offset Calculator for ProductLens 3.0.
Solves the offset mismatch / hallucination bug by strictly computing start_char and end_char in code.
Includes multi-tier normalized snapping and clause alignment.
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
from productlens.schemas.enums import VerticalDomain, normalize_category
from productlens.schemas.quads import AspectQuad, RawCompactQuad


def find_all_occurrences(text: str, sub: str, case_sensitive: bool = True) -> List[Tuple[int, int]]:
    """
    Finds all [start, end) character spans of sub in text.
    """
    if not sub:
        return []

    spans: List[Tuple[int, int]] = []
    if case_sensitive:
        start = 0
        while True:
            idx = text.find(sub, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(sub)))
            start = idx + 1
    else:
        pattern = re.escape(sub)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            spans.append((match.start(), match.end()))

    return spans


def find_normalized_spans(text: str, query: str, is_evidence: bool = False) -> List[Tuple[int, int]]:
    """
    Finds character spans of query in text with multi-tier deterministic fallback:
    1. Exact match (case-sensitive)
    2. Case-insensitive match
    3. Punctuation/quote-stripped match: strips edge quotes, brackets, and terminal punctuation
    4. Collapsed whitespace match: regex matching allowing multiple spaces/newlines
    5. Clause-level substring alignment (if is_evidence=True and evidence spans multiple clauses)
    """
    if not query:
        return []

    # Tier 1: Exact case-sensitive
    spans = find_all_occurrences(text, query, case_sensitive=True)
    if spans:
        return spans

    # Tier 2: Case-insensitive
    spans = find_all_occurrences(text, query, case_sensitive=False)
    if spans:
        return spans

    # Tier 3: Edge punctuation / quote stripping
    clean_query = query.strip(" \t\n\r\"'.,;:!?-–—()[]{}*`")
    if clean_query:
        spans = find_all_occurrences(text, clean_query, case_sensitive=False)
        if spans:
            return spans

    # Tier 4: Collapsed whitespace matching
    if clean_query:
        words = [re.escape(w) for w in clean_query.split() if w]
        if len(words) > 1:
            pattern = r"\s+".join(words)
            spans = [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
            if spans:
                return spans

    # Tier 5: Clause-level alignment for evidence
    if is_evidence and clean_query:
        # Split on clause punctuation (., ;, ,)
        clauses = re.split(r"[,;.]\s*", clean_query)
        for clause in clauses:
            c_clean = clause.strip(" \t\n\r\"'.,;:!?-–—()[]{}*`")
            words = [re.escape(w) for w in c_clean.split() if w]
            if len(words) >= 2:
                pattern = r"\s+".join(words)
                spans = [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
                if spans:
                    return spans

    return []


CATEGORY_KEYWORDS = {
    "DISPLAY": ["display", "screen", "panel", "oled", "brightness", "refresh rate", "resolution", "pixel"],
    "BATTERY": ["battery", "charger", "charging", "battery backup", "backup", "power"],
    "SOUND": ["sound", "audio", "speaker", "speakers", "volume", "bass", "mic", "microphone"],
    "PERFORMANCE": ["performance", "processor", "speed", "lag", "cpu", "gpu", "ssd", "ram", "thermal", "heating"],
    "BUILD_QUALITY": ["build", "quality", "construction", "body", "hinge", "chassis", "material", "finish"],
    "CONNECTIVITY": ["connectivity", "bluetooth", "wifi", "ports", "port", "usb", "hdmi", "headphone jack"],
    "SOFTWARE": ["software", "os", "app", "firmware", "windows", "driver", "ui"],
    "DELIVERY_INSTALLATION": ["delivery", "installation", "shipping", "courier", "package"],
    "VALUE": ["price", "value", "money", "worth", "cost", "pricing"],
    "FABRIC_MATERIAL": ["fabric", "material", "cloth", "cotton", "polyester", "wool", "silk"],
    "FIT_SIZING": ["sizing", "size", "fit", "fitting", "cut"],
    "COLOR_PRINT": ["color", "colour", "print", "shade", "dye", "pattern"],
    "STITCHING_DURABILITY": ["stitching", "stitch", "durability", "seam", "thread", "dhaaga"],
    "COMFORT": ["comfort", "comfortable", "feel"],
    "DELIVERY": ["delivery", "shipping", "courier", "package"],
}


class SubstringValidator:
    """
    Deterministic validator that resolves exact character spans for aspect and evidence.
    Applies multi-tier normalized snapping, clause alignment, and enum-fallback before hard rejection.
    """

    @classmethod
    def validate_quad(
        cls,
        text: str,
        raw: RawCompactQuad,
        vertical: VerticalDomain | str,
        case_sensitive: bool = True,
    ) -> Tuple[Optional[AspectQuad], Optional[str]]:
        """
        Locates evidence and aspect within text, validates containment,
        and constructs a fully verified AspectQuad.
        Returns: (AspectQuad, None) on success or (None, error_str) on failure.
        """
        normalized_category = normalize_category(raw.category, vertical)

        # 0. Enum-to-Noun Fallback: If raw.aspect is an uppercase category tag not present in text,
        # extract the matching aspect noun from evidence or treat as implicit
        aspect_query = raw.aspect
        if aspect_query and aspect_query.isupper() and not find_normalized_spans(text, aspect_query):
            cat_key = str(raw.category).upper()
            found_kw = None
            ev_text = raw.evidence or ""
            for kw in CATEGORY_KEYWORDS.get(cat_key, []):
                if re.search(r"\b" + re.escape(kw) + r"\b", ev_text, re.IGNORECASE):
                    found_kw = kw
                    break
            if found_kw:
                aspect_query = found_kw
            elif cat_key in ["VALUE", "COMFORT"]:
                aspect_query = None

        # 1. Locate Evidence (exact, case-insensitive, edge-stripped, or clause-aligned)
        if case_sensitive:
            evidence_spans = find_all_occurrences(text, raw.evidence, case_sensitive=True)
            if not evidence_spans:
                evidence_spans = find_normalized_spans(text, raw.evidence, is_evidence=True)
        else:
            evidence_spans = find_normalized_spans(text, raw.evidence, is_evidence=True)

        if not evidence_spans:
            return None, f"Evidence '{raw.evidence}' not found in review text."

        # 2. Implicit Aspect case (aspect is None)
        if aspect_query is None:
            best_ev_start, best_ev_end = evidence_spans[0]
            exact_evidence = text[best_ev_start:best_ev_end]
            quad = AspectQuad(
                aspect=None,
                category=normalized_category,
                sentiment=raw.sentiment,
                confidence=raw.confidence,
                evidence=exact_evidence,
                aspect_start=None,
                aspect_end=None,
                evidence_start=best_ev_start,
                evidence_end=best_ev_end,
            )
            return quad, None

        # 3. Explicit Aspect case: locate aspect within evidence span
        if case_sensitive:
            aspect_spans = find_all_occurrences(text, aspect_query, case_sensitive=True)
            if not aspect_spans:
                aspect_spans = find_normalized_spans(text, aspect_query, is_evidence=False)
        else:
            aspect_spans = find_normalized_spans(text, aspect_query, is_evidence=False)

        if not aspect_spans:
            return None, f"Aspect '{aspect_query}' not found in review text."

        # Disambiguate: Find pair where aspect is strictly inside evidence span
        matched_pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        for ev_s, ev_e in evidence_spans:
            for asp_s, asp_e in aspect_spans:
                if ev_s <= asp_s and asp_e <= ev_e:
                    matched_pair = ((asp_s, asp_e), (ev_s, ev_e))
                    break
            if matched_pair:
                break

        # Fallback if aspect and evidence are distinct clauses in review text
        if matched_pair is None:
            # Pick first occurrence of each
            matched_pair = (aspect_spans[0], evidence_spans[0])

        (asp_start, asp_end), (ev_start, ev_end) = matched_pair
        exact_aspect = text[asp_start:asp_end]
        exact_evidence = text[ev_start:ev_end]

        try:
            quad = AspectQuad(
                aspect=exact_aspect,
                category=normalized_category,
                sentiment=raw.sentiment,
                confidence=raw.confidence,
                evidence=exact_evidence,
                aspect_start=asp_start,
                aspect_end=asp_end,
                evidence_start=ev_start,
                evidence_end=ev_end,
            )
            return quad, None
        except Exception as e:
            return None, f"Quad construction failed: {str(e)}"
