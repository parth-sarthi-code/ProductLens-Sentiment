"""
Shared utilities for ProductLens-Sentiment.

Centralised here to avoid duplication between inference (engine.py, pipeline.py)
and any future modules that need the same helpers.
"""
from __future__ import annotations

import re
from typing import Tuple


def find_aspect_span(text: str, aspect: str) -> Tuple[int, int]:
    """
    Locates the character span of *aspect* within *text*.

    Strategy:
      1. Word-boundary regex match (case-insensitive) — highest precision.
      2. Plain substring search fallback for hyphenated or special-char terms.
      3. If the aspect is not found at all, returns (0, len(text)) so the
         whole text is used as the anchor, which is always safe.

    Returns:
        (start_char, end_char) — zero-indexed, end is exclusive.
    """
    match = re.search(r"\b" + re.escape(aspect) + r"\b", text, re.IGNORECASE)
    if match:
        return match.start(), match.end()

    idx = text.lower().find(aspect.lower())
    if idx != -1:
        return idx, idx + len(aspect)

    return 0, len(text)
