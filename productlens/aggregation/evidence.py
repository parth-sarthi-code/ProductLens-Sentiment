"""
Evidence Selection and Deduplication.
Extracts 2-3 high-confidence, non-duplicate verbatim quotes per sentiment polarity.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from productlens.schemas.enums import Sentiment
from productlens.schemas.quads import AspectQuad
from productlens.aggregation.schemas import EvidenceSnippet


def compute_string_jaccard(s1: str, s2: str) -> float:
    """Computes word-level Jaccard similarity between two quotes."""
    w1 = set(re.findall(r"\w+", s1.lower()))
    w2 = set(re.findall(r"\w+", s2.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


class EvidenceSelector:
    """Selects top non-duplicate evidence quotes for a feature."""

    def __init__(self, max_quotes_per_sentiment: int = 3, similarity_threshold: float = 0.60):
        self.max_quotes = max_quotes_per_sentiment
        self.similarity_threshold = similarity_threshold

    def select_evidence(
        self,
        quad_tuples: List[Tuple[AspectQuad, str]],  # (quad, review_id)
    ) -> Dict[str, List[EvidenceSnippet]]:
        """
        Groups evidence by sentiment, sorts by confidence/length, and deduplicates.
        """
        candidates: Dict[str, List[EvidenceSnippet]] = {
            "positive": [],
            "negative": [],
            "neutral": [],
            "mixed": [],
        }

        for q, rev_id in quad_tuples:
            ev_text = (q.evidence or "").strip()
            if not ev_text:
                continue
            conf = float(q.confidence) if q.confidence is not None else 1.0
            sent_str = q.sentiment.value if isinstance(q.sentiment, Sentiment) else str(q.sentiment)
            if sent_str in candidates:
                candidates[sent_str].append(
                    EvidenceSnippet(
                        review_id=rev_id,
                        evidence=ev_text,
                        sentiment=sent_str,
                        confidence=conf,
                    )
                )

        selected: Dict[str, List[EvidenceSnippet]] = {}

        for sent_str, snippets in candidates.items():
            if not snippets:
                continue

            # Sort by confidence descending, then by length (prefer complete informative clauses)
            snippets.sort(key=lambda s: (s.confidence, len(s.evidence)), reverse=True)

            kept: List[EvidenceSnippet] = []
            for candidate in snippets:
                # Check for redundancy against already selected quotes
                is_duplicate = False
                c_text = candidate.evidence.lower()
                for existing in kept:
                    e_text = existing.evidence.lower()
                    # Check substring containment or high Jaccard similarity
                    if c_text in e_text or e_text in c_text:
                        is_duplicate = True
                        break
                    if compute_string_jaccard(c_text, e_text) >= self.similarity_threshold:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    kept.append(candidate)
                    if len(kept) >= self.max_quotes:
                        break

            if kept:
                selected[sent_str] = kept

        return selected
