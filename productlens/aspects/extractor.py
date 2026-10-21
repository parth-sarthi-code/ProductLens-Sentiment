from __future__ import annotations

import re
from typing import List, Set

from productlens.schemas import AspectMention

# Filter Vocabularies
_PRONOUNS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "nobody", "nothing", "none",
}

_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
}

_GENERIC_NOUNS: Set[str] = {
    "thing", "things", "stuff", "item", "items", "product", "products",
    "everything", "nothing", "something", "anything", "aspect", "aspects",
    "feature", "features", "lot", "bit", "way", "part", "parts", "side",
    "point", "points", "time", "times", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "one", "ones", "people", "person",
    "case", "cases", "use", "uses", "review", "reviews", "star", "stars",
    "purchase", "purchases", "buy", "order", "orders", "price", "money",
    "amount", "deal", "reason", "reasons", "problem", "problems", "issue",
}

_OPINION_WORDS: Set[str] = {
    "good", "bad", "great", "terrible", "awful", "excellent", "poor", "amazing",
    "horrible", "wonderful", "fantastic", "decent", "superb", "fine", "mediocre",
    "subpar", "outstanding", "impressive", "disappointing", "pleasing", "perfect",
    "unacceptable", "satisfactory", "unsatisfactory", "exceptional", "abysmal",
    "flawless", "dreadful", "gorgeous", "lovely", "happy", "unhappy",
}

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

# Regex pattern for candidate multi-word or single-word nouns
_NOUN_PHRASE_REGEX = re.compile(r"\b[A-Za-z]+(?:[- ][A-Za-z]+){0,2}\b")


class AspectExtractor:
    """
    Extracts high-precision aspect mentions from product review text.
    Combines vocabulary-guided gazetteer scanning, noun-phrase heuristics,
    and semantic noise filtration.
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

        # Step 1: Exact matches against domain gazetteer (high precision)
        lower_text = clean_text.lower()
        for aspect in sorted(self.known_aspects, key=len, reverse=True):
            for match in re.finditer(r"\b" + re.escape(aspect) + r"\b", lower_text):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    # Check overlap with existing spans
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

        # Step 2: Noun-phrase heuristics for discovering novel aspects
        for match in _NOUN_PHRASE_REGEX.finditer(clean_text):
            span = (match.start(), match.end())
            term = match.group(0).strip().lower()

            if len(term) < self.min_length or len(term) > self.max_length:
                continue

            words = term.split()
            # Filter if all words are stopwords, pronouns, generics, or opinion words
            if all(
                w in _STOPWORDS or w in _PRONOUNS or w in _GENERIC_NOUNS or w in _OPINION_WORDS
                for w in words
            ):
                continue

            # First and last words cannot be stopwords or pronouns
            if words[0] in _STOPWORDS or words[0] in _PRONOUNS or words[-1] in _STOPWORDS:
                continue

            # Check overlap with already extracted spans
            overlap = any(s < span[1] and span[0] < e for s, e in seen_spans)
            if not overlap and term in _COMMON_PRODUCT_ASPECTS:
                seen_spans.add(span)
                candidates.append(
                    AspectMention(
                        aspect=clean_text[match.start() : match.end()],
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.85,
                    )
                )

        # Sort candidates chronologically by start character
        candidates.sort(key=lambda x: x.start_char)
        return candidates
