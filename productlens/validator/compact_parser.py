"""
Compact format parser and validator for ProductLens 3.0.
Handles decoding: aspect|category|sentiment|confidence|evidence
"""

from __future__ import annotations
from typing import List, Tuple
from productlens.schemas.enums import VerticalDomain
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator


class CompactFormatParser:
    """
    Parses compact quad lines and passes them through deterministic validation.
    """

    @classmethod
    def parse_line(cls, line: str) -> Tuple[Optional[RawCompactQuad], Optional[str]]:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            return None, "Empty or comment line"
        try:
            raw = RawCompactQuad.from_compact_line(cleaned)
            return raw, None
        except Exception as e:
            return None, str(e)

    @classmethod
    def parse_and_validate_review(
        cls,
        text: str,
        compact_output: str,
        vertical: VerticalDomain | str,
    ) -> Tuple[List[AspectQuad], int, int]:
        """
        Parses multi-line compact output for a single review text.
        Returns: (validated_quads, total_candidate_lines, parse_success_count)
        """
        lines = [line.strip() for line in compact_output.strip().split("\n") if line.strip()]
        total_candidates = len(lines)
        validated_quads: List[AspectQuad] = []
        parse_success_count = 0

        for line in lines:
            raw, err = cls.parse_line(line)
            if raw is None:
                continue
            parse_success_count += 1

            quad, val_err = SubstringValidator.validate_quad(text, raw, vertical)
            if quad is not None:
                validated_quads.append(quad)

        return validated_quads, total_candidates, parse_success_count
