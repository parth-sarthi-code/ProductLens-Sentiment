from __future__ import annotations

import time
from typing import List, Optional, Union

from productlens.schemas import (
    AspectMention,
    AspectSentimentResult,
    ReviewAnalysisReport,
    BatchAnalysisReport,
)
from productlens.aspects.extractor import AspectExtractor
from productlens.sentiment.pipeline import ProductLensSentimentPipeline
from productlens.utils import find_aspect_span


def extract_evidence_snippet(text: str, start_char: int, end_char: int, max_length: int = 160) -> str:
    """
    Self-extracts the surrounding sentence, line, or clause containing the target aspect.
    """
    delimiters = [".", "!", "?", "\n", ";"]

    # Locate sentence left boundary
    left = 0
    for d in delimiters:
        pos = text.rfind(d, 0, start_char)
        if pos != -1 and pos + 1 > left:
            left = pos + 1

    # Locate sentence right boundary
    right = len(text)
    for d in delimiters:
        pos = text.find(d, end_char)
        if pos != -1 and pos + 1 < right:
            right = pos + 1

    snippet = text[left:right].strip()

    # Fallback to local window if snippet is empty or excessively long
    if not snippet:
        snippet = text[max(0, start_char - 40) : min(len(text), end_char + 40)].strip()
    elif len(snippet) > max_length:
        s_local = max(0, start_char - 60)
        e_local = min(len(text), end_char + 60)
        prefix = "..." if s_local > 0 else ""
        suffix = "..." if e_local < len(text) else ""
        snippet = f"{prefix}{text[s_local:e_local].strip()}{suffix}"

    return snippet


class ProductLensEngine:
    """
    Unified Production Facade for ProductLens.
    Coordinates aspect extraction, evidence snippet extraction,
    and aspect-targeted sentiment analysis using a single unified calibrated model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path, dict]] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.extractor = AspectExtractor()
        self.pipeline = ProductLensSentimentPipeline(
            checkpoint_path=checkpoint_path,
            device=device,
            **kwargs,
        )

    def analyze_review(
        self,
        text: str,
        category: Optional[str] = None,
        aspects: Optional[List[str]] = None,
    ) -> ReviewAnalysisReport:
        """
        Analyzes a customer review text end-to-end, extracting aspect terms,
        evidence clauses, and routed sentiment predictions.
        """
        start_time = time.perf_counter()

        # Step 1: Aspect extraction (if not explicitly provided)
        if aspects:
            aspect_mentions: List[AspectMention] = []
            for asp in aspects:
                s_c, e_c = find_aspect_span(text, asp)
                aspect_mentions.append(AspectMention(aspect=asp, start_char=s_c, end_char=e_c, confidence=1.0))
        else:
            aspect_mentions = self.extractor.extract_aspects(text)

        # Fallback if no specific aspect detected
        if not aspect_mentions:
            aspect_mentions = [
                AspectMention(
                    aspect="general product experience",
                    start_char=0,
                    end_char=min(len(text), 30),
                    confidence=0.5,
                )
            ]

        # Step 2: Predict sentiment & extract evidence for each aspect
        results: List[AspectSentimentResult] = []
        pos_count, neg_count, neu_count = 0, 0, 0

        aspect_tuples = [(m.aspect, m.start_char, m.end_char) for m in aspect_mentions]
        predictions = self.pipeline.predict_aspects(
            text=text,
            aspects=aspect_tuples,
            category=category,
        )

        for mention, pred in zip(aspect_mentions, predictions):
            # Self-extract evidence snippet
            evidence = extract_evidence_snippet(text, mention.start_char, mention.end_char)

            if pred.sentiment == "positive":
                pos_count += 1
            elif pred.sentiment == "negative":
                neg_count += 1
            else:
                neu_count += 1

            results.append(
                AspectSentimentResult(
                    aspect=mention.aspect,
                    start_char=mention.start_char,
                    end_char=mention.end_char,
                    evidence_snippet=evidence,
                    sentiment=pred.sentiment,
                    confidence=pred.confidence,
                    probabilities=pred.probabilities,
                    routed_expert=pred.routed_expert,
                    routing_reason=pred.routing_reason,
                    aspect_mask_fallback=pred.aspect_mask_fallback,
                    calibrated=pred.calibrated,
                )
            )

        # Step 3: Compute overall summary sentiment
        total_polar = pos_count + neg_count
        if (
            pos_count > 0
            and neg_count > 0
            and total_polar > 0
            and abs(pos_count - neg_count) / total_polar <= 0.25
        ):
            summary = "mixed"
        elif pos_count >= neg_count and pos_count >= neu_count and pos_count > 0:
            summary = "positive"
        elif neg_count > pos_count and neg_count >= neu_count:
            summary = "negative"
        else:
            summary = "neutral"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ReviewAnalysisReport(
            review_text=text,
            category=category,
            aspect_count=len(results),
            results=results,
            summary_sentiment=summary,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def analyze_batch(
        self,
        reviews: List[Union[str, dict]],
        default_category: Optional[str] = None,
    ) -> BatchAnalysisReport:
        """
        Runs batch analysis on a list of reviews.
        """
        start_time = time.perf_counter()
        reports: List[ReviewAnalysisReport] = []
        total_aspects = 0

        for item in reviews:
            if isinstance(item, dict):
                text = item.get("text", "")
                cat = item.get("category", default_category)
                asps = item.get("aspects", None)
            else:
                text = str(item)
                cat = default_category
                asps = None

            report = self.analyze_review(text, category=cat, aspects=asps)
            reports.append(report)
            total_aspects += report.aspect_count

        elapsed_sec = time.perf_counter() - start_time
        throughput = len(reviews) / elapsed_sec if elapsed_sec > 0 else 0.0

        return BatchAnalysisReport(
            total_reviews=len(reviews),
            total_aspects=total_aspects,
            reports=reports,
            throughput_reviews_per_sec=round(throughput, 2),
        )
