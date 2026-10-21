from __future__ import annotations

import time
from typing import List, Optional, Union
from pathlib import Path

from productlens.schemas import (
    AspectMention,
    AspectSentimentResult,
    ReviewAnalysisReport,
    BatchAnalysisReport,
)
from productlens.aspects.extractor import AspectExtractor
from productlens.sentiment.pipeline import ProductLensMoEPipeline


class ProductLensEngine:
    """
    Unified Production Facade for ProductLens.
    Coordinates aspect extraction, MoE routing, and aspect-targeted sentiment analysis.
    """

    def __init__(
        self,
        checkpoint_paths: Optional[dict] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
    ):
        self.extractor = AspectExtractor()
        self.pipeline = ProductLensMoEPipeline(
            checkpoint_paths=checkpoint_paths,
            device=device,
            lazy_load=lazy_load,
        )

    def analyze_review(
        self,
        text: str,
        category: Optional[str] = None,
        aspects: Optional[List[str]] = None,
    ) -> ReviewAnalysisReport:
        """
        Analyzes a customer review text end-to-end.

        Args:
            text: Raw customer review text.
            category: Optional Amazon/e-commerce category (e.g. 'Electronics', 'Clothing').
            aspects: Optional list of explicit aspect terms to evaluate (if pre-extracted).

        Returns:
            ReviewAnalysisReport with per-aspect sentiment, confidence, and expert routing.
        """
        start_time = time.perf_counter()

        # Step 1: Aspect extraction (if not explicitly provided)
        if aspects:
            aspect_mentions = [
                AspectMention(aspect=asp, start_char=0, end_char=len(asp), confidence=1.0)
                for asp in aspects
            ]
        else:
            aspect_mentions = self.extractor.extract_aspects(text)

        # Fallback if no specific aspect detected: evaluate the entire review as an overarching aspect
        if not aspect_mentions:
            aspect_mentions = [
                AspectMention(
                    aspect="general product experience",
                    start_char=0,
                    end_char=min(len(text), 30),
                    confidence=0.5,
                )
            ]

        # Step 2: Predict sentiment for each aspect
        results: List[AspectSentimentResult] = []
        pos_count, neg_count, neu_count = 0, 0, 0

        for mention in aspect_mentions:
            pred = self.pipeline.predict(
                text=text,
                aspect=mention.aspect,
                category=category,
                aspect_start=mention.start_char,
                aspect_end=mention.end_char,
            )

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
                    sentiment=pred.sentiment,
                    confidence=pred.confidence,
                    probabilities=pred.probabilities,
                    routed_expert=pred.routed_expert,
                    routing_reason=pred.routing_reason,
                )
            )

        # Step 3: Compute overall summary sentiment
        if pos_count > 0 and neg_count > 0:
            summary = "mixed"
        elif pos_count > neg_count and pos_count >= neu_count:
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
