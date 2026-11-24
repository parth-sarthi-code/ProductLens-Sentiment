"""
Stage 1 Evaluation Runner for ProductLens 3.0.
Executes benchmarks against gold set, disaggregating English vs. Hinglish and hard cases.
Supports explicit multiset matching and implicit category tracking.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional
from productlens.gold.schema import GoldReview
from productlens.gold.manager import GoldSetManager
from productlens.schemas.quads import AspectQuad
from productlens.evaluation.metrics import (
    compute_explicit_aspect_spans_f1,
    compute_implicit_category_f1,
    compute_sentiment_macro_f1,
    compute_expected_calibration_error,
    compute_parse_and_offset_rates,
)


class Stage1Evaluator:
    """Evaluates ABSA systems and compact parses on the Stage 1 Gold Set."""

    def __init__(self, reviews: Optional[List[GoldReview]] = None):
        self.reviews = reviews if reviews is not None else GoldSetManager.load_reviews()

    def evaluate_predictions(
        self,
        predict_fn: Callable[[GoldReview], List[AspectQuad]],
    ) -> Dict[str, any]:
        """
        Runs prediction function over the gold set and calculates comprehensive Stage 1 metrics.
        """
        start_time = time.perf_counter()

        gold_quads_all: List[List[AspectQuad]] = []
        pred_quads_all: List[List[AspectQuad]] = []

        # Subsets for disaggregated evaluation
        lang_groups: Dict[str, Dict[str, List[List[AspectQuad]]]] = {
            "en": {"gold": [], "pred": []},
            "hi-en": {"gold": [], "pred": []},
        }
        hard_case_groups: Dict[str, Dict[str, List[List[AspectQuad]]]] = {
            "hard": {"gold": [], "pred": []},
            "standard": {"gold": [], "pred": []},
        }

        confidences: List[float] = []
        correctness: List[int] = []

        for r in self.reviews:
            gold_quads = r.quads
            pred_quads = predict_fn(r)

            gold_quads_all.append(gold_quads)
            pred_quads_all.append(pred_quads)

            # Route by language
            lang_key = r.language if r.language in lang_groups else "en"
            lang_groups[lang_key]["gold"].append(gold_quads)
            lang_groups[lang_key]["pred"].append(pred_quads)

            # Route by hard case
            hard_key = "hard" if r.is_hard_case else "standard"
            hard_case_groups[hard_key]["gold"].append(gold_quads)
            hard_case_groups[hard_key]["pred"].append(pred_quads)

            # Calibration data: collect on matched explicit spans and implicit categories
            pred_explicit_pool = [q for q in pred_quads if not q.is_implicit and q.aspect is not None]
            for gq in [q for q in gold_quads if not q.is_implicit and q.aspect is not None]:
                g_key = (gq.aspect.lower().strip(), gq.aspect_start, gq.aspect_end)
                for idx, pq in enumerate(pred_explicit_pool):
                    p_key = (pq.aspect.lower().strip(), pq.aspect_start, pq.aspect_end)
                    if g_key == p_key:
                        confidences.append(pq.confidence)
                        correctness.append(1 if pq.sentiment == gq.sentiment else 0)
                        pred_explicit_pool.pop(idx)
                        break

            pred_implicit_pool = [q for q in pred_quads if q.is_implicit]
            for gq in [q for q in gold_quads if q.is_implicit]:
                for idx, pq in enumerate(pred_implicit_pool):
                    if gq.category == pq.category:
                        confidences.append(pq.confidence)
                        correctness.append(1 if pq.sentiment == gq.sentiment else 0)
                        pred_implicit_pool.pop(idx)
                        break

        elapsed = time.perf_counter() - start_time
        throughput_sec_per_500 = (elapsed / len(self.reviews)) * 500 if self.reviews else 0.0

        # Helper to compute composite bundle for any group
        def evaluate_group(gold_list: List[List[AspectQuad]], pred_list: List[List[AspectQuad]]) -> Dict[str, any]:
            exp_metrics = compute_explicit_aspect_spans_f1(gold_list, pred_list)
            imp_metrics = compute_implicit_category_f1(gold_list, pred_list)
            sent_metrics = compute_sentiment_macro_f1(gold_list, pred_list)
            return {
                **exp_metrics,
                **imp_metrics,
                **sent_metrics,
                "total_matched_quads": exp_metrics["explicit_matched_count"] + imp_metrics["implicit_matched_count"],
            }

        overall_metrics = evaluate_group(gold_quads_all, pred_quads_all)
        ece = compute_expected_calibration_error(confidences, correctness)

        en_metrics = evaluate_group(lang_groups["en"]["gold"], lang_groups["en"]["pred"])
        hi_metrics = evaluate_group(lang_groups["hi-en"]["gold"], lang_groups["hi-en"]["pred"])
        hard_metrics = evaluate_group(hard_case_groups["hard"]["gold"], hard_case_groups["hard"]["pred"])

        return {
            "total_reviews": len(self.reviews),
            "total_matched_quads": overall_metrics["total_matched_quads"],
            "explicit_matched_count": overall_metrics["explicit_matched_count"],
            "implicit_matched_count": overall_metrics["implicit_matched_count"],
            "overall": {
                **overall_metrics,
                "expected_calibration_error": ece,
            },
            "english": en_metrics,
            "hinglish": hi_metrics,
            "hard_cases": hard_metrics,
            "latency": {
                "elapsed_seconds": round(elapsed, 4),
                "seconds_per_500_reviews": round(throughput_sec_per_500, 4),
            },
        }
