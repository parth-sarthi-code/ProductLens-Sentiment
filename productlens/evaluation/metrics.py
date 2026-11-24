"""
Evaluation metrics for ProductLens 3.0:
Explicit Aspect F1 (multiset Counter), Implicit Category F1, Sentiment Macro-F1 (explicit + implicit),
Offset Validity, Compact Parse Rate, ECE, and ARTS Accuracy.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score
from productlens.schemas.enums import Sentiment
from productlens.schemas.quads import AspectQuad


def compute_char_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """
    Computes Character-level Intersection over Union (IoU) between two [start, end) character spans.
    """
    start1, end1 = span1
    start2, end2 = span2
    inter = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return inter / union if union > 0 else 0.0


def compute_explicit_aspect_spans_f1(
    gold_quads_list: List[List[AspectQuad]],
    pred_quads_list: List[List[AspectQuad]],
    iou_threshold: float = 0.5,
) -> Dict[str, any]:
    """
    Computes Precision, Recall, and F1 for explicit aspects:
      1. Strict Exact Offset Matching: multiset matching by (aspect.lower().strip(), start, end)
      2. Overlap IoU Matching: greedy bipartite matching where character IoU >= iou_threshold
    """
    tp = 0
    fp = 0
    fn = 0
    overlap_tp = 0
    total_gold_explicit = 0
    total_pred_explicit = 0

    for gold_quads, pred_quads in zip(gold_quads_list, pred_quads_list):
        gold_explicit = [q for q in gold_quads if not q.is_implicit and q.aspect is not None]
        pred_explicit = [q for q in pred_quads if not q.is_implicit and q.aspect is not None]

        total_gold_explicit += len(gold_explicit)
        total_pred_explicit += len(pred_explicit)

        # 1. Strict exact offset matching
        gold_counts = Counter(
            (q.aspect.lower().strip(), q.aspect_start, q.aspect_end)
            for q in gold_explicit
        )
        pred_counts = Counter(
            (q.aspect.lower().strip(), q.aspect_start, q.aspect_end)
            for q in pred_explicit
        )

        matched_counts = gold_counts & pred_counts
        matched_in_review = sum(matched_counts.values())

        tp += matched_in_review
        fp += sum((pred_counts - gold_counts).values())
        fn += sum((gold_counts - pred_counts).values())

        # 2. Overlap IoU >= 0.5 greedy matching
        candidate_pairs = []
        for g_idx, gq in enumerate(gold_explicit):
            if gq.aspect_start is None or gq.aspect_end is None:
                continue
            for p_idx, pq in enumerate(pred_explicit):
                if pq.aspect_start is None or pq.aspect_end is None:
                    continue
                iou = compute_char_iou((gq.aspect_start, gq.aspect_end), (pq.aspect_start, pq.aspect_end))
                if iou >= iou_threshold:
                    candidate_pairs.append((iou, g_idx, p_idx))

        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        used_gold = set()
        used_pred = set()
        for iou, g_idx, p_idx in candidate_pairs:
            if g_idx not in used_gold and p_idx not in used_pred:
                used_gold.add(g_idx)
                used_pred.add(p_idx)
                overlap_tp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    overlap_prec = overlap_tp / total_pred_explicit if total_pred_explicit > 0 else 0.0
    overlap_rec = overlap_tp / total_gold_explicit if total_gold_explicit > 0 else 0.0
    overlap_f1 = (
        2 * overlap_prec * overlap_rec / (overlap_prec + overlap_rec)
        if (overlap_prec + overlap_rec) > 0
        else 0.0
    )

    return {
        "aspect_precision": round(precision, 4),
        "aspect_recall": round(recall, 4),
        "aspect_f1": round(f1, 4),
        "strict_aspect_f1": round(f1, 4),
        "overlap_aspect_precision": round(overlap_prec, 4),
        "overlap_aspect_recall": round(overlap_rec, 4),
        "overlap_aspect_f1": round(overlap_f1, 4),
        "explicit_matched_count": tp,
        "explicit_overlap_matched_count": overlap_tp,
        "total_gold_explicit": total_gold_explicit,
        "total_pred_explicit": total_pred_explicit,
    }


def compute_implicit_category_f1(
    gold_quads_list: List[List[AspectQuad]],
    pred_quads_list: List[List[AspectQuad]],
) -> Dict[str, any]:
    """
    Computes Precision, Recall, and F1 for implicit aspects (aspect: null)
    matched by canonical category.
    """
    tp = 0
    fp = 0
    fn = 0
    total_gold_implicit = 0
    total_pred_implicit = 0

    for gold_quads, pred_quads in zip(gold_quads_list, pred_quads_list):
        gold_implicit = [q for q in gold_quads if q.is_implicit]
        pred_implicit = [q for q in pred_quads if q.is_implicit]

        total_gold_implicit += len(gold_implicit)
        total_pred_implicit += len(pred_implicit)

        gold_counts = Counter(q.category for q in gold_implicit)
        pred_counts = Counter(q.category for q in pred_implicit)

        matched_counts = gold_counts & pred_counts
        matched_in_review = sum(matched_counts.values())

        tp += matched_in_review
        fp += sum((pred_counts - gold_counts).values())
        fn += sum((gold_counts - pred_counts).values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "implicit_category_precision": round(precision, 4),
        "implicit_category_recall": round(recall, 4),
        "implicit_category_f1": round(f1, 4),
        "implicit_matched_count": tp,
        "total_gold_implicit": total_gold_implicit,
        "total_pred_implicit": total_pred_implicit,
    }


def compute_sentiment_macro_f1(
    gold_quads_list: List[List[AspectQuad]],
    pred_quads_list: List[List[AspectQuad]],
) -> Dict[str, any]:
    """
    Computes Sentiment Macro-F1 across all matched aspect spans (both explicit and implicit)
    across all 4 classes: positive, negative, neutral, mixed.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    for gold_quads, pred_quads in zip(gold_quads_list, pred_quads_list):
        # 1. Match explicit aspects: multiset matching by (aspect.lower(), start, end)
        pred_explicit_pool = [q for q in pred_quads if not q.is_implicit and q.aspect is not None]
        for gq in [q for q in gold_quads if not q.is_implicit and q.aspect is not None]:
            g_key = (gq.aspect.lower().strip(), gq.aspect_start, gq.aspect_end)
            for idx, pq in enumerate(pred_explicit_pool):
                p_key = (pq.aspect.lower().strip(), pq.aspect_start, pq.aspect_end)
                if g_key == p_key:
                    y_true.append(gq.sentiment.value)
                    y_pred.append(pq.sentiment.value)
                    pred_explicit_pool.pop(idx)
                    break

        # 2. Match implicit aspects: match by category
        pred_implicit_pool = [q for q in pred_quads if q.is_implicit]
        for gq in [q for q in gold_quads if q.is_implicit]:
            for idx, pq in enumerate(pred_implicit_pool):
                if gq.category == pq.category:
                    y_true.append(gq.sentiment.value)
                    y_pred.append(pq.sentiment.value)
                    pred_implicit_pool.pop(idx)
                    break

    if not y_true:
        return {"sentiment_macro_f1": 0.0, "matched_aspect_count": 0}

    present_classes = sorted(list(set(y_true) | set(y_pred)))
    macro_f1 = f1_score(y_true, y_pred, labels=present_classes, average="macro", zero_division=0)

    # Per-class F1 for all defined sentiments
    all_classes = [s.value for s in Sentiment]
    per_class = f1_score(y_true, y_pred, labels=all_classes, average=None, zero_division=0)
    per_class_f1 = {f"f1_{cls_name}": round(float(score), 4) for cls_name, score in zip(all_classes, per_class)}

    res = {
        "sentiment_macro_f1": round(float(macro_f1), 4),
        "matched_aspect_count": len(y_true),
    }
    res.update(per_class_f1)
    return res


def compute_expected_calibration_error(
    confidences: List[float],
    correctness: List[int],
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    if not confidences or not correctness or len(confidences) != len(correctness):
        return 0.0

    conf_arr = np.array(confidences)
    corr_arr = np.array(correctness)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (conf_arr > bin_lower) & (conf_arr <= bin_upper) if i > 0 else (conf_arr >= bin_lower) & (conf_arr <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(conf_arr[in_bin])
            avg_accuracy = np.mean(corr_arr[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return round(float(ece), 4)


def compute_parse_and_offset_rates(
    total_raw_lines: int,
    parsed_lines: int,
    validated_quads: int,
) -> Dict[str, float]:
    """
    Calculates parse rate and deterministic offset validity rate.
    """
    parse_rate = (parsed_lines / total_raw_lines) if total_raw_lines > 0 else 1.0
    offset_validity_rate = (validated_quads / parsed_lines) if parsed_lines > 0 else 1.0

    return {
        "parse_rate": round(parse_rate, 4),
        "offset_validity_rate": round(offset_validity_rate, 4),
    }
