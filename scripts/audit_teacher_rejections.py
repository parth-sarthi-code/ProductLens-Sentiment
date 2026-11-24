#!/usr/bin/env python3
"""
ProductLens 3.0: Audit Teacher Validator Rejections.
Analyzes why raw candidate quads were rejected by SubstringValidator.
Categorizes rejections into:
  - evidence_not_found
  - aspect_not_found
  - containment_mismatch
  - invalid_category
  - offset_mismatch (passes validator but offsets differ from gold annotations)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import AspectQuad, RawCompactQuad
from productlens.validator.substring import SubstringValidator, find_all_occurrences
from productlens.gold.manager import GoldSetManager
from productlens.evaluation.metrics import compute_char_iou


def audit_predictions(predictions_path: Path):
    if not predictions_path.exists():
        print(f"❌ Predictions file not found at: {predictions_path}")
        return

    reviews = GoldSetManager.load_reviews()
    reviews_by_id = {r.review_id: r for r in reviews}

    total_reviews = 0
    total_raw_quads = 0
    total_valid_quads = 0
    rejections_by_reason = Counter()
    rejection_examples = []
    offset_mismatches = []

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            review_id = data["review_id"]
            review = reviews_by_id.get(review_id)
            if not review:
                continue
            total_reviews += 1

            raw_text = data.get("raw_text", "")
            try:
                parsed_json = json.loads(raw_text)
                raw_quads_data = parsed_json.get("quads", [])
            except Exception:
                raw_quads_data = []

            for rq in raw_quads_data:
                total_raw_quads += 1
                aspect = rq.get("aspect")
                evidence = rq.get("evidence", "")
                category = rq.get("category", "")
                sentiment_val = rq.get("sentiment", "neutral")
                confidence = rq.get("confidence", 1.0)

                try:
                    raw_c = RawCompactQuad(
                        aspect=aspect,
                        category=category,
                        sentiment=Sentiment(sentiment_val),
                        confidence=confidence,
                        evidence=evidence,
                    )
                except Exception as e:
                    rejections_by_reason["schema_validation_error"] += 1
                    rejection_examples.append({
                        "reason": "schema_validation_error",
                        "review_id": review_id,
                        "review_text": review.text,
                        "predicted_aspect": aspect,
                        "predicted_evidence": evidence,
                        "error": str(e),
                    })
                    continue

                v_quad, err = SubstringValidator.validate_quad(review.text, raw_c, review.vertical)
                if v_quad is not None:
                    total_valid_quads += 1
                    if not v_quad.is_implicit and v_quad.aspect is not None:
                        # Check if it matches gold exact offset
                        gold_exact_matches = [
                            g for g in review.quads
                            if not g.is_implicit and g.aspect is not None
                            and g.aspect.lower().strip() == v_quad.aspect.lower().strip()
                            and g.aspect_start == v_quad.aspect_start
                            and g.aspect_end == v_quad.aspect_end
                        ]
                        if not gold_exact_matches:
                            # Check if IoU >= 0.5 with any gold quad
                            for g in review.quads:
                                if not g.is_implicit and g.aspect is not None and g.aspect_start is not None:
                                    iou = compute_char_iou(
                                        (v_quad.aspect_start, v_quad.aspect_end),
                                        (g.aspect_start, g.aspect_end),
                                    )
                                    if iou > 0.0:
                                        offset_mismatches.append({
                                            "review_id": review_id,
                                            "review_text": review.text,
                                            "pred_aspect": v_quad.aspect,
                                            "pred_span": (v_quad.aspect_start, v_quad.aspect_end),
                                            "gold_aspect": g.aspect,
                                            "gold_span": (g.aspect_start, g.aspect_end),
                                            "iou": round(iou, 3),
                                        })
                else:
                    # Categorize rejection error
                    if not evidence or not find_all_occurrences(review.text, evidence, case_sensitive=False):
                        reason = "evidence_not_found"
                    elif aspect is not None and not find_all_occurrences(review.text, aspect, case_sensitive=False):
                        reason = "aspect_not_found"
                    elif "category" in str(err).lower():
                        reason = "invalid_category"
                    else:
                        reason = "containment_or_other_mismatch"

                    rejections_by_reason[reason] += 1
                    rejection_examples.append({
                        "reason": reason,
                        "review_id": review_id,
                        "review_text": review.text,
                        "predicted_aspect": aspect,
                        "predicted_evidence": evidence,
                        "error": err,
                    })

    print("=" * 72)
    print("🔍 ProductLens 3.0: Teacher Rejection & Error Audit Report")
    print("=" * 72)
    print(f"Total Reviews Audited:         {total_reviews}")
    print(f"Total Raw Quads Generated:     {total_raw_quads}")
    print(f"Total Validated Quads:         {total_valid_quads}")
    total_rejected = total_raw_quads - total_valid_quads
    pass_rate = (total_valid_quads / total_raw_quads * 100) if total_raw_quads > 0 else 0.0
    print(f"Validator Pass Rate:           {pass_rate:.2f}% ({total_valid_quads}/{total_raw_quads})")
    print(f"Total Rejected Quads:          {total_rejected}")
    print("\n--- Rejection Breakdown by Failure Mode ---")
    for reason, count in rejections_by_reason.most_common():
        pct = (count / total_rejected * 100) if total_rejected > 0 else 0.0
        print(f"  • {reason:<32} {count:>4} ({pct:>5.1f}%)")

    print(f"\n--- Offset Mismatches (Valid Substring, but IoU > 0 with Gold): {len(offset_mismatches)} ---")
    for m in offset_mismatches[:10]:
        print(f"  [{m['review_id']}] Pred: '{m['pred_aspect']}' {m['pred_span']} vs Gold: '{m['gold_aspect']}' {m['gold_span']} (IoU: {m['iou']})")
        print(f"    Text: \"{m['review_text']}\"")

    print("\n--- Top 20 Detailed Rejection Pairs ---")
    for idx, ex in enumerate(rejection_examples[:20], 1):
        print(f"\n[{idx}] Failure Mode: {ex['reason']}")
        print(f"    Review ID:   {ex['review_id']}")
        print(f"    Review Text: \"{ex['review_text']}\"")
        print(f"    Pred Aspect:   '{ex['predicted_aspect']}'")
        print(f"    Pred Evidence: '{ex['predicted_evidence']}'")
        print(f"    Error Note:    {ex['error']}")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Teacher Predictions Rejections")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "gold_eval" / "teacher_gold_predictions.jsonl",
        help="Path to teacher predictions JSONL file",
    )
    args = parser.parse_args()
    audit_predictions(args.predictions)
