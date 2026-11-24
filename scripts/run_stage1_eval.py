"""
Stage 1 End-to-End Verification and Evaluation Harness CLI.
Verifies contract, deterministic offsets, compact parsing, benchmarks, and gold metrics.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from productlens.gold.manager import GoldSetManager
from productlens.validator.compact_parser import CompactFormatParser
from productlens.benchmarks.loader import BenchmarkLoader
from productlens.evaluation.runner import Stage1Evaluator


def main():
    print("=" * 72)
    print("🎯 ProductLens 3.0: Stage 1 Contract and Foundations Verification")
    print("=" * 72)

    # 1. Load Gold Set and Manifest
    reviews = GoldSetManager.load_reviews()
    manifest = GoldSetManager.get_manifest(reviews)
    explicit_count = sum(sum(1 for q in r.quads if not q.is_implicit) for r in reviews)
    implicit_count = sum(sum(1 for q in r.quads if q.is_implicit) for r in reviews)

    print(f"\n[1] Gold Set Loaded:")
    print(f"    • Total Reviews:          {manifest.total_reviews}")
    print(f"    • English Reviews:        {manifest.english_reviews}")
    print(f"    • Hinglish Reviews:       {manifest.hinglish_reviews}")
    print(f"    • Hard Cases:             {manifest.hard_cases}")
    print(f"    • Unique Products:        {manifest.products_count}")
    print(f"    • Total Aspect Quads:     {manifest.total_quads}")
    print(f"      - Explicit Quads:       {explicit_count}")
    print(f"      - Implicit Quads:       {implicit_count}")

    # 2. Verify 100% Character Offset Substring Validity
    print(f"\n[2] Verifying Substring Offset Integrity...")
    offset_errors = 0
    total_quads_checked = 0

    for r in reviews:
        for q in r.quads:
            total_quads_checked += 1
            # Check evidence span
            ev_slice = r.text[q.evidence_start : q.evidence_end]
            if ev_slice != q.evidence:
                offset_errors += 1
                print(f"❌ Offset mismatch on evidence in review {r.review_id}: expected '{q.evidence}', got '{ev_slice}'")

            # Check aspect span
            if q.aspect is not None:
                asp_slice = r.text[q.aspect_start : q.aspect_end]
                if asp_slice != q.aspect:
                    offset_errors += 1
                    print(f"❌ Offset mismatch on aspect in review {r.review_id}: expected '{q.aspect}', got '{asp_slice}'")

    if offset_errors == 0:
        print(f"    ✅ 100% Offset Validity! All {total_quads_checked} quads match exact review substrings.")
    else:
        print(f"    ❌ Found {offset_errors} offset errors out of {total_quads_checked} quads.")

    # 3. Test Compact Format Roundtrip & Parser
    print(f"\n[3] Testing Compact Format (aspect|category|sentiment|confidence|evidence)...")
    compact_lines = []
    for r in reviews[:50]:
        for q in r.quads:
            compact_lines.append(q.to_compact_line())

    compact_block = "\n".join(compact_lines)
    parsed_quads, total_lines, parsed_count = CompactFormatParser.parse_and_validate_review(
        text=reviews[0].text,
        compact_output=compact_block,
        vertical=reviews[0].vertical,
    )
    print(f"    • Tested {len(compact_lines)} compact lines across sample reviews.")
    print(f"    • Parse Success: {parsed_count}/{len(compact_lines)} ({parsed_count/len(compact_lines)*100:.1f}%)")

    # 4. Verify Active Benchmark Datasets
    print(f"\n[4] Verifying Curated Benchmark Parquets...")
    benchmarks = BenchmarkLoader.load_all_active_splits()
    for name, splits in benchmarks.items():
        split_names = list(splits.keys())
        counts = {s: len(splits[s]) for s in split_names}
        print(f"    • {name:<16}: {counts}")

    # 5. Run Evaluation Harness on Gold Set
    print(f"\n[5] Running Stage 1 Evaluation Harness on Gold Set (Multiset + Implicit Category)...")
    evaluator = Stage1Evaluator(reviews)

    # Identity evaluation (oracle baseline)
    eval_results = evaluator.evaluate_predictions(lambda r: r.quads)

    ov = eval_results["overall"]
    print("\n" + "=" * 72)
    print("📊 Stage 1 Baseline Evaluation Results (Gold Oracle Baseline)")
    print("=" * 72)
    print(f"Total Reviews:                 {eval_results['total_reviews']}")
    print(f"Total Gold Quads:              {manifest.total_quads}")
    print(f"Explicit Matched Count:        {ov['explicit_matched_count']} / {ov['total_gold_explicit']}")
    print(f"Implicit Matched Count:        {ov['implicit_matched_count']} / {ov['total_gold_implicit']}")
    print(f"Total Matched Quads:           {ov['total_matched_quads']} / {manifest.total_quads} ({ov['total_matched_quads']/manifest.total_quads*100:.1f}%)")
    print(f"--- Core Metrics ---")
    print(f"Explicit Aspect F1:            {ov['aspect_f1'] * 100:.2f}% (P: {ov['aspect_precision']*100:.2f}%, R: {ov['aspect_recall']*100:.2f}%)")
    print(f"Implicit Category F1:          {ov['implicit_category_f1'] * 100:.2f}% (P: {ov['implicit_category_precision']*100:.2f}%, R: {ov['implicit_category_recall']*100:.2f}%)")
    print(f"Overall Sentiment Macro-F1:    {ov['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Expected Calib Error (ECE):    {ov['expected_calibration_error']:.4f}")
    print(f"--- Disaggregated Subsets ---")
    print(f"English Aspect F1:             {eval_results['english']['aspect_f1'] * 100:.2f}% | Sent F1: {eval_results['english']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Hinglish Aspect F1:            {eval_results['hinglish']['aspect_f1'] * 100:.2f}% | Sent F1: {eval_results['hinglish']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Hard Cases Aspect F1:          {eval_results['hard_cases']['aspect_f1'] * 100:.2f}% | Sent F1: {eval_results['hard_cases']['sentiment_macro_f1'] * 100:.2f}%")
    print(f"Latency (sec / 500 revs):      {eval_results['latency']['seconds_per_500_reviews']:.4f}s")
    print("=" * 72)

    # Assert 100% reconciliation
    assert ov['total_matched_quads'] == manifest.total_quads, (
        f"Discrepancy: {ov['total_matched_quads']} matched vs {manifest.total_quads} total!"
    )
    print(f"🎉 Stage 1 Exit Criteria Met Successfully! ({ov['explicit_matched_count']} Explicit + {ov['implicit_matched_count']} Implicit = {ov['total_matched_quads']} Quads Reconciled)\n")


if __name__ == "__main__":
    main()
