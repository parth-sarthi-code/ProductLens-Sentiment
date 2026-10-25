#!/usr/bin/env python3
"""
Evaluates ProductLens unified sentiment engine on a real-world multi-aspect gaming laptop review.
Demonstrates:
1. Automatic aspect extraction & self-extracted evidence snippets.
2. Aspect-targeted sentiment classification with calibrated probabilities.
3. Clean, high-throughput unified inference without MoE routing overhead.
"""

from pathlib import Path
import json
from productlens.engine import ProductLensEngine

BASE_DIR = Path(__file__).resolve().parent.parent

REVIEW_TEXT = """Good laptop at this price range

1.Gaming (8/10)
I played
Genshin (max graphics 60fps)
Elden ring (max + rtx off 60fps)
Dying light 1 (max+rtx on 70+fps)
Witcher 3 (max+rtx low 70-75 fps, max + rtx high 50-60 fps)
Valorant (max 144-200fps)
Overwatch 2 (max 70+fps)

2.Display is very great and video quality at peak beacuse of 2k res , but mine has more screen bleeding and i got one dead pixel (my fault because I didn't notice before).

3.speaker (7/10)
Average speaker

4.Heating (7/10)
Heat management is average
When I play games at high graphics it goes upto 90-95° sometimes at max fan speed. (Not a big deal,heating in normal in all gaming laptops)

5.keyboard and touch pad (4/10)
Tbh worst keyboard and touch pad at this price and low quality.
Now some keys are already started to not work.

MAIN THING

6.battery (4/10)
Worst battery life not even 90 minutes of playing or streaming videos. (Not a big problem because gaming laptops need active charging)

Conclusion : good laptop at this price (8/10)

*IF YOU ARE GOING TO BUY* Then don't forget to check if it has more screen bleeding. (All LCD displays have screen bleeding and it's common but buy which has the slightest screen bleeding)
Mine has more screen bleeding in all four sides but only noticeable in dark room"""

TARGET_ASPECTS = [
    {"aspect": "Gaming", "expected_human": "positive"},
    {"aspect": "Display", "expected_human": "positive"},
    {"aspect": "screen bleeding", "expected_human": "negative"},
    {"aspect": "dead pixel", "expected_human": "negative"},
    {"aspect": "speaker", "expected_human": "neutral"},
    {"aspect": "Heating", "expected_human": "neutral"},
    {"aspect": "keyboard", "expected_human": "negative"},
    {"aspect": "touch pad", "expected_human": "negative"},
    {"aspect": "battery", "expected_human": "negative"},
    {"aspect": "price", "expected_human": "positive"},
]


def main():
    engine = ProductLensEngine()

    print("=== 1. Running ProductLensEngine Auto-Extraction & Unified Analysis ===")
    auto_report = engine.analyze_review(text=REVIEW_TEXT)
    print(f"Extracted {auto_report.aspect_count} aspects automatically.")
    print(f"Summary Sentiment: {auto_report.summary_sentiment} (Execution: {auto_report.execution_time_ms:.1f}ms)")

    target_results = []
    print("\n=== 2. Evaluating Specific Target Aspects ===")
    for item in TARGET_ASPECTS:
        asp = item["aspect"]
        expected = item["expected_human"]
        pred = engine.pipeline.predict(text=REVIEW_TEXT, aspect=asp)
        target_results.append({
            "aspect": asp,
            "expected_human": expected,
            "predicted_sentiment": pred.sentiment,
            "confidence": round(pred.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in pred.probabilities.items()},
        })
        print(f"• {asp:18} | Expected: {expected:8} | Pred: {pred.sentiment:8} ({pred.confidence:.1%})")

    samples_dir = BASE_DIR / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    json_path = samples_dir / "laptop_review_benchmark_sample.json"

    sample_output = {
        "review_text": REVIEW_TEXT,
        "auto_extracted_aspects": [r.model_dump() for r in auto_report.results],
        "summary_sentiment": auto_report.summary_sentiment,
        "target_aspect_evaluations": target_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)

    # Generate Markdown summary
    md_lines = [
        "# Real-World Multi-Aspect Review Analysis Sample",
        "",
        "### Review Subject: Gaming Laptop Multi-Aspect Evaluation",
        "",
        f"**Summary Sentiment**: **{auto_report.summary_sentiment.upper()}** (Processed in {auto_report.execution_time_ms:.1f}ms)",
        "",
        "| Target Aspect | Human Ground Truth | Predicted Sentiment | Calibrated Confidence |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for item in target_results:
        icon = "🟢" if item["predicted_sentiment"] == "positive" else ("🔴" if item["predicted_sentiment"] == "negative" else "⚪")
        md_lines.append(
            f"| **{item['aspect']}** | `{item['expected_human'].upper()}` | {icon} **{item['predicted_sentiment']}** | {item['confidence'] * 100:.1f}% |"
        )

    md_lines.extend([
        "",
        "### Auto-Extracted Aspects by ProductLensEngine:",
        "",
        "| Extracted Aspect Span | Predicted Sentiment | Confidence | Evidence Snippet |",
        "| :--- | :---: | :---: | :--- |",
    ])

    for res in auto_report.results:
        icon = "🟢" if res.sentiment == "positive" else ("🔴" if res.sentiment == "negative" else "⚪")
        md_lines.append(
            f"| `{res.aspect}` | {icon} **{res.sentiment}** | {res.confidence * 100:.1f}% | *\"{res.evidence_snippet}\"* |"
        )

    md_path = samples_dir / "laptop_review_benchmark_sample.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved sample JSON to: {json_path}")
    print(f"Saved sample Markdown to: {md_path}")


if __name__ == "__main__":
    main()
