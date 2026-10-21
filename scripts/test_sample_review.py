#!/usr/bin/env python3
"""
Evaluates all models on a real-world multi-aspect gaming laptop review.
Compares:
1. Generic Foundation Model (MAMS)
2. Tech Specialist (Laptop14)
3. Home Specialist (Television)
4. Apparel Specialist (T-Shirt)
5. ProductLens MoE Unified Pipeline
"""

import json
from pathlib import Path
from productlens.engine import ProductLensEngine
from productlens.sentiment.pipeline import ProductLensMoEPipeline

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
    engine = ProductLensEngine(lazy_load=True)
    pipeline = engine.pipeline

    # 1. Run full auto-extraction analysis via ProductLensEngine
    print("=== 1. Running ProductLensEngine Auto-Extraction & MoE Analysis ===")
    auto_report = engine.analyze_review(text=REVIEW_TEXT, category="Electronics")
    print(f"Extracted {auto_report.aspect_count} aspects automatically.")
    print(f"Summary Sentiment: {auto_report.summary_sentiment} (Execution: {auto_report.execution_time_ms}ms)")

    # 2. Run multi-model comparison across target aspects
    models_to_test = {
        "Generic Foundation (MAMS)": "fallback",
        "Tech Specialist (Laptop14)": "tech",
        "Home Specialist (Television)": "home",
        "Apparel Specialist (T-Shirt)": "apparel",
        "ProductLens MoE (Auto Routed)": None,
    }

    comparison_results = []

    for item in TARGET_ASPECTS:
        asp = item["aspect"]
        expected = item["expected_human"]

        row = {
            "aspect": asp,
            "expected_human": expected,
            "predictions": {},
        }

        for model_label, force_key in models_to_test.items():
            pred = pipeline.predict(
                text=REVIEW_TEXT,
                aspect=asp,
                category="Electronics",
                force_expert=force_key,
            )
            row["predictions"][model_label] = {
                "sentiment": pred.sentiment,
                "confidence": round(pred.confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in pred.probabilities.items()},
                "routed_expert": pred.routed_expert,
                "routing_reason": pred.routing_reason,
            }

        comparison_results.append(row)

    sample_output = {
        "review_text": REVIEW_TEXT,
        "auto_extracted_aspects": [r.model_dump() for r in auto_report.results],
        "summary_sentiment": auto_report.summary_sentiment,
        "target_aspect_comparison": comparison_results,
    }

    # Save JSON sample
    json_path = Path("/home/parth/ProductLens-Sentiment/samples/laptop_review_benchmark_sample.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Real-World Multi-Aspect Review Benchmark Sample",
        "",
        "### Review Subject: Gaming Laptop Multi-Aspect Evaluation",
        "",
        "| Target Aspect | Human Ground Truth | Generic Foundation (MAMS) | Tech Specialist (Laptop14) | Home Specialist | Apparel Specialist | ProductLens MoE (Routed) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in comparison_results:
        asp = item["aspect"]
        exp = item["expected_human"]
        gen = item["predictions"]["Generic Foundation (MAMS)"]
        tech = item["predictions"]["Tech Specialist (Laptop14)"]
        home = item["predictions"]["Home Specialist (Television)"]
        app = item["predictions"]["Apparel Specialist (T-Shirt)"]
        moe = item["predictions"]["ProductLens MoE (Auto Routed)"]

        def fmt(p):
            icon = "🟢" if p["sentiment"] == "positive" else ("🔴" if p["sentiment"] == "negative" else "⚪")
            return f"{icon} **{p['sentiment']}** ({p['confidence'] * 100:.1f}%)"

        md_lines.append(
            f"| **{asp}** | **{exp.upper()}** | {fmt(gen)} | {fmt(tech)} | {fmt(home)} | {fmt(app)} | {fmt(moe)} |"
        )

    md_lines.extend([
        "",
        "### Auto-Extracted Aspects by ProductLensEngine:",
        "",
        "| Extracted Aspect Span | Predicted Sentiment | Confidence | Routed Specialist |",
        "| :--- | :---: | :---: | :--- |",
    ])

    for res in auto_report.results:
        icon = "🟢" if res.sentiment == "positive" else ("🔴" if res.sentiment == "negative" else "⚪")
        md_lines.append(
            f"| `{res.aspect}` | {icon} **{res.sentiment}** | {res.confidence * 100:.1f}% | `{res.routed_expert}` ({res.routing_reason}) |"
        )

    md_lines.append(f"\n**Aggregated Summary Sentiment**: **{auto_report.summary_sentiment.upper()}**")

    md_path = Path("/home/parth/ProductLens-Sentiment/samples/laptop_review_benchmark_sample.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved sample JSON to: {json_path}")
    print(f"Saved sample Markdown to: {md_path}")
    print("\n" + "\n".join(md_lines[:20]))


if __name__ == "__main__":
    main()
