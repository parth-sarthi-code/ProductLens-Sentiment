#!/usr/bin/env python3
"""
Runs end-to-end evaluation with self-extracted evidence snippets on one sample from every benchmark database:
1. Tech Hardware (Laptop14)
2. Home Electronics (Television)
3. Apparel & Fashion (T-Shirt)
4. MAMS Challenge (Universal Multi-Aspect)
"""

import json
from pathlib import Path
from productlens.engine import ProductLensEngine

BENCHMARK_SAMPLES = [
    {
        "database": "Tech Hardware (Laptop14)",
        "domain_category": "Electronics",
        "text": "Boot time is super fast , around anywhere from 35 seconds to 1 minute . The keyboard is backlit and keys have good travel , but the touchpad feels sticky .",
        "target_aspects": ["Boot time", "keyboard", "touchpad"],
    },
    {
        "database": "Home Electronics (Television)",
        "domain_category": "Home_and_Kitchen",
        "text": "very good, cast screen working properly, best sound and picture quality. Remote sensor has slight lag though.",
        "target_aspects": ["cast screen", "sound", "picture quality", "Remote sensor"],
    },
    {
        "database": "Apparel & Fashion (T-Shirt)",
        "domain_category": "Clothing_Shoes_and_Jewelry",
        "text": "cloth is very comfortable and soft, but when we clean the tshirt the colored dye is released and collar stretched out.",
        "target_aspects": ["cloth", "colored dye", "collar"],
    },
    {
        "database": "MAMS Challenge (Universal Multi-Aspect)",
        "domain_category": None,
        "text": "The food was served promptly and appetizers were divine , but the wait staff was rather cold and the noise level was unbearable .",
        "target_aspects": ["food", "appetizers", "wait staff", "noise level"],
    },
]


def main():
    print("=== Evaluating Samples from All 4 Benchmark Databases with Self-Extracted Evidence ===")
    engine = ProductLensEngine(lazy_load=True)

    all_database_reports = []

    md_lines = [
        "# End-to-End Multi-Database Evaluation: Self-Extracted Evidence & MoE Analysis",
        "",
        "This document demonstrates the full ProductLens ABSA pipeline evaluated on real-world samples from each benchmark database.",
        "**Key Feature**: The `evidence_snippet` is now **self-extracted by the engine** alongside each aspect mention.",
        "",
    ]

    for item in BENCHMARK_SAMPLES:
        db_name = item["database"]
        cat = item["domain_category"]
        text = item["text"]
        aspects = item["target_aspects"]

        print(f"\nProcessing Database: {db_name}...")
        report = engine.analyze_review(text=text, category=cat, aspects=aspects)

        db_entry = {
            "database": db_name,
            "category": cat,
            "review_text": text,
            "summary_sentiment": report.summary_sentiment,
            "execution_time_ms": report.execution_time_ms,
            "aspect_results": [r.model_dump() for r in report.results],
        }
        all_database_reports.append(db_entry)

        md_lines.extend([
            f"## Database: {db_name}",
            f"> **Review Text**: *\"{text}\"*",
            f"> **Input Category Tag**: `{cat or 'None (Universal Fallback)'}` | **Summary Sentiment**: **`{report.summary_sentiment.upper()}`** | **Latency**: `{report.execution_time_ms}ms`",
            "",
            "| Aspect Term | Self-Extracted Evidence Snippet | Predicted Sentiment | Confidence | Routed Specialist Expert |",
            "| :--- | :--- | :---: | :---: | :--- |",
        ])

        for res in report.results:
            icon = "🟢" if res.sentiment == "positive" else ("🔴" if res.sentiment == "negative" else "⚪")
            md_lines.append(
                f"| **`{res.aspect}`** | *\"{res.evidence_snippet}\"* | {icon} **{res.sentiment.upper()}** | {res.confidence * 100:.1f}% | `{res.routed_expert}` ({res.routing_reason}) |"
            )

        md_lines.append("\n---\n")

    # Save JSON Output
    json_path = Path("/home/parth/ProductLens-Sentiment/samples/cross_database_evidence_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_database_reports, f, indent=2)

    # Save Markdown Output
    md_path = Path("/home/parth/ProductLens-Sentiment/samples/cross_database_evidence_samples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nSaved cross-database JSON report to: {json_path}")
    print(f"Saved cross-database Markdown report to: {md_path}")
    print("\n" + "\n".join(md_lines[:45]))


if __name__ == "__main__":
    main()
