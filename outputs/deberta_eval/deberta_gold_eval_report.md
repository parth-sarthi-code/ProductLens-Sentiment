# 🔬 Evaluation Report: `yangheng/deberta-v3-base-absa-v1.1` on Gold Benchmark

> **Task:** Aspect Polarity Classification (APC) given ground-truth aspect terms
> **Gold Dataset:** 350 Curated E-Commerce Customer Reviews (Multilingual & Hard Cases)
> **Total Aspects Evaluated:** 539 aspect mentions across Electronics & Apparel
> **Throughput:** 316.6 aspects/sec (1.7s total)

## 📊 Disaggregated Performance Summary

| Evaluation Slice | Instances | Accuracy | Sentiment Macro-F1 | F1 (Positive) | F1 (Negative) | F1 (Neutral) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | `539` | **`83.3%`** | **`69.8%`** | `87.56%` | `87.76%` | `34.09%` |
| **English Subset** | `420` | `84.76%` | `71.88%` | `90.07%` | `90.68%` | `34.88%` |
| **Hinglish Subset** | `119` | `78.15%` | `52.12%` | `82.35%` | `74.0%` | `0.0%` |
| **Hard Cases (Sarcasm/Negation)** | `514` | `82.68%` | `69.48%` | `86.82%` | `87.52%` | `34.09%` |

## 🔍 Comparative Analysis: DeBERTa-v3 (APC) vs. ProductLens 3.0 (End-to-End)

| Capability / Dimension | `yangheng/deberta-v3-base-absa-v1.1` | ProductLens 3.0 (`LFM3-1.7B` Student) |
| :--- | :--- | :--- |
| **Pipeline Type** | Aspect Polarity Classification (APC) only | **End-to-End Quad Extraction & Scorecard** |
| **Aspect Extraction** | ❌ Cannot extract aspects (requires pre-given terms) | ✅ Extracts exact verbatim aspect spans |
| **Character Offsets** | ❌ None | ✅ 100% exact offsets verified by SubstringValidator |
| **Category Taxonomy** | ❌ No category classification | ✅ Categorizes into canonical e-commerce ontology |
| **Verbatim Evidence Quotes** | ❌ None | ✅ Extracts exact supporting evidence strings |
| **Hinglish Sentiment Macro-F1** | `52.12%` | `67.20%` (F1 on extracted Hinglish quads) |
| **Overall Sentiment Macro-F1** | `69.8%` (Oracle aspects) | `79.30%` (End-to-end extracted quads) |

## 💡 Presentation Takeaways
1. **Complementary Roles:** DeBERTa-v3 is a specialized classifier, but in production, customer reviews do not come with pre-tagged aspect terms. ProductLens 3.0 solves the complete discovery + categorization + evidence quote extraction pipeline.
2. **Hinglish Performance Gap:** Standard English ABSA models like DeBERTa degrade on code-switched Hindi-English phrasing, whereas our fine-tuned student model handles mixed-code tokens robustly.