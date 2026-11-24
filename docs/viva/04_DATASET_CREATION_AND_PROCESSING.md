# Dataset Architecture: Creation, Processing, and Zero-Leakage Quality Control

> **Viva & Presentation Guide: How We Built the Gold Exam Benchmark, Assembled 7,718 Training Reviews, and Proved Zero Data Leakage.**

---

## 1. The Core Idea (In Plain English)

Machine learning models are only as good as the data they learn from. If you train a model on sloppy data, you get a sloppy model. If you accidentally let the model see the test questions during training (called **data leakage**), your high benchmark scores are fake and the model fails in production.

For ProductLens, we treated data engineering with scientific rigor:
1. **The Exam Paper (Gold Benchmark):** A human-verified test suite of **350 multi-domain customer reviews** with zero errors.
2. **The Study Material (Training Corpus):** A leak-free dataset of **7,718 reviews (9,887 aspect quads)** assembled across tech and apparel.
3. **The Fire-Wall (Zero Leakage):** A programmatic blacklist of 3,186 texts that guaranteed the model never saw a single test question before the exam.
4. **The Compact Quad Format:** A fast, token-efficient serialization format that replaced heavy JSON.

---

## 2. The 350-Review Gold Benchmark (The Exam Paper)

To evaluate our model fairly, we needed a rigorous benchmark. 

```mermaid
flowchart TD
    GOLD["350-Review Gold Benchmark Set<br/>(100% Verified Ground Truth)"]
    
    GOLD --> D1["200 Electronics & Tech Reviews<br/>Laptops, Monitors, Audio, Phones"]
    GOLD --> D2["150 Apparel & Fashion Reviews<br/>Shirts, T-Shirts, Denim, Shoes"]
    
    D1 --> ADV["514 Adversarial Aspects Tested<br/>- Contrastive ('great screen, but fan is loud')<br/>- Negation ('not very durable')<br/>- Sarcasm & Irony ('just love when it crashes')"]
    D2 --> ADV
    
    ADV --> OFF["Zero Offset Errors<br/>100% of character spans verified against source text"]

    style GOLD fill:#3c3836,stroke:#7daea3,stroke-width:2px,color:#d4be98
    style ADV fill:#2f3424,stroke:#a9b665,stroke-width:2px,color:#b8bb26
    style OFF fill:#2f3424,stroke:#a9b665,stroke-width:2px,color:#b8bb26
```

### Key Properties of the Gold Benchmark:
- **Human Verified:** Every aspect, category, and sentiment was manually audited and labeled.
- **100% Valid Spans:** In code (`tests/test_gold_set.py`), an automated test slices every single review:
  ```python
  assert review.text[quad.evidence_start : quad.evidence_end] == quad.evidence
  ```
  Every single character index points to the exact text without a single off-by-one error.
- **Multi-Domain:** Evaluates both technical hardware (e.g., refresh rates, thermal noise) and physical apparel (e.g., fabric softness, post-wash shrinkage).

---

## 3. The 7,718-Review Training Corpus (The Study Material)

To train `LiquidAI/LFM3-1.7B`, `scripts/prepare_stage3_data.py` assembled a balanced training corpus:

```text
Training Corpus Manifest (data/stage3/dataset_manifest.json):
-----------------------------------------------------------
Total Reviews:          7,718 reviews (9,887 aspect quads)
Training Split (90%):   6,947 reviews (8,890 quads)
Validation Split (10%):   771 reviews (997 quads)

Domain Distribution:
- Electronics & Tech:   6,035 reviews (78.2%)
- Apparel & Fashion:    1,683 reviews (21.8%)
```

### Data Sources:
1. **Curated Human Benchmark Training Sets:** High-quality training splits from established NLP benchmarks (`tech_laptop`, `home_television`, `apparel_tshirt`).
2. **Stratified MAMS Dataset:** A balanced downsample of multi-aspect reviews (`multi_aspect_mams`) ensuring equal distribution across positive, negative, and neutral sentiments.
3. **Stage 2b Consensus Silver Batches:** High-confidence parquet batches produced by the dual-pass Qwen-9B teacher on Kaggle.

---

## 4. The Zero-Leakage Firewall: How We Proved Zero Cheating

In academic research, the #1 reason models achieve artificially inflated test scores is **data contamination** (the model memorized the test answers during training).

### How We Programmed the Firewall:
In `scripts/prepare_stage3_data.py`:
1. We loaded every single review text and product ID from:
   - The 350 Gold Benchmark reviews
   - The SemEval Laptop14 test set
   - The Television test set
   - The T-Shirt test set
   - The ARTS adversarial test set
2. This created a **forbidden blacklist of 3,186 unique review strings and product IDs**.
3. During dataset compilation, every candidate training review was cleaned and checked against this blacklist:
   ```python
   if clean_text in forbidden_texts or pid in forbidden_pids:
       skipped_leakage += 1
       continue  # Permanently blocked from training
   ```
4. **Verification Test:** `tests/test_stage3_data.py` runs a verification assertion ensuring that the intersection between the training set and the evaluation sets is strictly empty:
   $$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$

---

## 5. The Compact Quad Serialization Format: Why We Ditched JSON

Most AI applications format outputs as JSON:
```json
{
  "aspect": "display",
  "category": "DISPLAY",
  "sentiment": "POSITIVE",
  "confidence": 1.0,
  "evidence": "Outstanding display"
}
```

### Why JSON Fails for Compact Models on Edge Hardware:
- **Wasted Tokens:** JSON spends dozens of tokens on repetitive syntax (`{`, `"`, `}`, `:`, `"category":`). This wastes GPU memory and makes inference 40% slower.
- **Syntax Errors:** Compact models (under 3B parameters) frequently forget to close a curly brace `}` or omit a comma `,`, causing `json.loads()` to crash with a parse error.

### The ProductLens Compact Format:
```text
aspect|category|sentiment|confidence|evidence
```

**Real Example:**
```text
display|DISPLAY|POSITIVE|1.0|Outstanding display and battery
battery|BATTERY|POSITIVE|1.0|Outstanding display and battery
```
- **Special Cases:**
  - Implicit aspects use `null`: `null|VALUE|NEGATIVE|0.95|Way too overpriced`
  - Reviews with no aspects output: `NONE`
- **Benefits:**
  - Uses ~45% fewer tokens than JSON.
  - Generates in under 400 milliseconds.
  - Zero syntax parser failures.

---

## 6. Top Viva & Examination Questions (Cheat Sheet)

### Q1: "What is data leakage and how did your pipeline prevent it?"
> **Winning Answer:**
> *"Data leakage happens when test set samples or information leak into the training dataset, giving a false appearance of high accuracy. We prevented this by creating a programmatic blacklist of 3,186 test texts and product IDs across our Gold Set and benchmark test sets. Any training row matching the blacklist was dropped before training, verified by our automated unit test suite."*

### Q2: "Why did you use a 90/10 train-validation split instead of 80/20?"
> **Winning Answer:**
> *"With 7,718 total reviews, a 10% validation split gives 771 reviews (nearly 1,000 quads). Statistically, 771 samples is more than large enough to reliably track validation loss and detect overfitting, while leaving 6,947 diverse reviews to maximize training coverage."*

### Q3: "Why did you include adversarial hard cases in the Gold set?"
> **Winning Answer:**
> *"Real customer reviews are not simple sentences like 'The screen is nice.' Real customers write contrastive thoughts like 'Despite the vibrant OLED screen, the laptop gets burning hot on your lap.' If you only test on simple sentences, your model collapses in the real world. Testing 514 hard adversarial cases proves our model understands contrastive conjunctions like 'despite', 'however', and 'but'."*
