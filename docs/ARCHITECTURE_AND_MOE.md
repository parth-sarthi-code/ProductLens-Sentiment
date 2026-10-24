# ProductLens Architecture, Mixture of Experts (MoE), and Empirical Benchmark Synthesis

This document provides a comprehensive technical walkthrough of the **neural architecture**, **MoE implementation**, **generic vs specialist training**, and **empirical benchmark comparison** in the ProductLens Sentiment Analysis platform.

---

## 1. Model Architecture

Traditional sentiment classifiers pass the generic `[CLS]` token representation through a linear classifier:

$$y = \text{Softmax}(W \cdot h_{\text{[CLS]}} + b)$$

In multi-aspect e-commerce reviews (e.g., *"The battery lasts all day, but the screen is washed out"*), `[CLS]` aggregates global sentence sentiment, causing strong negative clauses to contaminate positive aspect predictions.

### 1.1 Aspect-Targeted Attention Pooling (`AspectTargetedAttentionPooler`)
Located in [`productlens/sentiment/model.py`](../productlens/sentiment/model.py).

Instead of `[CLS]` pooling, we condition the self-attention pooling directly on the target aspect span:

$$\text{AttentionScores}_{i,j} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + \beta \cdot M_j$$

Where:
* $h \in \mathbb{R}^{B \times L \times d}$ are the encoder hidden states (RoBERTa backbone).
* $M_j \in \{0, 1\}$ is a binary subword mask indicating whether token $j$ overlaps with the target aspect span $[s_{\text{aspect}}, e_{\text{aspect}})$.
* $\beta = 3.5$ is the **Aspect Boost Factor**, which injects a strong inductive bias forcing attention to remain tethered to the aspect clause.

The final representation combines the context-weighted sequence representation with the mean aspect span vector:

$$h_{\text{aspect\_span}} = \frac{\sum_{j=1}^L M_j h_j}{\max(1, \sum_{j=1}^L M_j)}$$

$$h_{\text{final}} = \frac{1}{2} \left( h_{\text{context\_pooled}} + h_{\text{aspect\_span}} \right)$$

### 1.2 Supervised Contrastive Loss (SCL) & Projection Head
Located in [`productlens/training/loss.py`](../productlens/training/loss.py).

To prevent representation collapse on under-represented classes and enforce domain-invariant sentiment geometry, we append a 128-dimensional projection head $z = g(h_{\text{final}}) \in \mathbb{R}^{128}$ and train with SCL ($\tau = 0.07$):

$$\mathcal{L}_{\text{SCL}} = -\sum_{i=1}^B \frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Weighted\_CE}} + \lambda_{\text{SCL}} \mathcal{L}_{\text{SCL}} \quad (\lambda_{\text{SCL}} = 0.10)$$

---

## 2. How Mixture of Experts (MoE) is Implemented

The MoE system dynamically routes reviews to domain-specialist models using a 3-level hierarchical dispatch engine.

```mermaid
flowchart TD
    Review["Customer Review + Target Aspect"] --> Router{"MoERouter Dispatch Engine"}
    
    Router -->|"Level 1: Category Tag (Electronics/Computers)"| Tech["Tech Specialist (Laptop14)<br/>Acc: 81.97%"]
    Router -->|"Level 1: Category Tag (Home/Kitchen/Appliances)"| Home["Home Specialist (Television)<br/>Acc: 90.87%"]
    Router -->|"Level 1: Category Tag (Clothing/Apparel/Shoes)"| Apparel["Apparel Specialist (T-Shirt)<br/>Acc: 92.37%"]
    
    Router -.->|"Level 2: Lexical Keyword Fallback"| Tech
    Router -.->|"Level 2: Lexical Keyword Fallback"| Apparel
    Router -.->|"Level 3: Universal Fallback (Uncertain/General)"| MAMS["Generic Foundation Expert<br/>Challenge Acc: 82.49%"]

    Tech --> Output["Final Predicted Sentiment + Calibrated Confidence"]
    Home --> Output
    Apparel --> Output
    MAMS --> Output
```

### 2.1 The 3-Level Routing Hierarchy (`MoERouter`)
Located in [`productlens/sentiment/router.py`](../productlens/sentiment/router.py).

1. **Level 1: Metadata Gating ($O(1)$ Zero Latency)**:
   * Amazon and Shopify reviews inherently include department metadata. If `category` matches known taxonomies (`Electronics`, `Clothing`, `Home_and_Kitchen`), the query routes instantly to that domain specialist with zero neural gate latency.
2. **Level 2: Lexical / Semantic Keyword Gating**:
   * If `category` is missing (`None`), the router scans the target aspect and review context against domain lexicons (e.g., *"touchpad"*, *"battery"* $\rightarrow$ Tech; *"fabric"*, *"wash"* $\rightarrow$ Apparel).
3. **Level 3: Universal MAMS Foundation Fallback**:
   * If domain signals are ambiguous or conflicting, queries route to the **Generic Foundation Expert**, ensuring robust handling of general multi-aspect reviews.

### 2.2 Inference Pipeline & Lazy Loading (`ProductLensMoEPipeline`)
Located in [`productlens/sentiment/pipeline.py`](../productlens/sentiment/pipeline.py).

* Models are loaded on-demand (`lazy_load=True`) and cached in GPU/CPU memory to minimize startup latency and VRAM footprint.
* The pipeline transparently prepares token encodings, computes character-to-subword span masks, dispatches to the routed specialist, and returns calibrated probabilities.

---

## 3. Generic Foundation Model vs Category Specialists

### 3.1 The Generic Foundation Model (`foundation_mams`)
* **Training Data**: 11,186 multi-aspect review sentences from the MAMS Challenge benchmark.
* **Objective**: Universal, domain-agnostic aspect sentiment classification.
* **Strengths**: Robust across syntactically complex multi-aspect sentences (achieves **83.83%** on MAMS challenge).
* **Limitation**: Drops to **73.04%** on Tech Hardware and **78.56%** on Home Electronics because domain-specific technical jargon is underrepresented in generic corpora.

### 3.2 Category Specialist Models
* **Tech Hardware Specialist (`expert_tech`)**: Fine-tuned on SemEval Laptop14 hardware reviews. Reaches **81.97%** (+8.93% over generic).
* **Home Electronics Specialist (`expert_home`)**: Fine-tuned on television/appliance reviews. Reaches **90.87%** (+12.31% over generic).
* **Apparel Specialist (`expert_apparel`)**: Fine-tuned on clothing and garment reviews. Reaches **92.37%** (+4.23% over generic).

---

## 4. Master Empirical Benchmark Comparison

All models were evaluated across the 4 held-out benchmark test sets (3,388 total samples) using sample-weighted composite accuracy:

| Model Architecture (Rows) \ Domain (Cols) | Laptop14 (Tech, 638) | Television (Home, 942) | T-Shirt (Apparel, 472) | MAMS Challenge (1,336) | E-Commerce Composite (2,052) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Generic Foundation (MAMS)** | 73.04% | 78.56% | 88.14% | **83.83% 🏆** | 79.05% |
| **Tech Specialist (`Laptop14`)** | **81.97% 🏆** | 88.54% | 91.95% | 58.83% | 87.28% |
| **Home Specialist (`Television`)** | 75.71% | **90.87% 🏆** | 89.83% | 49.93% | 85.92% |
| **Apparel Specialist (`T-Shirt`)** | 67.40% | 89.17% | **92.37% 🏆** | 47.23% | 83.14% |
| **ProductLens MoE (no category)** | 75.08% | 83.44% | 88.98% | 82.49% | **82.12% (+3.07% Net Lift)** |
| **ProductLens MoE (with category)** | **81.97% 🏆** | **90.87% 🏆** | **92.37% 🏆** | 82.49% | **88.45% 🏆 (+9.40% Net Lift)** |

---

## 5. Mathematical Proof: Why We Choose MoE Over Monolithic Models

### 5.1 The Diagonal Dominance Principle
In the transfer matrix, the maximum value in every column lies strictly along the main diagonal:

$$\max_i T_{i,j} = T_{j,j} \quad \forall j \in \{\text{Tech}, \text{Home}, \text{Apparel}, \text{MAMS}\}$$

Every vertical is best served by its dedicated specialist:
* Tech: Specialist achieves **81.97%** vs Generic **73.04%** (**+8.93% lift**).
* Home: Specialist achieves **90.87%** vs Generic **78.56%** (**+12.31% lift**).
* Apparel: Specialist achieves **92.37%** vs Generic **88.14%** (**+4.23% lift**).

### 5.2 Negative Cross-Domain Gradient Interference
In a monolithic model, the loss gradient is a composite across disparate domains:

$$\nabla_\theta \mathcal{L}_{\text{mono}} = \mathbb{E}_{x \sim \text{Tech}}[\nabla \mathcal{L}] + \mathbb{E}_{x \sim \text{Apparel}}[\nabla \mathcal{L}] + \mathbb{E}_{x \sim \text{Home}}[\nabla \mathcal{L}]$$

Because polarities and linguistic cues conflict (e.g., *"thin"* or *"tight"* is positive in apparel but negative in tech), gradient updates actively cancel each other out:

$$\langle \nabla \mathcal{L}_{\text{Tech}}, \nabla \mathcal{L}_{\text{Apparel}} \rangle < 0$$

By decomposing the system into isolated domain experts $E_k(\cdot)$, we guarantee:

$$\nabla_{\theta_k} \mathcal{L}_k \cdot \nabla_{\theta_j} \mathcal{L}_j = 0 \quad (\forall k \neq j)$$

This completely eliminates cross-domain gradient conflict and allows each domain specialist to achieve maximal accuracy.

---

## 6. How to Reproduce & Retrain

### Run the Benchmark Evaluation Locally
```bash
python scripts/evaluate_comparison.py
```
This loads the generic model, the specialist models, and the MoE pipeline, evaluates them on all 4 benchmark test sets, and outputs the comparison table above.

### Retrain Any Model from Scratch
```bash
python scripts/train.py --domain generic    # Retrains Generic Foundation Model
python scripts/train.py --domain tech       # Retrains Tech Specialist
python scripts/train.py --domain home       # Retrains Home Specialist
python scripts/train.py --domain apparel    # Retrains Apparel Specialist
```
