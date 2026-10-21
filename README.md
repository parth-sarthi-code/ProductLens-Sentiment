# ProductLens: Aspect-Based Sentiment Analysis Platform

Production-grade Aspect-Based Sentiment Analysis (ABSA) platform powered by a **Hierarchical Mixture of Experts (MoE)** architecture with domain-specialist RoBERTa backbones and pure $O(1)$ zero-latency metadata routing.

---

## 1. Key Performance Benchmarks

| Domain Vertical | Specialist Model | Held-Out Test Accuracy | In-Domain Win Lift |
| :--- | :--- | :---: | :---: |
| **Tech Hardware** | RoBERTa Tech Specialist (`Laptop14`) | **84.17%** | +12.38% over foundation |
| **Home Electronics** | RoBERTa Home Specialist (`Television`) | **89.81%** | +10.72% over foundation |
| **Apparel & Fashion** | RoBERTa Apparel Specialist (`T-Shirt`) | **94.07%** | +6.57% over foundation |
| **Universal Fallback** | RoBERTa MAMS Foundation | **84.21%** | Top general multi-aspect handling |
| **E-Commerce Composite** | **ProductLens MoE Unified System** | **89.04% 🏆** | **+10.39% absolute gain** |

---

## 2. Architecture & Flow

```mermaid
flowchart TD
    Review["Customer Review Text"] --> Extractor["Aspect Extractor<br/>(Noun Phrase & Gazetteer Filter)"]
    Review --> Router{"MoE Router"}
    
    Category["Optional Department / Category Tag"] --> Router
    
    Router -->|"Level 1: Exact Metadata Tag"| Specialists
    Router -->|"Level 2: Lexical Keyword Match"| Specialists
    Router -->|"Level 3: General Fallback"| Fallback["Universal MAMS Foundation"]
    
    subgraph Specialists ["Domain-Specialist RoBERTa Suite"]
        Tech["Tech Expert (84.2%)"]
        Home["Home Expert (89.8%)"]
        Apparel["Apparel Expert (94.1%)"]
    end

    Extractor --> AspectPairs["Aspect Mentions & Spans"]
    AspectPairs --> Pooler["Aspect-Targeted Attention Pooling (Boost=3.5)"]
    Specialists --> Pooler
    Fallback --> Pooler
    
    Pooler --> Head["Classification MLP + Softmax"]
    Head --> Output["Aspect-Level Sentiments + Calibrated Confidence"]
```

---

## 3. Quickstart

### Installation
```bash
cd /home/parth/ProductLens-Sentiment
pip install -e .
```

### Python SDK Usage
```python
from productlens.engine import ProductLensEngine

engine = ProductLensEngine()

# Analyze a single review
report = engine.analyze_review(
    text="The display on this laptop is crystal clear, but the battery drains fast.",
    category="Electronics"
)

print(f"Summary Sentiment: {report.summary_sentiment}")
for res in report.results:
    print(f"Aspect: {res.aspect} | Sentiment: {res.sentiment} ({res.confidence:.1%}) | Expert: {res.routed_expert}")
```

### Running the FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation will be available at: `http://localhost:8000/docs`

---

## 4. Running Tests
```bash
pytest tests/
```
