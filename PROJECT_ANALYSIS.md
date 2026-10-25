# ProductLens-Sentiment: Architecture & LLM Context Document

## 1. Executive Summary
- **Repository**: [ProductLens-Sentiment](file:///home/parth/Projects/ProductLens-Sentiment)
- **Domain**: Aspect-Based Sentiment Analysis (ABSA) for e-commerce reviews.
- **Core Architecture**: Unified calibrated DeBERTa-v3 architecture with Query-Key Cross-Attention Evidence Pooling, Aspect-Centered Windowing, and Ordinal Wasserstein Loss.
- **Key Task**: Given customer reviews, extract aspect targets, ground contextual opinion snippets, and predict aspect-level sentiment labels (Positive, Neutral, Negative) with calibrated confidence scores without multi-model routing overhead.

---

## 2. Directory Structure

```
ProductLens-Sentiment/
├── backend/
│   ├── api/
│   │   └── v1/
│   │       ├── analyze.py                # Review analysis REST endpoints
│   │       └── health.py                 # Service health check endpoints
│   ├── schemas/
│   │   └── requests.py                   # Pydantic request & response DTOs
│   └── main.py                           # FastAPI application entry & lifespan manager
├── productlens/
│   ├── aspects/
│   │   └── extractor.py                  # Aspect extraction (noun chunking, gazetteers)
│   ├── sentiment/
│   │   ├── model.py                      # DeBERTa-v3 backbone with aspect cross-attention pooling
│   │   ├── pipeline.py                   # Unified high-throughput inference pipeline
│   │   └── calibration.py                # Temperature calibration module
│   ├── training/
│   │   ├── dataset.py                    # PyTorch Dataset for ABSA tokenization & windowing
│   │   ├── loss.py                       # Ordinal Wasserstein & SCL loss functions
│   │   └── trainer.py                    # AspectSentimentTrainer with mixed precision
│   ├── engine.py                         # Unified ProductLensEngine facade
│   └── schemas.py                        # Domain dataclasses & schemas
├── configs/
│   ├── app_config.yaml                   # Model weights paths and runtime params
│   └── training_configs.yaml             # Hyperparameters for unified model training
├── data/benchmarks/                      # Benchmark datasets (Tech, Home, Apparel, ARTS)
├── models/unified_backbone/              # Unified model checkpoint & calibration parameters
```
├── scripts/                              # Training, evaluation & verification CLI scripts
└── tests/                                # Pytest test suite
```

---

## 3. Pipeline Flow

```mermaid
flowchart TD
    Review["Input: Customer Review + Optional Category"] --> Extractor["Aspect Extractor (productlens/aspects/extractor.py)"]
    Review --> Router{"MoE Router (productlens/sentiment/router.py)"}
    
    Extractor --> Spans["Aspect Mentions & Evidence Snippets"]
    
    Router -->|"Level 1: Exact Category Tag"| TechExpert["Tech Specialist (DeBERTa-v3)"]
    Router -->|"Level 1: Exact Category Tag"| HomeExpert["Home Specialist (DeBERTa-v3)"]
    Router -->|"Level 1: Exact Category Tag"| ApparelExpert["Apparel Specialist (DeBERTa-v3)"]
    Router -->|"Level 2: Keyword Match"| Experts["Specialist Suite"]
    Router -->|"Level 3: Universal Fallback"| Fallback["Universal MAMS Foundation"]
    
    Spans --> Pipeline["MoE Pipeline (productlens/sentiment/pipeline.py)"]
    TechExpert --> Pipeline
    HomeExpert --> Pipeline
    ApparelExpert --> Pipeline
    Fallback --> Pipeline
    
    Pipeline --> Pooling["Aspect-Targeted Attention Pooling (Boost=3.5)"]
    Pooling --> Classifier["Classification Head"]
    Classifier --> Report["Output: ReviewAnalysisReport"]
```

---

## 4. Key Modules & Symbol Reference

### Engine Facade
- **[`ProductLensEngine`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/engine.py#L52)**: High-level orchestrator.
  - `analyze_review(text: str, category: Optional[str] = None) -> ReviewAnalysisReport`
  - `analyze_batch(texts: List[str], categories: Optional[List[str]] = None) -> BatchAnalysisReport`
- **[`extract_evidence_snippet`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/engine.py#L17)**: Extracts contextual sentence/clause containing the aspect mention.

### Aspect Extraction
- **[`AspectExtractor`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/aspects/extractor.py)**: Identifies aspect terms using linguistic heuristics, noun phrases, and domain gazetteers.

### Sentiment Classification & Pipeline
- **[`ProductLensSentimentPipeline`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/sentiment/pipeline.py)**: Manages batch tokenization (`[CLS] text [SEP] aspect [SEP]`), aspect windowing, and high-throughput batched tensor inference.
- **[`AspectTargetedSentimentModel`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/sentiment/model.py)**: DeBERTa-v3 backbone with Query-Key cross-attention pooling targeted at aspect token spans and Tripartite Context Fusion.
- **[`TemperatureScaler`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/sentiment/calibration.py)**: Post-hoc temperature calibration minimizing Negative Log-Likelihood and Expected Calibration Error.

---

## 5. Domain Schemas ([`productlens/schemas.py`](file:///home/parth/Projects/ProductLens-Sentiment/productlens/schemas.py))

- **`AspectMention`**:
  - `aspect: str`: Extracted term (e.g., "battery life", "screen")
  - `start_char: int`, `end_char: int`: Character offsets in source review
  - `evidence_snippet: str`: Surrounding sentence or clause
- **`AspectSentimentResult`**:
  - `aspect: str`
  - `sentiment: str`: `"positive"` | `"negative"` | `"neutral"`
  - `confidence: float`: Softmax probability score $[0.0, 1.0]$
  - `routed_expert: str`: Name of expert model executing inference
  - `evidence_snippet: str`
- **`ReviewAnalysisReport`**:
  - `text: str`: Original review text
  - `category: Optional[str]`: Input or inferred category
  - `results: List[AspectSentimentResult]`: Per-aspect predictions
  - `summary_sentiment: str`: Aggregated review-level polarity
  - `latency_ms: float`: Total processing time in milliseconds

---

## 6. Service & Entry Points

- **FastAPI Application**: [backend/main.py](file:///home/parth/Projects/ProductLens-Sentiment/backend/main.py)
  - `GET /api/v1/health`: Service status and loaded expert weights.
  - `POST /api/v1/analyze`: Accepts `{ "text": str, "category": Optional[str] }` and returns `ReviewAnalysisReport`.
  - `POST /api/v1/analyze/batch`: Batch review processing.
- **CLI Scripts**:
  - `scripts/train_all_experts.py`: Trains all domain specialist heads.
  - `scripts/evaluate_comparison.py`: Benchmarks MoE system vs single generic baseline.
  - `scripts/test_sample_review.py`: Quick verification script.
