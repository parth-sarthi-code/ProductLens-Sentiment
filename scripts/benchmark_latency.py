#!/usr/bin/env python3
"""
Multi-Aspect Inference Latency Benchmark.
Measures latency comparison between sequential predict() and optimized batched predict_aspects().
"""

import time
import numpy as np
import torch

from productlens.sentiment.pipeline import ProductLensMoEPipeline
from productlens.engine import ProductLensEngine


def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Multi-Aspect Inference Speed Benchmark (device={device}) ===")

    pipeline = ProductLensMoEPipeline(lazy_load=True, single_active_expert=True, device=device)

    # Multi-aspect review with 4 aspects
    review_text = (
        "The battery life is exceptional and lasts two whole days on one charge. "
        "However, the screen resolution is somewhat pixelated and blurry. "
        "The keyboard typing experience is tactile and smooth. "
        "Finally, the internal speakers produce muffled and tinny audio."
    )
    aspects = [
        ("battery life", 4, 16),
        ("screen resolution", 80, 97),
        ("keyboard typing experience", 140, 166),
        ("internal speakers", 217, 234),
    ]

    # Warmup
    _ = pipeline.predict_aspects(text=review_text, aspects=aspects, category="Electronics")

    n_runs = 5

    # 1. Unoptimized Sequential predict()
    seq_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for asp, s, e in aspects:
            _ = pipeline.predict(text=review_text, aspect=asp, category="Electronics", aspect_start=s, aspect_end=e)
        seq_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Optimized Batched predict_aspects()
    batch_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = pipeline.predict_aspects(text=review_text, aspects=aspects, category="Electronics")
        batch_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. Engine end-to-end
    engine = ProductLensEngine(device=device, lazy_load=True)
    engine_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = engine.analyze_review(text=review_text, category="Electronics")
        engine_times.append((time.perf_counter() - t0) * 1000.0)

    avg_seq = float(np.mean(seq_times))
    avg_batch = float(np.mean(batch_times))
    avg_engine = float(np.mean(engine_times))
    speedup = avg_seq / avg_batch

    print(f"\nSequential predict (4 aspects, N={n_runs}): {avg_seq:.2f} ms")
    print(f"Batched predict_aspects (4 aspects, N={n_runs}): {avg_batch:.2f} ms")
    print(f"Speedup: {speedup:.2f}x faster ({((avg_seq - avg_batch) / avg_seq) * 100:.1f}% latency reduction)")
    print(f"Engine analyze_review end-to-end: {avg_engine:.2f} ms")


if __name__ == "__main__":
    benchmark()
