"""
Benchmark loaders for ProductLens 3.0.
Loads curated human-labeled benchmarks (Laptop14, Television, T-Shirt, ARTS Laptop).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from productlens.schemas.enums import VerticalDomain

DEFAULT_BENCHMARK_BASE = Path("/home/parth/Projects/ProductLens-Sentiment/data/benchmarks")

BENCHMARK_PATHS: Dict[str, Dict[str, Path]] = {
    "tech_laptop": {
        "train": DEFAULT_BENCHMARK_BASE / "tech_laptop" / "train.parquet",
        "test": DEFAULT_BENCHMARK_BASE / "tech_laptop" / "test.parquet",
        "vertical": VerticalDomain.ELECTRONICS_TECH,
    },
    "home_television": {
        "train": DEFAULT_BENCHMARK_BASE / "home_television" / "train.parquet",
        "test": DEFAULT_BENCHMARK_BASE / "home_television" / "test.parquet",
        "vertical": VerticalDomain.ELECTRONICS_TECH,
    },
    "apparel_tshirt": {
        "train": DEFAULT_BENCHMARK_BASE / "apparel_tshirt" / "train.parquet",
        "test": DEFAULT_BENCHMARK_BASE / "apparel_tshirt" / "test.parquet",
        "vertical": VerticalDomain.APPAREL_FASHION,
    },
    "arts_laptop": {
        "test": DEFAULT_BENCHMARK_BASE / "arts_laptop" / "test.parquet",
        "vertical": VerticalDomain.ELECTRONICS_TECH,
    },
    "multi_aspect_mams": {
        "train": DEFAULT_BENCHMARK_BASE / "multi_aspect_mams" / "train.parquet",
        "test": DEFAULT_BENCHMARK_BASE / "multi_aspect_mams" / "test.parquet",
        "val": DEFAULT_BENCHMARK_BASE / "multi_aspect_mams" / "val.parquet",
        "vertical": VerticalDomain.ELECTRONICS_TECH,
    },
}


class BenchmarkLoader:
    """Provides unified access to curated train and test benchmark datasets."""

    @staticmethod
    def load_dataset(name: str, split: str = "train") -> pd.DataFrame:
        if name not in BENCHMARK_PATHS:
            raise KeyError(f"Unknown benchmark dataset: '{name}'. Available: {list(BENCHMARK_PATHS.keys())}")
        paths = BENCHMARK_PATHS[name]
        if split not in paths:
            raise ValueError(f"Split '{split}' not available for '{name}'. Available: {[k for k in paths if k != 'vertical']}")

        filepath = paths[split]
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark file not found at: {filepath}")

        df = pd.read_parquet(filepath)
        return df

    @classmethod
    def get_vertical(cls, name: str) -> VerticalDomain:
        if name not in BENCHMARK_PATHS:
            raise KeyError(f"Unknown benchmark dataset: '{name}'")
        return BENCHMARK_PATHS[name]["vertical"]

    @classmethod
    def load_all_active_splits(cls) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Loads all active train and test sets according to the focused catalog."""
        datasets = {}
        for name, splits in BENCHMARK_PATHS.items():
            datasets[name] = {}
            for split in ["train", "test"]:
                if split in splits:
                    datasets[name][split] = pd.read_parquet(splits[split])
        return datasets
