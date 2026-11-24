"""
Unit and integration tests for benchmark loaders and adapter.
"""

import pytest
from productlens.benchmarks.loader import BenchmarkLoader, BENCHMARK_PATHS
from productlens.benchmarks.adapter import BenchmarkAdapter
from productlens.schemas.enums import VerticalDomain


def test_benchmark_paths_exist():
    for name, splits in BENCHMARK_PATHS.items():
        for split, path in splits.items():
            if split != "vertical":
                assert path.exists(), f"Missing parquet file for {name}:{split} at {path}"


def test_load_tech_laptop():
    df_train = BenchmarkLoader.load_dataset("tech_laptop", "train")
    df_test = BenchmarkLoader.load_dataset("tech_laptop", "test")
    assert len(df_train) == 2266
    assert len(df_test) == 638
    assert "aspect" in df_train.columns
    assert "sentiment" in df_train.columns


def test_load_television_and_tshirt():
    df_tv = BenchmarkLoader.load_dataset("home_television", "train")
    df_tshirt = BenchmarkLoader.load_dataset("apparel_tshirt", "train")
    assert len(df_tv) == 3705
    assert len(df_tshirt) == 1867


def test_adapter_laptop_sample():
    df = BenchmarkLoader.load_dataset("tech_laptop", "train")
    sample_row = df.iloc[0].to_dict()
    quad, err = BenchmarkAdapter.row_to_quad(sample_row, VerticalDomain.ELECTRONICS_TECH)
    assert err is None
    assert quad is not None
    assert quad.aspect == sample_row["aspect"]
    assert sample_row["text"][quad.aspect_start:quad.aspect_end] == sample_row["aspect"]


def test_adapter_arts_adversarial_sample():
    df = BenchmarkLoader.load_dataset("arts_laptop", "test")
    sample_row = df.iloc[0].to_dict()
    quad, err = BenchmarkAdapter.row_to_quad(sample_row, VerticalDomain.ELECTRONICS_TECH)
    assert err is None
    assert quad is not None
    assert quad.aspect == sample_row["aspect"]
    assert sample_row["text"][quad.aspect_start:quad.aspect_end] == sample_row["aspect"]
