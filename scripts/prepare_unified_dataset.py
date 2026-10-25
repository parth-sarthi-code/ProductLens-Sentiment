"""
Prepare consolidated multi-domain e-commerce datasets and adversarial ARTS test sets.
Creates:
- data/benchmarks/unified_ecommerce/train.parquet
- data/benchmarks/unified_ecommerce/test.parquet
- data/benchmarks/arts_laptop/test.parquet
"""

import urllib.request
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    print("Consolidating multi-domain e-commerce datasets...")
    
    # 1. Consolidate training data
    train_sources = [
        ("tech", BASE_DIR / "data/benchmarks/tech_laptop/train.parquet"),
        ("home", BASE_DIR / "data/benchmarks/home_television/train.parquet"),
        ("apparel", BASE_DIR / "data/benchmarks/apparel_tshirt/train.parquet"),
    ]
    
    train_dfs = []
    for domain, p in train_sources:
        df = pd.read_parquet(p)
        df["domain"] = domain
        train_dfs.append(df)
        
    unified_train = pd.concat(train_dfs, ignore_index=True)
    out_dir = BASE_DIR / "data/benchmarks/unified_ecommerce"
    out_dir.mkdir(parents=True, exist_ok=True)
    unified_train_path = out_dir / "train.parquet"
    unified_train.to_parquet(unified_train_path, index=False)
    print(f"Saved unified train to {unified_train_path}: {len(unified_train)} samples")
    print("Class distribution:", unified_train["sentiment"].value_counts().to_dict())

    # 2. Consolidate test data
    test_sources = [
        ("tech", BASE_DIR / "data/benchmarks/tech_laptop/test.parquet"),
        ("home", BASE_DIR / "data/benchmarks/home_television/test.parquet"),
        ("apparel", BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet"),
    ]
    
    test_dfs = []
    for domain, p in test_sources:
        df = pd.read_parquet(p)
        df["domain"] = domain
        test_dfs.append(df)
        
    unified_test = pd.concat(test_dfs, ignore_index=True)
    unified_test_path = out_dir / "test.parquet"
    unified_test.to_parquet(unified_test_path, index=False)
    print(f"Saved unified test to {unified_test_path}: {len(unified_test)} samples")
    print("Class distribution:", unified_test["sentiment"].value_counts().to_dict())

    # 3. Download and prepare ARTS laptop adversarial test set
    arts_dir = BASE_DIR / "data/benchmarks/arts_laptop"
    arts_dir.mkdir(parents=True, exist_ok=True)
    arts_test_path = arts_dir / "test.parquet"
    
    arts_url = "https://raw.githubusercontent.com/yangheng95/ABSADatasets/v1.2/datasets/apc_datasets/110.SemEval/111.arts_laptop14/laptop_arts_test.dat"
    print(f"Downloading ARTS laptop adversarial test set from {arts_url}...")
    req = urllib.request.Request(arts_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8")
        
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    records = []
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            records.append({
                "text": lines[i],
                "aspect": lines[i+1],
                "sentiment": lines[i+2].lower(),
                "domain": "tech_adversarial",
            })
            
    arts_df = pd.DataFrame(records)
    arts_df.to_parquet(arts_test_path, index=False)
    print(f"Saved ARTS adversarial test set to {arts_test_path}: {len(arts_df)} samples")
    print("Class distribution:", arts_df["sentiment"].value_counts().to_dict())

if __name__ == "__main__":
    main()
