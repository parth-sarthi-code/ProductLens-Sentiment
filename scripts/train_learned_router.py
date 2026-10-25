#!/usr/bin/env python3
"""
Train Learned Gating Network for Autonomous Text-Based MoE Routing.

Trains a lightweight 2-layer MLP (768 -> 128 -> 4) taking pooled DeBERTa
text embeddings to predict the target expert:
  0: tech
  1: home
  2: apparel
  3: fallback

Saves checkpoint to models/learned_router.pt.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from productlens.constants import DEFAULT_BACKBONE
from productlens.sentiment.model import AspectTargetedSentimentModel
from productlens.sentiment.router import LearnedGatingNetwork

BASE_DIR = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DOMAINS = [
    ("tech", BASE_DIR / "data/benchmarks/tech_laptop/train.parquet", BASE_DIR / "data/benchmarks/tech_laptop/test.parquet"),
    ("home", BASE_DIR / "data/benchmarks/home_television/train.parquet", BASE_DIR / "data/benchmarks/home_television/test.parquet"),
    ("apparel", BASE_DIR / "data/benchmarks/apparel_tshirt/train.parquet", BASE_DIR / "data/benchmarks/apparel_tshirt/test.parquet"),
    ("fallback", BASE_DIR / "data/benchmarks/multi_aspect_mams/train.parquet", BASE_DIR / "data/benchmarks/multi_aspect_mams/test.parquet"),
]

DOMAIN2ID = {"tech": 0, "home": 1, "apparel": 2, "fallback": 3}


def extract_embeddings(df: pd.DataFrame, model: AspectTargetedSentimentModel, tokenizer: AutoTokenizer, max_n: int = 1000):
    sub_df = df.iloc[:max_n]
    all_embs = []
    with torch.no_grad():
        for _, row in sub_df.iterrows():
            text = str(row["text"])
            aspect = str(row["aspect"])
            enc = tokenizer(text, aspect, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            out = model.base_model.deberta(enc["input_ids"], enc["attention_mask"])
            emb = model.base_model.pooler(out[0]).squeeze(0).cpu()
            all_embs.append(emb)
    return torch.stack(all_embs)


def main():
    print(f"=== Training Learned Gating Router (Device: {DEVICE}) ===")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE)
    feature_extractor = AspectTargetedSentimentModel().to(DEVICE)
    feature_extractor.eval()

    train_embs, train_labels = [], []
    test_embs, test_labels = [], []

    for dom, tr_path, te_path in DOMAINS:
        print(f"Extracting embeddings for domain: {dom.upper()}...")
        tr_df = pd.read_parquet(tr_path)
        te_df = pd.read_parquet(te_path)

        tr_e = extract_embeddings(tr_df, feature_extractor, tokenizer, max_n=600)
        te_e = extract_embeddings(te_df, feature_extractor, tokenizer, max_n=300)

        train_embs.append(tr_e)
        train_labels.extend([DOMAIN2ID[dom]] * len(tr_e))
        test_embs.append(te_e)
        test_labels.extend([DOMAIN2ID[dom]] * len(te_e))

    X_train = torch.cat(train_embs, dim=0)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.cat(test_embs, dim=0)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    print(f"\nExtracted Training Features: {X_train.shape}, Test Features: {X_test.shape}")

    # Train Gating Network
    gate = LearnedGatingNetwork(hidden_size=768, num_experts=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-2)

    train_ds = TensorDataset(X_train, y_train)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    print("\nTraining Gating Network...")
    for epoch in range(1, 16):
        gate.train()
        total_loss = 0.0
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            logits = gate.gate(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate
    gate.eval()
    with torch.no_grad():
        test_logits = gate.gate(X_test.to(DEVICE))
        test_preds = torch.argmax(test_logits, dim=-1).cpu().numpy()

    acc = accuracy_score(y_test.numpy(), test_preds) * 100
    print(f"\nLearned Gating Router Test Accuracy: {acc:.2f}%")
    print("\nClassification Report across Domains:")
    target_names = ["tech", "home", "apparel", "fallback"]
    print(classification_report(y_test.numpy(), test_preds, target_names=target_names))

    # Save
    out_path = BASE_DIR / "models/learned_router.pt"
    torch.save(gate.state_dict(), out_path)
    print(f"Successfully saved learned router checkpoint to {out_path}")


if __name__ == "__main__":
    main()
