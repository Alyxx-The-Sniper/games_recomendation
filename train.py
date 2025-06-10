#!/usr/bin/env python3

import argparse
import json
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_CKPT = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(texts, tokenizer, model, device):
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded)
    return model_output.last_hidden_state[:, 0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Create FAISS index from game descriptions JSON.")
    parser.add_argument("--json_path", type=str, default="steam_games.json", help="Input JSON file path.")
    parser.add_argument("--faiss_path", type=str, default="steam_games.faiss", help="Output FAISS file path.")
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        games = json.load(f)

    df = pd.DataFrame(games)
    dataset = Dataset.from_pandas(df)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
    model = AutoModel.from_pretrained(MODEL_CKPT).to(DEVICE)

    print("Computing embeddings...")
    embeddings = get_embeddings(df["game_description"].tolist(), tokenizer, model, DEVICE)
    dataset = dataset.add_column("embeddings", embeddings.tolist())

    print("Building FAISS index...")
    dataset.add_faiss_index(column="embeddings")

    print(f"Saving FAISS index to {args.faiss_path}...")
    dataset.save_faiss_index("embeddings", args.faiss_path)

    print("FAISS index created successfully.")

if __name__ == "__main__":
    main()
