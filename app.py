#!/usr/bin/env python3
"""
Flask web app for semantic search over Steam game descriptions.
Configuration is via environment variables, not command-line arguments.
"""

import os
import json
import pandas as pd
import torch
from flask import Flask, render_template, request
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

# --- OpenMP workaround (for some PyTorch/Transformers setups) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Model Configuration ---
MODEL_CKPT = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configurable paths via environment variables (with sensible defaults) ---
JSON_PATH = os.getenv("JSON_PATH", "steam_games.json") # if not set use default steam_games.json
FAISS_PATH = os.getenv("FAISS_PATH", "steam_games.faiss") # if not set use default steam_games.faiss
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# --- Load Data & FAISS Index ---
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        games = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"{JSON_PATH} not found. Please provide a valid JSON file (set JSON_PATH env var).")

df = pd.DataFrame(games)
ds = Dataset.from_pandas(df)
try:
    ds.load_faiss_index("embeddings", FAISS_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load FAISS index from {FAISS_PATH}. Please provide a valid index (set FAISS_PATH env var).") from e

# --- Load Model & Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
model = AutoModel.from_pretrained(MODEL_CKPT).to(DEVICE)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def embed_query(text: str):
    enc = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    return cls_pooling(out).cpu().numpy()[0]

# --- Flask App ---
app = Flask(__name__, template_folder="templates")  # Ensure 'templates/home.html' exists

@app.route('/', methods=['GET', 'POST'])
def home():
    results = []
    if request.method == 'POST':
        query = request.form['query'].strip()
        if query:
            q_emb = embed_query(query)
            scores, retrieved_ds = ds.get_nearest_examples("embeddings", q_emb, k=5)
            for name, desc, dist in zip(
                retrieved_ds["game_name"],
                retrieved_ds["game_description"],
                scores
            ):
                results.append({
                    "name": name,
                    "description": desc,
                    "distance": float(dist)
                })
    return render_template("home.html", results=results)

if __name__ == '__main__':
    print(f"🚀 Starting Flask server at http://localhost:{PORT}")
    print(f"Using JSON: {JSON_PATH}")
    print(f"Using FAISS: {FAISS_PATH}")
    print("If you see OpenMP errors, they are safe to ignore for small Flask apps.")
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
