#!/usr/bin/env python3
"""
Flask web app for semantic search over Steam game descriptions.
Configuration is via environment variables.
"""

import os
# --- Disable extra thread pools in tokenizers and OpenMP duplicate lib errors ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, request
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel

# --- Model Configuration ---
MODEL_CKPT = os.getenv("MODEL_CKPT", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
DEVICE = torch.device("cpu")  # Force CPU-only to save memory

# --- Configurable paths via environment variables ---
JSON_PATH = os.getenv("JSON_PATH", "steam_games.json")
FAISS_PATH = os.getenv("FAISS_PATH", "steam_games.faiss")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# --- Lazy load placeholders ---
tokenizer = None
model = None

def load_model():
    """Lazy-load the tokenizer and model into memory when first needed."""
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
        model = AutoModel.from_pretrained(MODEL_CKPT).to(DEVICE)

# --- Embedding helper ---
def cls_pooling(model_output):
    """Extract [CLS] token embeddings from model output."""
    return model_output.last_hidden_state[:, 0]

def embed_query(text: str):
    """Generate a sentence embedding for the input text."""
    load_model()
    enc = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    return cls_pooling(out).cpu().numpy()[0]

# --- Load Data & FAISS Index when app starts ---
try:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        games = json.load(f)
except FileNotFoundError:
    raise RuntimeError(f"{JSON_PATH} not found. Please set the JSON_PATH environment variable.")

# Build a Hugging Face Dataset and load the FAISS index
df = pd.DataFrame(games)
ds = Dataset.from_pandas(df)
try:
    ds.load_faiss_index("embeddings", FAISS_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load FAISS index from {FAISS_PATH}: {e}")

# --- Flask App Initialization ---
app = Flask(__name__, template_folder="templates")

@app.route('/', methods=['GET', 'POST'])
def home():
    """Render the search form and display top-5 semantic matches."""
    results = []
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            q_emb = embed_query(query)
            scores, retrieved = ds.get_nearest_examples("embeddings", q_emb, k=5)
            for name, desc, dist in zip(
                retrieved["game_name"], retrieved["game_description"], scores
            ):
                results.append({
                    "name": name,
                    "description": desc,
                    "distance": float(dist)
                })
    return render_template("home.html", results=results)

if __name__ == '__main__':
    print(f"🚀 Starting Flask server at http://localhost:{PORT}")
    app.run(debug=DEBUG, host='0.0.0.0', port=PORT)
