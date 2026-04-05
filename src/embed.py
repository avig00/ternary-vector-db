"""
embed.py — Generate float32 embeddings from MS MARCO passages.

Usage:
    python src/embed.py [--limit 50000] [--batch-size 64]
"""

import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

EMBED_DIR = Path("embeddings/float32")
DATA_DIR = Path("data/raw")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_passages(limit: int = 50_000) -> list[str]:
    """Load MS MARCO passages from HuggingFace datasets."""
    from datasets import load_dataset

    print(f"Loading MS MARCO dataset (first {limit} examples)...")
    dataset = load_dataset("ms_marco", "v1.1", split=f"train[:{limit}]")

    passages = []
    for item in tqdm(dataset, desc="Extracting passages"):
        # Each item has a 'passages' dict with 'passage_text' list
        for text in item["passages"]["passage_text"]:
            passages.append(text)
            if len(passages) >= limit:
                break
        if len(passages) >= limit:
            break

    print(f"Loaded {len(passages)} passages.")
    return passages


def generate_embeddings(passages: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode passages using SentenceTransformer (384-dim float32)."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(passages)} passages in batches of {batch_size}...")
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize separately for FAISS
    )
    print(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32)


def save_embeddings(embeddings: np.ndarray, passages: list[str]):
    """Save embeddings and passages to disk."""
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    emb_path = EMBED_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {emb_path}  ({embeddings.nbytes / 1e6:.1f} MB)")

    passages_path = DATA_DIR / "passages.npy"
    np.save(passages_path, np.array(passages, dtype=object))
    print(f"Saved passages: {passages_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate float32 embeddings")
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    passages = load_passages(args.limit)
    embeddings = generate_embeddings(passages, args.batch_size)
    save_embeddings(embeddings, passages)

    print("\nDone. Embedding stats:")
    print(f"  Shape : {embeddings.shape}")
    print(f"  dtype : {embeddings.dtype}")
    print(f"  Mean  : {embeddings.mean():.4f}")
    print(f"  Std   : {embeddings.std():.4f}")
    print(f"  Min   : {embeddings.min():.4f}")
    print(f"  Max   : {embeddings.max():.4f}")


if __name__ == "__main__":
    main()
