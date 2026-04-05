"""
baseline.py — FAISS float32 baseline index with cosine similarity.

Uses IndexFlatIP (inner product) after L2-normalizing vectors,
which is equivalent to cosine similarity.
"""

import numpy as np
import time
import faiss
from pathlib import Path
from typing import Optional


class FAISSBaseline:
    """
    FAISS flat inner-product index (cosine similarity after normalization).

    Provides the same .add() / .search() / .batch_search() interface
    as TernaryIndex so they can be swapped in benchmarks.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self._n = 0
        self._d = 0
        self._normalized_vecs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray):
        """
        L2-normalize and add float32 embeddings to a FAISS flat IP index.

        Args:
            embeddings: Float32 array of shape (N, D)
        """
        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        self._normalized_vecs = vecs

        self._n, self._d = vecs.shape
        self.index = faiss.IndexFlatIP(self._d)
        self.index.add(vecs)

        print(f"FAISSBaseline: stored {self._n} vectors of dim {self._d}")
        print(f"  Memory: {self.memory_mb:.2f} MB")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest vectors (cosine similarity).

        Args:
            query: Float32 array of shape (D,) or (1, D)
            k: Number of results

        Returns:
            (indices, scores) — both shape (k,)
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Call .add() first.")

        q = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return indices[0], scores[0]

    def batch_search(
        self, queries: np.ndarray, k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for multiple queries at once.

        Args:
            queries: Float32 array of shape (Q, D)
            k: Number of results per query

        Returns:
            (all_indices, all_scores) — both shape (Q, k)
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Call .add() first.")

        q = queries.astype(np.float32).copy()
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, k)
        return indices, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "embeddings/float32"):
        """Save FAISS index to directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "faiss.index"))
        print(f"Saved FAISS index to {p / 'faiss.index'}")

    def load(self, path: str = "embeddings/float32"):
        """Load FAISS index from directory."""
        p = Path(path)
        idx_path = p / "faiss.index"
        self.index = faiss.read_index(str(idx_path))
        self._n = self.index.ntotal
        self._d = self.index.d
        print(f"Loaded FAISS index from {idx_path}: {self._n} vectors")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"{self._n} vectors × {self._d} dims" if self.index else "empty"
        return f"FAISSBaseline({status})"

    @property
    def memory_mb(self) -> float:
        """Approximate RAM used by the float32 vectors."""
        if self.index is None:
            return 0.0
        # FAISS FlatIP stores raw float32 vectors
        return (self._n * self._d * 4) / 1e6


def benchmark_throughput(baseline: FAISSBaseline, queries: np.ndarray, k: int = 10) -> dict:
    """
    Measure query latency and throughput over a set of queries.

    Returns dict with: latency_ms_mean, latency_ms_p95, qps
    """
    n_queries = len(queries)
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        baseline.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    total_s = latencies.sum() / 1000

    return {
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "qps": n_queries / total_s,
    }
