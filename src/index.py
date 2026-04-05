"""
index.py — Ternary vector index with dot-product search.

Storage: int8 arrays — the ternary values {-1, 0, +1} fit in int8.
Search:  ternary dot product, which counts agreements (+1·+1 or -1·-1)
         minus disagreements (+1·-1), ignoring zeros.
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

from src.quantize import quantize, sparsity


class TernaryIndex:
    """
    In-memory ternary vector index.

    Stores vectors as int8 and uses integer dot product for search.
    Supports batch queries and returns both indices and scores.
    """

    def __init__(self, delta: float = 0.1):
        """
        Args:
            delta: Threshold for quantization. Values in (-delta, +delta) → 0.
        """
        self.delta = delta
        self.vectors: Optional[np.ndarray] = None  # shape (N, D), dtype int8
        self._n = 0
        self._d = 0

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray):
        """
        Quantize and store float32 embeddings.

        Args:
            embeddings: Float32 array of shape (N, D)
        """
        self.vectors = quantize(embeddings, self.delta)
        self._n, self._d = self.vectors.shape
        print(f"TernaryIndex: stored {self._n} vectors of dim {self._d}")
        print(f"  δ={self.delta}, sparsity={sparsity(self.vectors):.1%}")
        print(f"  Memory: {self.vectors.nbytes / 1e6:.2f} MB")

    def add_precomputed(self, ternary: np.ndarray):
        """Add already-quantized int8 vectors directly."""
        assert ternary.dtype == np.int8, "Expected int8 array"
        self.vectors = ternary
        self._n, self._d = self.vectors.shape

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest vectors to query.

        Args:
            query: Float32 array of shape (D,) or (1, D)
            k: Number of results to return

        Returns:
            (indices, scores) — both shape (k,), sorted descending by score
        """
        if self.vectors is None:
            raise RuntimeError("Index is empty. Call .add() first.")

        q = query.reshape(1, -1)
        q_ternary = quantize(q, self.delta).astype(np.int16)  # (1, D)

        # Dot product: (N, D) · (D, 1) → (N, 1)
        scores = (self.vectors.astype(np.int16) @ q_ternary.T).flatten()

        k = min(k, self._n)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return top_k_idx, scores[top_k_idx]

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
        if self.vectors is None:
            raise RuntimeError("Index is empty. Call .add() first.")

        q_ternary = quantize(queries, self.delta).astype(np.int16)  # (Q, D)

        # (N, D) · (D, Q) → (N, Q), then transpose → (Q, N)
        scores_matrix = (self.vectors.astype(np.int16) @ q_ternary.T).T  # (Q, N)

        k = min(k, self._n)
        all_indices = np.zeros((len(queries), k), dtype=np.int64)
        all_scores = np.zeros((len(queries), k), dtype=np.int32)

        for i, scores in enumerate(scores_matrix):
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            all_indices[i] = top_k_idx
            all_scores[i] = scores[top_k_idx]

        return all_indices, all_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "embeddings/ternary"):
        """Save index vectors and delta to directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "embeddings.npy", self.vectors)
        np.save(p / "delta.npy", np.array([self.delta]))
        print(f"Saved index to {p}")

    def load(self, path: str = "embeddings/ternary"):
        """Load index vectors and delta from directory."""
        p = Path(path)
        self.vectors = np.load(p / "embeddings.npy")
        self.delta = float(np.load(p / "delta.npy")[0])
        self._n, self._d = self.vectors.shape
        print(f"Loaded index from {p}: {self._n} vectors, δ={self.delta}")

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"{self._n} vectors × {self._d} dims" if self.vectors is not None else "empty"
        return f"TernaryIndex(delta={self.delta}, {status})"

    @property
    def memory_mb(self) -> float:
        if self.vectors is None:
            return 0.0
        return self.vectors.nbytes / 1e6


def benchmark_throughput(index: TernaryIndex, queries: np.ndarray, k: int = 10) -> dict:
    """
    Measure query latency and throughput over a set of queries.

    Returns dict with: latency_ms_mean, latency_ms_p95, qps
    """
    n_queries = len(queries)
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        index.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    total_s = latencies.sum() / 1000

    return {
        "latency_ms_mean": float(latencies.mean()),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "qps": n_queries / total_s,
    }
