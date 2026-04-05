"""
benchmark.py — Head-to-head evaluation: FAISS float32 vs. TernaryIndex vs. Reranker.

Metrics captured for all systems:
  - RAM usage (MB)
  - Index build time (s)
  - Query latency: mean, p50, p95 (ms)
  - Queries per second (QPS)
  - Recall@10 (fraction of true top-10 recovered)

Usage:
    python src/benchmark.py [--n-queries 500] [--k 10] [--delta 0.02]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from src.baseline import FAISSBaseline
from src.index import TernaryIndex
from src.quantize import quantize, sparsity
from src.rerank import RerankingPipeline

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def process_memory_mb() -> float:
    """Current RSS of this process in MB."""
    return psutil.Process().memory_info().rss / 1e6


def compute_recall_at_k(
    true_indices: np.ndarray,
    pred_indices: np.ndarray,
    k: int = 10,
) -> float:
    """
    Recall@k: fraction of true top-k results found in predicted top-k.

    Args:
        true_indices: (Q, k) ground truth indices
        pred_indices: (Q, k) predicted indices
        k: cutoff

    Returns:
        Mean recall across all queries (0.0 – 1.0)
    """
    recalls = []
    for true, pred in zip(true_indices, pred_indices):
        true_set = set(true[:k])
        pred_set = set(pred[:k])
        recalls.append(len(true_set & pred_set) / k)
    return float(np.mean(recalls))


def sample_query_embeddings(
    embeddings: np.ndarray,
    n: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (query_embeddings, query_ids) — random sample from corpus."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(embeddings), size=n, replace=False)
    return embeddings[idx], idx


# ---------------------------------------------------------------------------
# Build + measure helpers
# ---------------------------------------------------------------------------

def build_faiss(embeddings: np.ndarray) -> tuple[FAISSBaseline, float, float]:
    """
    Build and time the FAISS index. Returns (index, build_time_s, ram_mb).
    """
    mem_before = process_memory_mb()
    t0 = time.perf_counter()
    baseline = FAISSBaseline()
    baseline.add(embeddings)
    build_time = time.perf_counter() - t0
    ram_used = process_memory_mb() - mem_before
    # Use direct byte count for the stored vectors (more reliable than RSS delta)
    return baseline, build_time, baseline.memory_mb


def build_ternary(embeddings: np.ndarray, delta: float) -> tuple[TernaryIndex, float, float]:
    """
    Build and time the TernaryIndex. Returns (index, build_time_s, ram_mb).
    """
    t0 = time.perf_counter()
    idx = TernaryIndex(delta=delta)
    idx.add(embeddings)
    build_time = time.perf_counter() - t0
    return idx, build_time, idx.memory_mb


def measure_latency(index, queries: np.ndarray, k: int = 10) -> dict:
    """Run per-query search and collect latency stats."""
    latencies_ms = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": float(np.percentile(arr, 50)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "qps": len(queries) / (arr.sum() / 1000),
    }


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------

def evaluate_recall(
    baseline: FAISSBaseline,
    candidate_idx,
    queries: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Recall@k for any index vs. FAISS ground truth.

    Ground truth = FAISS top-k. Works with TernaryIndex or RerankingPipeline.
    """
    print(f"Computing Recall@{k} over {len(queries)} queries...")
    true_indices, _ = baseline.batch_search(queries, k=k)
    pred_indices, _ = candidate_idx.batch_search(queries, k=k)
    return compute_recall_at_k(true_indices, pred_indices, k)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_benchmark(
    embeddings_path: str = "embeddings/float32/embeddings.npy",
    n_queries: int = 500,
    k: int = 10,
    delta: float = 0.02,
    candidates: int = 100,
    output_csv: str = "results/benchmark_results.csv",
):
    print("=" * 60)
    print("Ternary Vector DB — Benchmark")
    print("=" * 60)

    # Load embeddings
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    print(f"  Shape: {embeddings.shape}  ({embeddings.nbytes / 1e6:.1f} MB)")

    # Sample queries
    queries, query_ids = sample_query_embeddings(embeddings, n=n_queries)
    print(f"  Sampled {n_queries} queries (ids: {query_ids[:5]}...)")

    # ------------------------------------------------------------------
    # FAISS baseline
    # ------------------------------------------------------------------
    print("\n--- FAISS Float32 Baseline ---")
    faiss_idx, faiss_build_s, faiss_ram = build_faiss(embeddings)
    faiss_latency = measure_latency(faiss_idx, queries, k=k)

    print(f"  Build time : {faiss_build_s:.3f}s")
    print(f"  RAM        : {faiss_ram:.1f} MB")
    print(f"  Latency    : {faiss_latency['latency_ms_mean']:.2f}ms mean")
    print(f"  QPS        : {faiss_latency['qps']:.0f}")

    # ------------------------------------------------------------------
    # Ternary index
    # ------------------------------------------------------------------
    print(f"\n--- Ternary Index (δ={delta}) ---")
    ternary_idx, ternary_build_s, ternary_ram = build_ternary(embeddings, delta=delta)
    ternary_latency = measure_latency(ternary_idx, queries, k=k)

    ternary_sparsity = sparsity(ternary_idx.vectors)
    print(f"  Build time : {ternary_build_s:.3f}s")
    print(f"  RAM        : {ternary_ram:.1f} MB")
    print(f"  Sparsity   : {ternary_sparsity:.1%}")
    print(f"  Latency    : {ternary_latency['latency_ms_mean']:.2f}ms mean")
    print(f"  QPS        : {ternary_latency['qps']:.0f}")

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    recall = evaluate_recall(faiss_idx, ternary_idx, queries, k=k)
    print(f"\n  Recall@{k}   : {recall:.1%}")

    # ------------------------------------------------------------------
    # Reranking pipeline
    # ------------------------------------------------------------------
    print(f"\n--- Ternary + Float Reranker (candidates={candidates}) ---")
    reranker = RerankingPipeline(ternary_idx, embeddings, candidates=candidates)
    rerank_latency = measure_latency(reranker, queries, k=k)
    rerank_recall = evaluate_recall(faiss_idx, reranker, queries, k=k)

    print(f"  RAM (index only) : {ternary_idx.memory_mb:.1f} MB")
    print(f"  Latency    : {rerank_latency['latency_ms_mean']:.2f}ms mean")
    print(f"  QPS        : {rerank_latency['qps']:.0f}")
    print(f"  Recall@{k}   : {rerank_recall:.1%}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("%-25s %12s %12s %12s" % ("Metric", "FAISS", "Ternary", "Reranker"))
    print("-" * 72)
    print("%-25s %12.1f %12.1f %12.1f" % ("RAM (MB)", faiss_ram, ternary_ram, ternary_ram))
    print("%-25s %12.3f %12.3f %12s" % ("Build time (s)", faiss_build_s, ternary_build_s, "—"))
    print("%-25s %12.2f %12.2f %12.2f" % ("Latency mean (ms)", faiss_latency["latency_ms_mean"], ternary_latency["latency_ms_mean"], rerank_latency["latency_ms_mean"]))
    print("%-25s %12.0f %12.0f %12.0f" % ("QPS", faiss_latency["qps"], ternary_latency["qps"], rerank_latency["qps"]))
    print("%-25s %12s %11.1f%% %11.1f%%" % ("Recall@10", "100.0%", recall * 100, rerank_recall * 100))
    print("%-25s %12s %11.1f%% %12s" % ("Sparsity", "0.0%", ternary_sparsity * 100, "—"))
    print("=" * 72)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    results = {
        "delta": delta,
        "n_corpus": len(embeddings),
        "n_queries": n_queries,
        "k": k,
        # FAISS
        "faiss_ram_mb": faiss_ram,
        "faiss_build_s": faiss_build_s,
        "faiss_latency_ms_mean": faiss_latency["latency_ms_mean"],
        "faiss_latency_ms_p50": faiss_latency["latency_ms_p50"],
        "faiss_latency_ms_p95": faiss_latency["latency_ms_p95"],
        "faiss_qps": faiss_latency["qps"],
        # Ternary
        "ternary_ram_mb": ternary_ram,
        "ternary_build_s": ternary_build_s,
        "ternary_latency_ms_mean": ternary_latency["latency_ms_mean"],
        "ternary_latency_ms_p50": ternary_latency["latency_ms_p50"],
        "ternary_latency_ms_p95": ternary_latency["latency_ms_p95"],
        "ternary_qps": ternary_latency["qps"],
        "ternary_sparsity": ternary_sparsity,
        # Reranker
        "rerank_candidates": candidates,
        "rerank_latency_ms_mean": rerank_latency["latency_ms_mean"],
        "rerank_qps": rerank_latency["qps"],
        "rerank_recall_at_k": rerank_recall,
        # Comparison
        "recall_at_k": recall,
        "memory_compression": faiss_ram / max(ternary_ram, 1e-9),
        "speedup": ternary_latency["qps"] / max(faiss_latency["qps"], 1e-9),
    }

    csv_path = Path(output_csv)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)

    print(f"\nResults saved to {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ternary vs. FAISS benchmark")
    parser.add_argument("--embeddings", default="embeddings/float32/embeddings.npy")
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--output", default="results/benchmark_results.csv")
    args = parser.parse_args()

    run_benchmark(
        embeddings_path=args.embeddings,
        n_queries=args.n_queries,
        k=args.k,
        delta=args.delta,
        candidates=args.candidates,
        output_csv=args.output,
    )
