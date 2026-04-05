"""
quantize.py — Float32 → Ternary {-1, 0, +1} conversion.

The delta threshold δ controls sparsity:
    value < -δ   →  -1  (strong negative)
    -δ ≤ v ≤ δ   →   0  (weak / irrelevant)
    value > +δ   →  +1  (strong positive)
"""

import numpy as np
from pathlib import Path

TERNARY_DIR = Path("embeddings/ternary")


def quantize(embeddings: np.ndarray, delta: float) -> np.ndarray:
    """
    Quantize float embeddings to ternary {-1, 0, +1} using threshold δ.

    Args:
        embeddings: Float array of shape (N, D) or (D,)
        delta: Threshold value. Dimensions in [-δ, +δ] become 0.

    Returns:
        int8 array of same shape with values in {-1, 0, +1}
    """
    ternary = np.zeros_like(embeddings, dtype=np.int8)
    ternary[embeddings > delta] = 1
    ternary[embeddings < -delta] = -1
    return ternary


def sparsity(ternary: np.ndarray) -> float:
    """Returns the fraction of values that are zero."""
    return float((ternary == 0).mean())


def memory_mb(arr: np.ndarray) -> float:
    """Return the memory footprint of an array in megabytes."""
    return arr.nbytes / 1e6


def delta_sweep(
    embeddings: np.ndarray,
    deltas: np.ndarray | None = None,
) -> list[dict]:
    """
    Sweep over a range of δ values and record sparsity + memory for each.

    Args:
        embeddings: Float32 array of shape (N, D)
        deltas: 1D array of δ values to test. Defaults to linspace(0.01, 0.5, 50).

    Returns:
        List of dicts with keys: delta, sparsity, memory_mb
    """
    if deltas is None:
        deltas = np.linspace(0.01, 0.5, 50)

    results = []
    for d in deltas:
        t = quantize(embeddings, float(d))
        results.append(
            {
                "delta": float(d),
                "sparsity": sparsity(t),
                "memory_mb": memory_mb(t),
            }
        )
    return results


def save_ternary(ternary: np.ndarray, delta: float):
    """Save ternary embeddings and the chosen delta to disk."""
    TERNARY_DIR.mkdir(parents=True, exist_ok=True)
    path = TERNARY_DIR / "embeddings.npy"
    np.save(path, ternary)
    np.save(TERNARY_DIR / "delta.npy", np.array([delta]))
    print(f"Saved ternary embeddings: {path}  ({ternary.nbytes / 1e6:.1f} MB)")
    print(f"Saved delta: {delta}")


def load_ternary() -> tuple[np.ndarray, float]:
    """Load ternary embeddings and the stored delta from disk."""
    path = TERNARY_DIR / "embeddings.npy"
    delta_path = TERNARY_DIR / "delta.npy"
    if not path.exists():
        raise FileNotFoundError(f"Ternary embeddings not found at {path}. Run quantize first.")
    ternary = np.load(path)
    delta = float(np.load(delta_path)[0]) if delta_path.exists() else 0.1
    return ternary, delta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize float embeddings to ternary")
    parser.add_argument("--delta", type=float, default=0.1, help="Threshold δ")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="embeddings/float32/embeddings.npy",
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    embeddings = np.load(args.embeddings)
    print(f"Shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    ternary = quantize(embeddings, args.delta)
    s = sparsity(ternary)
    print(f"\nδ = {args.delta}")
    print(f"  Sparsity  : {s:.1%}")
    print(f"  Float mem : {memory_mb(embeddings):.1f} MB")
    print(f"  Ternary mem: {memory_mb(ternary):.1f} MB")
    print(f"  Compression: {memory_mb(embeddings) / memory_mb(ternary):.1f}x")

    save_ternary(ternary, args.delta)
