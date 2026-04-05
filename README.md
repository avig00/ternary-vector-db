# Ternary-Native Vector Database

A vector search engine that converts standard float32 embeddings into ternary vectors {-1, 0, +1}, demonstrating dramatically reduced memory usage compared to FAISS with near-perfect recall via a two-stage reranking pipeline.

---

## Results

All benchmarks run on 50,000 MS MARCO passages, `all-MiniLM-L6-v2` embeddings (384-dim), 500 queries, k=10.

| Metric | FAISS Float32 | Ternary (δ=0.02) | Ternary + Reranker |
|---|---|---|---|
| Index RAM | 76.8 MB | 19.2 MB | 19.2 MB |
| QPS | 209 | 54 | 55 |
| Recall@10 | 100% (reference) | 86.5% | 99.9% |
| Sparsity | 0% | 31% | — |
| Compression | — | 4x | 4x |

### What the experiment showed

The 4x memory reduction is exact and free. int8 stores 4 bytes fewer per value than float32 — no approximation, no algorithmic tradeoff. This holds regardless of δ or embedding model.

Raw ternary recall tops out at ~86% for this model. `all-MiniLM-L6-v2` produces unit-normalized embeddings in a tight range (−0.28 to +0.28, std=0.05). Any δ > 0.05 zeros out most of the signal; any δ < 0.01 turns every dimension into ±1, losing magnitude distinction. The useful window is narrow, and the max recoverable recall within it is ~87%.

The more practical result is the reranking pipeline: use ternary to fetch the top-100 candidates, then rerank with full float cosine similarity. That gets to 99.9% Recall@10 at the same 4x memory footprint. The ternary index acts as a fast, cheap filter; the float reranker restores precision.

Ternary is conceptually faster than float search, but the speedup only materializes with SIMD-optimized bitpacking. Pure NumPy int16 matmul does not exploit sparsity and was slower than FAISS's heavily tuned float kernels on this hardware.

---

## Why Ternary? The Math Behind the Choice

### Radix Economy and the Proximity to e

The question "what base should a number system use?" has a precise mathematical answer. To represent an integer $N$ in base $r$ requires $\lceil \log_r N \rceil = \lceil \ln N / \ln r \rceil$ digits, and each digit requires $r$ distinguishable states. The total representational cost — states × digits — is proportional to:

$$E(r) = \frac{r}{\ln r}$$

To find the base that minimizes this, differentiate and set to zero:

$$\frac{dE}{dr} = \frac{\ln r - 1}{\ln^2 r} = 0 \implies \ln r = 1 \implies r = e \approx 2.718$$

The mathematically optimal radix is $e$, the base of the natural logarithm. Since we need an integer base, the question becomes which integer is closest to $e$:

$$|e - 2| = 0.718 \quad \text{(binary)}$$
$$|e - 3| = 0.282 \quad \text{(ternary)}$$

Ternary is 2.5x closer to the theoretical optimum than binary. This is the *radix economy*, and it means base-3 encodes information more efficiently per unit of hardware complexity than base-2. Each ternary trit carries $\log_2 3 \approx 1.585$ bits of information — 58% more than a binary bit — while requiring only 50% more states.

In concrete terms: a 384-dimensional ternary vector carries the equivalent of $384 \times \log_2 3 \approx 609$ bits of directional information, versus 384 bits for binary. This is why ternary quantization can preserve so much recall despite the dramatic compression.

### Why $\{-1, 0, +1\}$ and Not $\{0, 1, 2\}$?

Both are valid ternary alphabets. The choice of $\{-1, 0, +1\}$ is not arbitrary — it is the only one that makes semantic sense for similarity search.

Neural network embeddings are trained to be approximately zero-centered. A dimension with a large positive value indicates one semantic direction; a large negative value indicates the opposite. The value set $\{-1, 0, +1\}$ mirrors this structure exactly. The set $\{0, 1, 2\}$ is asymmetric — it treats zero and "mild positive" as the two low-signal states, which has no geometric meaning in embedding space.

The dot product also works out correctly. With $\{-1, 0, +1\}$, the inner product between two ternary vectors $\mathbf{a}$ and $\mathbf{b}$ is:

$$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = |\{i : a_i = b_i \neq 0\}| - |\{i : a_i \neq b_i, \, a_i b_i \neq 0\}|$$

That is: agreements minus disagreements, ignoring dimensions where either vector is zero. This is a natural measure of directional alignment — exactly what cosine similarity computes in float space. With $\{0, 1, 2\}$, the dot product produces values of $0, 1, 2, 4$ for the four non-zero combinations, which conflates magnitude with direction and has no clean geometric interpretation.

There is also a subtler point about what zero means. In $\{0, 1, 2\}$, zero is ambiguous — it could mean "no signal" or simply the lowest positive value. In $\{-1, 0, +1\}$, zero unambiguously means "this dimension carries no information about the query", which is exactly what the $\delta$ threshold enforces. The resulting sparsity concentrates the dot product on dimensions that actually discriminate between documents, rather than computing over noise.

Finally, $\{-1, 0, +1\}$ is closed under negation: flipping all signs gives another valid vector representing the opposite semantic direction. This is not true for $\{0, 1, 2\}$, where negation takes you outside the alphabet. Signed symmetry matters for representing contrast and opposition in semantic space.

---

## How It Works

The quantizer maps each float dimension to {-1, 0, +1} using a threshold δ:

```
value < -δ   →  -1   (strong negative signal)
-δ ≤ v ≤ +δ  →   0   (weak/irrelevant — zeroed out)
value > +δ   →  +1   (strong positive signal)
```

The δ threshold controls sparsity. A larger δ zeros out more dimensions (more compression, lower recall). The `02_quantization` notebook sweeps δ from 0.005 to 0.30 and identifies the elbow where recall starts to collapse.

For `all-MiniLM-L6-v2`, δ = 0.02 is optimal: 31% sparsity, 86.5% Recall@10.

### Two-Stage Reranking Pipeline

```
Query (float) ──→ Ternary search (top-100) ──→ Float cosine rerank (top-10)
                  [fast, low-memory]            [precise, small candidate set]
```

Stage 1 uses the 19.2 MB ternary index to quickly eliminate 99.8% of the corpus. Stage 2 applies exact cosine similarity only to the 100 survivors, which is cheap. Combined result: 99.9% Recall@10 at 4x lower memory than a pure float index.

---

## Project Structure

```
ternary-vector-db/
├── src/
│   ├── embed.py        # Generate float32 embeddings from MS MARCO
│   ├── quantize.py     # Float → ternary conversion + delta sweep
│   ├── index.py        # TernaryIndex class (.add / .search / .batch_search)
│   ├── baseline.py     # FAISSBaseline class (same interface)
│   ├── rerank.py       # Two-stage reranking pipeline
│   └── benchmark.py    # Head-to-head evaluation: FAISS vs Ternary vs Reranker
├── notebooks/
│   ├── 01_exploration.ipynb   # Dataset inspection + embedding distribution
│   ├── 02_quantization.ipynb  # Delta sweep, optimal threshold selection
│   └── 03_results.ipynb       # Final benchmark charts
├── results/
│   ├── chart1_recall_vs_delta.png
│   ├── chart2_memory_comparison.png
│   ├── chart3_throughput_recall.png
│   ├── delta_sweep.csv
│   └── benchmark_results.csv
├── embeddings/
│   ├── float32/        # Saved float32 .npy arrays (76.8 MB)
│   └── ternary/        # Saved int8 .npy arrays + chosen delta (19.2 MB)
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate embeddings (~5 min)
python src/embed.py --limit 50000

# 2. Quantize to ternary
python src/quantize.py --delta 0.02

# 3. Run benchmark
PYTHONPATH=. python src/benchmark.py --delta 0.02 --candidates 100

# 4. Explore interactively
jupyter notebook notebooks/
```

Run notebooks in order: `01_exploration` → `02_quantization` → `03_results`

---

## Stack

- Python 3.10, NumPy, PyTorch
- sentence-transformers (`all-MiniLM-L6-v2`, 384-dim)
- FAISS (cosine similarity baseline)
- HuggingFace datasets (MS MARCO passage corpus)

---

## Connection to Ternary SNNs

This project is a direct precursor to a Ternary Spiking Neural Network implementation. The δ threshold here maps to the membrane firing threshold in a spiking neuron; ternary weights {-1, 0, +1} are identical to ternary synaptic weights in TWN-style networks; and the sparsity pattern mirrors sparse spike trains in biological neural networks, where most neurons are silent most of the time.

The radix economy argument applies there too: ternary synapses carry more information per weight than binary synapses, which is part of why biological synapses are graded rather than all-or-nothing.

Reference: [Ternary Weight Networks (TWN, Li et al. 2016)](https://arxiv.org/abs/1605.04711)
