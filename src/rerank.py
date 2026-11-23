"""Fusion and re-ranking: RRF, the tag overlap shown next to results, alpha-QE, Rocchio, MMR.

The re-rankers work on L2-normalized vectors of one representation (README: Feedback, Diversity):
- alpha-QE (automatic pseudo-relevance feedback): q' = norm(q + sum_{i<n} max(s_i, 0)^alpha x_i)
  over the top n results, s_i = cos(q, x_i).
- Rocchio (user-marked): q' = norm(alpha q + beta mean(relevant) - gamma mean(non-relevant)).
- MMR (Carbonell & Goldstein 1998): pick argmax lambda rel(d) - (1 - lambda) max_s sim(d, s),
  rel = previous score min-max scaled (so RRF scores compare with cosines), sim = cosine.
"""
import numpy as np

from src.schema import RERANKERS  # noqa: F401  (re-exported)


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank from 1.
    Sorted by fused score descending, image id ascending on ties."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[int(d)] = scores.get(int(d), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def shared_tags(query_terms: list[str], doc_terms: list[str]) -> list[str]:
    """Query terms that the document also has, in query order."""
    doc = set(doc_terms)
    return [t for t in query_terms if t in doc]


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else fallback).astype(np.float32)


def alpha_qe(query: np.ndarray, neighbors: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """query (d,), neighbors (m, d) = vectors of the initial top results in rank order."""
    nb = np.asarray(neighbors, dtype=np.float32)[:n]
    if len(nb) == 0:
        return np.asarray(query, dtype=np.float32)
    weights = np.clip(nb @ query, 0, None) ** alpha
    return _unit(query + weights @ nb, query)


def rocchio(query: np.ndarray, relevant: np.ndarray, nonrelevant: np.ndarray, alpha: float,
            beta: float, gamma: float) -> np.ndarray:
    """relevant / nonrelevant: (m, d) vectors of the results the user marked (either may be empty)."""
    q = alpha * np.asarray(query, dtype=np.float32)
    if len(relevant):
        q = q + beta * np.asarray(relevant, dtype=np.float32).mean(0)
    if len(nonrelevant):
        q = q - gamma * np.asarray(nonrelevant, dtype=np.float32).mean(0)
    return _unit(q, np.asarray(query, dtype=np.float32))


def mmr(hits: list[tuple[int, float]], vectors: np.ndarray, lam: float,
        depth: int) -> list[tuple[int, float]]:
    """Re-order the first `depth` hits (image id, score); vectors[i] belongs to hits[i].
    Later hits keep their order below them. Returned scores are the MMR values at selection."""
    head, tail = hits[:depth], hits[depth:]
    n = len(head)
    if n <= 1 or lam >= 1.0:
        return list(hits)
    s = np.array([sc for _, sc in head], dtype=np.float64)
    rel = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones(n)
    emb = np.asarray(vectors[:n], dtype=np.float32)
    sim = emb @ emb.T
    chosen = [int(np.argmax(rel))]
    picked = [lam * rel[chosen[0]]]
    max_sim = sim[chosen[0]].astype(np.float64)
    remaining = np.ones(n, dtype=bool)
    remaining[chosen[0]] = False
    for _ in range(n - 1):
        val = np.where(remaining, lam * rel - (1 - lam) * max_sim, -np.inf)
        j = int(np.argmax(val))
        chosen.append(j)
        picked.append(float(val[j]))
        remaining[j] = False
        max_sim = np.maximum(max_sim, sim[j])
    out = [(head[j][0], float(v)) for j, v in zip(chosen, picked)]
    floor = min(picked) - 1.0
    return out + [(d, floor - i * 1e-6) for i, (d, _) in enumerate(tail)]
