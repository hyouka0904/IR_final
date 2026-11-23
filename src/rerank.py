"""Reciprocal rank fusion and the tag overlap shown next to results."""


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
