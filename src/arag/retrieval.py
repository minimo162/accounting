"""Retrieval data structures and helpers."""

from dataclasses import dataclass, field
import math
import re


_TOKEN_RE = re.compile(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize_for_bm25(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [tok for tok in tokens if tok]


@dataclass
class SearchResult:
    chunk_id: str
    parent_id: str
    score: float
    source: str
    snippet: str
    text: str
    metadata: dict = field(default_factory=dict)


class ChunkCorpus:
    """Access helpers for parent/child chunk structures."""

    def __init__(self, chunks: list[dict]):
        self.chunk_map = {chunk["id"]: chunk for chunk in chunks}
        self.parent_chunks = [chunk for chunk in chunks if chunk.get("level", "parent") == "parent"]
        self.child_chunks = [chunk for chunk in chunks if chunk.get("level", "parent") == "child"]
        self.searchable_chunks = self.child_chunks or self.parent_chunks
        self.parent_map = {chunk["id"]: chunk for chunk in self.parent_chunks} or self.chunk_map
        self.children_by_parent: dict[str, list[dict]] = {}
        for chunk in self.searchable_chunks:
            parent_id = chunk.get("parent_id", chunk["id"])
            self.children_by_parent.setdefault(parent_id, []).append(chunk)
        self.parent_order_by_file: dict[str, list[str]] = {}
        for chunk in self.parent_chunks or self.searchable_chunks:
            self.parent_order_by_file.setdefault(chunk.get("file", ""), []).append(chunk["id"])
        for ids in self.parent_order_by_file.values():
            ids.sort(key=self._parent_sort_key)

    @staticmethod
    def _parent_sort_key(chunk_id: str) -> tuple[int, str]:
        m = re.search(r":p?(\d+)$", chunk_id)
        num = int(m.group(1)) if m else math.inf
        return num, chunk_id

    def get_parent(self, chunk_id: str) -> dict | None:
        chunk = self.chunk_map.get(chunk_id)
        if not chunk:
            return None
        parent_id = chunk.get("parent_id", chunk["id"])
        return self.parent_map.get(parent_id, chunk)

    def get_adjacent_parent_ids(self, chunk_id: str) -> tuple[str | None, str | None]:
        parent = self.get_parent(chunk_id)
        if not parent:
            return None, None
        ids = self.parent_order_by_file.get(parent.get("file", ""), [])
        try:
            idx = ids.index(parent["id"])
        except ValueError:
            return None, None
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None
        return prev_id, next_id

    def dedupe_to_parents(self, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        best_by_parent: dict[str, SearchResult] = {}
        for result in results:
            current = best_by_parent.get(result.parent_id)
            if current is None or result.score > current.score:
                best_by_parent[result.parent_id] = result
        deduped = sorted(best_by_parent.values(), key=lambda item: item.score, reverse=True)
        return deduped[:top_k]


class BM25Index:
    """Minimal BM25 implementation over searchable chunks."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.docs = [tokenize_for_bm25(chunk["text"]) for chunk in chunks]
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in self.docs:
            freqs: dict[str, int] = {}
            for token in doc:
                freqs[token] = freqs.get(token, 0) + 1
            self.tf.append(freqs)
            for token in freqs:
                self.df[token] = self.df.get(token, 0) + 1
        self.n_docs = len(self.docs)

    def score(self, query: str) -> list[tuple[int, float]]:
        query_terms = tokenize_for_bm25(query)
        scores: list[tuple[int, float]] = []
        for idx, freqs in enumerate(self.tf):
            score = 0.0
            doc_len = self.doc_lens[idx]
            for term in query_terms:
                df = self.df.get(term)
                if not df:
                    continue
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-6))
                score += idf * (tf * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores


def reciprocal_rank_fusion(rankings: list[list[SearchResult]], rrf_k: int = 60) -> list[SearchResult]:
    fused: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.parent_id] = scores.get(item.parent_id, 0.0) + 1.0 / (rrf_k + rank)
            if item.parent_id not in fused or item.score > fused[item.parent_id].score:
                fused[item.parent_id] = item
    final = []
    for parent_id, item in fused.items():
        final.append(
            SearchResult(
                chunk_id=item.chunk_id,
                parent_id=parent_id,
                score=scores[parent_id],
                source=item.source,
                snippet=item.snippet,
                text=item.text,
                metadata=item.metadata,
            )
        )
    final.sort(key=lambda item: item.score, reverse=True)
    return final
