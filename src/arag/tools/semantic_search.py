"""Semantic retrieval over searchable child chunks."""

import logging
import pickle
from typing import Any

import numpy as np
import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag, is_clearly_low_value
from ..context import AgentContext
from ..retrieval import ChunkCorpus, SearchResult

logger = logging.getLogger(__name__)
_tokenizer = tiktoken.get_encoding("cl100k_base")


class SemanticSearchTool(BaseTool):
    def __init__(self, index_path: str, embed_fn, corpus: ChunkCorpus):
        self._embed_fn = embed_fn
        self._corpus = corpus
        self._load_index(index_path)

    def _load_index(self, path: str):
        from pathlib import Path

        p = Path(path)
        npz_path = p.with_name("sentence_index.npz")
        meta_path = p.with_name("sentence_meta.pkl")

        if npz_path.exists() and meta_path.exists():
            data = np.load(str(npz_path))
            self._embeddings = data["embeddings"].astype(np.float32)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        else:
            with open(path, "rb") as f:
                meta = pickle.load(f)
            self._embeddings = meta["embeddings"]

        self._texts = meta["sentences"]
        self._text_to_chunk = meta["sentence_to_chunk"]
        self._chunks = meta["chunks"]
        logger.info(f"Loaded semantic index: {len(self._texts)} entries")

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "意味検索で関連する会計基準を探します。自然文や概念質問向けです。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_vec = self._embed_fn(query)
        if query_vec is None:
            return []
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm
        similarities = self._embeddings @ query_vec

        scored: list[SearchResult] = []
        for idx, similarity in enumerate(similarities):
            chunk_id = self._text_to_chunk[idx]
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue
            if is_clearly_low_value(chunk.get("text", "")):
                continue
            parent = self._corpus.get_parent(chunk_id) or chunk
            scored.append(
                SearchResult(
                    chunk_id=chunk_id,
                    parent_id=parent["id"],
                    score=float(similarity),
                    source=parent.get("source", chunk.get("source", "")),
                    snippet=self._texts[idx][:300],
                    text=parent.get("text", chunk.get("text", "")),
                    metadata=parent,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return self._corpus.dedupe_to_parents(scored, top_k)

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        query = kwargs.get("query", "")
        top_k = min(int(kwargs.get("top_k", 10)), 30)
        if not query:
            return "検索クエリを指定してください。", {"error": "no query"}
        ranked = self.search(query, top_k)
        if not ranked:
            return "関連する文書が見つかりませんでした。", {"matches": 0}

        lines = []
        chunk_ids = []
        snippets = []
        for item in ranked:
            chunk_ids.append(item.parent_id)
            tag = get_chunk_tag(item.text)
            tag_str = f" {tag}" if tag else ""
            lines.append(f"[Chunk {item.parent_id}] (semantic: {item.score:.3f}){tag_str} {item.source}")
            lines.append(f"  > {item.snippet}")
            snippets.append(item.snippet)

        unread_ids = [cid for cid in chunk_ids if not context.is_chunk_read(cid) and not get_chunk_tag((self._corpus.get_parent(cid) or {}).get("text", ""))]
        if unread_ids:
            lines.append(f"\n--- {len(unread_ids)}件の未読チャンクがあります。read_chunkで全文を取得してください ---")
            lines.append(f"read_chunk(chunk_ids={unread_ids})")

        retrieved_tokens = len(_tokenizer.encode("\n".join(snippets))) if snippets else 0
        context.add_retrieval_log(
            tool_name=self.name,
            tokens=retrieved_tokens,
            metadata={"query": query, "chunks_found": len(ranked), "chunk_ids": chunk_ids},
        )
        return "\n".join(lines), {
            "matches": len(ranked),
            "query": query,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
        }
