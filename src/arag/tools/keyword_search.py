"""Keyword and BM25-style retrieval over searchable child chunks."""

import re
from typing import Any

import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag, is_clearly_low_value
from ..context import AgentContext
from ..retrieval import BM25Index, ChunkCorpus, SearchResult

_SENTENCE_RE = re.compile(r"[。．.!！?？\n]+")
_tokenizer = tiktoken.get_encoding("cl100k_base")


class KeywordSearchTool(BaseTool):
    def __init__(self, corpus: ChunkCorpus):
        self._corpus = corpus
        self._chunks = corpus.searchable_chunks
        self._chunk_map = corpus.chunk_map
        self._bm25 = BM25Index(self._chunks)

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "キーワード完全一致とBM25で会計基準を検索します。条項番号や基準名向けです。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["keywords"],
            },
        }

    def search(self, keywords: list[str], top_k: int = 10) -> list[SearchResult]:
        query = " ".join(keywords)
        bm25_scores = self._bm25.score(query)
        scored: list[SearchResult] = []
        for idx, bm25_score in bm25_scores:
            chunk = self._chunks[idx]
            text = chunk["text"]
            if is_clearly_low_value(text):
                continue
            parent = self._corpus.get_parent(chunk["id"]) or chunk
            snippets: list[str] = []
            for kw in keywords:
                for sent in _SENTENCE_RE.split(text):
                    if kw.lower() in sent.lower() and len(sent.strip()) > 5:
                        snippet = sent.strip()[:200]
                        if snippet not in snippets:
                            snippets.append(snippet)
            if not snippets:
                snippets = [text[:200]]
            exact_bonus = sum(0.3 for kw in keywords if kw in text or kw in parent.get("source", ""))
            scored.append(
                SearchResult(
                    chunk_id=chunk["id"],
                    parent_id=parent["id"],
                    score=float(bm25_score + exact_bonus),
                    source=parent.get("source", chunk.get("source", "")),
                    snippet=" / ".join(snippets[:3]),
                    text=parent.get("text", text),
                    metadata=parent,
                )
            )
        return self._corpus.dedupe_to_parents(scored, top_k)

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        keywords = kwargs.get("keywords", [])
        top_k = min(int(kwargs.get("top_k", 10)), 30)
        if not keywords:
            return "キーワードを指定してください。", {"error": "no keywords"}
        top = self.search([str(kw) for kw in keywords], top_k)
        if not top:
            return "指定されたキーワードに一致する文書が見つかりませんでした。", {"matches": 0}

        lines = []
        snippets = []
        chunk_ids = []
        for item in top:
            chunk_ids.append(item.parent_id)
            tag = get_chunk_tag(item.text)
            tag_str = f" {tag}" if tag else ""
            lines.append(f"[Chunk {item.parent_id}] (keyword: {item.score:.3f}){tag_str} {item.source}")
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
            metadata={"keywords": keywords, "chunks_found": len(top), "chunk_ids": chunk_ids},
        )
        return "\n".join(lines), {
            "matches": len(top),
            "keywords": keywords,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
        }
