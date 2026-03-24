"""Semantic search tool using Gemini embeddings at chunk level."""

import logging
import pickle
from typing import Any

import numpy as np
import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag, is_clearly_low_value
from ..context import AgentContext

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")


class SemanticSearchTool(BaseTool):
    def __init__(self, index_path: str, embed_fn):
        """
        Args:
            index_path: Path to sentence_index.pkl
            embed_fn: Callable that takes a query string and returns a numpy vector
        """
        self._embed_fn = embed_fn
        self._load_index(index_path)

    def _load_index(self, path: str):
        from pathlib import Path
        p = Path(path)
        npz_path = p.with_name("sentence_index.npz")
        meta_path = p.with_name("sentence_meta.pkl")

        if npz_path.exists() and meta_path.exists():
            data = np.load(str(npz_path))
            self._embeddings: np.ndarray = data["embeddings"].astype(np.float32)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._texts: list[str] = meta["sentences"]
            self._text_to_chunk: list[str] = meta["sentence_to_chunk"]
            self._chunks: dict[str, dict] = meta["chunks"]
        else:
            with open(path, "rb") as f:
                index = pickle.load(f)
            self._texts = index["sentences"]
            self._embeddings = index["embeddings"]
            self._text_to_chunk = index["sentence_to_chunk"]
            self._chunks = index["chunks"]
        logger.info(f"Loaded index: {len(self._texts)} entries, {len(self._chunks)} chunks")

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "semantic_search",
            "description": (
                "意味的類似性で文書を検索します。自然言語の質問や概念的な検索に適しています。\n"
                "Search documents by semantic similarity. "
                "Good for natural language queries and conceptual searches.\n"
                "STRATEGY:\n"
                "- Phrase your query as a natural language question or description\n"
                "- Try different phrasings if initial results aren't relevant\n"
                "- Good for finding related concepts even without exact term matches"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索クエリ / Search query in natural language",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返す結果の最大数 / Max results (default 15, max 30). 複雑な質問では多めに設定してください。",
                        "default": 15,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        query: str = kwargs.get("query", "")
        top_k: int = min(kwargs.get("top_k", 15), 30)

        if not query:
            return "検索クエリを指定してください。", {"error": "no query"}

        # Embed query
        query_vec = self._embed_fn(query)
        if query_vec is None:
            return "埋め込みの生成に失敗しました。", {"error": "embedding failed"}

        query_vec = query_vec / np.linalg.norm(query_vec)

        # Cosine similarity against all chunk embeddings
        similarities = self._embeddings @ query_vec

        # Build (index, chunk_id, similarity) list, excluding clearly low-value chunks
        scored = []
        for idx in range(len(similarities)):
            chunk_id = self._text_to_chunk[idx]
            chunk_text = self._chunks.get(chunk_id, {}).get("text", "")
            if is_clearly_low_value(chunk_text):
                continue
            scored.append((idx, chunk_id, float(similarities[idx])))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[2], reverse=True)
        ranked = scored[:top_k]

        if not ranked:
            return "関連する文書が見つかりませんでした。", {"matches": 0}

        lines = []
        all_matched_text = []
        chunk_ids = []
        for idx, chunk_id, score in ranked:
            chunk = self._chunks.get(chunk_id, {})
            source = chunk.get("source", "")
            chunk_ids.append(chunk_id)
            tag = get_chunk_tag(chunk.get("text", ""))
            tag_str = f" {tag}" if tag else ""
            lines.append(f"[Chunk {chunk_id}] (similarity: {score:.3f}){tag_str} {source}")

            # Show first 300 chars as snippet
            snippet = self._texts[idx][:300]
            lines.append(f"  > {snippet}")
            all_matched_text.append(snippet)

        result = "\n".join(lines)

        # Token counting
        retrieved_tokens = len(_tokenizer.encode("\n".join(all_matched_text))) if all_matched_text else 0

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(ranked),
                "chunk_ids": chunk_ids,
            },
        )

        return result, {
            "matches": len(ranked),
            "query": query,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
        }
