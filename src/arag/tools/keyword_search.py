"""Keyword search tool for exact lexical matching in Japanese text."""

import re
from typing import Any

import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag, is_clearly_low_value
from ..context import AgentContext

# Japanese-aware sentence splitting
_SENTENCE_RE = re.compile(r'[。．.！！\?？\n]+')

_tokenizer = tiktoken.get_encoding("cl100k_base")


class KeywordSearchTool(BaseTool):
    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._chunk_map = {c["id"]: c for c in chunks}

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "keyword_search",
            "description": (
                "キーワードで文書を検索します。会計基準の用語、条項番号、"
                "特定のフレーズなどの完全一致検索に適しています。\n"
                "Search documents by exact keyword matching. "
                "Good for accounting terms, article numbers, specific phrases.\n"
                "STRATEGY:\n"
                "- Use specific accounting terms (e.g., '減損', 'のれん', '収益認識')\n"
                "- Use standard numbers (e.g., '第29号', '第34号')\n"
                "- Combine multiple related keywords for better results"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "検索キーワードのリスト / List of keywords to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返す結果の最大数 / Max results to return (default 5, max 20)",
                        "default": 10,
                    },
                },
                "required": ["keywords"],
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        keywords: list[str] = kwargs.get("keywords", [])
        top_k: int = min(kwargs.get("top_k", 10), 20)

        if not keywords:
            return "キーワードを指定してください。", {"error": "no keywords"}

        scored: list[tuple[str, float, list[str]]] = []

        for chunk in self._chunks:
            if is_clearly_low_value(chunk["text"]):
                continue
            text = chunk["text"].lower()
            score = 0.0
            matched_sentences = []

            for kw in keywords:
                kw_lower = kw.lower()
                count = text.count(kw_lower)
                if count > 0:
                    score += count * len(kw)
                    # Extract sentences containing keyword
                    sentences = _SENTENCE_RE.split(chunk["text"])
                    for sent in sentences:
                        if kw_lower in sent.lower() and len(sent.strip()) > 5:
                            snippet = sent.strip()[:200]
                            if snippet not in matched_sentences:
                                matched_sentences.append(snippet)
                            if len(matched_sentences) >= 5:
                                break

            if score > 0:
                scored.append((chunk["id"], score, matched_sentences))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        if not top:
            return "指定されたキーワードに一致する文書が見つかりませんでした。", {"matches": 0}

        lines = []
        all_matched_text = []
        chunk_ids = []
        for chunk_id, score, matched_sentences in top:
            chunk = self._chunk_map[chunk_id]
            source = chunk.get("source", "")
            chunk_ids.append(chunk_id)
            tag = get_chunk_tag(chunk.get("text", ""))
            tag_str = f" {tag}" if tag else ""
            lines.append(f"[Chunk {chunk_id}] (score: {score:.1f}){tag_str} {source}")
            for s in matched_sentences[:5]:
                lines.append(f"  > {s}")
                all_matched_text.append(s)

        result = "\n".join(lines)

        # Token counting for retrieval tracking
        retrieved_tokens = len(_tokenizer.encode("\n".join(all_matched_text))) if all_matched_text else 0

        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(top),
                "chunk_ids": chunk_ids,
            },
        )

        return result, {
            "matches": len(top),
            "keywords": keywords,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
        }
