"""Read chunk tool for full content retrieval with adjacent chunk expansion."""

from typing import Any

import tiktoken

from .base import BaseTool
from ..context import AgentContext
from ..retrieval import ChunkCorpus

_tokenizer = tiktoken.get_encoding("cl100k_base")


class ReadChunkTool(BaseTool):
    def __init__(self, corpus: ChunkCorpus):
        self._corpus = corpus
        self._chunk_map = corpus.chunk_map

    @property
    def name(self) -> str:
        return "read_chunk"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "read_chunk",
            "description": (
                "チャンクIDを指定して全文を取得します。keyword_searchやsemantic_searchで"
                "見つけたチャンクの詳細を読むために使用してください。\n"
                "Read full content of specific chunks by their IDs. "
                "Use after search to read details of found chunks.\n"
                "STRATEGY:\n"
                "- Always read promising chunks identified by your searches\n"
                "- Make sure to read the most relevant chunks to gather complete information\n"
                "- If information seems incomplete or truncated, read adjacent chunks (± 1)\n"
                "- Reading full text is essential for accurate answers\n"
                "- Adjacent chunk IDs use file:page format (e.g., chunk 'foo.pdf:5' has neighbors 'foo.pdf:4' and 'foo.pdf:6')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "読み取るチャンクIDのリスト / List of chunk IDs to read",
                    },
                },
                "required": ["chunk_ids"],
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        chunk_ids: list[str] = kwargs.get("chunk_ids", [])

        if not chunk_ids:
            return "チャンクIDを指定してください。", {"error": "no chunk_ids"}

        lines = []
        new_count = 0
        skipped = 0
        total_tokens = 0

        for cid in chunk_ids:
            parent = self._corpus.get_parent(cid)
            if parent is None:
                lines.append(f"[Chunk {cid}] Not found.")
                continue

            parent_id = parent["id"]
            if context.is_chunk_read(parent_id):
                lines.append(f"[Chunk {parent_id}] (既読 / already read)")
                skipped += 1
                continue

            text = parent["text"]
            source = parent.get("source", "")
            chunk_tokens = len(_tokenizer.encode(text))
            total_tokens += chunk_tokens

            context.mark_chunk_read(parent_id, chunk_tokens)
            new_count += 1

            # Show adjacent chunk info for context expansion
            prev_id, next_id = self._corpus.get_adjacent_parent_ids(parent_id)
            adj_info = ""
            if prev_id or next_id:
                parts = []
                if prev_id:
                    parts.append(f"前: Chunk {prev_id}")
                if next_id:
                    parts.append(f"次: Chunk {next_id}")
                adj_info = f" [隣接チャンク: {', '.join(parts)}]"

            lines.append(f"=== Chunk {parent_id} | {source}{adj_info} ===")
            lines.append(text)
            lines.append("")

        # Show cumulative read count to help agent gauge progress
        total_read = len(context.read_chunk_ids)
        lines.append(f"[累計読了チャンク数: {total_read}]")

        result = "\n".join(lines)

        context.add_retrieval_log(
            tool_name="read_chunk",
            tokens=total_tokens,
            metadata={
                "chunk_ids": chunk_ids,
                "new_chunks_read": new_count,
                "skipped_already_read": skipped,
            },
        )

        return result, {
            "new": new_count,
            "skipped": skipped,
            "retrieved_tokens": total_tokens,
        }
