"""Read chunk tool for full content retrieval with adjacent chunk expansion."""

from typing import Any

import tiktoken

from .base import BaseTool
from ..context import AgentContext

_tokenizer = tiktoken.get_encoding("cl100k_base")


class ReadChunkTool(BaseTool):
    def __init__(self, chunks: list[dict]):
        self._chunk_map = {c["id"]: c for c in chunks}
        # Build adjacency: map chunk_id -> (prev_id, next_id)
        self._chunk_ids = [c["id"] for c in chunks]
        self._id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

    def _get_adjacent_ids(self, chunk_id: str) -> tuple[str | None, str | None]:
        """Get the previous and next chunk IDs for context expansion."""
        idx = self._id_to_idx.get(chunk_id)
        if idx is None:
            return None, None
        prev_id = self._chunk_ids[idx - 1] if idx > 0 else None
        next_id = self._chunk_ids[idx + 1] if idx < len(self._chunk_ids) - 1 else None
        # Only return adjacent if from the same file
        chunk = self._chunk_map[chunk_id]
        if prev_id and self._chunk_map[prev_id].get("file") != chunk.get("file"):
            prev_id = None
        if next_id and self._chunk_map[next_id].get("file") != chunk.get("file"):
            next_id = None
        return prev_id, next_id

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
                "- Adjacent chunk IDs are typically the current ID ± 1 (e.g., chunk '50' has neighbors '49' and '51')"
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
            if cid not in self._chunk_map:
                lines.append(f"[Chunk {cid}] Not found.")
                continue

            if context.is_chunk_read(cid):
                lines.append(f"[Chunk {cid}] (既読 / already read)")
                skipped += 1
                continue

            chunk = self._chunk_map[cid]
            text = chunk["text"]
            source = chunk.get("source", "")
            chunk_tokens = len(_tokenizer.encode(text))
            total_tokens += chunk_tokens

            context.mark_chunk_read(cid, chunk_tokens)
            new_count += 1

            # Show adjacent chunk info for context expansion
            prev_id, next_id = self._get_adjacent_ids(cid)
            adj_info = ""
            if prev_id or next_id:
                parts = []
                if prev_id:
                    parts.append(f"前: Chunk {prev_id}")
                if next_id:
                    parts.append(f"次: Chunk {next_id}")
                adj_info = f" [隣接チャンク: {', '.join(parts)}]"

            lines.append(f"=== Chunk {cid} | {source}{adj_info} ===")
            lines.append(text)
            lines.append("")

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
