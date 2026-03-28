"""Agent execution context for tracking retrieval state."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalLog:
    """Structured log entry for a single retrieval operation."""
    tool_name: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Tracks per-query state: which chunks have been read, token counts, retrieval logs."""

    def __init__(self):
        self.read_chunk_ids: set[str] = set()
        self.searched_chunk_ids: list[str] = []  # ordered; preserves first-seen rank
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: list[RetrievalLog] = []
        self.search_history: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []

    def mark_chunk_read(self, chunk_id: str, token_count: int = 0):
        self.read_chunk_ids.add(chunk_id)

    def add_searched_chunks(self, chunk_ids: list[str]):
        """Record chunk IDs returned by a search (for fallback references)."""
        seen = set(self.searched_chunk_ids)
        for cid in chunk_ids:
            if cid not in seen:
                self.searched_chunk_ids.append(cid)
                seen.add(cid)

    def is_chunk_read(self, chunk_id: str) -> bool:
        return chunk_id in self.read_chunk_ids

    def add_retrieval_log(
        self, tool_name: str, tokens: int, metadata: dict[str, Any] | None = None
    ):
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {},
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens

    def add_trajectory_entry(
        self,
        loop: int,
        tool_name: str,
        arguments: dict,
        tool_result: str,
        tool_log: dict[str, Any],
    ):
        self.trajectory.append({
            "loop": loop,
            "tool_name": tool_name,
            "arguments": arguments,
            "tool_result": tool_result,
            **tool_log,
        })

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids),
            "retrieval_logs": [
                {"tool": log.tool_name, "tokens": log.tokens, **log.metadata}
                for log in self.retrieval_logs
            ],
        }

    def reset(self):
        self.read_chunk_ids.clear()
        self.searched_chunk_ids.clear()
        self.total_retrieved_tokens = 0
        self.retrieval_logs.clear()
        self.search_history.clear()
        self.trajectory.clear()
