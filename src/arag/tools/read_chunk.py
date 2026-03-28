"""Read chunk tool for query-aware parent chunk retrieval."""

import re
from typing import Any

import tiktoken

from .base import BaseTool
from ..context import AgentContext
from ..retrieval import ChunkCorpus

_tokenizer = tiktoken.get_encoding("cl100k_base")


class ReadChunkTool(BaseTool):
    _GENERIC_QUERY_TERMS = {
        "会計", "会計基準", "基準", "内容", "教えて", "教えてください", "について",
        "改正", "変更", "見直し", "概要", "ポイント", "処理", "方法",
    }

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
                "見つけたチャンクの重要箇所を読むために使用してください。質問に近い箇所を優先して返します。\n"
                "Read query-aware evidence from specific chunks by their IDs. "
                "Use after search to read the most relevant passages.\n"
                "STRATEGY:\n"
                "- Always read promising chunks identified by your searches\n"
                "- The tool returns the most relevant evidence spans from the parent chunk\n"
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
        compressed_any = False

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
            query = context.current_search_query or context.question
            excerpt, compressed = self._build_excerpt(text, query, context.question_complexity)
            chunk_tokens = len(_tokenizer.encode(excerpt))
            total_tokens += chunk_tokens
            compressed_any = compressed_any or compressed

            context.mark_chunk_read(parent_id, chunk_tokens)
            context.set_evidence_note(parent_id, excerpt)
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
            if compressed:
                lines.append("[重要箇所を抽出]")
            lines.append(excerpt)
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
                "compressed": compressed_any,
            },
        )

        return result, {
            "new": new_count,
            "skipped": skipped,
            "retrieved_tokens": total_tokens,
        }

    @classmethod
    def _extract_query_terms(cls, query: str) -> list[str]:
        if not query:
            return []
        terms = []
        for term in re.findall(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,}", query):
            if term in cls._GENERIC_QUERY_TERMS:
                continue
            if term not in terms:
                terms.append(term)
        return terms[:6]

    @classmethod
    def _sentence_units(cls, text: str) -> list[str]:
        units: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for part in re.split(r"(?<=[。！？])\s+|(?<=。)|(?<=！)|(?<=？)", line):
                sentence = part.strip()
                if sentence:
                    units.append(sentence)
        return units

    @classmethod
    def _score_unit(cls, unit: str, query_terms: list[str]) -> int:
        if not query_terms:
            return 0
        score = 0
        for term in query_terms:
            if term in unit:
                score += 3 if len(term) >= 4 else 2
        if "第" in unit and any("第" in term for term in query_terms):
            score += 1
        return score

    @classmethod
    def _excerpt_budget(cls, complexity: str) -> int:
        if complexity == "simple":
            return 1200
        if complexity == "complex":
            return 2200
        return 1600

    @classmethod
    def _build_excerpt(cls, text: str, query: str, complexity: str) -> tuple[str, bool]:
        budget = cls._excerpt_budget(complexity)
        if len(text) <= budget:
            return text, False

        query_terms = cls._extract_query_terms(query)
        units = cls._sentence_units(text)
        if not units:
            return text[:budget].rstrip(), True

        selected_indexes: set[int] = set()
        if query_terms:
            scored_units = sorted(
                ((cls._score_unit(unit, query_terms), idx) for idx, unit in enumerate(units)),
                key=lambda item: (item[0], -item[1]),
                reverse=True,
            )
            for score, idx in scored_units[:6]:
                if score <= 0:
                    continue
                selected_indexes.add(idx)
                if idx > 0:
                    selected_indexes.add(idx - 1)
                if idx + 1 < len(units):
                    selected_indexes.add(idx + 1)

        if not selected_indexes:
            selected_indexes.update(range(min(6, len(units))))

        ordered_units = [units[idx] for idx in sorted(selected_indexes)]
        excerpt_parts: list[str] = []
        current_len = 0
        for unit in ordered_units:
            addition = unit if not excerpt_parts else f"\n{unit}"
            if excerpt_parts and current_len + len(addition) > budget:
                break
            excerpt_parts.append(unit)
            current_len += len(addition)

        excerpt = "\n".join(excerpt_parts).strip()
        if not excerpt:
            excerpt = text[:budget].rstrip()
        if excerpt != text:
            excerpt = f"{excerpt}\n[抜粋]"
        return excerpt, True
