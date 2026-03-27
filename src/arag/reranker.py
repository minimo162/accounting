"""Candidate rerankers."""

import json
import logging
import re

from .llm import LLMClient
from .retrieval import SearchResult, tokenize_for_bm25

logger = logging.getLogger(__name__)


class BaseReranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class HeuristicReranker(BaseReranker):
    """Cheap reranker combining lexical, metadata, and semantic priors."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = set(tokenize_for_bm25(query))
        reranked: list[SearchResult] = []
        for result in results:
            lexical_overlap = len(query_terms & set(tokenize_for_bm25(result.text[:500])))
            metadata_bonus = 0.0
            meta = result.metadata or {}
            for key in ("standard_no", "doc_type", "doc_title", "section_title"):
                value = str(meta.get(key, ""))
                if value and value in query:
                    metadata_bonus += 0.35
            final_score = result.score + lexical_overlap * 0.08 + metadata_bonus
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    parent_id=result.parent_id,
                    score=final_score,
                    source=result.source,
                    snippet=result.snippet,
                    text=result.text,
                    metadata=result.metadata,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked


class LLMReranker(BaseReranker):
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.fallback = HeuristicReranker()

    @staticmethod
    def _unique_candidates(results: list[SearchResult], limit: int) -> list[SearchResult]:
        seen: set[str] = set()
        unique: list[SearchResult] = []
        for result in results:
            if result.parent_id in seen:
                continue
            seen.add(result.parent_id)
            unique.append(result)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _extract_ordered_ids(raw_content: str, allowed_ids: set[str]) -> list[str]:
        content = raw_content.strip()
        if not content:
            raise ValueError("empty reranker response")

        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", content)
            content = re.sub(r"\n?```$", "", content).strip()

        candidates = [content]
        array_match = re.search(r"\[[\s\S]*\]", content)
        if array_match:
            candidates.insert(0, array_match.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                ordered: list[str] = []
                seen: set[str] = set()
                for item in parsed:
                    item_str = str(item).strip()
                    if item_str in allowed_ids and item_str not in seen:
                        ordered.append(item_str)
                        seen.add(item_str)
                if ordered:
                    return ordered

        ordered = []
        seen = set()
        for line in content.splitlines():
            for candidate_id in re.findall(r"[\w.\-]+:\d+", line):
                if candidate_id in allowed_ids and candidate_id not in seen:
                    ordered.append(candidate_id)
                    seen.add(candidate_id)
        if ordered:
            return ordered

        raise ValueError(f"unparseable reranker response: {raw_content[:200]!r}")

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        candidates = self._unique_candidates(results, limit=12)
        payload = [
            {
                "id": result.parent_id,
                "source": result.source,
                "snippet": result.snippet[:240],
                "doc_title": (result.metadata or {}).get("doc_title", ""),
                "section_title": (result.metadata or {}).get("section_title", ""),
                "doc_type": (result.metadata or {}).get("doc_type", ""),
            }
            for result in candidates
        ]
        allowed_ids = {item["id"] for item in payload}
        prompt = (
            "質問に最も関連する候補順に並べ替えてください。"
            "回答は候補 id の JSON 配列だけにしてください。"
            "説明文、コードブロック、前置きは禁止です。"
            '例: ["doc.pdf:10", "doc.pdf:4"]\n\n'
            f"質問: {query}\n\n候補:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = self.llm.chat(messages=[{"role": "user", "content": prompt}], tools=None, temperature=0.0, max_tokens=120)
            raw_content = response["message"].get("content", "")
            ordered_ids = self._extract_ordered_ids(raw_content, allowed_ids)
            order = {parent_id: idx for idx, parent_id in enumerate(ordered_ids)}
            ranked = sorted(results, key=lambda item: (order.get(item.parent_id, 10**6), -item.score))
            boosted: list[SearchResult] = []
            for item in ranked:
                boost = 1.0 / (1 + order.get(item.parent_id, len(ranked)))
                boosted.append(
                    SearchResult(
                        chunk_id=item.chunk_id,
                        parent_id=item.parent_id,
                        score=item.score + boost,
                        source=item.source,
                        snippet=item.snippet,
                        text=item.text,
                        metadata=item.metadata,
                    )
                )
            return boosted
        except Exception as e:
            logger.warning(f"LLM rerank failed, falling back to heuristic rerank: {e}")
            return self.fallback.rerank(query, results)
