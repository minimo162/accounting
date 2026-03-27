"""Candidate rerankers."""

import json
import logging

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

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        payload = [
            {
                "id": result.parent_id,
                "source": result.source,
                "text": result.text[:800],
                "metadata": result.metadata,
            }
            for result in results[:12]
        ]
        prompt = (
            "質問に最も関連する候補順に並べ替えてください。"
            "JSON配列で parent_id だけ返してください。\n\n"
            f"質問: {query}\n\n候補:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            response = self.llm.chat(messages=[{"role": "user", "content": prompt}], tools=None, temperature=0.0, max_tokens=300)
            ordered_ids = json.loads(response["message"].get("content", "[]"))
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
            logger.warning(f"LLM rerank failed, falling back to original order: {e}")
            return results
