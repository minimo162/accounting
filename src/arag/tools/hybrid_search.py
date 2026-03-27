"""Hybrid retrieval with query expansion and reranking."""

import re
from typing import Any

import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag
from .keyword_search import KeywordSearchTool
from .semantic_search import SemanticSearchTool
from ..config import RetrievalConfig
from ..context import AgentContext
from ..query_rewrite import QueryExpander
from ..reranker import BaseReranker
from ..retrieval import reciprocal_rank_fusion, tokenize_for_bm25

_tokenizer = tiktoken.get_encoding("cl100k_base")


class HybridSearchTool(BaseTool):
    def __init__(
        self,
        semantic_tool: SemanticSearchTool,
        keyword_tool: KeywordSearchTool,
        query_expander: QueryExpander,
        reranker: BaseReranker,
        config: RetrievalConfig,
    ):
        self.semantic_tool = semantic_tool
        self.keyword_tool = keyword_tool
        self.query_expander = query_expander
        self.reranker = reranker
        self.config = config

    @property
    def name(self) -> str:
        return "hybrid_search"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": "dense検索とkeyword検索を統合し、必要ならquery expansionとrerankも行います。通常はこのツールを優先してください。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        query = kwargs.get("query", "")
        top_k = min(int(kwargs.get("top_k", self.config.final_top_k)), 30)
        if not query:
            return "検索クエリを指定してください。", {"error": "no query"}
        final, expansions, hyde_doc = self.search(query, top_k)
        if not final:
            return "関連する文書が見つかりませんでした。", {"matches": 0}

        lines = []
        snippets = []
        chunk_ids = []
        for item in final:
            chunk_ids.append(item.parent_id)
            tag = get_chunk_tag(item.text)
            tag_str = f" {tag}" if tag else ""
            lines.append(f"[Chunk {item.parent_id}] (hybrid: {item.score:.3f}){tag_str} {item.source}")
            lines.append(f"  > {item.snippet}")
            snippets.append(item.snippet)

        unread_ids = [cid for cid in chunk_ids if not context.is_chunk_read(cid) and not get_chunk_tag((self.semantic_tool._corpus.get_parent(cid) or {}).get("text", ""))]
        if unread_ids:
            lines.append(f"\n--- {len(unread_ids)}件の未読チャンクがあります。read_chunkで全文を取得してください ---")
            lines.append(f"read_chunk(chunk_ids={unread_ids})")

        retrieved_tokens = len(_tokenizer.encode("\n".join(snippets))) if snippets else 0
        context.add_retrieval_log(
            tool_name=self.name,
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "expansions": expansions,
                "hyde_used": bool(hyde_doc),
                "chunks_found": len(final),
                "chunk_ids": chunk_ids,
            },
        )
        return "\n".join(lines), {
            "matches": len(final),
            "query": query,
            "expansions": expansions,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
        }

    def search(self, query: str, top_k: int) -> tuple[list, list[str], str | None]:
        expansions = self.query_expander.expand(query)
        hyde_doc = self.query_expander.generate_hypothetical_document(query)

        semantic_rankings = []
        keyword_rankings = []
        for expanded in expansions:
            semantic_rankings.append(self.semantic_tool.search(expanded, self.config.semantic_top_k))
            keyword_rankings.append(self.keyword_tool.search([expanded], self.config.keyword_top_k))
        if hyde_doc:
            semantic_rankings.append(self.semantic_tool.search(hyde_doc, self.config.semantic_top_k))

        fused = reciprocal_rank_fusion(semantic_rankings + keyword_rankings, rrf_k=self.config.rrf_k)
        fused = self._apply_exact_match_boosts(query, fused)
        fused = self._apply_change_intent_boosts(query, fused)
        fused = self._apply_topic_alignment_boosts(query, fused)
        reranked = self.reranker.rerank(query, fused[: self.config.rerank_top_n])
        final = reranked[:top_k]
        return final, expansions, hyde_doc

    @staticmethod
    def _extract_exact_terms(query: str) -> list[str]:
        patterns = [
            r"企業会計基準第\d+号",
            r"適用指針第\d+号",
            r"実務対応報告第\d+号",
            r"会計基準第\d+号",
            r"第\d+項",
            r"BC\d+",
        ]
        terms: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, query):
                if match not in terms:
                    terms.append(match)
        return terms

    def _apply_exact_match_boosts(self, query: str, results: list) -> list:
        exact_terms = self._extract_exact_terms(query)
        if not exact_terms:
            return results

        boosted = []
        for item in results:
            source = item.source or ""
            metadata = item.metadata or {}
            haystacks = [
                source,
                str(metadata.get("source", "")),
                str(metadata.get("doc_title", "")),
                str(metadata.get("section_title", "")),
                str(metadata.get("standard_no", "")),
            ]
            bonus = 0.0
            for term in exact_terms:
                if any(term in haystack for haystack in haystacks):
                    bonus += 0.6
            boosted.append(
                item.__class__(
                    chunk_id=item.chunk_id,
                    parent_id=item.parent_id,
                    score=item.score + bonus,
                    source=item.source,
                    snippet=item.snippet,
                    text=item.text,
                    metadata=item.metadata,
                )
            )
        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted

    @staticmethod
    def _is_change_query(query: str) -> bool:
        return any(term in query for term in ("改正", "変更", "見直し", "新基準", "改訂"))

    def _apply_change_intent_boosts(self, query: str, results: list) -> list:
        if not self._is_change_query(query):
            return results

        positive_terms = ("改正", "変更", "見直し", "導入", "廃止", "新た", "経過措置", "適用初年度")
        negative_terms = ("議決", "委員", "名簿")
        boosted = []

        for item in results:
            source = item.source or ""
            text_window = f"{source} {item.snippet[:260]} {item.text[:800]}"
            bonus = 0.0

            for term in positive_terms:
                if term in text_window:
                    bonus += 0.25
            for term in negative_terms:
                if term in text_window:
                    bonus -= 0.5

            boosted.append(
                item.__class__(
                    chunk_id=item.chunk_id,
                    parent_id=item.parent_id,
                    score=item.score + bonus,
                    source=item.source,
                    snippet=item.snippet,
                    text=item.text,
                    metadata=item.metadata,
                )
            )

        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted

    @staticmethod
    def _query_anchor_terms(query: str) -> list[str]:
        generic_terms = {
            "改正", "改正点", "変更", "変更点", "見直し", "新基準", "改訂",
            "教えて", "内容", "記載", "取扱い", "方法", "基準", "会計基準",
        }
        anchors: list[str] = []
        for token in tokenize_for_bm25(query):
            if len(token) < 2 or token in generic_terms:
                continue
            if token not in anchors:
                anchors.append(token)
        return anchors[:4]

    def _apply_topic_alignment_boosts(self, query: str, results: list) -> list:
        anchors = self._query_anchor_terms(query)
        if not anchors:
            return results

        boosted = []
        for item in results:
            text_window = f"{item.source} {item.snippet[:260]} {item.text[:800]}"
            hits = sum(1 for term in anchors if term in text_window)
            bonus = hits * 0.2
            if hits == 0:
                bonus -= 0.35
            boosted.append(
                item.__class__(
                    chunk_id=item.chunk_id,
                    parent_id=item.parent_id,
                    score=item.score + bonus,
                    source=item.source,
                    snippet=item.snippet,
                    text=item.text,
                    metadata=item.metadata,
                )
            )

        boosted.sort(key=lambda item: item.score, reverse=True)
        return boosted
