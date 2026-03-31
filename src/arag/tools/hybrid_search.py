"""Hybrid retrieval with query expansion and reranking."""

import json
import re
from typing import Any

import tiktoken

from .base import BaseTool
from .filters import get_chunk_tag, is_clearly_low_value
from .keyword_search import KeywordSearchTool
from .semantic_search import SemanticSearchTool
from ..config import RetrievalConfig
from ..context import AgentContext
from ..query_rewrite import QueryExpander, QueryProfile
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

        cache_key = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False, sort_keys=True)
        cached = context.get_cached_tool_result(self.name, cache_key)
        if cached is not None:
            result_text, tool_log = cached
            context.set_current_search_query(str(tool_log.get("effective_query") or query))
            return result_text, {**tool_log, "cached": True}

        final, expansions, hyde_doc, search_info = self._search_with_details(query, top_k)
        if not final:
            context.set_current_search_query(query)
            return "関連する文書が見つかりませんでした。", {"matches": 0}

        effective_query = (
            str(search_info.get("corrective_query") or "").strip()
            or str(search_info.get("profile", {}).get("canonical_focus_query") or "").strip()
            or query
        )
        context.set_current_search_query(effective_query)

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
        confidence_score = float(search_info.get("confidence", 0.0))
        confidence_label = "高" if confidence_score >= 0.7 else "中" if confidence_score >= 0.45 else "低"
        lines.append(f"[検索信頼度: {confidence_label}]")
        if search_info.get("corrective_query"):
            lines.append(f"[補正検索: {search_info['corrective_query']}]")

        context.add_searched_chunks(chunk_ids)
        context.add_search_entry(
            {
                "query": query,
                "profile": search_info.get("profile", {}),
                "confidence": confidence_score,
                "corrective_query": search_info.get("corrective_query"),
                "effective_query": effective_query,
                "chunk_ids": chunk_ids,
                "exact_evidence_found": bool(search_info.get("exact_evidence_found")),
                "exact_shortfall": bool(search_info.get("exact_shortfall")),
                "exact_doc_hits": int(search_info.get("exact_doc_hits", 0) or 0),
                "exact_section_hits": int(search_info.get("exact_section_hits", 0) or 0),
            }
        )
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
                "confidence": confidence_score,
                "corrective_query": search_info.get("corrective_query"),
                "query_complexity": search_info.get("profile", {}).get("complexity"),
                "effective_query": effective_query,
                "exact_evidence_found": bool(search_info.get("exact_evidence_found")),
                "exact_shortfall": bool(search_info.get("exact_shortfall")),
                "exact_doc_hits": int(search_info.get("exact_doc_hits", 0) or 0),
                "exact_section_hits": int(search_info.get("exact_section_hits", 0) or 0),
            },
        )
        result_text = "\n".join(lines)
        tool_log = {
            "matches": len(final),
            "query": query,
            "effective_query": effective_query,
            "expansions": expansions,
            "retrieved_tokens": retrieved_tokens,
            "chunk_ids": chunk_ids,
            "confidence": confidence_score,
            "corrective_query": search_info.get("corrective_query"),
            "profile": search_info.get("profile", {}),
            "exact_evidence_found": bool(search_info.get("exact_evidence_found")),
            "exact_shortfall": bool(search_info.get("exact_shortfall")),
            "exact_doc_hits": int(search_info.get("exact_doc_hits", 0) or 0),
            "exact_section_hits": int(search_info.get("exact_section_hits", 0) or 0),
        }
        context.set_cached_tool_result(self.name, cache_key, result_text, tool_log)
        return result_text, tool_log

    def search(self, query: str, top_k: int) -> tuple[list, list[str], str | None]:
        final, expansions, hyde_doc, _ = self._search_with_details(query, top_k)
        return final, expansions, hyde_doc

    def _search_with_details(self, query: str, top_k: int) -> tuple[list, list[str], str | None, dict[str, Any]]:
        profile = self.query_expander.analyze(query)
        expansions = self.query_expander.expand(query)
        is_exact_query = self.query_expander.is_exact_query(query)
        hyde_doc = None if is_exact_query else self.query_expander.generate_hypothetical_document(query)
        semantic_query = self._select_semantic_query(query, expansions, profile)

        semantic_rankings = []
        keyword_rankings = []
        keyword_limit = min(self.config.keyword_top_k, max(top_k * 2, 8))
        semantic_limit = min(self.config.semantic_top_k, max(top_k * 2, 8))

        if is_exact_query or profile.search_mode == "keyword_first":
            exact_term_sets: list[list[str]] = []
            if is_exact_query:
                primary_keyword_query = query
            elif profile.detail_seeking and profile.corrective_query:
                primary_keyword_query = profile.corrective_query
            else:
                primary_keyword_query = profile.canonical_focus_query or profile.corrective_query or query
            if is_exact_query:
                exact_term_sets = QueryExpander.exact_keyword_term_sets(primary_keyword_query)
                keyword_terms = exact_term_sets[0] if exact_term_sets else self.query_expander.exact_keyword_terms(primary_keyword_query)
            else:
                keyword_terms = [term for term in primary_keyword_query.split() if term] or [primary_keyword_query]
            primary_keyword = self.keyword_tool.search(keyword_terms, keyword_limit)
            if primary_keyword:
                keyword_rankings.append(primary_keyword)
            if is_exact_query and not self._rankings_have_exact_evidence(query, keyword_rankings):
                for term_set in exact_term_sets[1:3]:
                    ranking = self.keyword_tool.search(term_set, keyword_limit)
                    if ranking:
                        keyword_rankings.append(ranking)
            if (
                not primary_keyword
                or (profile.complexity == "complex" and not is_exact_query)
                or self._should_backfill_semantic_for_keyword_focus(profile, keyword_terms, primary_keyword, is_exact_query)
                or (is_exact_query and not self._rankings_have_exact_evidence(query, keyword_rankings))
            ):
                semantic_rankings.append(self.semantic_tool.search(semantic_query, semantic_limit))
        else:
            if profile.search_mode != "keyword_first":
                semantic_rankings.append(self.semantic_tool.search(semantic_query, semantic_limit))
            for expanded in expansions:
                keyword_rankings.append(self.keyword_tool.search([expanded], keyword_limit))
            if hyde_doc:
                semantic_rankings.append(self.semantic_tool.search(hyde_doc, semantic_limit))

        reranked = self._rank_results(query, semantic_rankings, keyword_rankings)
        confidence = self._estimate_confidence(query, reranked)
        corrective_query = None
        if self._should_run_corrective_search(profile, reranked, confidence, is_exact_query):
            corrective_query = profile.corrective_query
            if corrective_query and corrective_query != query:
                corrective_keyword = self.keyword_tool.search([corrective_query], keyword_limit)
                if corrective_keyword:
                    keyword_rankings.append(corrective_keyword)
                elif profile.search_mode != "keyword_first":
                    semantic_rankings.append(self.semantic_tool.search(corrective_query, semantic_limit))
                reranked = self._rank_results(query, semantic_rankings, keyword_rankings)
                confidence = max(confidence, self._estimate_confidence(query, reranked))

        if self._should_rerank(query, reranked, top_k):
            reranked = self.reranker.rerank(query, reranked)
        reranked = self._filter_exact_mismatch_results(query, reranked)
        reranked = self._filter_change_delta_results(query, reranked)
        reranked = self._filter_low_value_parent_results(reranked)
        final = reranked[:top_k]
        exact_doc_hits = 0
        exact_section_hits = 0
        exact_evidence_found = False
        doc_terms, section_terms = QueryExpander.split_exact_constraints(query)
        if doc_terms or section_terms:
            for item in final:
                doc_hits, section_hits = self._exact_match_counts(query, item)
                exact_doc_hits = max(exact_doc_hits, doc_hits)
                exact_section_hits = max(exact_section_hits, section_hits)
                if doc_hits == len(doc_terms) and section_hits == len(section_terms):
                    exact_evidence_found = True
                    break
        return final, expansions, hyde_doc, {
            "confidence": confidence,
            "corrective_query": corrective_query,
            "profile": profile.to_dict(),
            "exact_evidence_found": exact_evidence_found,
            "exact_shortfall": bool((doc_terms or section_terms) and not exact_evidence_found),
            "exact_doc_hits": exact_doc_hits,
            "exact_section_hits": exact_section_hits,
        }

    @staticmethod
    def _select_semantic_query(query: str, expansions: list[str], profile: QueryProfile) -> str:
        if (profile.verification_mode or profile.judgment_validation) and profile.corrective_query:
            return profile.corrective_query
        if "会計基準" not in query:
            return query
        for variant in expansions:
            if "に関する会計基準" in variant:
                return variant
        return query

    def _should_rerank(self, query: str, results: list, top_k: int) -> bool:
        if not results or len(results) <= top_k:
            return False
        if getattr(self.reranker, "is_expensive", False) and self.query_expander.is_exact_query(query):
            return False
        return True

    def _rank_results(self, query: str, semantic_rankings: list[list], keyword_rankings: list[list]) -> list:
        fused = reciprocal_rank_fusion(semantic_rankings + keyword_rankings, rrf_k=self.config.rrf_k)
        fused = self._apply_exact_match_boosts(query, fused)
        fused = self._apply_exact_constraint_boosts(query, fused)
        fused = self._apply_change_intent_boosts(query, fused)
        fused = self._apply_topic_alignment_boosts(query, fused)
        fused = self._apply_focus_term_boosts(query, fused)
        return fused[: self.config.rerank_top_n]

    @classmethod
    def _should_backfill_semantic_for_keyword_focus(
        cls,
        profile: QueryProfile,
        keyword_terms: list[str],
        primary_keyword: list,
        is_exact_query: bool,
    ) -> bool:
        if is_exact_query or not primary_keyword or profile.search_mode != "keyword_first":
            return False

        focus_terms = [term for term in keyword_terms if len(term) >= 2][:6]
        if len(focus_terms) < 3:
            return False

        top = primary_keyword[0]
        haystack = f"{top.source} {top.snippet[:260]} {top.text[:800]}"
        focus_hits = sum(1 for term in focus_terms if term in haystack)
        required_hits = 3 if len(focus_terms) >= 5 else 2
        return focus_hits < required_hits

    def _estimate_confidence(self, query: str, results: list) -> float:
        if not results:
            return 0.0

        top = results[0]
        anchors = self._query_anchor_terms(query)
        top_window = f"{top.source} {top.snippet[:260]} {top.text[:800]}"
        anchor_hits = sum(1 for term in anchors if term in top_window)
        doc_terms, section_terms = self._split_exact_constraints(query)
        doc_hits, section_hits = self._exact_match_counts(query, top)
        exact_hit = (
            (doc_terms or section_terms)
            and doc_hits == len(doc_terms)
            and section_hits == len(section_terms)
        ) or any(term in top_window for term in self._extract_exact_terms(query))
        doc_type_match = self._doc_type_matches(
            self._target_doc_type(query),
            top.metadata or {},
            top.source,
        ) if self._target_doc_type(query) else False
        canonical_titles = self._canonical_title_phrases(query)
        file_name = str((top.metadata or {}).get("file", ""))
        aliases = self.semantic_tool._corpus.get_document_aliases(file_name) if file_name else set()
        title_hit = bool(canonical_titles and aliases and any(title in aliases for title in canonical_titles))
        gap = top.score - (results[1].score if len(results) > 1 else 0.0)

        confidence = 0.15
        if anchor_hits >= 1:
            confidence += 0.2
        if anchor_hits >= 2:
            confidence += 0.1
        if exact_hit:
            confidence += 0.25
        elif (doc_terms and doc_hits == len(doc_terms)) or (section_terms and section_hits == len(section_terms)):
            confidence += 0.12
        if doc_type_match:
            confidence += 0.1
        if title_hit:
            confidence += 0.25
        if gap >= 0.35:
            confidence += 0.1
        elif gap >= 0.15:
            confidence += 0.05
        return min(confidence, 1.0)

    @staticmethod
    def _should_run_corrective_search(profile: QueryProfile, results: list, confidence: float, is_exact_query: bool) -> bool:
        if is_exact_query:
            return False
        if not results:
            return bool(profile.corrective_query)
        if not profile.corrective_query:
            return False
        if profile.corrective_query == profile.query:
            return False
        if confidence < 0.45:
            return True
        if profile.complexity == "complex" and confidence < 0.6:
            return True
        return False

    @staticmethod
    def _extract_exact_terms(query: str) -> list[str]:
        return QueryExpander.extract_exact_terms(query)

    @staticmethod
    def _normalize_exact_text(text: str) -> str:
        return re.sub(r"[\s　]+", "", text or "")

    @classmethod
    def _split_exact_constraints(cls, query: str) -> tuple[list[str], list[str]]:
        return QueryExpander.split_exact_constraints(query)

    def _exact_match_counts(self, query: str, item) -> tuple[int, int]:
        doc_terms, section_terms = self._split_exact_constraints(query)
        meta = item.metadata or {}
        file_name = str(meta.get("file", ""))
        aliases = self.semantic_tool._corpus.get_document_aliases(file_name) if file_name else set()

        doc_hits = QueryExpander.count_exact_doc_hits(
            doc_terms,
            [
                item.source,
                str(meta.get("source", "")),
                str(meta.get("doc_title", "")),
                str(meta.get("standard_no", "")),
                *aliases,
            ],
        )
        section_hits = QueryExpander.count_exact_section_hits(
            section_terms,
            [
                item.source,
                str(meta.get("section_title", "")),
                item.snippet[:260],
                item.text[:1200],
            ],
        )
        return doc_hits, section_hits

    def _apply_exact_constraint_boosts(self, query: str, results: list) -> list:
        doc_terms, section_terms = self._split_exact_constraints(query)
        if not doc_terms and not section_terms:
            return results

        boosted = []
        for item in results:
            doc_hits, section_hits = self._exact_match_counts(query, item)
            bonus = doc_hits * 0.8 + section_hits * 0.55
            if doc_terms and doc_hits == 0:
                bonus -= 1.1
            elif doc_terms and doc_hits < len(doc_terms):
                bonus -= 0.35
            if section_terms and section_hits == 0:
                bonus -= 0.75
            elif section_terms and section_hits < len(section_terms):
                bonus -= 0.25
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

    def _filter_exact_mismatch_results(self, query: str, results: list) -> list:
        doc_terms, section_terms = self._split_exact_constraints(query)
        if not doc_terms and not section_terms:
            return results

        fully_matching = []
        doc_matching = []
        for item in results:
            doc_hits, section_hits = self._exact_match_counts(query, item)
            doc_ok = not doc_terms or doc_hits == len(doc_terms)
            section_ok = not section_terms or section_hits == len(section_terms)
            if doc_ok:
                doc_matching.append(item)
            if doc_ok and section_ok:
                fully_matching.append(item)

        if fully_matching:
            return fully_matching
        if doc_matching:
            return doc_matching
        return results

    def _apply_exact_match_boosts(self, query: str, results: list) -> list:
        exact_terms = self._extract_exact_terms(query)
        if not exact_terms:
            return results

        boosted = []
        for item in results:
            source = item.source or ""
            metadata = item.metadata or {}
            haystacks = [
                self._normalize_exact_text(source),
                self._normalize_exact_text(str(metadata.get("source", ""))),
                self._normalize_exact_text(str(metadata.get("doc_title", ""))),
                self._normalize_exact_text(str(metadata.get("section_title", ""))),
                self._normalize_exact_text(str(metadata.get("standard_no", ""))),
            ]
            bonus = 0.0
            for term in exact_terms:
                normalized_term = self._normalize_exact_text(term)
                if any(normalized_term in haystack for haystack in haystacks):
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

    def _rankings_have_exact_evidence(self, query: str, rankings: list[list]) -> bool:
        doc_terms, section_terms = QueryExpander.split_exact_constraints(query)
        if not doc_terms and not section_terms:
            return False
        for ranking in rankings:
            for item in ranking[:5]:
                doc_hits, section_hits = self._exact_match_counts(query, item)
                if doc_hits == len(doc_terms) and section_hits == len(section_terms):
                    return True
        return False

    @staticmethod
    def _is_change_query(query: str) -> bool:
        return any(term in query for term in ("改正", "変更", "見直し", "新基準", "改訂"))

    def _apply_change_intent_boosts(self, query: str, results: list) -> list:
        if not self._is_change_query(query):
            return results

        positive_terms = (
            "改正", "変更", "見直し", "導入", "廃止", "新た", "新設",
            "追加", "修正", "経過措置", "適用初年度", "適用時期",
        )
        negative_terms = ("議決", "委員", "名簿")
        unchanged_terms = (
            "変更していない",
            "変更してません",
            "改正前会計基準における定義を変更していない",
            "改正前会計基準の方法を変更していない",
        )
        soft_negative_terms = ("従来どおり", "従来通り", "踏襲", "同様")
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
            for term in unchanged_terms:
                if term in text_window:
                    bonus -= 0.8
            for term in soft_negative_terms:
                if term in text_window:
                    bonus -= 0.2

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
        focus_query = QueryExpander._domain_focus_variant(query) or query
        anchors = QueryExpander._extract_keywords(focus_query)
        if anchors:
            return anchors[:4]

        generic_terms = (
            "改正点", "変更点", "会計基準", "改正", "変更", "見直し", "新基準", "改訂",
            "教えてください", "教えて", "内容", "記載", "取扱い", "方法", "基準",
        )
        reduced = query
        for term in generic_terms:
            reduced = reduced.replace(term, " ")
        anchors: list[str] = []
        for token in re.findall(r"[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,8}", reduced):
            if len(token) < 2:
                continue
            if token not in anchors:
                anchors.append(token)
        return anchors[:4]

    @staticmethod
    def _filter_low_value_parent_results(results: list) -> list:
        filtered = [
            item
            for item in results
            if not is_clearly_low_value(f"{item.source}\n{item.snippet}\n{item.text[:400]}")
        ]
        return filtered or results

    def _apply_topic_alignment_boosts(self, query: str, results: list) -> list:
        anchors = self._query_anchor_terms(query)
        if not anchors:
            return results

        target_doc_type = self._target_doc_type(query)
        canonical_titles = self._canonical_title_phrases(query)
        boosted = []
        for item in results:
            text_window = f"{item.source} {item.snippet[:260]} {item.text[:800]}"
            hits = sum(1 for term in anchors if term in text_window)
            bonus = hits * 0.2
            if hits == 0:
                bonus -= 0.35
            if target_doc_type:
                meta = item.metadata or {}
                if self._doc_type_matches(target_doc_type, meta, item.source):
                    bonus += 0.25
                elif meta:
                    bonus -= 0.6
            file_name = str((item.metadata or {}).get("file", ""))
            aliases = self.semantic_tool._corpus.get_document_aliases(file_name) if file_name else set()
            if canonical_titles and aliases:
                if any(title in aliases for title in canonical_titles):
                    bonus += 1.0
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
    def _apply_focus_term_boosts(query: str, results: list) -> list:
        focus_query = QueryExpander._domain_focus_variant(query)
        if not focus_query:
            return results

        focus_terms = [term for term in focus_query.split() if len(term) >= 2]
        boosted = []
        for item in results:
            meta = item.metadata or {}
            title_window = " ".join(
                [
                    item.source,
                    str(meta.get("doc_title", "")),
                    str(meta.get("section_title", "")),
                ]
            )
            body_window = f"{item.snippet[:260]} {item.text[:800]}"
            title_hits = sum(1 for term in focus_terms if term in title_window)
            body_hits = sum(1 for term in focus_terms if term in body_window)
            bonus = title_hits * 0.45 + body_hits * 0.12
            if title_hits == 0 and body_hits <= 1:
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

    @staticmethod
    def _target_doc_type(query: str) -> str | None:
        if "実務対応報告" in query:
            return "実務対応報告"
        if "適用指針" in query:
            return "適用指針"
        if "会計基準" in query:
            return "企業会計基準"
        return None

    @staticmethod
    def _doc_type_matches(target_doc_type: str, metadata: dict, source: str) -> bool:
        haystack = " ".join(
            [
                str(metadata.get("doc_type", "")),
                str(metadata.get("standard_no", "")),
                str(metadata.get("doc_title", "")),
                source,
            ]
        ).strip()
        if target_doc_type == "企業会計基準":
            return "企業会計基準" in haystack and "適用指針" not in haystack and "実務対応報告" not in haystack
        return target_doc_type in haystack

    def _canonical_title_phrases(self, query: str) -> list[str]:
        canonical = QueryExpander._canonicalize_standard_aliases(query)
        return re.findall(r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に関する会計基準)", canonical)

    def _filter_change_delta_results(self, query: str, results: list) -> list:
        if not self._is_change_query(query):
            return results

        strong_terms = (
            "改正", "改正前", "改正後", "変更", "見直し", "新たに", "新設", "廃止",
            "従前", "従来", "今回", "経過措置", "適用初年度",
        )
        anchors = self._query_anchor_terms(query)
        kept = []
        for item in results:
            text_window = f"{item.source} {item.snippet[:260]} {item.text[:1200]}"
            has_change_term = any(term in text_window for term in strong_terms)
            has_anchor = any(term in text_window for term in anchors) if anchors else True
            if has_change_term and has_anchor:
                kept.append(item)

        return kept or results
