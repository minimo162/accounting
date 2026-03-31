"""ReAct-style agent loop for A-RAG with full context tracking."""

from collections import Counter
import json
import logging
import re
from typing import Any, AsyncGenerator

from .config import Config
from .context import AgentContext
from .llm import LLMClient
from .observability import get_monitor_case_id, get_request_id, question_sha1, structured_log
from .prompt import FINAL_COVERAGE_RULES, SYSTEM_PROMPT
from .query_rewrite import QueryExpander
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


WRAP_UP_HINT = (
    "【システム通知】検索ステップが多くなっています。"
    "これまでに収集した情報で十分回答できる場合は、追加検索せずに最終回答を提供してください。"
    "完全な情報が得られなくても、現在の情報に基づいて回答し、不足部分はその旨を明記してください。"
)


class Agent:
    NUDGE_AT_LOOP = 8  # After this many loops, hint the LLM to wrap up
    SEARCH_TOOL_NAMES = {"hybrid_search", "keyword_search", "semantic_search"}
    _SEARCH_STAGNATION_MIN_OVERLAP = 0.6
    _MAX_CROSS_REFERENCE_NUDGES = 2
    _VERIFICATION_MAX_LOOPS = 12
    _DETAIL_NOTE_MARKERS = (
        "場合", "とき", "要件", "条件", "例外", "ただし", "なお", "一方",
        "また", "比較", "違い", "区分", "判断", "経過措置", "適用時期",
        "別個", "見積り", "制約", "総額", "純額",
    )

    def __init__(self, config: Config, tools: ToolRegistry, chunk_map: dict[str, dict] | None = None, pdf_sources: dict[str, str] | None = None):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.tools = tools
        self.chunk_map = chunk_map or {}
        self.pdf_sources = pdf_sources or {}
        self.max_loops = config.agent.max_loops
        self.max_token_budget = config.agent.max_token_budget
        self.verbose = config.agent.verbose
        self.nudge_at_loop = config.agent.nudge_at_loop or self.NUDGE_AT_LOOP
        self.wrap_up_after_searches = config.agent.wrap_up_after_searches
        self.force_final_after_searches = config.agent.force_final_after_searches
        self.force_final_after_reads = config.agent.force_final_after_reads

    def _count_tool_calls(self, context: AgentContext, tool_name: str) -> int:
        return sum(1 for log in context.retrieval_logs if log.tool_name == tool_name)

    def _count_search_calls(self, context: AgentContext) -> int:
        return sum(1 for log in context.retrieval_logs if log.tool_name in self.SEARCH_TOOL_NAMES)

    @staticmethod
    def _is_verification_context(context: AgentContext) -> bool:
        return bool(context.query_profile.get("verification_mode") or context.verification_claims)

    def _loop_budget(self, context: AgentContext) -> int:
        if self._is_verification_context(context):
            return min(self.max_loops, self._VERIFICATION_MAX_LOOPS)
        return self.max_loops

    @staticmethod
    def _query_class(context: AgentContext) -> str:
        if QueryExpander.is_exact_query(context.question):
            return "exact"
        return context.question_complexity or "unknown"

    @staticmethod
    def _tool_call_counts(context: AgentContext) -> dict[str, int]:
        return dict(Counter(entry["tool_name"] for entry in context.trajectory))

    def _build_exploration_signature(self, context: AgentContext) -> str:
        segments: list[str] = []
        for idx, entry in enumerate(context.search_history[:4], start=1):
            query = str(entry.get("query", "")).strip()
            query = re.sub(r"\s+", " ", query)
            if len(query) > 36:
                query = query[:33].rstrip() + "..."
            segments.append(f"{idx}:{query or '-'}")
        if not segments:
            return "no_search"
        return " | ".join(segments)

    def _build_observability_summary(
        self,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        answer: str,
        references: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tool_call_counts = self._tool_call_counts(context)
        search_queries = [str(entry.get("query", "")).strip() for entry in context.search_history if str(entry.get("query", "")).strip()]
        confidences = [float(entry.get("confidence", 0.0)) for entry in context.search_history]
        corrective_queries = [
            str(entry.get("corrective_query", "")).strip()
            for entry in context.search_history
            if str(entry.get("corrective_query", "")).strip()
        ]
        coverage = context.get_evidence_coverage()
        cited_reference_count = self._count_cited_references(answer)

        return {
            "request_id": get_request_id() or None,
            "monitor_case_id": get_monitor_case_id() or None,
            "query_class": self._query_class(context),
            "query_complexity": context.question_complexity,
            "search_mode": str(context.query_profile.get("search_mode", "")) or None,
            "question_length": len(context.question),
            "question_sha1": question_sha1(context.question) if context.question else None,
            "loops": loops,
            "stop_reason": stop_reason,
            "tool_call_count": len(context.trajectory),
            "tool_call_counts": tool_call_counts,
            "search_count": self._count_search_calls(context),
            "read_chunk_count": len(context.read_chunk_ids),
            "read_document_count": tool_call_counts.get("read_document", 0),
            "retrieved_tokens": context.total_retrieved_tokens,
            "reference_count": len(references),
            "cited_reference_count": cited_reference_count,
            "zero_reference": cited_reference_count == 0,
            "coverage_ratio": coverage["coverage_ratio"],
            "covered_slot_count": len(coverage["covered_slots"]),
            "uncovered_slot_count": len(coverage["uncovered_slots"]),
            "search_queries": search_queries,
            "search_query_count": len(search_queries),
            "search_query_unique_count": len(set(search_queries)),
            "search_confidence_max": max(confidences) if confidences else 0.0,
            "search_confidence_min": min(confidences) if confidences else 0.0,
            "corrective_query_count": len(corrective_queries),
            "exploration_signature": self._build_exploration_signature(context),
        }

    def _log_completion(self, observability: dict[str, Any]) -> None:
        structured_log(logger, logging.INFO, "agent_run_complete", **observability)

    def _log_error(self, context: AgentContext, exc: Exception) -> None:
        structured_log(
            logger,
            logging.ERROR,
            "agent_run_error",
            request_id=get_request_id() or None,
            monitor_case_id=get_monitor_case_id() or None,
            query_class=self._query_class(context) if context.question else None,
            question_length=len(context.question or ""),
            question_sha1=question_sha1(context.question) if context.question else None,
            search_mode=str(context.query_profile.get("search_mode", "")) or None,
            tool_call_count=len(context.trajectory),
            search_count=self._count_search_calls(context),
            read_chunk_count=len(context.read_chunk_ids),
            retrieved_tokens=context.total_retrieved_tokens,
            error_type=type(exc).__name__,
            error=str(exc),
            exploration_signature=self._build_exploration_signature(context),
        )

    @staticmethod
    def _adaptive_budget(base: int, complexity: str) -> int:
        if base <= 0:
            return base
        if complexity == "simple":
            return max(1, base - 1)
        return base

    @staticmethod
    def _has_high_confidence_evidence(context: AgentContext) -> bool:
        recent = context.search_history[-2:]
        if not recent:
            return False
        best_confidence = max(float(entry.get("confidence", 0.0)) for entry in recent)
        coverage = context.get_evidence_coverage()
        covered_slots = len(coverage.get("covered_slots", []))
        return best_confidence >= 0.7 and (
            len(context.evidence_notes) >= 4 or covered_slots >= max(2, len(coverage.get("slots", [])) - 1)
        )

    @staticmethod
    def _best_search_confidence(context: AgentContext) -> float:
        if not context.search_history:
            return 0.0
        recent = context.search_history[-3:]
        return max(float(entry.get("confidence", 0.0)) for entry in recent)

    @staticmethod
    def _required_slot_labels(context: AgentContext) -> set[str]:
        required: set[str] = set()
        focus_text = " ".join(
            [
                context.question,
                str(context.query_profile.get("canonical_focus_query", "")),
                str(context.query_profile.get("corrective_query", "")),
            ]
        )
        if "ヘッジ" in focus_text and "要件" in focus_text and "有効性" in context.evidence_slot_terms:
            required.add("有効性")
        return required

    @staticmethod
    def _has_exact_clause_evidence(context: AgentContext) -> bool:
        return bool(context.exact_evidence_chunk_ids)

    @classmethod
    def _has_exact_shortfall(cls, context: AgentContext) -> bool:
        if not QueryExpander.is_exact_query(context.question):
            return False
        if cls._has_exact_clause_evidence(context):
            return False
        if any(bool(entry.get("exact_shortfall")) for entry in context.search_history[-2:]):
            return True
        return any(
            log.tool_name == "hybrid_search" and bool(log.metadata.get("exact_shortfall"))
            for log in context.retrieval_logs[-3:]
        )

    @classmethod
    def _search_results_are_stagnating(cls, context: AgentContext) -> bool:
        recent_entries = [entry for entry in context.search_history[-3:] if entry.get("chunk_ids")]
        if len(recent_entries) < 2:
            return False

        overlaps: list[float] = []
        for prev, curr in zip(recent_entries, recent_entries[1:]):
            prev_ids = set(str(chunk_id) for chunk_id in prev.get("chunk_ids", [])[:5] if str(chunk_id))
            curr_ids = set(str(chunk_id) for chunk_id in curr.get("chunk_ids", [])[:5] if str(chunk_id))
            if not prev_ids or not curr_ids:
                return False
            overlap = len(prev_ids & curr_ids) / min(len(prev_ids), len(curr_ids))
            overlaps.append(overlap)

        if not overlaps or min(overlaps) < cls._SEARCH_STAGNATION_MIN_OVERLAP:
            return False

        normalized_queries: list[str] = []
        for entry in recent_entries:
            query = (
                str(entry.get("effective_query", "")).strip()
                or str(entry.get("corrective_query", "")).strip()
                or str(entry.get("query", "")).strip()
            )
            if not query:
                continue
            normalized = re.sub(r"[\s　、。・/／()（）「」『』【】\[\]{}:：?？!！]+", " ", query)
            tokens = [token for token in normalized.split() if len(token) >= 2]
            normalized_queries.append(" ".join(tokens[:5]))

        return len(set(normalized_queries)) <= max(1, len(normalized_queries) - 1)

    @classmethod
    def _detail_rich_note_count(cls, context: AgentContext) -> int:
        return sum(
            1
            for note in context.evidence_notes.values()
            if len(str(note or "")) >= 120 or any(marker in str(note or "") for marker in cls._DETAIL_NOTE_MARKERS)
        )

    def _evidence_requirements(self, context: AgentContext) -> dict[str, float | int]:
        coverage = context.get_evidence_coverage()
        slot_count = len(coverage["slots"])
        complexity = context.question_complexity
        keyword_first = str(context.query_profile.get("search_mode", "")) == "keyword_first"

        if complexity == "simple":
            min_searches = 1
            min_reads = 1
            min_notes = 1
            required_slots = slot_count if slot_count <= 2 else 2
            min_confidence = 0.45 if keyword_first else 0.55
        elif complexity == "complex":
            min_searches = 1 if keyword_first and slot_count <= 4 else 2
            min_reads = 2 if slot_count <= 4 else 3
            min_notes = 2 if slot_count <= 4 else 3
            if slot_count == 0:
                required_slots = 0
            else:
                required_slots = max(2, (slot_count * 3 + 3) // 4)
            min_confidence = 0.55 if keyword_first else 0.6
        else:
            min_searches = 1 if keyword_first else 2
            min_reads = 2
            min_notes = 2
            if slot_count <= 3:
                required_slots = slot_count
            else:
                required_slots = max(2, (slot_count * 2 + 2) // 3)
            min_confidence = 0.5 if keyword_first else 0.55

        return {
            "min_searches": min_searches,
            "min_reads": min_reads,
            "min_notes": min_notes,
            "required_slots": required_slots,
            "min_confidence": min_confidence,
        }

    def _has_sufficient_evidence(self, context: AgentContext) -> bool:
        requirements = self._evidence_requirements(context)
        search_count = self._count_search_calls(context)
        read_count = self._count_tool_calls(context, "read_chunk")
        read_document_count = self._count_tool_calls(context, "read_document")
        coverage = context.get_evidence_coverage()
        covered_slots = len(coverage["covered_slots"])
        slot_count = len(coverage["slots"])
        uncovered_slot_count = max(slot_count - covered_slots, 0)
        required_slot_labels = self._required_slot_labels(context)
        keyword_first = str(context.query_profile.get("search_mode", "")) == "keyword_first"
        complexity = context.question_complexity
        exact_query = QueryExpander.is_exact_query(context.question)
        best_confidence = self._best_search_confidence(context)
        required_slots = int(requirements["required_slots"])
        detail_seeking = self._is_detail_seeking_context(context)
        detail_rich_notes = self._detail_rich_note_count(context)
        read_like_count = max(read_count, read_document_count) if exact_query else read_count
        allow_keyword_first_example_shortfall = (
            keyword_first
            and complexity == "moderate"
            and slot_count >= 3
            and uncovered_slot_count <= 1
            and covered_slots >= max(2, required_slots - 1)
            and search_count >= int(requirements["min_searches"])
            and read_count >= int(requirements["min_reads"])
            and len(context.evidence_notes) >= int(requirements["min_notes"])
            and best_confidence >= 0.4
        )
        allow_detail_keyword_shortfall = (
            detail_seeking
            and keyword_first
            and complexity == "moderate"
            and slot_count >= 2
            and uncovered_slot_count <= 1
            and covered_slots >= max(1, required_slots - 1)
            and search_count >= int(requirements["min_searches"])
            and read_like_count >= int(requirements["min_reads"])
            and len(context.evidence_notes) >= int(requirements["min_notes"])
            and detail_rich_notes >= max(2, len(context.evidence_notes) - 1)
            and best_confidence >= 0.35
        )
        allow_one_slot_shortfall = (
            complexity == "complex"
            and required_slots > 0
            and covered_slots >= required_slots
            and uncovered_slot_count <= 1
            and search_count >= 2
            and len(context.evidence_notes) >= int(requirements["min_notes"])
            and best_confidence >= 0.45
        )

        if search_count < int(requirements["min_searches"]):
            return False
        min_reads = int(requirements["min_reads"])
        if allow_one_slot_shortfall:
            min_reads = max(2, min_reads - 1)
        if read_like_count < min_reads:
            return False
        if len(context.evidence_notes) < int(requirements["min_notes"]):
            return False
        if exact_query and not self._has_exact_clause_evidence(context):
            return False
        if required_slot_labels and any(not context.evidence_slot_hits.get(label) for label in required_slot_labels):
            return False

        if (
            required_slots > 0
            and covered_slots < required_slots
            and not allow_keyword_first_example_shortfall
            and not allow_detail_keyword_shortfall
        ):
            return False

        if exact_query:
            return True

        if slot_count > 0 and covered_slots >= slot_count:
            return True

        if keyword_first and required_slots > 0:
            return True

        if allow_detail_keyword_shortfall:
            return True

        if best_confidence >= float(requirements["min_confidence"]):
            return True

        if allow_one_slot_shortfall:
            return True

        if self._search_results_are_stagnating(context):
            required_stagnation_slots = max(required_slots, slot_count - 1) if slot_count > 0 else required_slots
            required_note_surplus = 1 if complexity == "complex" else 0
            if (
                covered_slots >= required_stagnation_slots
                and len(context.evidence_notes) >= int(requirements["min_notes"]) + required_note_surplus
            ):
                return True

        return False

    def _build_exact_gap_message(self, context: AgentContext) -> dict[str, str] | None:
        if context.exact_gap_nudge_sent or not self._has_exact_shortfall(context):
            return None
        if self._count_search_calls(context) < 1:
            return None

        context.exact_gap_nudge_sent = True
        return {
            "role": "user",
            "content": (
                "【システム通知】指定条項の原文根拠がまだ取れていません。"
                "追加の hybrid_search は増やさず、候補文書が見えているなら read_document で原文を確認し、"
                "該当条項の記載を含む根拠を 1 件確保してください。"
                "別基準の一般論で埋めて最終回答しないでください。"
            ),
        }

    def _build_cross_reference_message(self, context: AgentContext) -> dict[str, str] | None:
        if context.cross_reference_nudges_sent >= self._MAX_CROSS_REFERENCE_NUDGES:
            return None
        if self._count_search_calls(context) < 1 or self._count_tool_calls(context, "read_chunk") < 1:
            return None

        searched_text = " ".join(
            filter(
                None,
                [
                    str(entry.get("query", "")).strip()
                    for entry in context.search_history
                ]
                + [
                    str(entry.get("effective_query", "")).strip()
                    for entry in context.search_history
                ],
            )
        )
        candidates: list[dict[str, Any]] = []
        for item in context.discovered_cross_references.values():
            doc_term = str(item.get("doc_term", "")).strip()
            if not doc_term or doc_term in searched_text:
                continue
            section_terms = [str(term).strip() for term in item.get("section_terms", []) if str(term).strip()]
            candidates.append(
                {
                    "doc_term": doc_term,
                    "section_terms": section_terms,
                    "search_query": " ".join([doc_term, *section_terms[:1]]).strip(),
                }
            )

        if not candidates:
            return None

        top = candidates[0]
        context.cross_reference_nudges_sent += 1
        suffix = f"（{context.cross_reference_nudges_sent}/{self._MAX_CROSS_REFERENCE_NUDGES}回目）"
        return {
            "role": "user",
            "content": (
                "【システム通知】読取済みの根拠本文から、未確認の参照先基準が見つかりました。"
                f"{suffix}\n"
                f"- 追加検索候補: {top['search_query']}\n"
                "この参照先だけを対象に 1 回だけ追加で hybrid_search し、必要なら read_chunk または read_document で確認してください。"
                "同じ参照先を繰り返し検索しないでください。"
            ),
        }

    def _build_coverage_gap_message(self, context: AgentContext) -> dict[str, str] | None:
        coverage = context.get_evidence_coverage()
        uncovered_slots = coverage["uncovered_slots"]
        if not uncovered_slots:
            return None

        search_count = self._count_search_calls(context)
        read_count = self._count_tool_calls(context, "read_chunk")
        if search_count < 2 or read_count < 2:
            return None

        signature = "|".join(uncovered_slots)
        if context.coverage_gap_nudge_signature == signature:
            return None
        context.coverage_gap_nudge_signature = signature
        slot_text = "、".join(uncovered_slots[:4])
        content = (
            "【システム通知】まだ根拠メモが足りない論点があります: "
            f"{slot_text}。追加検索する場合は、この未充足論点だけをクエリにしてください。"
            "既に候補文書は見えているのに該当論点の evidence note が取れていない場合に限り、"
            "read_document でその文書全体を確認してください。"
        )
        return {"role": "user", "content": content}

    def _coverage_focus_query(self, context: AgentContext) -> str:
        exact_terms = QueryExpander.extract_exact_terms(context.question)
        if exact_terms:
            return " ".join(exact_terms[:2])
        for key in ("canonical_focus_query", "corrective_query"):
            value = str(context.query_profile.get(key, "")).strip()
            if value:
                return value
        keywords = [str(keyword).strip() for keyword in context.query_profile.get("keywords", []) if str(keyword).strip()]
        if keywords:
            return " ".join(keywords[:4])
        return context.question.strip()

    def _coverage_plan_items(self, context: AgentContext) -> list[dict[str, Any]]:
        if not context.evidence_slots:
            return []

        ordered_note_ids = self._ordered_evidence_note_ids(context)
        fallback_ids = ordered_note_ids[:2] or context.searched_chunk_ids[:2]
        focus_query = self._coverage_focus_query(context)
        items: list[dict[str, Any]] = []

        for slot in context.evidence_slots:
            aliases = tuple(
                term for term in context.evidence_slot_terms.get(slot, (slot,)) if str(term).strip()
            ) or (slot,)
            supporting_ids = [
                chunk_id for chunk_id in ordered_note_ids if chunk_id in context.evidence_slot_hits.get(slot, set())
            ]
            items.append({
                "label": slot,
                "aliases": aliases,
                "supporting_ids": supporting_ids,
                "covered": bool(supporting_ids),
                "fallback_ids": fallback_ids,
                "search_query": " ".join(part for part in (focus_query, slot) if part).strip(),
            })
        return items

    def _build_coverage_review_message(self, context: AgentContext, stop_reason: str | None) -> dict[str, str] | None:
        if stop_reason is None or context.final_coverage_review_done:
            return None

        if self._has_exact_shortfall(context):
            search_count = self._count_search_calls(context)
            read_like_count = self._count_tool_calls(context, "read_chunk") + self._count_tool_calls(context, "read_document")
            if search_count < 1 or read_like_count < 1:
                return None
            return {
                "role": "user",
                "content": (
                    "【システム通知】最終回答の直前です。指定条項の原文根拠がまだ不足しています。\n"
                    "次のどちらかだけを選んでください。\n"
                    "- 候補文書に対して read_document を 1 回だけ使い、指定条項の本文を確認する\n"
                    "- これ以上原文根拠が増えないなら、最終回答で「今回確認できた根拠では不十分」と明示する\n"
                    "一般的な解説に逃げず、指定条項の有無を基準に判断してください。"
                ),
            }

        coverage = context.get_evidence_coverage()
        if not coverage["uncovered_slots"]:
            return None

        search_count = self._count_search_calls(context)
        read_count = self._count_tool_calls(context, "read_chunk")
        if search_count < 1 or read_count < 1:
            return None

        plan_lines = []
        for item in self._coverage_plan_items(context):
            status = "根拠あり" if item["covered"] else "根拠不足"
            refs = "、".join(item["supporting_ids"][:2]) if item["supporting_ids"] else "-"
            line = f"- {item['label']}: {status} / 確認済み chunk: {refs}"
            if not item["covered"] and item["search_query"]:
                line += f" / 追加検索候補: {item['search_query']}"
            plan_lines.append(line)

        content = (
            "【システム通知】最終回答の直前です。論点カバレッジを 1 回だけ再点検してください。\n"
            "未充足論点が残る場合は、次のどちらかだけを選んでください。\n"
            "- その未充足論点だけを対象に 1 回だけ追加検索し、必要なら read_chunk または read_document を続ける\n"
            "- これ以上根拠が増えないと判断したら、最終回答でその論点を「今回確認できた根拠では不十分」と明示する\n"
            "既に根拠がある論点は再検索しないでください。\n\n"
            "## 論点カバレッジ\n"
            + "\n".join(plan_lines)
            + "\n\n"
            + FINAL_COVERAGE_RULES.strip()
        )
        return {"role": "user", "content": content}

    def _force_stop_reason(self, context: AgentContext) -> str | None:
        if self._has_sufficient_evidence(context):
            return "evidence_sufficient"
        if self.force_final_after_searches <= 0 or self.force_final_after_reads <= 0:
            return None
        search_count = self._count_search_calls(context)
        read_count = self._count_tool_calls(context, "read_chunk")
        complexity = context.question_complexity
        search_budget = self._adaptive_budget(self.force_final_after_searches, complexity)
        read_budget = self._adaptive_budget(self.force_final_after_reads, complexity)
        if self._has_high_confidence_evidence(context):
            search_budget = max(2, search_budget - 1)
            read_budget = max(2, read_budget - 1)
        if search_count >= search_budget and read_count >= read_budget:
            return "retrieval_budget"
        return None

    def _should_nudge_wrap_up(self, context: AgentContext) -> bool:
        if context.wrap_up_nudged or self.wrap_up_after_searches <= 0:
            return False
        search_count = self._count_search_calls(context)
        read_count = self._count_tool_calls(context, "read_chunk")
        complexity = context.question_complexity
        search_budget = self._adaptive_budget(self.wrap_up_after_searches, complexity)
        read_budget = self._adaptive_budget(2, complexity)
        return search_count >= search_budget and read_count >= read_budget

    def _should_force_wrap_up(self, context: AgentContext) -> bool:
        return self._force_stop_reason(context) is not None

    def _maybe_nudge(self, messages: list[dict], loop_idx: int, context: AgentContext):
        """Inject a wrap-up hint if we've been searching too long."""
        exact_hint = self._build_exact_gap_message(context)
        if exact_hint is not None:
            messages.append(exact_hint)
        cross_reference_hint = self._build_cross_reference_message(context)
        if cross_reference_hint is not None:
            messages.append(cross_reference_hint)
        coverage_hint = self._build_coverage_gap_message(context)
        if coverage_hint is not None:
            messages.append(coverage_hint)
        if context.wrap_up_nudged:
            return
        if loop_idx == self.nudge_at_loop or self._should_nudge_wrap_up(context):
            messages.append({"role": "user", "content": WRAP_UP_HINT})
            context.wrap_up_nudged = True

    def _build_initial_messages(self, question: str, history: list[dict] | None = None) -> list[dict]:
        """Build initial messages with optional conversation history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            # Include recent conversation turns as context
            # Keep last 2 turns to avoid token overflow
            recent = history[-2:]
            context_text = "以下は直前の会話です。ユーザーの新しい質問が前の話題の続きかどうかを判断し、続きであれば前の文脈を踏まえて検索・回答してください。全く別の話題であれば、独立した質問として扱ってください。\n\n"
            for turn in recent:
                q = turn["question"] if isinstance(turn, dict) else turn.question
                a = turn["answer"] if isinstance(turn, dict) else turn.answer
                context_text += f"ユーザー: {q}\nアシスタント: {a[:500]}\n\n"
            messages.append({"role": "user", "content": context_text})
            messages.append({"role": "assistant", "content": "理解しました。前の会話の文脈を踏まえて、新しい質問に回答します。"})

        messages.append({"role": "user", "content": question})
        return messages

    def _build_verification_loop_message(self, context: AgentContext) -> dict[str, str] | None:
        if not self._is_verification_context(context) or not context.verification_results:
            return None

        lines = [
            "【判断検証モード】この質問は、ユーザーの判断を主張単位で照合するタスクです。",
            "次の順で進めてください。",
            "- まず主張ごとに `hybrid_search` を 1 回ずつ試し、必要なら `read_chunk` または `read_document` で根拠本文を確認する",
            "- 既に十分な根拠を読めた主張は再検索しない",
            "- 最終回答では、各主張に `○適切` / `△要注意` / `×不適切` のいずれかを付けて報告する",
            "",
            "## 検証対象の主張",
        ]
        for item in context.verification_results:
            refs = item.get("cited_references", [])
            ref_text = f" | 依拠条文: {', '.join(refs)}" if refs else ""
            query_text = str(item.get("search_query", "")).strip()
            lines.append(
                f"- 主張{item['index']}: {item['claim']}{ref_text}"
            )
            if query_text:
                lines.append(f"  推奨検索: {query_text}")
        return {"role": "user", "content": "\n".join(lines)}

    def _build_verification_result_message(self, context: AgentContext) -> dict[str, str] | None:
        if not self._is_verification_context(context) or not context.verification_results:
            return None

        lines = ["## 主張別照合メモ"]
        for item in context.verification_results:
            refs = item.get("cited_references", [])
            ref_text = f" | 依拠条文: {', '.join(refs)}" if refs else ""
            lines.append(
                f"- 主張{item['index']} | 判定候補: {item['judgment']} | 状態: {item['status']}{ref_text}"
            )
            lines.append(f"  - 内容: {item['claim']}")
            query_text = str(item.get("search_query", "")).strip()
            if query_text:
                lines.append(f"  - 推奨検索: {query_text}")
            evidence_ids = [str(chunk_id) for chunk_id in item.get("evidence_chunk_ids", []) if str(chunk_id)]
            if evidence_ids:
                refs_text = "".join(f"[{chunk_id}]" for chunk_id in evidence_ids[:2])
                lines.append(f"  - 根拠候補: {refs_text}")
            else:
                search_ids = [str(chunk_id) for chunk_id in item.get("search_chunk_ids", []) if str(chunk_id)]
                if search_ids:
                    refs_text = "".join(f"[{chunk_id}]" for chunk_id in search_ids[:2])
                    lines.append(f"  - 検索済み候補: {refs_text}")
                else:
                    lines.append("  - 根拠候補: 未確認")
        return {"role": "user", "content": "\n".join(lines)}

    @staticmethod
    def _has_verification_answer_sections(answer: str) -> bool:
        required = ("## 主張要約", "## 照合結果", "## 追加考慮事項", "## 参照")
        return all(section in answer for section in required)

    def _tool_schemas(self, context: AgentContext) -> list[dict[str, Any]]:
        return self.tools.get_schemas(context)

    def _seed_context(self, context: AgentContext, question: str):
        profile = QueryExpander(self.config.retrieval, self.llm).analyze(question)
        context.set_question(question, profile.to_dict())

    def run(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Synchronous run: returns final answer with metadata."""
        context = AgentContext()
        try:
            self._seed_context(context, question)
            messages = self._build_initial_messages(question, history)
            verification_message = self._build_verification_loop_message(context)
            if verification_message is not None:
                messages.append(verification_message)
            total_cost = 0.0
            loop_budget = self._loop_budget(context)

            for loop_idx in range(loop_budget):
                if self.verbose:
                    logger.info(f"Loop {loop_idx + 1}/{loop_budget}")

                self._maybe_nudge(messages, loop_idx, context)

                stop_reason = self._force_stop_reason(context)
                coverage_review = self._build_coverage_review_message(context, stop_reason)
                if coverage_review is not None:
                    messages.append(coverage_review)
                    context.final_coverage_review_done = True
                    stop_reason = None
                if stop_reason is not None:
                    answer, cost = self._force_final_answer(messages, context)
                    total_cost += cost
                    result = self._build_result(answer, context, loop_idx + 1, stop_reason, total_cost)
                    self._log_completion(result["observability"])
                    return result

                # Token budget check
                current_tokens = self.llm.count_message_tokens(messages)
                if current_tokens > self.max_token_budget:
                    logger.info(f"Token budget exceeded: {current_tokens} > {self.max_token_budget}")
                    answer, cost = self._force_final_answer(messages, context)
                    total_cost += cost
                    result = self._build_result(answer, context, loop_idx + 1, "budget_exceeded", total_cost)
                    self._log_completion(result["observability"])
                    return result

                tool_schemas = self._tool_schemas(context)
                response = self.llm.chat(messages=messages, tools=tool_schemas)
                message = response["message"]
                total_cost += response.get("cost", 0.0)
                messages.append(message)

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    result = self._build_result(
                        message.get("content", ""), context, loop_idx + 1, "natural", total_cost
                    )
                    if self._should_retry_natural_answer(result, context):
                        answer, cost = self._force_final_answer(messages, context)
                        total_cost += cost
                        result = self._build_result(answer, context, loop_idx + 1, "natural", total_cost)
                    self._log_completion(result["observability"])
                    return result

                # Execute tool calls
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}
                        logger.warning(f"Failed to parse tool arguments: {tc['function']['arguments']}")

                    if self.verbose:
                        logger.info(f"  Tool: {func_name}({func_args})")

                    result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                    # Trajectory logging
                    context.add_trajectory_entry(
                        loop=loop_idx + 1,
                        tool_name=func_name,
                        arguments=func_args,
                        tool_result=result_text,
                        tool_log=tool_log,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                        "_func_name": func_name,
                    })

            # Max loops exceeded
            answer, cost = self._force_final_answer(messages, context)
            total_cost += cost
            result = self._build_result(answer, context, loop_budget, "max_loops", total_cost)
            self._log_completion(result["observability"])
            return result
        except Exception as exc:
            self._log_error(context, exc)
            raise

    async def arun(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Async run."""
        context = AgentContext()
        try:
            self._seed_context(context, question)
            messages = self._build_initial_messages(question, history)
            verification_message = self._build_verification_loop_message(context)
            if verification_message is not None:
                messages.append(verification_message)
            total_cost = 0.0
            loop_budget = self._loop_budget(context)

            for loop_idx in range(loop_budget):
                self._maybe_nudge(messages, loop_idx, context)

                stop_reason = self._force_stop_reason(context)
                coverage_review = self._build_coverage_review_message(context, stop_reason)
                if coverage_review is not None:
                    messages.append(coverage_review)
                    context.final_coverage_review_done = True
                    stop_reason = None
                if stop_reason is not None:
                    answer, cost = await self._aforce_final_answer(messages, context)
                    total_cost += cost
                    result = self._build_result(answer, context, loop_idx + 1, stop_reason, total_cost)
                    self._log_completion(result["observability"])
                    return result

                # Token budget check
                current_tokens = self.llm.count_message_tokens(messages)
                if current_tokens > self.max_token_budget:
                    answer, cost = await self._aforce_final_answer(messages, context)
                    total_cost += cost
                    result = self._build_result(answer, context, loop_idx + 1, "budget_exceeded", total_cost)
                    self._log_completion(result["observability"])
                    return result

                tool_schemas = self._tool_schemas(context)
                response = await self.llm.achat(messages=messages, tools=tool_schemas)
                message = response["message"]
                total_cost += response.get("cost", 0.0)
                messages.append(message)

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    result = self._build_result(
                        message.get("content", ""), context, loop_idx + 1, "natural", total_cost
                    )
                    if self._should_retry_natural_answer(result, context):
                        answer, cost = await self._aforce_final_answer(messages, context)
                        total_cost += cost
                        result = self._build_result(answer, context, loop_idx + 1, "natural", total_cost)
                    self._log_completion(result["observability"])
                    return result

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                    context.add_trajectory_entry(
                        loop=loop_idx + 1,
                        tool_name=func_name,
                        arguments=func_args,
                        tool_result=result_text,
                        tool_log=tool_log,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                        "_func_name": func_name,
                    })

            answer, cost = await self._aforce_final_answer(messages, context)
            total_cost += cost
            result = self._build_result(answer, context, loop_budget, "max_loops", total_cost)
            self._log_completion(result["observability"])
            return result
        except Exception as exc:
            self._log_error(context, exc)
            raise

    async def arun_stream(self, question: str, history: list[dict] | None = None) -> AsyncGenerator[dict, None]:
        """Async streaming run - yields events for SSE.

        Tool-calling loops use non-streaming (need full response for tool parsing).
        Final answer is streamed token-by-token via answer_delta events.
        """
        context = AgentContext()
        try:
            self._seed_context(context, question)
            messages = self._build_initial_messages(question, history)
            verification_message = self._build_verification_loop_message(context)
            if verification_message is not None:
                messages.append(verification_message)
            total_cost = 0.0
            loop_budget = self._loop_budget(context)

            for loop_idx in range(loop_budget):
                status_msg = "調査中..." if loop_idx == 0 else f"調査中... (ステップ {loop_idx + 1})"
                yield {"type": "status", "data": status_msg}

                self._maybe_nudge(messages, loop_idx, context)

                stop_reason = self._force_stop_reason(context)
                coverage_review = self._build_coverage_review_message(context, stop_reason)
                if coverage_review is not None:
                    messages.append(coverage_review)
                    context.final_coverage_review_done = True
                    stop_reason = None
                if stop_reason is not None:
                    yield {"type": "status", "data": "回答を生成中..."}
                    async for event in self._astream_final_answer(messages, context, loop_idx + 1, stop_reason, total_cost):
                        yield event
                    return

                # Token budget check
                current_tokens = self.llm.count_message_tokens(messages)
                if current_tokens > self.max_token_budget:
                    yield {"type": "status", "data": "回答を生成中..."}
                    async for event in self._astream_final_answer(messages, context, loop_idx + 1, "budget_exceeded", total_cost):
                        yield event
                    return

                tool_schemas = self._tool_schemas(context)
                response = await self.llm.achat(messages=messages, tools=tool_schemas)
                message = response["message"]
                total_cost += response.get("cost", 0.0)
                messages.append(message)

                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    result = self._build_completed_result(
                        message.get("content", ""),
                        context,
                        loop_idx + 1,
                        "natural",
                        total_cost,
                    )
                    if self._should_retry_natural_answer(result, context):
                        answer, cost = await self._aforce_final_answer(messages, context)
                        total_cost += cost
                        result = self._build_completed_result(
                            answer,
                            context,
                            loop_idx + 1,
                            "natural",
                            total_cost,
                        )
                    yield {"type": "status", "data": "回答を生成中..."}
                    chunk_size = 8
                    for i in range(0, len(result["answer"]), chunk_size):
                        yield {"type": "answer_delta", "data": result["answer"][i:i + chunk_size]}
                    yield {"type": "answer_done", "data": result["answer"]}
                    for ref in result["references"]:
                        yield {"type": "reference", "data": ref}
                    self._log_completion(result["observability"])
                    yield {
                        "type": "done",
                        "data": {
                            "loops": result["loops"],
                            "stop_reason": result["stop_reason"],
                            "source_url_map": result["source_url_map"],
                            "chunks_read_count": result["chunks_read_count"],
                            "read_chunk_count": result["read_chunk_count"],
                            "total_cost": result["total_cost"],
                            "total_retrieved_tokens": result["total_retrieved_tokens"],
                            "evidence_coverage": result.get("evidence_coverage", {}),
                            "uncertainty": result.get("uncertainty", {}),
                            "request_id": result.get("request_id"),
                            "query_class": result.get("query_class"),
                        },
                    }
                    return

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    yield {
                        "type": "tool_call",
                        "data": {"tool": func_name, "args": func_args},
                    }
                    result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                    context.add_trajectory_entry(
                        loop=loop_idx + 1,
                        tool_name=func_name,
                        arguments=func_args,
                        tool_result=result_text,
                        tool_log=tool_log,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                        "_func_name": func_name,
                    })

            # Max loops - force final answer with streaming
            yield {"type": "status", "data": "回答を生成中..."}
            async for event in self._astream_final_answer(messages, context, loop_budget, "max_loops", total_cost):
                yield event
        except Exception as exc:
            self._log_error(context, exc)
            raise

    async def _astream_final_answer(
        self,
        messages: list[dict],
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> AsyncGenerator[dict, None]:
        """Force a final answer and simulate streaming output."""
        answer, cost = await self._aforce_final_answer(messages, context)
        total_cost += cost
        result = self._build_completed_result(answer, context, loops, stop_reason, total_cost)

        chunk_size = 8
        for i in range(0, len(result["answer"]), chunk_size):
            yield {"type": "answer_delta", "data": result["answer"][i:i + chunk_size]}
        yield {"type": "answer_done", "data": result["answer"]}
        for ref in result["references"]:
            yield {"type": "reference", "data": ref}
        self._log_completion(result["observability"])
        yield {
            "type": "done",
            "data": {
                "loops": result["loops"],
                "stop_reason": result["stop_reason"],
                "source_url_map": result["source_url_map"],
                "chunks_read_count": result["chunks_read_count"],
                "read_chunk_count": result["read_chunk_count"],
                "total_cost": result["total_cost"],
                "total_retrieved_tokens": result["total_retrieved_tokens"],
                "evidence_coverage": result.get("evidence_coverage", {}),
                "uncertainty": result.get("uncertainty", {}),
                "request_id": result.get("request_id"),
                "query_class": result.get("query_class"),
            },
        }

    @staticmethod
    def _note_limits(complexity: str, *, detail_seeking: bool = False) -> tuple[int, int, int]:
        if complexity == "complex":
            return (7, 1200, 6800) if detail_seeking else (6, 900, 5200)
        if complexity == "simple":
            return 4, 500, 2400
        return (6, 950, 4700) if detail_seeking else (5, 700, 3600)

    @staticmethod
    def _is_detail_seeking_context(context: AgentContext) -> bool:
        return bool(context.query_profile.get("detail_seeking")) or context.question_complexity == "complex"

    def _ordered_evidence_note_ids(self, context: AgentContext) -> list[str]:
        ordered_ids: list[str] = []
        seen: set[str] = set()
        for chunk_id in context.searched_chunk_ids:
            if chunk_id in context.evidence_notes and chunk_id not in seen:
                ordered_ids.append(chunk_id)
                seen.add(chunk_id)
        for chunk_id in context.evidence_notes:
            if chunk_id not in seen:
                ordered_ids.append(chunk_id)
                seen.add(chunk_id)
        return ordered_ids

    def _slot_focused_evidence_sections(
        self,
        context: AgentContext,
        per_note_chars: int,
        total_chars: int,
    ) -> list[str]:
        sections: list[str] = []
        used_chars = 0
        used_chunk_ids: set[str] = set()

        for item in self._coverage_plan_items(context):
            supporting_ids = [str(chunk_id) for chunk_id in item["supporting_ids"]]
            if not supporting_ids:
                continue
            chunk_id = supporting_ids[0]
            if chunk_id in used_chunk_ids:
                continue
            note = context.evidence_notes.get(chunk_id, "").strip()
            if not note:
                continue
            aliases = tuple(str(alias) for alias in item["aliases"])
            snippet = self._extract_slot_snippet(
                note,
                aliases,
                prefer_detail=self._is_detail_seeking_context(context),
            ).replace("[抜粋]", "").strip()
            if not snippet:
                continue
            if len(snippet) > per_note_chars:
                snippet = snippet[:per_note_chars].rstrip() + "..."
            chunk = self.chunk_map.get(chunk_id, {})
            source = str(chunk.get("source", "")).strip()
            header = f"- {item['label']} | {chunk_id}"
            if source:
                header = f"{header} | {source}"
            section = f"{header}\n{snippet}"
            if sections and used_chars + len(section) > total_chars:
                break
            sections.append(section)
            used_chars += len(section)
            used_chunk_ids.add(chunk_id)

        return sections

    def _build_evidence_note_message(self, context: AgentContext) -> dict[str, str] | None:
        if not context.evidence_notes:
            return None

        detail_seeking = self._is_detail_seeking_context(context)
        max_notes, per_note_chars, total_chars = self._note_limits(
            context.question_complexity,
            detail_seeking=detail_seeking,
        )
        used_chars = 0
        sections = self._slot_focused_evidence_sections(context, per_note_chars, total_chars)
        if sections:
            used_chars = sum(len(section) for section in sections)
        used_chunk_ids = {
            match.group(1)
            for section in sections
            for match in re.finditer(r"- [^\n|]+\| ([^\s|]+)", section)
        }
        coverage = context.get_evidence_coverage()
        allow_fallback_note_dump = not sections or (
            context.question_complexity == "complex" and len(sections) < 2
        )

        for chunk_id in self._ordered_evidence_note_ids(context):
            if not allow_fallback_note_dump:
                break
            if chunk_id in used_chunk_ids:
                continue
            note = context.evidence_notes.get(chunk_id, "").strip()
            if not note:
                continue
            note = note.replace("\n[抜粋]", "").replace("[抜粋]", "").strip()
            if len(note) > per_note_chars:
                note = note[:per_note_chars].rstrip() + "..."
            chunk = self.chunk_map.get(chunk_id, {})
            source = str(chunk.get("source", "")).strip()
            header = f"- {chunk_id}"
            if source:
                header = f"{header} | {source}"
            section = f"{header}\n{note}"
            if sections and used_chars + len(section) > total_chars:
                break
            sections.append(section)
            used_chars += len(section)
            if len(sections) >= max_notes:
                break

        if not sections:
            return None

        content = (
            "以下は read_chunk で抽出済みの重要箇所メモです。"
            "追加検索は行わず、このメモを優先して最終回答を作成してください。\n"
            "長文質問や詳説要求では、このメモに含まれる具体的な変更点・要件・条件・例外・経過措置を落とさずに整理してください。\n"
            "各記述の末尾には、見出し行に書かれた chunk ID をそのまま付けてください。\n\n"
        )
        if coverage["covered_slots"]:
            content += "## 充足済み論点\n" + "、".join(coverage["covered_slots"]) + "\n\n"
        if coverage["uncovered_slots"]:
            content += (
                "## 未充足論点\n"
                + "、".join(coverage["uncovered_slots"])
                + "。この論点は十分な根拠メモがないため、断定せず不足として扱ってください。\n\n"
            )
        content += (
            "## 重要箇所メモ\n"
            + "\n\n".join(sections)
        )
        return {"role": "user", "content": content}

    def _build_coverage_plan_message(self, context: AgentContext) -> dict[str, str] | None:
        items = self._coverage_plan_items(context)
        if not items:
            return None

        lines = []
        for item in items:
            status = "根拠あり" if item["covered"] else "根拠不足"
            refs = "、".join(item["supporting_ids"][:2]) if item["supporting_ids"] else "-"
            line = f"- {item['label']}: {status} / 根拠 chunk: {refs}"
            if not item["covered"] and item["search_query"]:
                line += f" / 追加検索候補: {item['search_query']}"
            lines.append(line)

        content = (
            "以下は質問から抽出した想定論点です。最終回答では、この論点順に coverage を点検してください。\n\n"
            "## 想定論点\n"
            + "\n".join(lines)
            + "\n\n"
            + FINAL_COVERAGE_RULES.strip()
        )
        return {"role": "user", "content": content}

    @staticmethod
    def _exact_clause_key_terms(note: str) -> list[str]:
        mappings = (
            ("通常の売買取引に係る方法に準じ", "通常の売買処理"),
            ("通常の賃貸借取引に係る方法に準じ", "通常の賃貸借取引"),
            ("ファイナンス・リース取引", "ファイナンス・リース"),
            ("オペレーティング・リース取引", "オペレーティング・リース"),
        )
        terms: list[str] = []
        for raw, label in mappings:
            if raw in note and label not in terms:
                terms.append(label)
        return terms

    def _build_exact_evidence_message(self, context: AgentContext) -> dict[str, str] | None:
        if not context.exact_evidence_chunk_ids:
            return None

        lines = [
            "【exact clause 要点】指定条項の原文根拠があります。条項回答では、原文の言い換えとして短い会計用語も落とさず含めてください。"
        ]
        added = 0
        for chunk_id in self._ordered_evidence_note_ids(context):
            if chunk_id not in context.exact_evidence_chunk_ids:
                continue
            note = context.evidence_notes.get(chunk_id, "")
            key_terms = self._exact_clause_key_terms(note)
            if not key_terms:
                continue
            lines.append(f"- {chunk_id}: 要点語 {', '.join(key_terms)}")
            added += 1
            if added >= 2:
                break

        if added == 0:
            return None
        return {"role": "user", "content": "\n".join(lines)}

    def _build_final_answer_messages(self, messages: list[dict], context: AgentContext) -> list[dict]:
        """Force the LLM to produce a final answer without tool calls."""
        force_prompt = (
            "これ以上ツールを呼び出さないでください。"
            "これまでに収集した情報に基づいて、最終的な回答を提供してください。"
            "情報が不十分な場合は、その旨を明記した上で、得られた情報の範囲で回答してください。"
            "推測は避け、文書に基づいた回答のみを行ってください。"
            "見出し以外の本文、箇条書き、まとめ文の末尾には必ず参照したチャンクIDを付け、付けられない文は出力しないでください。"
            "既に抽出済みの重要箇所メモがあれば、それを優先して detail を保って回答してください。"
            "複合質問では、想定論点ごとに少なくとも 1 行は残し、根拠が不足する論点は推測せず不足を明示してください。"
        )
        messages_copy = list(messages)
        coverage_message = self._build_coverage_plan_message(context)
        if coverage_message is not None:
            messages_copy.append(coverage_message)
        note_message = self._build_evidence_note_message(context)
        if note_message is not None:
            messages_copy.append(note_message)
        exact_evidence_message = self._build_exact_evidence_message(context)
        if exact_evidence_message is not None:
            messages_copy.append(exact_evidence_message)
        verification_message = self._build_verification_result_message(context)
        if verification_message is not None:
            messages_copy.append(verification_message)
        if self._is_detail_seeking_context(context):
            messages_copy.append(
                {
                    "role": "user",
                    "content": (
                        "【詳細回答ルール】詳しく/違い/どのような場合といった質問です。"
                        "各論点では、原則だけで終わらせず、確認できた範囲で条件・例外・判断基準・比較観点のうち"
                        "該当するものを少なくとも 1 つ含めてください。"
                    ),
                }
            )
        if self._is_verification_context(context):
            messages_copy.append(
                {
                    "role": "user",
                    "content": (
                        "【判断検証の最終回答ルール】最終回答は必ず次の 4 セクションで構成してください。"
                        "`## 主張要約` → `## 照合結果` → `## 追加考慮事項` → `## 参照`。"
                        "各主張について、確認できた根拠に基づき `○適切` / `△要注意` / `×不適切` のいずれかを付け、"
                        "判定理由の本文行末には必ず引用を付けてください。"
                        "根拠が弱い場合は `△要注意` とし、参照条文とズレる場合だけ `×不適切` を使ってください。"
                    ),
                }
            )
        if context.exact_doc_terms or context.exact_section_terms:
            messages_copy.append(
                {
                    "role": "user",
                    "content": (
                        "【exact query ルール】質問で指定された基準番号・条項番号に対応する根拠がある場合だけ、その指定箇所の内容を直接回答してください。"
                        "指定箇所の根拠が確認できない場合は、別基準や現行基準の一般論で補わず、"
                        "『今回確認できた根拠では不十分』と明示してください。"
                    ),
                }
            )
        messages_copy.append({"role": "user", "content": force_prompt})
        return messages_copy

    def _force_final_answer(self, messages: list[dict], context: AgentContext) -> tuple[str, float]:
        messages_copy = self._build_final_answer_messages(messages, context)
        response = self.llm.chat(messages=messages_copy, tools=None, temperature=0.0)
        return response["message"].get("content", ""), response.get("cost", 0.0)

    async def _aforce_final_answer(self, messages: list[dict], context: AgentContext) -> tuple[str, float]:
        """Async force final answer."""
        messages_copy = self._build_final_answer_messages(messages, context)
        response = await self.llm.achat(messages=messages_copy, tools=None, temperature=0.0)
        return response["message"].get("content", ""), response.get("cost", 0.0)

    # Chunk ID patterns shared across ref-handling methods
    _CHUNK_SUFFIX = r':(?:p|c)?\d+(?:-\d+)*'
    _FILENAME = r'[\w\-]+\.(?:pdf|xml)'

    def _citation_group_key(self, chunk_id: str) -> str:
        chunk = self.chunk_map.get(chunk_id)
        if not chunk:
            return chunk_id
        return chunk.get("parent_id", chunk_id)

    def _preferred_citation_chunk(self, current_id: str, candidate_id: str) -> str:
        current = self.chunk_map.get(current_id)
        candidate = self.chunk_map.get(candidate_id)
        if not current or not candidate:
            return candidate_id if candidate else current_id

        current_parent = current.get("parent_id", current_id)
        candidate_parent = candidate.get("parent_id", candidate_id)
        current_is_parent = current_parent == current_id
        candidate_is_parent = candidate_parent == candidate_id

        # Prefer the more specific child chunk when the same parent is cited twice.
        if current_is_parent and not candidate_is_parent:
            return candidate_id
        return current_id

    def _number_chunk_refs(self, text: str) -> tuple[str, list[str]]:
        """Replace inline chunk ID refs with [N] citation markers.

        Scans the answer for bracketed chunk ID references and converts them to
        sequential `[1]`, `[2]` … markers.  Returns the modified text and an
        ordered list of displayed chunk IDs (index 0 → citation [1], etc.).
        """
        # Single ID in brackets: （filename.pdf:p5）→ [N]
        single = re.compile(
            rf'[（\(\[【]\s*({self._FILENAME}{self._CHUNK_SUFFIX})\s*[）\)\]】]'
        )
        # Multiple IDs in one bracket: （id1、id2）→ [N][M]
        multi = re.compile(
            rf'[（\(\[【]\s*({self._FILENAME}{self._CHUNK_SUFFIX}(?:\s*[、，;]\s*{self._FILENAME}{self._CHUNK_SUFFIX})+)\s*[）\)\]】]'
        )
        参照_form = re.compile(
            rf'参照:\s*({self._FILENAME}{self._CHUNK_SUFFIX})\s*;?'
        )

        ordered_keys: list[str] = []
        num_map: dict[str, int] = {}
        display_chunk_by_key: dict[str, str] = {}

        def _num(cid: str) -> str:
            key = self._citation_group_key(cid)
            if key not in num_map:
                ordered_keys.append(key)
                num_map[key] = len(ordered_keys)
                display_chunk_by_key[key] = cid
            else:
                display_chunk_by_key[key] = self._preferred_citation_chunk(
                    display_chunk_by_key[key],
                    cid,
                )
            return f'[{num_map[key]}]'

        def replace_multi(m: re.Match) -> str:
            ids = re.findall(rf'{self._FILENAME}{self._CHUNK_SUFFIX}', m.group(1))
            return ''.join(_num(cid) for cid in ids)

        def replace_single(m: re.Match) -> str:
            return _num(m.group(1))

        text = multi.sub(replace_multi, text)
        text = single.sub(replace_single, text)
        text = 参照_form.sub(replace_single, text)
        text = re.sub(r'(\[\d+\])(?:\s*\1)+', r'\1', text)
        ordered = [display_chunk_by_key[key] for key in ordered_keys]
        return text, ordered

    @classmethod
    def _strip_chunk_refs(cls, text: str) -> str:
        """Remove any remaining un-numbered chunk ID references from the answer."""
        _s = cls._CHUNK_SUFFIX
        _f = cls._FILENAME

        # Any leftover bracketed forms
        text = re.sub(rf'[（\(\[【]\s*{_f}{_s}\s*[）\)\]】]', '', text)
        # Bare Chunk keyword refs
        text = re.sub(r'[【\[\(]\s*Chunk\s*[\w\-:\.]+\s*[】\]\)]', '', text)
        text = re.sub(r'\bChunk\s*\d+\b', '', text)
        # Bare filename:page refs
        text = re.sub(rf'\b{_f}{_s}\b', '', text)
        # Orphaned 参照:
        text = re.sub(r'参照:\s*;?', '', text)
        # Orphaned bare page-number refs like (-40), （-39）, (p40), （-40-41） left by model
        # Handle ASCII and full-width parens, various dash/minus chars, ASCII and full-width digits
        # All dash/minus variants including U+2011 NON-BREAKING HYPHEN used by some models
        _dash = r'[-－−‐\u2011–—\u2212]?'
        _any_dash = r'[-－−‐\u2011–—\u2212]'
        _digs = r'[0-9０-９]+'
        # Keep numbered citation markers like [1], but strip bare page refs in parentheses.
        text = re.sub(
            rf'[（(]\s*{_dash}\s*p?{_digs}(?:\s*{_any_dash}\s*p?{_digs})*\s*[）)]',
            '', text
        )
        # Square-bracket cleanup only targets explicit page-like forms such as [p5] or [-21].
        text = re.sub(
            rf'\[\s*(?:{_any_dash}\s*p?{_digs}(?:\s*{_any_dash}\s*p?{_digs})*|p{_digs}(?:\s*{_any_dash}\s*p?{_digs})*)\s*\]',
            '', text
        )
        # Strip trailing Japanese/ASCII comma before closing paren: （第25項、） → （第25項）
        text = re.sub(r'([（(][^）)\n]{2,})[、，,]\s*([）)])', r'\1\2', text)
        # Clean up residual empty or punctuation-only parentheses like （、）（;）（ ）
        text = re.sub(r'[（(][\s、;,・]*[）)]', '', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([。、，,.])', r'\1', text)
        return text

    @classmethod
    def _strip_inline_citation_labels(cls, text: str) -> str:
        """Remove citation-only parentheticals and page labels from answer prose."""
        dash = r'[-－−‐\u2011–—\u2212]'
        standard = (
            r'(?:企業会計基準適用指針|企業会計基準|実務対応報告|移管指針|監査基準|会計基準)'
            r'\s*第\s*\d+(?:[\u2010\-]\d+)?\s*号'
            r'|IFRS\s*第?\s*\d+(?:[\u2010\-]\d+)?\s*号?'
        )
        article = r'第\s*\d+\s*(?:項|条|号)'
        bc = rf'BC\s*\d+(?:\s*{dash}\s*\d+)?'
        page = rf'(?:[Pp]{{1,2}}\.?\s*\d+(?:\s*{dash}\s*\d+)?|(?:ページ|頁)\s*\d+)'
        citation_content = rf'(?:{standard}|{article}|{bc}|{page}|[、，,・／/\s]+)+'

        text = re.sub(rf'[（(]\s*(?:{citation_content})\s*[）)]', '', text)
        text = re.sub(
            rf'\s*[,、，]?\s*(?:[Pp]{{1,2}}\.?\s*\d+(?:\s*{dash}\s*\d+)?|#page=\d+)(?=\s*\[\d+\])',
            '',
            text,
        )
        text = re.sub(r'\s+(\[\d+\])', r'\1', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([。、，,.])', r'\1', text)
        return text

    @staticmethod
    def _drop_uncited_lines(text: str) -> str:
        """Keep only headings that lead to cited content and cited body lines."""
        kept_lines: list[str] = []
        pending_headings: list[str] = []
        heading_only_bullet = re.compile(
            r"^-\s+\*\*(?P<label>[^*]+)\*\*(?P<suffix>（[^）]+）)?$"
        )

        def flush_pending_headings() -> None:
            nonlocal pending_headings
            if not pending_headings:
                return
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            kept_lines.extend(pending_headings)
            pending_headings = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                if kept_lines and kept_lines[-1] != "":
                    kept_lines.append("")
                continue

            heading_bullet_match = heading_only_bullet.match(stripped)
            if heading_bullet_match:
                label = heading_bullet_match.group("label").strip()
                suffix = (heading_bullet_match.group("suffix") or "").strip()
                pending_headings.append(f"## {label}{suffix}".rstrip())
                continue

            if re.match(r"^#{1,6}\s+", stripped):
                pending_headings.append(stripped)
                continue

            if re.search(r"\[\d+\]", stripped):
                flush_pending_headings()
                kept_lines.append(stripped)

        while kept_lines and kept_lines[-1] == "":
            kept_lines.pop()
        return "\n".join(kept_lines)

    @staticmethod
    def _strip_reasoning_preamble(text: str) -> str:
        """Remove model internal reasoning/tool-call leakage before the actual answer.

        Some LLMs emit planning text like "We have many chunks. Need to read them."
        or raw JSON tool-call arguments as part of the final text content.
        Detect these patterns and strip everything up to the start of the real answer.
        """
        # Remove JSON blobs that look like tool call arguments
        text = re.sub(r'\{[^{}]*"chunk_ids"[^{}]*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"query"[^{}]*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"ids"[^{}]*\}', '', text, flags=re.DOTALL)
        # Remove common English reasoning phrases emitted by the LLM
        text = re.sub(r'We(?:\s+have|\s+need\s+to|\s+got)[^\n.]*\.', '', text)
        text = re.sub(r'Need\s+to\s+\w+[^\n.]*\.', '', text)
        text = re.sub(r'Let\s+me\s+\w+[^\n.]*\.', '', text)
        # If there's still a block of ASCII-only preamble before the Japanese answer,
        # find the first line with Japanese text or a ## heading and discard everything before it.
        lines = text.splitlines()
        japanese_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('##') or re.search(r'[\u3040-\u9fff\u30a0-\u30ff\u4e00-\u9fff]', stripped):
                japanese_start = i
                break
        if japanese_start is not None and japanese_start > 0:
            preamble = '\n'.join(lines[:japanese_start])
            # Only strip preamble if it looks like leaked reasoning (contains JSON chars or English words)
            if re.search(r'[{}\[\]]', preamble) or re.search(r'\b[A-Za-z]{4,}\b', preamble):
                text = '\n'.join(lines[japanese_start:])
        # Handle the case where ASCII planning text and Japanese answer are on the same line
        # e.g. "We have the content.## 減損の注記様式" or "We have the content.減損..."
        inline_match = re.match(r'^[A-Za-z\s.,!?:;\[\]{}"\']+?(##\s|\S*[\u3040-\u9fff\u30a0-\u30ff\u4e00-\u9fff])', text)
        if inline_match:
            text = text[inline_match.start(1):]
        return text

    @staticmethod
    def _note_units(note: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？])\s+|(?<=。)|(?<=！)|(?<=？)|\n+", note)
            if sentence.strip() and sentence.strip() != "[抜粋]"
        ]

    @classmethod
    def _extract_slot_snippet(
        cls,
        note: str,
        aliases: tuple[str, ...],
        *,
        prefer_detail: bool = False,
    ) -> str:
        sentences = cls._note_units(note)
        if not sentences:
            return note.strip().replace("[抜粋]", "")
        focus_idx = 0
        for idx, sentence in enumerate(sentences):
            if any(alias in sentence for alias in aliases):
                focus_idx = idx
                break
        if not prefer_detail:
            return sentences[focus_idx]

        selected_indexes: list[int] = [focus_idx]
        for idx in (focus_idx - 1, focus_idx + 1, focus_idx + 2):
            if idx < 0 or idx >= len(sentences) or idx in selected_indexes:
                continue
            sentence = sentences[idx]
            if any(marker in sentence for marker in cls._DETAIL_NOTE_MARKERS):
                selected_indexes.append(idx)
        if len(selected_indexes) == 1 and focus_idx + 1 < len(sentences):
            selected_indexes.append(focus_idx + 1)
        return "\n".join(sentences[idx] for idx in sorted(selected_indexes[:3]))

    def _answer_mentions_slot_with_citation(self, text: str, aliases: tuple[str, ...]) -> bool:
        chunk_ref_re = re.compile(rf"{self._FILENAME}{self._CHUNK_SUFFIX}|\[\d+\]")
        in_slot_section = False
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                in_slot_section = any(alias in stripped for alias in aliases)
                continue
            if any(alias in stripped for alias in aliases) and chunk_ref_re.search(stripped):
                return True
            if in_slot_section and chunk_ref_re.search(stripped):
                return True
        return False

    def _build_slot_repair_line(self, item: dict[str, Any], context: AgentContext) -> str | None:
        label = str(item["label"])
        aliases = tuple(str(alias) for alias in item["aliases"])
        supporting_ids = [str(chunk_id) for chunk_id in item["supporting_ids"]]
        if supporting_ids:
            chunk_id = supporting_ids[0]
            note = context.evidence_notes.get(chunk_id, "").strip()
            if not note:
                return None
            snippet = self._extract_slot_snippet(
                note,
                aliases,
                prefer_detail=self._is_detail_seeking_context(context),
            ).replace("[抜粋]", "").strip()
            snippet = snippet.rstrip("。！？")
            if not snippet:
                return None
            if not any(alias in snippet for alias in aliases):
                snippet = f"{label}については、{snippet}"
            return f"- {snippet}[{chunk_id}]。"

        fallback_ids = [str(chunk_id) for chunk_id in item.get("fallback_ids", []) if str(chunk_id)]
        if not fallback_ids:
            return None
        refs = "".join(f"[{chunk_id}]" for chunk_id in fallback_ids[:2])
        return (
            f"- {label}: 今回確認できた根拠では不十分であり、この論点を裏付ける十分な記載を確認できませんでした"
            f"{refs}。"
        )

    def _build_exact_repair_line(self, context: AgentContext) -> str | None:
        if not self._has_exact_shortfall(context):
            return None
        label = " ".join([*context.exact_doc_terms[:1], *context.exact_section_terms[:1]]).strip() or "指定条項"
        fallback_ids = list(context.exact_evidence_chunk_ids) or context.searched_chunk_ids[:2]
        refs = "".join(f"[{chunk_id}]" for chunk_id in fallback_ids[:2] if chunk_id)
        return (
            f"- {label}: 今回確認できた根拠では不十分であり、指定条項の原文を裏付ける十分な記載を確認できませんでした"
            f"{refs}。"
        )

    def _repair_answer_coverage(self, answer: str, context: AgentContext) -> str:
        items = self._coverage_plan_items(context)

        repair_lines: list[str] = []
        for item in items:
            aliases = tuple(str(alias) for alias in item["aliases"])
            if self._answer_mentions_slot_with_citation(answer, aliases):
                continue
            line = self._build_slot_repair_line(item, context)
            if line and line not in repair_lines:
                repair_lines.append(line)

        exact_line = self._build_exact_repair_line(context)
        if exact_line and "今回確認できた根拠では不十分" not in answer:
            repair_lines.append(exact_line)

        if not repair_lines:
            return answer

        cleaned = answer.rstrip()
        if cleaned:
            cleaned += "\n\n"
        cleaned += "## 論点カバレッジ\n" + "\n".join(repair_lines)
        return cleaned

    @staticmethod
    def _verification_judgment_present(answer: str) -> bool:
        return any(marker in answer for marker in ("○適切", "△要注意", "×不適切"))

    def _verification_refs(self, item: dict[str, Any], context: AgentContext) -> list[str]:
        refs: list[str] = []
        for bucket in ("exact_evidence_chunk_ids", "evidence_chunk_ids", "search_chunk_ids"):
            for raw in item.get(bucket, []) or []:
                chunk_id = str(raw).strip()
                if chunk_id and chunk_id not in refs:
                    refs.append(chunk_id)
        for chunk_id in context.searched_chunk_ids:
            normalized = str(chunk_id).strip()
            if normalized and normalized not in refs:
                refs.append(normalized)
            if len(refs) >= 2:
                break
        return refs[:2]

    def _verification_reason_line(self, item: dict[str, Any], context: AgentContext) -> str:
        refs = self._verification_refs(item, context)
        ref_text = "".join(f"[{chunk_id}]" for chunk_id in refs)
        cited_refs = [str(ref).strip() for ref in item.get("cited_references", []) if str(ref).strip()]
        cited_text = f"依拠条文: {', '.join(cited_refs)}。 " if cited_refs else ""
        note = ""
        if refs:
            note = context.evidence_notes.get(refs[0], "").strip()
        snippet = ""
        if note:
            snippet = self._note_units(note)[0].rstrip("。！？")
        judgment = str(item.get("judgment", "△要注意"))
        if judgment == "○適切":
            reason = "確認できた根拠と主張の方向性は整合しています"
        elif judgment == "×不適切":
            reason = "今回確認できた根拠では主張をそのまま採用するのは難しいです"
        else:
            reason = "関連する根拠は見つかったものの、適用関係の確認がまだ不十分です"
        if snippet:
            reason = f"{reason}。{snippet}"
        return f"{cited_text}{reason}{ref_text}".strip()

    def _build_verification_structured_answer(self, context: AgentContext) -> str:
        if not context.verification_results:
            return ""

        summary_lines = ["## 主張要約"]
        result_lines = ["## 照合結果"]
        consideration_lines = ["## 追加考慮事項"]
        reference_lines = ["## 参照"]

        for item in context.verification_results:
            refs = self._verification_refs(item, context)
            ref_text = "".join(f"[{chunk_id}]" for chunk_id in refs)
            cited_refs = [str(ref).strip() for ref in item.get("cited_references", []) if str(ref).strip()]
            cited_text = f"（依拠: {', '.join(cited_refs)}）" if cited_refs else ""
            summary_lines.append(f"- 主張{item['index']}: {item['claim']}{cited_text}{ref_text}")
            result_lines.append(
                f"- 主張{item['index']}: {item.get('judgment', '△要注意')}。{self._verification_reason_line(item, context)}"
            )

        if any(str(item.get("judgment", "")) != "○適切" for item in context.verification_results):
            for item in context.verification_results:
                if str(item.get("judgment", "")) == "○適切":
                    continue
                refs = self._verification_refs(item, context)
                ref_text = "".join(f"[{chunk_id}]" for chunk_id in refs)
                consideration_lines.append(
                    f"- 主張{item['index']}: 原文の条項対応と適用場面を追加確認してください{ref_text}"
                )
        else:
            refs = self._verification_refs(context.verification_results[0], context)
            ref_text = "".join(f"[{chunk_id}]" for chunk_id in refs)
            consideration_lines.append(f"- 今回確認した範囲では、主張同士の大きな矛盾は見当たりません{ref_text}")

        seen_refs: set[str] = set()
        for item in context.verification_results:
            refs = self._verification_refs(item, context)
            cited_refs = [str(ref).strip() for ref in item.get("cited_references", []) if str(ref).strip()]
            label = " / ".join(cited_refs) if cited_refs else f"主張{item['index']}の確認根拠"
            ref_text = "".join(f"[{chunk_id}]" for chunk_id in refs)
            key = f"{label}|{ref_text}"
            if key in seen_refs:
                continue
            seen_refs.add(key)
            reference_lines.append(f"- {label}{ref_text}")

        return "\n".join(summary_lines + [""] + result_lines + [""] + consideration_lines + [""] + reference_lines)

    def _sanitize_answer(self, text: str, *, number_refs: bool = False) -> tuple[str, list[str]] | str:
        """Normalize answer formatting before returning it to clients.

        When ``number_refs=True`` returns ``(text, ordered_chunk_ids)`` so
        callers can build a numbered reference list matched to inline [N] markers.
        Otherwise returns just the text string (backward-compatible default).
        """
        text = self._strip_reasoning_preamble(text)
        cited_ids: list[str] = []
        if number_refs:
            # Drop model-authored numeric citations so only chunk-backed markers remain.
            text = re.sub(r"\[(\d+)\]", "", text)
            text, cited_ids = self._number_chunk_refs(text)
        text = self._strip_chunk_refs(text)
        text = self._strip_inline_citation_labels(text)
        if number_refs:
            text = self._drop_uncited_lines(text)

        heading_only_bullet = re.compile(
            r"^-\s+\*\*(?P<label>[^*]+)\*\*(?P<suffix>（[^）]+）)?$"
        )
        normalized_lines = []
        previous_blank = False
        for raw_line in text.splitlines():
            line = re.sub(r'^[ \t]+(?=-\s)', '', raw_line)
            line = re.sub(r'[ \t]+$', '', line)
            match = heading_only_bullet.match(line)
            if match:
                label = match.group("label").strip()
                suffix = (match.group("suffix") or "").strip()
                line = f"## {label}{suffix}".rstrip()
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            normalized_lines.append("" if is_blank else line)
            previous_blank = is_blank

        result = "\n".join(normalized_lines).strip()
        if number_refs:
            return result, cited_ids
        return result

    def _finalize_answer(self, answer: str, context: AgentContext) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
        if self._is_verification_context(context):
            if not self._has_verification_answer_sections(answer) or not self._verification_judgment_present(answer):
                answer = self._build_verification_structured_answer(context) or answer
        else:
            answer = self._repair_answer_coverage(answer, context)
        sanitized_answer, cited_ids = self._sanitize_answer(answer, number_refs=True)
        references, source_url_map = self._get_referenced_chunks(context, cited_ids)
        return sanitized_answer, references, source_url_map

    def _get_referenced_chunks(
        self,
        context: AgentContext,
        cited_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Get chunks for references panel in citation order.

        Priority:
        1. Inline-cited chunk IDs (from [N] markers in answer) — in citation order
        2. Explicitly read via read_chunk tool
        3. Fallback: top search results
        """
        seen: set[str] = set()
        ids_to_show: list[str] = []

        # 1. Inline citations from answer body (determines [N] numbering)
        for cid in (cited_ids or []):
            if cid not in seen:
                ids_to_show.append(cid)
                seen.add(cid)

        # 2. read_chunk calls only when the answer has no inline citations
        if not ids_to_show:
            for cid in context.read_chunk_ids:
                if cid not in seen:
                    ids_to_show.append(cid)
                    seen.add(cid)

        # 3. Search fallback when nothing was explicitly cited or read
        if not ids_to_show:
            for cid in context.searched_chunk_ids[:10]:
                if cid not in seen:
                    ids_to_show.append(cid)
                    seen.add(cid)

        refs = []
        seen_parents: set[str] = set()
        source_url_map: dict[str, str] = {}
        for chunk_id in ids_to_show:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            parent_id = chunk.get("parent_id", chunk_id)
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            ref = {
                "id": chunk_id,
                "source": chunk.get("source", ""),
                "text": chunk["text"],
            }
            ref.update(self._reference_card_fields(chunk))
            filename = chunk.get("file", "")
            if filename and filename in self.pdf_sources:
                base_url = self.pdf_sources[filename]
                pdf_page = chunk.get("pdf_page")
                ref["url"] = f"{base_url}#page={pdf_page}" if pdf_page and pdf_page > 1 else base_url
                source = chunk.get("source", "")
                source_label = source.split(">")[0].strip()
                if source_label and source_label not in source_url_map:
                    source_url_map[source_label] = ref["url"]
                # Also keep a generic "第N号" key for short citations.
                for m in re.finditer(r"第\s*(\d+(?:[\u2010\-]\d+)?)\s*号", source):
                    key = f"第{m.group(1).replace(' ', '')}号"
                    if key not in source_url_map:
                        source_url_map[key] = ref["url"]
            ref["display_number"] = len(refs) + 1
            refs.append(ref)
        return refs, source_url_map

    @staticmethod
    def _count_cited_references(answer: str) -> int:
        """Count unique visible citations in the answer.

        Prefer numbered markers like [1], [2] that the app renders inline.
        Fall back to explicit standard/article patterns for answers without
        numbered chunk refs.
        """
        numbered = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        if numbered:
            return len(numbered)

        patterns = [
            r"企業会計基準第\s*\d+\s*号",
            r"企業会計基準適用指針第\s*\d+\s*号",
            r"実務対応報告第\s*\d+\s*号",
            r"IFRS\s*第?\s*\d+\s*号",
            r"第\s*\d+\s*項",
            r"第\s*\d+\s*条",
            r"BC\s*\d+(?:\s*[-‑–]\s*\d+)?",
        ]
        citations: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, answer):
                normalized = re.sub(r"\s+", "", match)
                citations.add(normalized)
        return len(citations)

    @staticmethod
    def _compact_label(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip(" >\u3000")

    @classmethod
    def _clean_reference_label(cls, text: str) -> str:
        cleaned = cls._compact_label(text)
        if not cleaned:
            return ""
        if re.fullmatch(r"[-\u2010-\u2015\s\d.]+", cleaned):
            return ""
        return cleaned

    @classmethod
    def _short_reference_label(cls, text: str, limit: int = 44) -> str:
        cleaned = cls._clean_reference_label(text)
        if not cleaned:
            return ""
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"

    @classmethod
    def _infer_reference_doc_type(cls, chunk: dict[str, Any]) -> str:
        explicit = cls._compact_label(chunk.get("doc_type", ""))
        if explicit:
            return explicit
        haystack = cls._compact_label(chunk.get("doc_title", "")) or cls._compact_label(chunk.get("source", ""))
        for token in (
            "企業会計基準適用指針",
            "企業会計基準",
            "実務対応報告",
            "会計制度委員会報告",
            "企業会計原則",
            "原価計算基準",
            "注解",
            "法令",
            "IFRS関連情報",
            "中小企業会計",
        ):
            if token and token in haystack:
                return token
        return "参考資料"

    @classmethod
    def _reference_card_fields(cls, chunk: dict[str, Any]) -> dict[str, Any]:
        source = cls._compact_label(chunk.get("source", ""))
        doc_title = cls._compact_label(chunk.get("doc_title", ""))
        if not doc_title and source:
            doc_title = cls._compact_label(source.split(">")[0])
        section_title = cls._clean_reference_label(chunk.get("section_title", ""))
        if not section_title and ">" in source:
            section_title = cls._clean_reference_label(source.split(">", 1)[1])
        page_label = ""
        try:
            pdf_page = int(chunk.get("pdf_page") or 0)
        except (TypeError, ValueError):
            pdf_page = 0
        if pdf_page > 0:
            page_label = f"p.{pdf_page}"
        section_label = cls._short_reference_label(section_title)
        if not section_label and page_label:
            section_label = page_label
        return {
            "doc_type": cls._infer_reference_doc_type(chunk),
            "doc_title": doc_title or source,
            "section_title": section_title,
            "section_label": section_label,
            "page_label": page_label,
            "standard_no": cls._compact_label(chunk.get("standard_no", "")) or None,
        }

    @staticmethod
    def _extract_insufficient_points(answer: str) -> list[str]:
        points: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"^\s*[-・•*]\s*([^:：\n]+?)\s*[:：]\s*今回確認できた根拠では不十分", answer, flags=re.MULTILINE):
            label = re.sub(r"\s+", " ", match.group(1)).strip()
            if label and label not in seen:
                seen.add(label)
                points.append(label)
        return points

    def _build_uncertainty_summary(self, answer: str, context: AgentContext) -> dict[str, Any]:
        coverage = context.get_evidence_coverage()
        insufficient_points: list[str] = []
        seen: set[str] = set()
        for label in list(coverage.get("uncovered_slots", [])) + self._extract_insufficient_points(answer):
            normalized = re.sub(r"\s+", " ", str(label or "")).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                insufficient_points.append(normalized)

        present = bool(insufficient_points) or "今回確認できた根拠では不十分" in answer
        note = ""
        if present and insufficient_points:
            note = "未確定の論点があります。参照カードは確認できた範囲の原典です。"
        elif present:
            note = "一部の記述は、今回確認できた根拠だけでは十分に裏付けられていません。"

        return {
            "present": present,
            "insufficient_points": insufficient_points,
            "covered_points": list(coverage.get("covered_slots", [])),
            "coverage_ratio": float(coverage.get("coverage_ratio", 0.0)),
            "note": note,
        }

    def _build_answer_summary(self, answer: str, context: AgentContext) -> dict[str, Any]:
        summary = context.get_summary()
        summary["read_chunk_count"] = len(context.read_chunk_ids)
        summary["chunks_read_count"] = self._count_cited_references(answer)
        return summary

    @staticmethod
    def _iter_cited_answer_lines(answer: str) -> list[str]:
        lines: list[str] = []
        for raw_line in answer.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.search(r"\[\d+\]", stripped):
                lines.append(stripped)
        return lines

    @staticmethod
    def _strip_numeric_citations(text: str) -> str:
        return re.sub(r"\[\d+\]", "", text)

    def _natural_answer_retry_reasons(self, result: dict[str, Any], context: AgentContext) -> list[str]:
        answer = str(result.get("answer", ""))
        if self._is_verification_context(context):
            reasons: list[str] = []
            if not self._has_verification_answer_sections(answer):
                reasons.append("verification_format")
            if int(result.get("cited_reference_count", 0)) < max(1, min(2, len(context.verification_results) or 1)):
                reasons.append("verification_citations")
            return reasons
        if QueryExpander.is_exact_query(context.question):
            if self._has_exact_clause_evidence(context):
                return []
            return ["missing_exact_evidence"]
        if context.question_complexity == "simple":
            return []
        if len(context.read_chunk_ids) < 2 or len(context.evidence_notes) < 2:
            return []

        reasons: list[str] = []
        cited_reference_count = int(result.get("cited_reference_count", 0))
        if cited_reference_count < 2:
            reasons.append("low_citations")

        cited_lines = self._iter_cited_answer_lines(answer)
        coverage = context.get_evidence_coverage()
        required_line_count = 2
        if self._is_detail_seeking_context(context):
            required_line_count = max(2, len(coverage.get("covered_slots", [])) or len(coverage.get("slots", [])) or 2)
        if len(cited_lines) < required_line_count:
            reasons.append("too_few_cited_lines")

        if self._is_detail_seeking_context(context):
            substantive_chars = sum(len(self._strip_numeric_citations(line).strip()) for line in cited_lines)
            minimum_chars = 60 if context.question_complexity == "complex" else 40
            if substantive_chars < minimum_chars:
                reasons.append("too_little_detail")

            covered_slots = list(coverage.get("covered_slots", []))
            covered_mentions = sum(
                1
                for slot in covered_slots
                if self._answer_mentions_slot_with_citation(
                    answer,
                    context.evidence_slot_terms.get(slot, (slot,)),
                )
            )
            if covered_slots and covered_mentions < len(covered_slots):
                reasons.append("covered_slots_missing")

        return reasons

    def _should_retry_natural_answer(self, result: dict[str, Any], context: AgentContext) -> bool:
        return bool(self._natural_answer_retry_reasons(result, context))

    def _build_completed_result(
        self,
        answer: str,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> dict[str, Any]:
        sanitized_answer, references, source_url_map = self._finalize_answer(answer, context)
        summary = self._build_answer_summary(sanitized_answer, context)
        uncertainty = self._build_uncertainty_summary(sanitized_answer, context)
        observability = self._build_observability_summary(
            context=context,
            loops=loops,
            stop_reason=stop_reason,
            answer=sanitized_answer,
            references=references,
        )
        return {
            "answer": sanitized_answer,
            "loops": loops,
            "stop_reason": stop_reason,
            "total_cost": total_cost,
            **summary,
            "cited_reference_count": self._count_cited_references(sanitized_answer),
            "read_chunk_count": len(context.read_chunk_ids),
            "trajectory": context.trajectory,
            "references": references,
            "source_url_map": source_url_map,
            "uncertainty": uncertainty,
            "request_id": observability.get("request_id"),
            "query_class": observability["query_class"],
            "observability": observability,
        }

    def _build_result(
        self,
        answer: str,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> dict[str, Any]:
        return self._build_completed_result(answer, context, loops, stop_reason, total_cost)
