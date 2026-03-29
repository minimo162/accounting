"""Agent execution context for tracking retrieval state."""

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class RetrievalLog:
    """Structured log entry for a single retrieval operation."""
    tool_name: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Tracks per-query state: which chunks have been read, token counts, retrieval logs."""

    _GENERIC_SLOT_TERMS = {
        "教えて",
        "教えてください",
        "どのように",
        "ですか",
        "ますか",
        "ついて",
        "会計",
        "会計基準",
        "基準",
        "処理",
        "説明",
        "詳しく",
        "主な",
        "扱い",
        "影響",
        "改正点",
        "改正",
        "変更",
        "要件",
        "違い",
        "比較",
        "場合",
        "条件",
        "例外",
        "観点",
    }
    _SLOT_ALIASES: dict[str, tuple[str, ...]] = {
        "借手": ("借手",),
        "貸手": ("貸手",),
        "経過措置": ("経過措置", "適用初年度", "初度適用", "適用時期"),
        "関連基準": (
            "関連基準",
            "関連する会計基準",
            "他の会計基準",
            "固定資産の減損",
            "減損会計基準",
            "資産除去債務",
            "財務諸表等規則",
        ),
        "使用権資産": ("使用権資産",),
        "リース負債": ("リース負債",),
        "短期リース": ("短期リース",),
        "少額リース": ("少額リース", "少額資産"),
        "本人": ("本人",),
        "代理人": ("代理人",),
        "総額": ("総額",),
        "純額": ("純額",),
        "支配": ("支配",),
        "履行義務": ("履行義務",),
        "保守サービス": ("保守サービス", "保守契約"),
        "値引き": ("値引き", "割引"),
        "変動対価": ("変動対価",),
        "制約": ("制約", "重要な戻入れ"),
        "契約変更": ("契約変更",),
        "取引価格": ("取引価格",),
        "一定の期間": ("一定の期間", "一定期間"),
        "一時点": ("一時点",),
        "進捗": ("進捗", "進捗度"),
        "ヘッジ": ("ヘッジ", "繰延ヘッジ", "ヘッジ会計"),
        "有効性": ("有効性",),
        "文書化": ("文書化", "正式な文書", "リスク管理方針"),
        "事前": ("事前", "事前テスト", "開始時"),
        "事後": ("事後", "事後テスト"),
        "リスク": ("リスク", "ヘッジ対象", "ヘッジ手段"),
        "繰延税金資産": ("繰延税金資産",),
        "繰延税金負債": ("繰延税金負債",),
        "税率": ("税率", "法定実効税率"),
        "税率変更": ("税率変更",),
        "回収可能性": ("回収可能性",),
        "課税所得": ("課税所得",),
        "スケジューリング": ("スケジューリング",),
        "減損": ("減損",),
        "兆候": ("兆候",),
        "回収可能価額": ("回収可能価額",),
        "使用価値": ("使用価値",),
        "正味売却価額": ("正味売却価額",),
        "研究開発費": ("研究開発費",),
        "ソフトウェア": ("ソフトウェア",),
        "発生時": ("発生時",),
        "費用": ("費用",),
        "資産": ("資産", "資産計上"),
        "数理計算上の差異": ("数理計算上の差異",),
        "退職給付": ("退職給付",),
        "未認識": ("未認識",),
        "平均残存勤務期間": ("平均残存勤務期間",),
    }
    _SLOT_SUFFIX_RE = re.compile(
        r"(の会計処理|の処理|の扱い|への影響|について|とは|は|を|に分けて.*|まで|も|など)$"
    )
    _EXACT_REFERENCE_RE = re.compile(
        r"(?:企業会計基準|企業会計基準適用指針|適用指針|実務対応報告|会計基準)第?\s*\d+\s*号|第\s*\d+\s*(?:項|条|号)|BC\s*\d+"
    )

    def __init__(self):
        self.question: str = ""
        self.question_complexity: str = "moderate"
        self.query_profile: dict[str, Any] = {}
        self.current_search_query: str = ""
        self.read_chunk_ids: set[str] = set()
        self.searched_chunk_ids: list[str] = []  # ordered; preserves first-seen rank
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: list[RetrievalLog] = []
        self.search_history: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.tool_cache: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
        self.evidence_notes: dict[str, str] = {}
        self.evidence_note_sources: dict[str, str] = {}
        self.evidence_slots: list[str] = []
        self.evidence_slot_terms: dict[str, tuple[str, ...]] = {}
        self.evidence_slot_hits: dict[str, set[str]] = {}
        self.wrap_up_nudged: bool = False
        self.coverage_gap_nudge_signature: str = ""
        self.final_coverage_review_done: bool = False

    @classmethod
    def _normalize_slot_text(cls, text: str) -> str:
        return re.sub(r"[\s　]+", "", text or "")

    @classmethod
    def _clean_slot_phrase(cls, text: str) -> str:
        phrase = text.strip(" 、。・/／()（）「」『』")
        phrase = re.sub(r"(を|は|が|の)?教えてください$", "", phrase)
        phrase = re.sub(r"(を|は|が)?詳しく$", "", phrase)
        phrase = cls._SLOT_SUFFIX_RE.sub("", phrase).strip()
        return phrase

    @classmethod
    def _register_slot(
        cls,
        slots: list[str],
        slot_terms: dict[str, tuple[str, ...]],
        label: str,
        terms: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        normalized_label = cls._clean_slot_phrase(label)
        if (
            not normalized_label
            or cls._EXACT_REFERENCE_RE.search(normalized_label)
            or normalized_label in cls._GENERIC_SLOT_TERMS
        ):
            return
        if normalized_label in slot_terms:
            return
        ordered_terms: list[str] = []
        for term in (terms or [normalized_label]):
            normalized_term = cls._clean_slot_phrase(term)
            if not normalized_term:
                continue
            if normalized_term not in ordered_terms:
                ordered_terms.append(normalized_term)
        if normalized_label not in ordered_terms:
            ordered_terms.insert(0, normalized_label)
        slots.append(normalized_label)
        slot_terms[normalized_label] = tuple(ordered_terms)

    @classmethod
    def _infer_evidence_slots(
        cls,
        question: str,
        query_profile: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, tuple[str, ...]]]:
        slots: list[str] = []
        slot_terms: dict[str, tuple[str, ...]] = {}
        normalized_question = cls._normalize_slot_text(question)

        for label, aliases in cls._SLOT_ALIASES.items():
            if any(cls._normalize_slot_text(alias) in normalized_question for alias in aliases):
                cls._register_slot(slots, slot_terms, label, list(aliases))

        for raw_segment in re.split(r"[・/／]", question):
            cleaned = cls._clean_slot_phrase(raw_segment)
            if not cleaned or cleaned in cls._GENERIC_SLOT_TERMS:
                continue
            for label, aliases in cls._SLOT_ALIASES.items():
                if any(alias in cleaned for alias in aliases):
                    cls._register_slot(slots, slot_terms, label, list(aliases))
                    break

        if not slots:
            keywords = list((query_profile or {}).get("keywords", []))
            complexity = str((query_profile or {}).get("complexity", "moderate"))
            keyword_budget = 2 if complexity == "simple" else 4 if complexity == "complex" else 3
            for keyword in keywords:
                cleaned = cls._clean_slot_phrase(str(keyword))
                if (
                    not cleaned
                    or cleaned in cls._GENERIC_SLOT_TERMS
                    or cls._EXACT_REFERENCE_RE.search(cleaned)
                ):
                    continue
                cls._register_slot(slots, slot_terms, cleaned, [cleaned])
                if len(slots) >= keyword_budget:
                    break

        return slots, slot_terms

    def set_question(self, question: str, query_profile: dict[str, Any] | None = None):
        self.question = question
        self.query_profile = query_profile or {}
        self.question_complexity = str(self.query_profile.get("complexity", "moderate"))
        self.current_search_query = question
        self.evidence_slots, self.evidence_slot_terms = self._infer_evidence_slots(question, self.query_profile)
        self.evidence_slot_hits = {slot: set() for slot in self.evidence_slots}
        self.coverage_gap_nudge_signature = ""
        self.final_coverage_review_done = False

    def set_current_search_query(self, query: str):
        self.current_search_query = query

    def add_search_entry(self, entry: dict[str, Any]):
        self.search_history.append(entry)

    def set_evidence_note(self, chunk_id: str, note: str, source: str = ""):
        self.evidence_notes[chunk_id] = note
        self.evidence_note_sources[chunk_id] = source
        for hits in self.evidence_slot_hits.values():
            hits.discard(chunk_id)

        haystack = self._normalize_slot_text(f"{source}\n{note}")
        for slot, terms in self.evidence_slot_terms.items():
            if any(self._normalize_slot_text(term) in haystack for term in terms):
                self.evidence_slot_hits.setdefault(slot, set()).add(chunk_id)

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

    def get_cached_tool_result(self, tool_name: str, cache_key: str) -> tuple[str, dict[str, Any]] | None:
        return self.tool_cache.get((tool_name, cache_key))

    def set_cached_tool_result(self, tool_name: str, cache_key: str, result_text: str, tool_log: dict[str, Any]):
        self.tool_cache[(tool_name, cache_key)] = (result_text, tool_log)

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids),
            "retrieval_logs": [
                {"tool": log.tool_name, "tokens": log.tokens, **log.metadata}
                for log in self.retrieval_logs
            ],
            "question_complexity": self.question_complexity,
            "evidence_coverage": self.get_evidence_coverage(),
        }

    def get_evidence_coverage(self) -> dict[str, Any]:
        covered_slots = [slot for slot in self.evidence_slots if self.evidence_slot_hits.get(slot)]
        uncovered_slots = [slot for slot in self.evidence_slots if slot not in covered_slots]
        return {
            "slots": list(self.evidence_slots),
            "covered_slots": covered_slots,
            "uncovered_slots": uncovered_slots,
            "coverage_ratio": (len(covered_slots) / len(self.evidence_slots)) if self.evidence_slots else 0.0,
        }

    def reset(self):
        self.question = ""
        self.question_complexity = "moderate"
        self.query_profile = {}
        self.current_search_query = ""
        self.read_chunk_ids.clear()
        self.searched_chunk_ids.clear()
        self.total_retrieved_tokens = 0
        self.retrieval_logs.clear()
        self.search_history.clear()
        self.trajectory.clear()
        self.tool_cache.clear()
        self.evidence_notes.clear()
        self.evidence_note_sources.clear()
        self.evidence_slots.clear()
        self.evidence_slot_terms.clear()
        self.evidence_slot_hits.clear()
        self.wrap_up_nudged = False
        self.coverage_gap_nudge_signature = ""
        self.final_coverage_review_done = False
