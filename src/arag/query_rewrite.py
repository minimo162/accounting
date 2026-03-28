"""Query expansion and HyDE helpers."""

from dataclasses import dataclass, field
import json
import logging
import re

from .config import RetrievalConfig
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryProfile:
    query: str
    complexity: str
    search_mode: str
    keywords: list[str] = field(default_factory=list)
    canonical_focus_query: str | None = None
    corrective_query: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "complexity": self.complexity,
            "search_mode": self.search_mode,
            "keywords": list(self.keywords),
            "canonical_focus_query": self.canonical_focus_query,
            "corrective_query": self.corrective_query,
        }


class QueryExpander:
    _COMPLEX_TERMS = (
        "改正", "変更", "見直し", "背景", "目的", "経過措置", "適用時期",
        "比較", "違い", "それぞれ", "併せて", "また", "及び", "ならびに",
    )
    _SIMPLE_TERMS = ("とは", "何か", "意味", "定義", "概要", "趣旨")

    def __init__(self, config: RetrievalConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"(企業会計基準第\d+号|適用指針第\d+号|実務対応報告第\d+号|第\d+項|[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,})", query)))

    @classmethod
    def profile(cls, query: str) -> QueryProfile:
        keywords = cls._extract_keywords(query)
        exact = cls.is_exact_query(query)
        canonical_focus = cls._canonical_title_focus_variant(query)
        corrective_query = canonical_focus or cls._anchor_focus_variant(query)
        complexity = cls._infer_complexity(query, keywords, exact)

        if exact:
            search_mode = "keyword_first"
        elif canonical_focus and complexity != "complex":
            search_mode = "keyword_first"
        elif any(term in query for term in cls._SIMPLE_TERMS) and not canonical_focus:
            search_mode = "semantic_first"
        else:
            search_mode = "balanced"

        return QueryProfile(
            query=query,
            complexity=complexity,
            search_mode=search_mode,
            keywords=keywords[:6],
            canonical_focus_query=canonical_focus,
            corrective_query=corrective_query,
        )

    @staticmethod
    def is_exact_query(query: str) -> bool:
        return bool(
            re.search(r"(企業会計基準第\d+号|適用指針第\d+号|実務対応報告第\d+号|会計基準第\d+号|第\d+項|BC\d+)", query)
        )

    def expand(self, query: str) -> list[str]:
        variants = [query]
        if self.config.enable_query_expansion:
            if self.is_exact_query(query):
                return variants
            variants.extend(self._heuristic_variants(query))
            if self.llm is not None and self.config.enable_llm_query_expansion:
                variants.extend(self._llm_variants(query))
        deduped: list[str] = []
        for variant in variants:
            variant = variant.strip()
            if variant and variant not in deduped:
                deduped.append(variant)
        return deduped[: self.config.expansion_max_variants]

    def generate_hypothetical_document(self, query: str) -> str | None:
        if not self.config.enable_hyde or self.llm is None or self.is_exact_query(query):
            return None
        prompt = (
            "次の日本の会計基準に関する質問について、検索用の仮想回答断片を150〜250字で書いてください。"
            "断定調でよく、条項番号や関連論点を自然に含めてください。JSONや箇条書きは不要です。\n\n"
            f"質問: {query}"
        )
        response = self.llm.chat(messages=[{"role": "user", "content": prompt}], tools=None, temperature=0.0, max_tokens=300)
        text = response["message"].get("content", "").strip()
        return text or None

    def _heuristic_variants(self, query: str) -> list[str]:
        keywords = self._extract_keywords(query)
        variants: list[str] = []
        canonical_focus = self._canonical_title_focus_variant(query)
        if canonical_focus:
            variants.append(canonical_focus)
        if len(keywords) >= 2:
            variants.append(" ".join(keywords[:4]))
        if any("第" in kw and "号" in kw for kw in keywords):
            variants.append(f"{query} 条項 経過措置 開示")
        if "会計" not in query:
            variants.append(f"{query} 会計基準")
        return variants

    @classmethod
    def _infer_complexity(cls, query: str, keywords: list[str], exact: bool) -> str:
        complex_hits = sum(1 for term in cls._COMPLEX_TERMS if term in query)
        separator_hits = sum(query.count(token) for token in ("と", "、", "/", "・"))
        if exact and len(keywords) <= 4:
            return "simple"
        if complex_hits >= 2 or (complex_hits >= 1 and (separator_hits >= 1 or len(keywords) >= 5)):
            return "complex"
        if any(term in query for term in cls._SIMPLE_TERMS) and len(keywords) <= 4:
            return "simple"
        if len(query) >= 35 or len(keywords) >= 6:
            return "complex"
        return "moderate"

    @staticmethod
    def _canonicalize_standard_aliases(query: str) -> str:
        def repl(match: re.Match[str]) -> str:
            subject = match.group(1)
            if subject.endswith("に関する"):
                return match.group(0)
            return f"{subject}に関する会計基準"

        return re.sub(r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+)会計基準", repl, query)

    @classmethod
    def _canonical_title_focus_variant(cls, query: str) -> str | None:
        canonical_query = cls._canonicalize_standard_aliases(query)
        if canonical_query == query:
            return None

        title_match = re.search(r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に関する会計基準)", canonical_query)
        if not title_match:
            return canonical_query

        focus_parts = [title_match.group(1)]
        modifiers = (
            "改正", "変更", "見直し", "背景", "目的", "借手", "貸手",
            "使用権資産", "リース負債", "経過措置", "適用時期",
        )
        for term in modifiers:
            if term in query and term not in focus_parts:
                focus_parts.append(term)
        return " ".join(focus_parts)

    @classmethod
    def _anchor_focus_variant(cls, query: str) -> str | None:
        generic_terms = (
            "教えてください", "教えて", "改正点", "変更点", "違い", "比較",
            "概要", "内容", "ポイント", "会計基準", "基準", "について",
        )
        reduced = query
        for term in generic_terms:
            reduced = reduced.replace(term, " ")
        keywords = cls._extract_keywords(reduced)
        anchor_terms = [term for term in keywords if len(term) >= 2][:4]
        if len(anchor_terms) < 2:
            return None
        return " ".join(anchor_terms)

    def _llm_variants(self, query: str) -> list[str]:
        prompt = (
            "次の質問を検索向けに日本語で言い換えてください。"
            "3個まで。短く、会計用語を優先してください。JSON配列だけを返してください。\n\n"
            f"質問: {query}"
        )
        try:
            response = self.llm.chat(messages=[{"role": "user", "content": prompt}], tools=None, temperature=0.0, max_tokens=200)
            content = response["message"].get("content", "").strip()
            variants = json.loads(content)
            if isinstance(variants, list):
                return [str(item) for item in variants]
        except Exception as e:
            logger.debug(f"Query rewrite failed: {e}")
        return []
