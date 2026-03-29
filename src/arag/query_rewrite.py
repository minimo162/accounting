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
    detail_seeking: bool = False
    detail_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "complexity": self.complexity,
            "search_mode": self.search_mode,
            "keywords": list(self.keywords),
            "canonical_focus_query": self.canonical_focus_query,
            "corrective_query": self.corrective_query,
            "detail_seeking": self.detail_seeking,
            "detail_terms": list(self.detail_terms),
        }


class QueryExpander:
    _EXACT_TITLE_PATTERNS = (
        r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に関する会計基準(?:の適用指針)?)",
        r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+取引に関する会計基準)",
        r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に係る会計基準)",
        r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に関する実務指針)",
    )
    _EXACT_REFERENCE_PATTERNS = (
        r"(企業会計基準第\s*\d+\s*号)",
        r"(企業会計基準適用指針第\s*\d+\s*号)",
        r"(適用指針第\s*\d+\s*号)",
        r"(実務対応報告第\s*\d+\s*号)",
        r"(会計基準第\s*\d+\s*号)",
        r"(第\s*\d+\s*(?:項|条|号))",
        r"(BC\s*\d+(?:\s*[-‑–]\s*\d+)?)",
    )
    _COMPLEX_TERMS = (
        "改正", "変更", "見直し", "背景", "目的", "経過措置", "適用時期",
        "比較", "違い", "それぞれ", "併せて", "また", "及び", "ならびに",
    )
    _DETAIL_SEEKING_TERMS = (
        "詳しく", "要件", "違い", "比較", "どのような場合", "判断",
        "経過措置", "手順", "例外", "条件", "観点", "識別", "見積り", "見積もり",
    )
    _DETAIL_HINT_TERMS = (
        "要件", "条件", "例外", "判断", "比較", "経過措置", "適用時期",
        "識別", "見積り", "見積もり", "場合", "区分", "差異",
    )
    _SIMPLE_TERMS = ("とは", "何か", "意味", "定義", "概要", "趣旨")
    _KEYWORD_PATTERNS = (
        r"(企業会計基準第\d+号)",
        r"(適用指針第\d+号)",
        r"(実務対応報告第\d+号)",
        r"(会計基準第\d+号)",
        r"(第\d+項)",
        r"(BC\d+)",
        r"([一-龥ぁ-んァ-ヶーA-Za-z0-9]+に関する会計基準(?:の適用指針)?)",
        r"(ヘッジ会計の適用要件)",
        r"(リスク管理方針文書(?:の記載事項)?)",
        r"(繰延ヘッジ)",
        r"(ヘッジ会計)",
        r"(有効性)",
        r"(事前テスト)",
        r"(事後テスト)",
        r"(使用権資産)",
        r"(リース負債)",
        r"(履行義務)",
        r"(本人)",
        r"(代理人)",
        r"(税効果会計)",
        r"(繰延税金資産)",
        r"(繰延税金負債)",
        r"(法定実効税率)",
        r"(収益認識基準)",
    )
    _FILLER_PHRASES = (
        "について",
        "における",
        "に関する",
        "を教えてください",
        "を教えて",
        "教えてください",
        "教えて",
        "どのように",
        "どう判断しますか",
        "どう判断する",
        "ですか",
        "ますか",
        "してください",
        "するための",
        "ための",
        "主な",
        "詳しく",
        "それぞれ",
    )
    _SEPARATOR_CHARS = "、。・/()（）「」『』【】[]{}:：?？!！,."

    def __init__(self, config: RetrievalConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm

    @classmethod
    def _extract_keywords(cls, query: str) -> list[str]:
        keywords: list[str] = []

        def add(term: str):
            normalized = term.strip()
            if normalized and normalized not in keywords:
                keywords.append(normalized)

        for pattern in cls._KEYWORD_PATTERNS:
            for match in re.findall(pattern, query):
                add(match)

        reduced = query
        for phrase in cls._FILLER_PHRASES:
            reduced = reduced.replace(phrase, " ")
        for ch in cls._SEPARATOR_CHARS:
            reduced = reduced.replace(ch, " ")
        for particle in ("は", "が", "を", "に", "で", "と", "の", "も", "へ", "や"):
            reduced = reduced.replace(particle, " ")

        for token in reduced.split():
            if len(token) < 2 or len(token) > 24:
                continue
            if re.fullmatch(r"[ぁ-ん]{2,}", token):
                continue
            add(token)

        return keywords[:8]

    @classmethod
    def profile(cls, query: str) -> QueryProfile:
        keywords = cls._extract_keywords(query)
        exact = cls.is_exact_query(query)
        complexity = cls._infer_complexity(query, keywords, exact)
        detail_seeking = cls._is_detail_seeking(query, exact, complexity)
        detail_terms = cls._detail_focus_terms(query)
        canonical_focus = cls._domain_focus_variant(query) or cls._canonical_title_focus_variant(query)
        corrective_query = cls._merge_focus_terms(
            canonical_focus or cls._anchor_focus_variant(query),
            detail_terms if detail_seeking else [],
        )

        if exact:
            search_mode = "keyword_first"
        elif canonical_focus and (
            complexity != "complex"
            or cls._prefer_keyword_first_for_complex(query, keywords)
            or detail_seeking
        ):
            search_mode = "keyword_first"
        elif detail_seeking and corrective_query and complexity != "simple":
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
            detail_seeking=detail_seeking,
            detail_terms=detail_terms,
        )

    @staticmethod
    def is_exact_query(query: str) -> bool:
        return bool(QueryExpander.extract_exact_terms(query))

    @classmethod
    def extract_exact_terms(cls, query: str) -> list[str]:
        terms: list[str] = []

        def add(term: str):
            normalized = re.sub(r"\s+", "", term.strip())
            if normalized and normalized not in terms:
                terms.append(normalized)

        for pattern in cls._EXACT_REFERENCE_PATTERNS:
            for match in re.findall(pattern, query):
                add(match)
        for pattern in cls._EXACT_TITLE_PATTERNS:
            for match in re.findall(pattern, query):
                add(match)
        filtered: list[str] = []
        for term in terms:
            if any(term != other and term in other for other in terms):
                continue
            filtered.append(term)
        return filtered

    @classmethod
    def exact_keyword_terms(cls, query: str) -> list[str]:
        terms = cls.extract_exact_terms(query)
        if terms:
            anchor_terms: list[str] = []
            generic_terms = {
                "処理", "扱い", "内容", "概要", "意味", "定義",
                "教えて", "ください", "どのように", "ですか", "ますか",
            }
            for keyword in cls._extract_keywords(query):
                normalized = re.sub(r"\s+", "", keyword)
                if any(
                    normalized == term
                    or normalized in term
                    or term in normalized
                    for term in terms
                ):
                    continue
                if normalized in generic_terms or len(normalized) < 2:
                    continue
                if keyword not in anchor_terms:
                    anchor_terms.append(keyword)
                if len(anchor_terms) >= 2:
                    break
            return terms + anchor_terms
        canonical_title = cls._canonical_title_focus_variant(query)
        if canonical_title:
            title_terms = cls.extract_exact_terms(canonical_title)
            if title_terms:
                return title_terms
        return [query.strip()]

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
        profile = self.profile(query)
        canonical_focus = profile.canonical_focus_query
        if profile.detail_seeking and profile.corrective_query and profile.corrective_query != query:
            variants.append(profile.corrective_query)
        if canonical_focus and canonical_focus not in variants:
            variants.append(canonical_focus)
        if profile.corrective_query and profile.corrective_query not in variants and profile.corrective_query != query:
            variants.append(profile.corrective_query)
        if len(keywords) >= 2:
            variants.append(" ".join(keywords[:4]))
        if any("第" in kw and "号" in kw for kw in keywords):
            variants.append(f"{query} 条項 経過措置 開示")
        if "会計" not in query:
            variants.append(f"{query} 会計基準")
        return variants

    @classmethod
    def _is_detail_seeking(cls, query: str, exact: bool, complexity: str) -> bool:
        if exact and complexity == "simple":
            return False
        if complexity == "complex":
            return True
        return any(term in query for term in cls._DETAIL_SEEKING_TERMS)

    @classmethod
    def _detail_focus_terms(cls, query: str) -> list[str]:
        terms: list[str] = []

        def add(term: str) -> None:
            if term and term not in terms:
                terms.append(term)

        for term in cls._DETAIL_HINT_TERMS:
            if term in {"見積り", "見積もり"}:
                continue
            if term in query:
                add(term)

        if any(term in query for term in ("見積り", "見積もり")):
            add("見積り")

        if "どのような場合" in query:
            add("場合")
        if "どのように" in query and "判断" in query:
            add("判断")
        if any(term in query for term in ("違い", "比較", "区分")):
            add("比較")

        return terms[:4]

    @staticmethod
    def _merge_focus_terms(base: str | None, extra_terms: list[str]) -> str | None:
        tokens: list[str] = []
        for part in (base or "").split():
            normalized = part.strip()
            if normalized and normalized not in tokens:
                tokens.append(normalized)
        for term in extra_terms:
            normalized = term.strip()
            if normalized and normalized not in tokens:
                tokens.append(normalized)
        return " ".join(tokens) or None

    @classmethod
    def _infer_complexity(cls, query: str, keywords: list[str], exact: bool) -> str:
        complex_hits = sum(1 for term in cls._COMPLEX_TERMS if term in query)
        separator_hits = sum(query.count(token) for token in ("と", "、", "/", "・"))
        if exact:
            exact_terms = set(cls.extract_exact_terms(query))
            non_exact_keywords = [
                keyword for keyword in keywords
                if not any(
                    (normalized_keyword := re.sub(r"\s+", "", keyword)) == exact_term
                    or normalized_keyword in exact_term
                    or exact_term in normalized_keyword
                    for exact_term in exact_terms
                )
            ]
            if len(non_exact_keywords) <= 3:
                return "simple"
            if len(non_exact_keywords) <= 4:
                return "moderate"
        if "短期リース" in query and any(term in query for term in ("少額リース", "少額資産")) and len(keywords) <= 5:
            return "simple"
        if all(term in query for term in ("本人", "代理人")) and len(keywords) <= 6:
            return "moderate"
        if "履行義務" in query and any(term in query for term in ("保守サービス", "値引き", "割引")) and len(keywords) <= 8:
            return "moderate"
        if "減損" in query and "兆候" in query and len(keywords) <= 7:
            return "moderate"
        if "変動対価" in query and len(keywords) <= 6:
            return "moderate"
        if "契約変更" in query and any(term in query for term in ("既存", "新しい契約", "別個")) and len(keywords) <= 8:
            return "moderate"
        if "研究開発費" in query and "ソフトウェア" in query and len(keywords) <= 7:
            return "moderate"
        if "税効果会計" in query and "税率" in query and len(keywords) <= 7:
            return "moderate"
        if complex_hits >= 2 or (complex_hits >= 1 and (separator_hits >= 1 or len(keywords) >= 5)):
            return "complex"
        if any(term in query for term in cls._SIMPLE_TERMS) and len(keywords) <= 4:
            return "simple"
        if len(query) >= 35 or len(keywords) >= 6:
            return "complex"
        return "moderate"

    @classmethod
    def _domain_focus_variant(cls, query: str) -> str | None:
        if "短期リース" in query and any(term in query for term in ("少額リース", "少額資産")):
            focus_parts = []
            if "借手" in query:
                focus_parts.append("借手")
            focus_parts.extend(["短期リース", "少額リース"])
            return " ".join(dict.fromkeys(focus_parts))
        if "減損" in query and "兆候" in query:
            focus_parts = []
            if "固定資産" in query:
                focus_parts.append("固定資産")
            focus_parts.extend(["減損", "兆候", "回収可能価額"])
            if "固定資産" in query or any(term in query for term in ("使用価値", "正味売却価額")):
                focus_parts.extend(["使用価値", "正味売却価額"])
            return " ".join(dict.fromkeys(focus_parts))
        if "変動対価" in query:
            focus_parts = []
            if "収益認識基準" in query:
                focus_parts.append("収益認識基準")
            focus_parts.extend(["変動対価", "見積り", "制約"])
            if "収益" in query:
                focus_parts.append("収益")
            return " ".join(dict.fromkeys(focus_parts))
        if "契約変更" in query:
            focus_parts = []
            if "収益認識基準" in query:
                focus_parts.append("収益認識基準")
            focus_parts.extend(["契約変更", "別個"])
            if any(term in query for term in ("既存", "継続")):
                focus_parts.append("既存")
            if any(term in query for term in ("新しい契約", "新規")):
                focus_parts.append("新しい契約")
            focus_parts.extend(["履行義務", "取引価格"])
            return " ".join(dict.fromkeys(focus_parts))
        if "研究開発費" in query:
            focus_parts = ["研究開発費", "発生時", "費用"]
            if "ソフトウェア" in query:
                focus_parts.extend(["ソフトウェア", "資産"])
            return " ".join(dict.fromkeys(focus_parts))
        if all(term in query for term in ("本人", "代理人")):
            focus_parts = []
            if "収益認識基準" in query:
                focus_parts.append("収益認識基準")
            focus_parts.extend(["本人", "代理人", "支配"])
            if any(term in query for term in ("区分", "判断")):
                focus_parts.extend(["総額", "純額"])
            return " ".join(dict.fromkeys(focus_parts))
        if "履行義務" in query:
            focus_parts = []
            if "収益認識基準" in query:
                focus_parts.append("収益認識基準")
            focus_parts.extend(["履行義務", "別個"])
            if "保守サービス" in query:
                focus_parts.append("保守サービス")
            if any(term in query for term in ("値引き", "割引")):
                focus_parts.append("値引き")
            if "契約" in query:
                focus_parts.append("契約")
            return " ".join(dict.fromkeys(focus_parts))
        if "繰延ヘッジ" in query or "ヘッジ会計" in query:
            focus_parts = ["ヘッジ会計"]
            if "繰延ヘッジ" in query:
                focus_parts.append("繰延ヘッジ")
            if any(term in query for term in ("要件", "適用")):
                focus_parts.extend(["ヘッジ会計の適用要件", "正式な文書", "有効性", "事前テスト", "事後テスト"])
            return " ".join(dict.fromkeys(focus_parts))
        if "税効果会計" in query and "税率" in query:
            focus_parts = ["税効果会計", "税率"]
            if "繰延税金資産" in query:
                focus_parts.append("繰延税金資産")
            if "繰延税金負債" in query or ("繰延税金資産" in query and "負債" in query):
                focus_parts.append("繰延税金負債")
            if "税率変更" in query or "変更時" in query:
                focus_parts.append("税率変更")
            return " ".join(dict.fromkeys(focus_parts))
        return None

    @classmethod
    def _prefer_keyword_first_for_complex(cls, query: str, keywords: list[str]) -> bool:
        return (
            "リース" in query
            and any(term in query for term in ("改正", "改正点", "経過措置"))
            and all(term in query for term in ("借手", "貸手"))
            and len(keywords) <= 8
        )

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
            "どのように", "ですか", "ますか", "主な", "するための",
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
