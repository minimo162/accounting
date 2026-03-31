"""Query expansion and HyDE helpers."""

from dataclasses import dataclass, field, replace
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
    verification_mode: bool = False
    verification_claims: list[dict[str, object]] = field(default_factory=list)
    judgment_validation: bool = False
    validation_claims: list[str] = field(default_factory=list)
    cited_references: list[str] = field(default_factory=list)

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
            "verification_mode": self.verification_mode,
            "verification_claims": list(self.verification_claims),
            "judgment_validation": self.judgment_validation,
            "validation_claims": list(self.validation_claims),
            "cited_references": list(self.cited_references),
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
    _JUDGMENT_VALIDATION_TERMS = (
        "妥当か", "妥当でしょうか", "正しいか", "正しいでしょうか", "問題ないか",
        "問題ないでしょうか", "確認して", "確認してください", "チェックして",
        "チェックしてください", "見てください", "適切か", "適切でしょうか",
    )
    _JUDGMENT_ASSERTION_TERMS = (
        "と判断", "と考え", "べき", "ではない", "だと思", "という理解",
        "という認識", "としている", "として扱", "と整理",
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

    def analyze(self, query: str) -> QueryProfile:
        profile = self.profile(query)
        if not profile.verification_mode:
            return profile
        verification_claims = self.decompose_verification_query(query, profile=profile)
        if verification_claims == profile.verification_claims:
            return profile
        return replace(profile, verification_claims=verification_claims)

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
        judgment_validation = cls._is_judgment_validation(query)
        validation_claims = cls._extract_validation_claims(query)
        cited_references = cls.extract_exact_terms(query)
        verification_claims = cls._build_verification_claims(
            validation_claims,
            cited_references,
        )
        complexity = cls._infer_complexity(query, keywords, exact, judgment_validation=judgment_validation)
        detail_seeking = cls._is_detail_seeking(query, exact, complexity)
        detail_terms = cls._detail_focus_terms(query)
        canonical_focus = (
            cls._validation_focus_variant(
                query,
                keywords=keywords,
                exact_terms=cited_references,
                validation_claims=validation_claims,
            )
            if judgment_validation
            else None
        ) or cls._domain_focus_variant(query) or cls._canonical_title_focus_variant(query)
        corrective_query = cls._merge_focus_terms(
            canonical_focus or cls._anchor_focus_variant(query),
            detail_terms if detail_seeking else [],
        )

        if exact:
            search_mode = "keyword_first"
        elif judgment_validation and (canonical_focus or corrective_query):
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
            verification_mode=judgment_validation,
            verification_claims=verification_claims,
            judgment_validation=judgment_validation,
            validation_claims=validation_claims,
            cited_references=cited_references,
        )

    @staticmethod
    def is_exact_query(query: str) -> bool:
        return bool(QueryExpander.extract_exact_terms(query))

    @classmethod
    def extract_exact_terms(cls, query: str) -> list[str]:
        terms: list[str] = []

        def add(term: str):
            cleaned = re.sub(r"^(?:依拠条文として|依拠条文|依拠|根拠として|根拠)\s*[:：]?\s*", "", term.strip())
            normalized = re.sub(r"\s+", "", cleaned)
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
    def split_exact_constraints(cls, query: str) -> tuple[list[str], list[str]]:
        doc_terms: list[str] = []
        section_terms: list[str] = []
        for term in cls.extract_exact_terms(query):
            normalized = cls.normalize_exact_text(term)
            if re.fullmatch(
                r"(?:企業会計基準|企業会計基準適用指針|適用指針|実務対応報告|会計基準)第\d+号",
                normalized,
            ):
                doc_terms.append(term)
            elif re.fullmatch(r"第\d+(?:項|条|号)", normalized) or re.fullmatch(r"BC\d+(?:[-‑–]\d+)?", normalized, flags=re.IGNORECASE):
                section_terms.append(term)
            else:
                doc_terms.append(term)
        return doc_terms, section_terms

    @staticmethod
    def normalize_exact_text(text: str) -> str:
        return re.sub(r"[\s　]+", "", text or "")

    @classmethod
    def _section_term_patterns(cls, term: str) -> list[re.Pattern[str]]:
        normalized = cls.normalize_exact_text(term)
        match = re.fullmatch(r"第(\d+)(項|条|号)", normalized)
        if match:
            number, suffix = match.groups()
            return [
                re.compile(rf"第\s*{re.escape(number)}\s*{re.escape(suffix)}"),
                re.compile(rf"(?<!\d){re.escape(number)}\s*{re.escape(suffix)}(?!\d)"),
                re.compile(rf"(?<!\d){re.escape(number)}\s*[\.．](?!\d)"),
                re.compile(rf"[（(]\s*{re.escape(number)}\s*[)）](?!\d)"),
            ]
        bc_match = re.fullmatch(r"BC(\d+)", normalized, flags=re.IGNORECASE)
        if bc_match:
            number = bc_match.group(1)
            return [
                re.compile(rf"BC\s*{re.escape(number)}", flags=re.IGNORECASE),
                re.compile(rf"結論の背景\s*{re.escape(number)}"),
            ]
        return [re.compile(re.escape(normalized))]

    @classmethod
    def count_exact_doc_hits(cls, doc_terms: list[str], texts: list[str]) -> int:
        if not doc_terms:
            return 0
        haystacks = [cls.normalize_exact_text(text) for text in texts if text]
        return sum(
            1
            for term in doc_terms
            if any(cls.normalize_exact_text(term) in haystack for haystack in haystacks)
        )

    @classmethod
    def count_exact_section_hits(cls, section_terms: list[str], texts: list[str]) -> int:
        if not section_terms:
            return 0
        haystacks = [text for text in texts if text]
        hits = 0
        for term in section_terms:
            patterns = cls._section_term_patterns(term)
            if any(any(pattern.search(text) for pattern in patterns) for text in haystacks):
                hits += 1
        return hits

    @classmethod
    def _exact_doc_term_variants(cls, term: str) -> list[str]:
        normalized = cls.normalize_exact_text(term)
        variants: list[str] = []

        def add(value: str) -> None:
            cleaned = value.strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(term)
        if normalized != term:
            add(normalized)

        reference_match = re.fullmatch(r"(.+第)(\d+)(号)", normalized)
        if reference_match:
            prefix, number, suffix = reference_match.groups()
            add(f"{prefix}{number}{suffix}")
            add(f"{prefix}{number} {suffix}")

        return variants

    @classmethod
    def _exact_section_term_variants(cls, term: str) -> list[str]:
        normalized = cls.normalize_exact_text(term)
        variants: list[str] = []

        def add(value: str) -> None:
            cleaned = value.strip()
            if cleaned and cleaned not in variants:
                variants.append(cleaned)

        add(term)
        if normalized != term:
            add(normalized)

        match = re.fullmatch(r"第(\d+)(項|条|号)", normalized)
        if match:
            number, suffix = match.groups()
            add(f"{number}{suffix}")
            add(f"{number}.")
            add(f"{number}．")

        bc_match = re.fullmatch(r"BC(\d+)", normalized, flags=re.IGNORECASE)
        if bc_match:
            number = bc_match.group(1)
            add(f"BC{number}")
            add(f"結論の背景 {number}")

        return variants

    @classmethod
    def exact_keyword_term_sets(cls, query: str) -> list[list[str]]:
        exact_terms = cls.extract_exact_terms(query)
        base_terms = cls.exact_keyword_terms(query)
        doc_terms, section_terms = cls.split_exact_constraints(query)
        anchor_terms = [
            term
            for term in base_terms
            if not any(
                cls.normalize_exact_text(term) == cls.normalize_exact_text(exact_term)
                for exact_term in exact_terms
            )
        ]

        candidate_sets: list[list[str]] = []

        def add(term_set: list[str]) -> None:
            deduped: list[str] = []
            for term in term_set:
                cleaned = str(term).strip()
                if cleaned and cleaned not in deduped:
                    deduped.append(cleaned)
            if not deduped:
                return
            signature = "\u241f".join(deduped)
            if any("\u241f".join(existing) == signature for existing in candidate_sets):
                return
            candidate_sets.append(deduped)

        if exact_terms:
            add(exact_terms)
        add(base_terms)

        expanded_terms: list[str] = []
        for term in doc_terms:
            expanded_terms.extend(cls._exact_doc_term_variants(term))
        for term in section_terms:
            expanded_terms.extend(cls._exact_section_term_variants(term))
        add(expanded_terms + anchor_terms[:2])

        if section_terms:
            section_focus_terms: list[str] = []
            for term in section_terms:
                section_focus_terms.extend(cls._exact_section_term_variants(term))
            add(section_focus_terms + anchor_terms[:3])

        return candidate_sets or [base_terms or [query.strip()]]

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

    def decompose_verification_query(
        self,
        query: str,
        *,
        profile: QueryProfile | None = None,
    ) -> list[dict[str, object]]:
        profile = profile or self.profile(query)
        if not profile.verification_mode:
            return []

        fallback = self._build_verification_claims(
            profile.validation_claims,
            profile.cited_references,
            query=query,
        )
        if self.llm is None or len(query) < 200:
            return fallback

        llm_claims = self._llm_verification_claims(query)
        if not llm_claims:
            return fallback
        normalized = self._normalize_verification_claims(
            llm_claims,
            query=query,
            fallback_references=profile.cited_references,
        )
        return normalized or fallback

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

    def _llm_verification_claims(self, query: str) -> list[dict[str, object]]:
        prompt = (
            "次の会計判断の確認依頼を、検索用の主張単位に分解してください。"
            "JSON配列だけを返してください。各要素は"
            ' {"claim": "...", "target_transaction": "...", "cited_references": ["..."] } '
            "の形式にしてください。主張は2〜4件まで、簡潔にしてください。\n\n"
            f"質問:\n{query}"
        )
        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
                max_tokens=600,
            )
            content = response["message"].get("content", "").strip()
            payload = json.loads(content)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except Exception as e:
            logger.debug(f"Verification decomposition failed: {e}")
        return []

    @classmethod
    def _is_detail_seeking(cls, query: str, exact: bool, complexity: str) -> bool:
        if exact and complexity == "simple":
            return False
        if complexity == "complex":
            return True
        return any(term in query for term in cls._DETAIL_SEEKING_TERMS)

    @classmethod
    def _is_judgment_validation(cls, query: str) -> bool:
        has_validation_request = any(term in query for term in cls._JUDGMENT_VALIDATION_TERMS)
        if not has_validation_request:
            return False
        return any(term in query for term in cls._JUDGMENT_ASSERTION_TERMS) or "依拠" in query

    @classmethod
    def _extract_validation_claims(cls, query: str) -> list[str]:
        if not cls._is_judgment_validation(query):
            return []

        claims: list[str] = []

        def add(text: str) -> None:
            cleaned = text.strip(" 　。.!！?？:：;；")
            cleaned = re.sub(r"^(私は|当社では|当社|実務上|なお)\s*", "", cleaned)
            cleaned = re.sub(r"(依拠|根拠)\s*[：:].*$", "", cleaned).strip()
            if len(cleaned) < 8:
                return
            if cleaned not in claims:
                claims.append(cleaned)

        for sentence in re.split(r"[。!?！？\n]+", query):
            normalized = sentence.strip()
            if not normalized:
                continue
            if any(term in normalized for term in cls._JUDGMENT_ASSERTION_TERMS):
                add(normalized)
                continue
            if "という理解" in normalized or "という認識" in normalized:
                add(normalized)

        return claims[:3]

    @classmethod
    def _build_verification_claims(
        cls,
        validation_claims: list[str],
        cited_references: list[str],
        *,
        query: str = "",
    ) -> list[dict[str, object]]:
        if not validation_claims:
            return []

        claims: list[dict[str, object]] = []
        for claim in validation_claims:
            target_transaction = cls._infer_target_transaction(claim, query=query)
            doc_terms, section_terms = cls._split_claim_references(
                cited_references or cls.extract_exact_terms(claim),
            )
            claims.append(
                {
                    "claim": claim,
                    "cited_references": list(cited_references),
                    "doc_terms": list(doc_terms),
                    "section_terms": list(section_terms),
                    "target_transaction": target_transaction,
                    "search_query": cls._build_claim_search_query(
                        claim,
                        doc_terms=doc_terms,
                        section_terms=section_terms,
                        target_transaction=target_transaction,
                    ),
                }
            )
        return claims

    @classmethod
    def _split_claim_references(cls, references: list[str]) -> tuple[list[str], list[str]]:
        doc_terms: list[str] = []
        section_terms: list[str] = []
        for ref in references:
            normalized = cls.normalize_exact_text(ref)
            if re.fullmatch(r"第\d+(?:項|条|号)", normalized) or re.fullmatch(r"BC\d+(?:[-‑–]\d+)?", normalized, flags=re.IGNORECASE):
                if ref not in section_terms:
                    section_terms.append(ref)
            elif ref not in doc_terms:
                doc_terms.append(ref)
        return doc_terms, section_terms

    @classmethod
    def _infer_target_transaction(cls, claim: str, *, query: str = "") -> str:
        haystack = f"{claim} {query}"
        candidates = [
            "土地譲渡",
            "土地再評価",
            "連結消去",
            "未実現損失",
            "未実現利益",
            "企業結合",
            "事業分離",
        ]
        hits = [term for term in candidates if term in haystack]
        if hits:
            return " / ".join(dict.fromkeys(hits))
        keywords = cls._extract_keywords(claim)
        return " ".join(keywords[:3])

    @classmethod
    def _build_claim_search_query(
        cls,
        claim: str,
        *,
        doc_terms: list[str],
        section_terms: list[str],
        target_transaction: str = "",
    ) -> str:
        keywords = cls._extract_keywords(claim)
        target_terms = [term for term in re.split(r"\s*/\s*|\s+", target_transaction) if term]
        base_terms = [*doc_terms[:2], *section_terms[:2], *target_terms[:3], *keywords[:4]]
        return cls._merge_focus_terms(" ".join(base_terms), []) or claim

    @classmethod
    def _normalize_verification_claims(
        cls,
        claims: list[dict[str, object]],
        *,
        query: str,
        fallback_references: list[str],
    ) -> list[dict[str, object]]:
        normalized_claims: list[dict[str, object]] = []
        for item in claims:
            claim = str(item.get("claim", "")).strip()
            if len(claim) < 8:
                continue
            raw_refs = item.get("cited_references", [])
            references = [str(ref).strip() for ref in raw_refs if str(ref).strip()] if isinstance(raw_refs, list) else []
            if not references:
                references = [
                    ref for ref in fallback_references
                    if cls.normalize_exact_text(ref) in cls.normalize_exact_text(query)
                ] or list(fallback_references)
            doc_terms, section_terms = cls._split_claim_references(references)
            target_transaction = str(item.get("target_transaction", "")).strip() or cls._infer_target_transaction(claim, query=query)
            normalized_claims.append(
                {
                    "claim": claim,
                    "cited_references": references,
                    "doc_terms": doc_terms,
                    "section_terms": section_terms,
                    "target_transaction": target_transaction,
                    "search_query": cls._build_claim_search_query(
                        claim,
                        doc_terms=doc_terms,
                        section_terms=section_terms,
                        target_transaction=target_transaction,
                    ),
                }
            )
        return normalized_claims[:4]

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
    def _infer_complexity(
        cls,
        query: str,
        keywords: list[str],
        exact: bool,
        *,
        judgment_validation: bool = False,
    ) -> str:
        if judgment_validation:
            if len(query) >= 40 or "依拠" in query or len(keywords) >= 5:
                return "complex"
            return "moderate"
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
    def _validation_focus_variant(
        cls,
        query: str,
        *,
        keywords: list[str],
        exact_terms: list[str],
        validation_claims: list[str],
    ) -> str | None:
        focus_parts: list[str] = []

        for term in exact_terms[:3]:
            if term not in focus_parts:
                focus_parts.append(term)

        claim_keywords = cls._extract_keywords(" ".join(validation_claims)) if validation_claims else []
        for term in claim_keywords + keywords:
            normalized = cls.normalize_exact_text(term)
            if not normalized:
                continue
            if any(normalized == cls.normalize_exact_text(existing) for existing in focus_parts):
                continue
            if term in {"確認", "妥当", "正しい", "問題", "依拠"}:
                continue
            focus_parts.append(term)
            if len(focus_parts) >= 7:
                break

        if not focus_parts:
            return None
        if "判断" in query and "判断" not in focus_parts:
            focus_parts.append("判断")
        return " ".join(focus_parts[:8])

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
