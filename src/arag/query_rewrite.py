"""Query expansion and HyDE helpers."""

import json
import logging
import re

from .config import RetrievalConfig
from .llm import LLMClient

logger = logging.getLogger(__name__)


class QueryExpander:
    def __init__(self, config: RetrievalConfig, llm: LLMClient | None = None):
        self.config = config
        self.llm = llm

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"(企業会計基準第\d+号|適用指針第\d+号|実務対応報告第\d+号|第\d+項|[一-龥ぁ-んァ-ヶーA-Za-z0-9]{2,})", query)))

    def expand(self, query: str) -> list[str]:
        variants = [query]
        if self.config.enable_query_expansion:
            variants.extend(self._heuristic_variants(query))
            if self.llm is not None:
                variants.extend(self._llm_variants(query))
        deduped: list[str] = []
        for variant in variants:
            variant = variant.strip()
            if variant and variant not in deduped:
                deduped.append(variant)
        return deduped[: self.config.expansion_max_variants]

    def generate_hypothetical_document(self, query: str) -> str | None:
        if not self.config.enable_hyde or self.llm is None:
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
        if len(keywords) >= 2:
            variants.append(" ".join(keywords[:4]))
        if any("第" in kw and "号" in kw for kw in keywords):
            variants.append(f"{query} 条項 経過措置 開示")
        if "会計" not in query:
            variants.append(f"{query} 会計基準")
        return variants

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
