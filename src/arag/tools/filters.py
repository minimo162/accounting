"""Chunk quality filtering and tagging for retrieval."""

import re

# Lines that are pure junk: year/era, page numbers, or 4 chars or fewer
_JUNK_LINE_RE = re.compile(
    r'^(?:'
    r'\d{4}\s*年.*'                         # 西暦年号行 (2024年 ...)
    r'|(?:平成|令和|昭和)\s*[\d０-９]+\s*年.*'  # 元号行
    r'|[-―─\s]*\d+[-―─\s]*'               # ページ番号行 (- 3 -)
    r'|.{1,4}'                              # 4文字以下の超短行
    r')$'
)


def _is_mostly_junk_lines(lines: list[str]) -> bool:
    """非空行の80%以上がジャンク行（日付・短語・ページ番号）なら True。"""
    if not lines:
        return False
    junk_count = sum(1 for line in lines if _JUNK_LINE_RE.match(line))
    return junk_count / len(lines) >= 0.8


def is_clearly_low_value(text: str) -> bool:
    """Hard filter: exclude chunks that are obviously not useful for answering questions.

    These are removed from search results entirely so they don't waste top_k slots.
    短さだけでは除外しない — 短い条文でも重要なものがある。
    """
    # Table of contents
    if "目次" in text or "目 次" in text:
        return True

    # Dot leaders (TOC-style listings)
    if text.count("・・") >= 5 or text.count("...") >= 10:
        return True

    # Mostly junk lines (表紙断片・奥付など)
    lines = [line for line in text.split("\n") if line.strip()]
    if lines and _is_mostly_junk_lines(lines):
        return True

    return False


def get_chunk_tag(text: str) -> str | None:
    """Soft tag for borderline chunks the agent should be cautious about.

    These remain in search results but are tagged so the agent can decide.
    """
    lines = [line for line in text.split("\n") if line.strip()]

    # Member lists / voting records
    if any(k in text[:200] for k in ["委員は、以下のとおり", "名全員の賛成により", "委員等名簿"]):
        return "⚠️名簿/議決"

    # Very short chunks (300字未満) — 重要な定義条文の可能性があるので除外はしないが注意タグ
    if len(text) < 300 and len(lines) < 8:
        return "⚠️短文"

    return None
