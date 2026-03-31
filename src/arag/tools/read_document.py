"""Read Document tool — モデルが文書全文を直接読むためのツール。

chunks.json から同一ファイルのチャンクを結合して全文を再構築する。
別途 full_texts.json は不要。
"""

from collections import defaultdict
import re
from typing import Any

from .base import BaseTool
from ..context import AgentContext
from ..query_rewrite import QueryExpander

_DEFAULT_MAX_CHARS = 40_000  # 旧来の全文読取上限


def _max_chars_for_context(context: AgentContext) -> int:
    complexity = context.question_complexity or "moderate"
    if complexity == "complex":
        max_chars = 24_000
    elif complexity == "simple":
        max_chars = 16_000
    else:
        max_chars = 20_000

    if len(context.read_chunk_ids) >= 2:
        max_chars = int(max_chars * 0.75)

    return max(8_000, min(max_chars, _DEFAULT_MAX_CHARS))


def _build_doc_index(chunks: list[dict]) -> dict[str, dict]:
    """chunks リストから {filename: {source, text, char_count}} を構築する。"""
    # filename ごとにチャンクを収集（chunk_num 順）
    by_file: dict[str, list[tuple[int, str, str, str, str]]] = defaultdict(list)
    for chunk in chunks:
        chunk_id: str = chunk.get("id", "")
        if ":" not in chunk_id:
            continue
        filename, num_str = chunk_id.rsplit(":", 1)
        try:
            num = int(re.sub(r"\D+", "", num_str) or "0")
        except ValueError:
            num = 0
        raw_source = chunk.get("source", filename).strip()
        doc_source = raw_source.split(" >")[0].strip()
        text = chunk.get("text", "")
        by_file[filename].append((num, chunk_id, doc_source, raw_source, text))

    docs: dict[str, dict] = {}
    for filename, items in by_file.items():
        items.sort(key=lambda x: x[0])
        full_text = "\n\n".join(text for _, _, _, _, text in items)
        docs[filename] = {
            "source": items[0][2] if items else filename,
            "text": full_text,
            "char_count": len(full_text),
            "segments": [
                {"chunk_id": chunk_id, "source": item_source, "text": text}
                for _, chunk_id, _, item_source, text in items
            ],
        }
    return docs


def _exact_segment_matches(context: AgentContext, filename: str, source: str, text: str) -> bool:
    doc_haystacks = [filename, source]
    section_haystacks = [source, text[:2400]]
    doc_hits = QueryExpander.count_exact_doc_hits(context.exact_doc_terms, doc_haystacks)
    section_hits = QueryExpander.count_exact_section_hits(context.exact_section_terms, section_haystacks)
    return (
        doc_hits == len(context.exact_doc_terms)
        and section_hits == len(context.exact_section_terms)
    )


def _extract_exact_excerpt(context: AgentContext, text: str) -> str:
    if not text:
        return ""

    match_start = 0
    if context.exact_section_terms:
        for term in context.exact_section_terms:
            for pattern in QueryExpander._section_term_patterns(term):
                match = pattern.search(text)
                if match:
                    match_start = max(0, match.start() - 120)
                    break
            if match_start:
                break
    excerpt = text[match_start: match_start + 1200].strip()
    if not excerpt:
        excerpt = text[:1200].strip()
    excerpt = re.sub(r"\n{3,}", "\n\n", excerpt)
    if len(excerpt) < len(text):
        excerpt = f"{excerpt}\n[抜粋]"
    return excerpt


class ReadDocumentTool(BaseTool):
    def __init__(self, chunks: list[dict]):
        self._docs = _build_doc_index(chunks)

    @property
    def name(self) -> str:
        return "read_document"

    def get_schema(self) -> dict[str, Any]:
        return {
            "name": "read_document",
            "description": (
                "文書名またはファイル名を指定して、文書の全文を取得します。\n"
                "keyword_search / semantic_search で関連チャンクが見つからない場合や、"
                "特定の会計基準・規則の全体像を把握したい場合に使用してください。\n"
                "例: '固定資産の減損', '財規', 'bac_genson_kijun.pdf'\n\n"
                "引数 name を省略するか空文字にすると、利用可能な文書の一覧を返します。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "読みたい文書名またはファイル名（部分一致）。"
                            "空文字または省略で文書一覧を表示。"
                        ),
                        "default": "",
                    },
                    "offset": {
                        "type": "integer",
                        "description": (
                            "読み始める文字位置（デフォルト: 0）。"
                            "大きな文書の続きを読む場合に使用。"
                        ),
                        "default": 0,
                    },
                },
                "required": [],
            },
        }

    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        name: str = kwargs.get("name", "") or ""
        offset: int = int(kwargs.get("offset", 0))

        # 一覧表示
        if not name.strip():
            lines = ["## 利用可能な文書一覧\n"]
            for filename, doc in sorted(self._docs.items(), key=lambda x: x[1]["source"]):
                lines.append(f"- **{doc['source']}** (`{filename}`, {doc['char_count']:,}文字)")
            return "\n".join(lines), {"action": "list", "count": len(self._docs)}

        # 検索（大文字小文字を無視した部分一致）
        name_lower = name.lower()
        matches = [
            (fn, doc)
            for fn, doc in self._docs.items()
            if name_lower in fn.lower() or name_lower in doc.get("source", "").lower()
        ]

        if not matches:
            available = sorted(f"{d['source']} ({fn})" for fn, d in self._docs.items())
            hint = "\n".join(f"- {s}" for s in available[:20])
            if len(available) > 20:
                hint += f"\n... 他 {len(available)-20} 件"
            return (
                f"文書 '{name}' が見つかりませんでした。\n\n"
                f"利用可能な文書（抜粋）:\n{hint}\n\n"
                "より具体的な名前で再試行してください。"
            ), {"matched": []}

        if len(matches) > 5:
            options = "\n".join(f"- {doc['source']} (`{fn}`)" for fn, doc in matches[:10])
            if len(matches) > 10:
                options += f"\n... 他 {len(matches)-10} 件"
            return (
                f"{len(matches)} 件の文書が一致しました。より具体的な名前を指定してください:\n{options}"
            ), {"matched": [fn for fn, _ in matches]}

        # 文書内容を返す
        parts = []
        max_chars = _max_chars_for_context(context)
        for filename, doc in matches:
            source = doc.get("source", filename)
            text = doc.get("text", "")
            total = len(text)
            chunk = text[offset: offset + max_chars]
            end = offset + len(chunk)

            header = f"=== {source} ({filename}) | 全{total:,}文字 ==="
            if total > max_chars or offset > 0:
                header += f"\n[読取範囲: {offset:,}〜{end:,}文字目]"
            if end < total:
                header += (
                    f"\n[※ 続きがあります。続きを読む場合: "
                    f'read_document(name="{filename}", offset={end})]'
                )

            parts.append(f"{header}\n\n{chunk}")

            if context.exact_doc_terms or context.exact_section_terms:
                for segment in doc.get("segments", []):
                    segment_source = str(segment.get("source", source))
                    segment_text = str(segment.get("text", ""))
                    if not _exact_segment_matches(context, filename, segment_source, segment_text):
                        continue
                    chunk_id = str(segment.get("chunk_id", ""))
                    excerpt = _extract_exact_excerpt(context, segment_text)
                    if chunk_id and excerpt:
                        context.mark_chunk_read(chunk_id, len(excerpt) // 4)
                        context.set_evidence_note(chunk_id, excerpt, source=segment_source)
                    break

        result = "\n\n---\n\n".join(parts)
        context.add_retrieval_log(
            tool_name="read_document",
            tokens=len(result) // 4,
            metadata={"name": name, "matched": [fn for fn, _ in matches], "offset": offset, "max_chars": max_chars},
        )
        return result, {"matched": [fn for fn, _ in matches], "offset": offset, "max_chars": max_chars}
