"""Read Document tool — モデルが文書全文を直接読むためのツール。

chunks.json から同一ファイルのチャンクを結合して全文を再構築する。
別途 full_texts.json は不要。
"""

from collections import defaultdict
from typing import Any

from .base import BaseTool
from ..context import AgentContext

_MAX_CHARS = 40_000  # 1回に返す最大文字数（約10,000トークン相当）


def _build_doc_index(chunks: list[dict]) -> dict[str, dict]:
    """chunks リストから {filename: {source, text, char_count}} を構築する。"""
    # filename ごとにチャンクを収集（chunk_num 順）
    by_file: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for chunk in chunks:
        chunk_id: str = chunk.get("id", "")
        if ":" not in chunk_id:
            continue
        filename, num_str = chunk_id.rsplit(":", 1)
        try:
            num = int(num_str)
        except ValueError:
            num = 0
        source = chunk.get("source", filename).split(" >")[0].strip()
        text = chunk.get("text", "")
        by_file[filename].append((num, source, text))

    docs: dict[str, dict] = {}
    for filename, items in by_file.items():
        items.sort(key=lambda x: x[0])
        source = items[0][1] if items else filename
        full_text = "\n\n".join(t for _, _, t in items)
        docs[filename] = {
            "source": source,
            "text": full_text,
            "char_count": len(full_text),
        }
    return docs


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
        for filename, doc in matches:
            source = doc.get("source", filename)
            text = doc.get("text", "")
            total = len(text)
            chunk = text[offset: offset + _MAX_CHARS]
            end = offset + len(chunk)

            header = f"=== {source} ({filename}) | 全{total:,}文字 ==="
            if total > _MAX_CHARS or offset > 0:
                header += f"\n[読取範囲: {offset:,}〜{end:,}文字目]"
            if end < total:
                header += (
                    f"\n[※ 続きがあります。続きを読む場合: "
                    f'read_document(name="{filename}", offset={end})]'
                )

            parts.append(f"{header}\n\n{chunk}")

        result = "\n\n---\n\n".join(parts)
        context.add_retrieval_log(
            tool_name="read_document",
            tokens=len(result) // 4,
            metadata={"name": name, "matched": [fn for fn, _ in matches], "offset": offset},
        )
        return result, {"matched": [fn for fn, _ in matches], "offset": offset}
