"""全PDFとe-Gov XMLから全文テキストを抽出して full_texts.json に保存する。

chunks.json（チャンク分割＋埋め込み用）とは別に、モデルが文書全体を読むための
ファイルを生成する。fitz（PyMuPDF）を使い、LiteParse のレイアウト処理を省いた
素直なテキスト抽出を行う。

出力形式:
  {
    "<filename>": {
      "source": "文書名（人間が読める名前）",
      "text": "全文テキスト",
      "char_count": 12345
    },
    ...
  }
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz  # PyMuPDF


def _extract_pdf_text(pdf_path: Path) -> str:
    """fitz で PDF 全ページのテキストを抽出して連結する。"""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        t = page.get_text("text").strip()
        if t:
            pages.append(t)
    doc.close()
    return "\n\n".join(pages)


def _extract_xml_text(xml_path: Path) -> str:
    """e-Gov XML から全条文テキストを抽出する。"""
    def _collect(elem) -> list[str]:
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            parts.extend(_collect(child))
            if child.tail:
                parts.append(child.tail)
        return parts

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    raw = "".join(_collect(root))
    # 連続空白・改行を整理
    raw = re.sub(r'\n\s*\n+', '\n\n', raw)
    raw = re.sub(r'  +', ' ', raw)
    return raw.strip()


def _guess_title(text: str, filename: str) -> str:
    """テキスト冒頭から文書タイトルを推定する。"""
    std_markers = [
        '企業会計基準', '適用指針', '実務対応報告', '監査基準',
        'サステナビリティ', '財務諸表', '連結', '固定資産の減損',
        '退職給付', 'リース', '税効果', '金融商品', '内部統制',
        '財規', '連結財規', '中間財規',
    ]
    for line in text.split('\n')[:40]:
        line = line.strip()
        if len(line) < 5 or len(line) > 80:
            continue
        if any(m in line for m in std_markers):
            return line
    return filename


def build_full_texts(pdf_dir: str, egov_cache_dir: str, output_path: str):
    pdf_dir = Path(pdf_dir)
    egov_cache = Path(egov_cache_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    full_texts: dict[str, dict] = {}

    # --- PDFs ---
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    print(f"Processing {len(pdf_files)} PDFs...")
    for pdf_path in pdf_files:
        text = _extract_pdf_text(pdf_path)
        if not text.strip():
            print(f"  [skip] {pdf_path.name}: no text")
            continue
        title = _guess_title(text, pdf_path.stem)
        full_texts[pdf_path.name] = {
            "source": title,
            "text": text,
            "char_count": len(text),
        }
        print(f"  {pdf_path.name}: {len(text):,} chars — {title[:40]}")

    # e-Gov XMLs は条文数が膨大（財規 5MB+）のため chunks.json で十分にカバー済み。
    # read_document ツールには含めない。

    output.write_text(json.dumps(full_texts, ensure_ascii=False, indent=2), encoding="utf-8")
    total_chars = sum(v["char_count"] for v in full_texts.values())
    print(f"\nSaved {len(full_texts)} documents to {output}")
    print(f"Total text: {total_chars:,} chars ({total_chars // 1024} KB)")


if __name__ == "__main__":
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    egov_cache = sys.argv[2] if len(sys.argv) > 2 else "data/egov_cache"
    output = sys.argv[3] if len(sys.argv) > 3 else "data/full_texts.json"
    build_full_texts(pdf_dir, egov_cache, output)
