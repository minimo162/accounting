"""Process PDFs into chunks using LiteParse spatial text extraction.

Design philosophy (LiteParse):
- Preserve layout rather than detect structure
- No Markdown conversion, no table detection — avoid failure modes
- Spatial text (indentation, column spacing) is kept as-is
- LLMs/embeddings understand ASCII layout natively

Chunking strategy:
- Extract spatial text from each page via LiteParse
- Remove page numbers, join all pages with \\n (no artificial gaps)
- Try article-header splitting first (第X条/X./（X）), fall back to blank-line splitting
- Merge small sections, split oversized ones
- Add 100-char overlap from previous chunk to preserve cross-reference context

Chunk ID format: "{filename}:{chunk_num}"
"""

import json
import re
import sys
from pathlib import Path

from liteparse import LiteParse

sys.path.insert(0, str(Path(__file__).parent))
from process_egov import get_egov_covered_pdfs


# Page number patterns at page boundaries
_PAGE_NUM_RE = re.compile(
    r'(?:^|\n)'           # start of line
    r'\s*-\s*\n'          # "  -"
    r'\s*\d+\s*\n'        # "  123"
    r'\s*-\s*'            # "  -"
    r'(?:\n|$)',          # end of line
    re.MULTILINE,
)
# Inline page numbers like "- 3 -" on a single line
_PAGE_NUM_INLINE_RE = re.compile(r'^\s*-\s*\d+\s*-\s*$', re.MULTILINE)

# Article/section header patterns for Japanese legal/accounting documents
# Matches lines that start a new logical unit
_ARTICLE_HEADER_RE = re.compile(
    r'^(?:'
    r'第\s*[一二三四五六七八九十百千\d]+\s*(?:条|章|節|款|目)'  # 第X条/章/節
    r'|\d+\.\s+\S'                    # 1. text (段落番号、後続に非空白)
    r'|[１-９][０-９]*\.\s+\S'        # 全角数字段落番号
    r'|\（\d+\）'                     # （1）（2）...
    r')',
    re.MULTILINE,
)

# Subtitle extraction: prefer article number lines over plain first lines
_SUBTITLE_RE = re.compile(
    r'^(?:'
    r'第\s*[一二三四五六七八九十百千\d]+\s*(?:条|章|節|款|目)'
    r'|\d+\.\s'
    r'|\（\d+\）'
    r')'
)

# Overlap added to the beginning of each chunk from the tail of the previous
_OVERLAP_CHARS = 100


def _remove_page_numbers(text: str) -> str:
    """Remove page number patterns from text."""
    text = _PAGE_NUM_RE.sub('\n', text)
    text = _PAGE_NUM_INLINE_RE.sub('', text)
    return text


def _extract_title(text: str) -> str:
    """Extract document title from first non-junk lines."""
    std_patterns = ['企業会計基準', '適用指針', '実務対応報告', '監査基準', '中間監査',
                    '期中レビュー', '内部統制', 'サステナビリティ', '財務諸表',
                    '連結', '減損', '退職給付', 'リース', '税効果', '金融商品']
    lines = text.split("\n")[:30]

    # Pass 1: find line with standard identifier
    for line in lines:
        line = line.strip()
        if line and any(p in line for p in std_patterns):
            return line

    # Pass 2: find first non-junk line
    for line in lines:
        line = line.strip()
        if not line or len(line) < 6:
            continue
        if re.match(r'^[\d\s]*\d{4}\s*年', line):
            continue
        if re.match(r'^(平成|令和|昭和)\s*[\d０-９]+\s*年', line):
            continue
        return line

    return ""


def _extract_subtitle(section_text: str) -> str:
    """条文番号パターンを優先して subtitle を返す。なければ最初の非空行。"""
    for line in section_text.split("\n"):
        line = line.strip()
        if not line or len(line) <= 3:
            continue
        if _SUBTITLE_RE.match(line):
            return line[:60]
        # 条文番号なし → 最初の非空行をそのまま使う
        return line[:60]
    return ""


def _split_by_article_headers(text: str) -> list[tuple[int, str]]:
    """条文ヘッダー行 (第X条/X./（X）) で分割し (start_pos, section_text) を返す。

    ヘッダーが2つ以上検出されなければ空リストを返し、呼び出し側が空行分割に切り替える。
    """
    sections: list[tuple[int, str]] = []
    current_start = 0
    current_lines: list[str] = []
    pos = 0

    for line in text.split('\n'):
        stripped = line.strip()
        if current_lines and _ARTICLE_HEADER_RE.match(stripped):
            sec = '\n'.join(current_lines).strip()
            if sec:
                sections.append((current_start, sec))
            current_start = pos
            current_lines = []
        current_lines.append(line)
        pos += len(line) + 1  # +1 for \n

    if current_lines:
        sec = '\n'.join(current_lines).strip()
        if sec:
            sections.append((current_start, sec))

    # ヘッダーが1つ以下なら検出失敗とみなしてフォールバックを促す
    if len(sections) <= 1:
        return []
    return sections


def _merge_and_size(
    raw_sections: list[tuple[int, str]],
    max_chars: int = 2000,
    min_chars: int = 200,
) -> list[tuple[int, str]]:
    """raw_sections をサイズ制約に合わせてマージ・分割する。

    Returns (start_pos, text) pairs.
    """
    chunks: list[tuple[int, str]] = []
    buffer = ""
    buffer_start = 0

    def flush():
        nonlocal buffer, buffer_start
        if not buffer.strip():
            buffer = ""
            return
        if len(buffer.strip()) >= min_chars:
            chunks.append((buffer_start, buffer.strip()))
        elif chunks:
            prev_start, prev_text = chunks[-1]
            chunks[-1] = (prev_start, prev_text + "\n\n" + buffer.strip())
        else:
            chunks.append((buffer_start, buffer.strip()))
        buffer = ""

    for section_start, section in raw_sections:
        if len(buffer) + len(section) > max_chars and buffer:
            flush()

        # Oversized section: split at single blank lines
        if len(section) > max_chars:
            if buffer:
                flush()
            sub_pos = section_start
            for sub in re.split(r'\n\s*\n', section):
                if len(buffer) + len(sub) > max_chars and buffer:
                    flush()
                if not buffer:
                    buffer_start = sub_pos
                buffer += "\n\n" + sub if buffer else sub
                sub_pos += len(sub) + 2  # approximate
            continue

        if not buffer:
            buffer_start = section_start
        buffer += "\n\n" + section if buffer else section

    flush()
    return chunks


def _split_by_blank_lines(text: str, max_chars: int = 2000, min_chars: int = 200) -> list[str]:
    """Split text by consecutive blank lines (2+), then merge/split to target size."""
    return [t for _, t in _split_by_blank_lines_with_positions(text, max_chars, min_chars)]


def _split_by_blank_lines_with_positions(
    text: str, max_chars: int = 2000, min_chars: int = 200
) -> list[tuple[int, str]]:
    """Split text, returning (start_pos_in_text, section_text) pairs.

    First tries article-header splitting; falls back to blank-line splitting.
    start_pos is the character offset in the original text (for PDF page mapping).
    """
    # Try article-header splitting first
    raw_sections = _split_by_article_headers(text)

    # Fall back to blank-line splitting if no headers detected
    if not raw_sections:
        raw_sections = []
        last_end = 0
        for m in re.finditer(r'\n\s*\n\s*\n', text):
            raw = text[last_end:m.start()]
            stripped = raw.strip()
            if stripped:
                offset = raw.index(stripped[0]) if stripped else 0
                raw_sections.append((last_end + offset, stripped))
            last_end = m.end()
        raw = text[last_end:]
        stripped = raw.strip()
        if stripped:
            offset = raw.index(stripped[0]) if stripped else 0
            raw_sections.append((last_end + offset, stripped))

    return _merge_and_size(raw_sections, max_chars, min_chars)


def _parse_pdf(lp: LiteParse, pdf_path: Path) -> tuple[str, list[tuple[int, int]]]:
    """Extract spatial text from PDF.

    Returns:
        (full_text, page_breaks) where page_breaks is a list of
        (char_offset, pdf_page_num) indicating where each PDF page starts
        in full_text (1-indexed page numbers).
    """
    try:
        result = lp.parse(str(pdf_path))
    except Exception as e:
        print(f"  Warning: LiteParse failed for {pdf_path.name}: {e}")
        # Fallback to PyMuPDF — no per-page tracking in fallback
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_data: list[tuple[int, str]] = []
            for i in range(len(doc)):
                t = doc[i].get_text("text").strip()
                if t:
                    page_data.append((i + 1, t))
            doc.close()
            return _join_pages(page_data)
        except Exception:
            return "", []

    page_data = []
    for page_num, page in enumerate(result.pages, 1):
        cleaned = _remove_page_numbers(page.text).rstrip()
        if cleaned.strip():
            page_data.append((page_num, cleaned))

    return _join_pages(page_data)


def _join_pages(page_data: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Join (page_num, text) pairs into full_text with page break positions."""
    page_breaks: list[tuple[int, int]] = []
    parts: list[str] = []
    pos = 0
    for page_num, text in page_data:
        page_breaks.append((pos, page_num))
        parts.append(text)
        pos += len(text) + 1  # +1 for the \n separator
    return "\n".join(parts), page_breaks


def _lookup_pdf_page(char_pos: int, page_breaks: list[tuple[int, int]]) -> int:
    """Return the PDF page number corresponding to char_pos."""
    page_num = 1
    for break_pos, pnum in page_breaks:
        if break_pos <= char_pos:
            page_num = pnum
        else:
            break
    return page_num


def create_chunks(pdf_dir: str, output_path: str):
    """Process all PDFs into spatial-layout chunks."""
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))

    # e-Gov XML でカバーされる PDF はスキップ (XML の方が構造的に正確)
    egov_covered = get_egov_covered_pdfs()

    lp = LiteParse()
    chunks = []

    for pdf_path in pdf_files:
        if pdf_path.name in egov_covered:
            print(f"Skipping (covered by e-Gov XML): {pdf_path.name}")
            continue
        print(f"Processing: {pdf_path.name}")
        full_text, page_breaks = _parse_pdf(lp, pdf_path)
        if not full_text.strip():
            print(f"  Warning: No text extracted from {pdf_path.name}")
            continue

        title = _extract_title(full_text)
        source_name = title or pdf_path.stem

        sections_with_pos = _split_by_blank_lines_with_positions(full_text)

        file_chunks: list[dict] = []
        for i, (section_start, section_text) in enumerate(sections_with_pos, 1):
            subtitle = _extract_subtitle(section_text)
            if subtitle and subtitle != source_name and not subtitle.startswith(source_name):
                source = f"{source_name} > {subtitle}"
            else:
                source = source_name

            pdf_page = _lookup_pdf_page(section_start, page_breaks)

            file_chunks.append({
                "id": f"{pdf_path.name}:{i}",
                "text": section_text,
                "source": source,
                "file": pdf_path.name,
                "page": i,
                "pdf_page": pdf_page,
            })

        # Add overlap: prepend tail of previous chunk to each chunk
        prev_tail = ""
        for chunk in file_chunks:
            if prev_tail:
                chunk["text"] = prev_tail + "\n...\n" + chunk["text"]
            prev_tail = chunk["text"][-_OVERLAP_CHARS:]

        chunks.extend(file_chunks)
        print(f"  {len(file_chunks)} chunks")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(chunks)} chunks from {len(pdf_files)} PDFs")
    print(f"Output: {output}")


if __name__ == "__main__":
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.json"
    create_chunks(pdf_dir, output)
