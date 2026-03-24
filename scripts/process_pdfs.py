"""Process PDFs into chunks using LiteParse spatial text extraction.

Design philosophy (LiteParse):
- Preserve layout rather than detect structure
- No Markdown conversion, no table detection — avoid failure modes
- Spatial text (indentation, column spacing) is kept as-is
- LLMs/embeddings understand ASCII layout natively

Chunking strategy:
- Extract spatial text from each page via LiteParse
- Remove page numbers, join all pages with \n (no artificial gaps)
- Split on consecutive blank lines (2+) — natural section boundaries in the layout
- Merge small sections, split oversized ones

Chunk ID format: "{filename}:{chunk_num}"
"""

import json
import re
import sys
from pathlib import Path

from liteparse import LiteParse


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


def _split_by_blank_lines(text: str, max_chars: int = 2000, min_chars: int = 200) -> list[str]:
    """Split text by consecutive blank lines (2+), then merge/split to target size."""
    # Split on 2+ consecutive blank lines (natural layout gaps)
    sections = re.split(r'\n\s*\n\s*\n', text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    buffer = ""

    def flush():
        nonlocal buffer
        if not buffer.strip():
            buffer = ""
            return
        if len(buffer.strip()) >= min_chars:
            chunks.append(buffer.strip())
        elif chunks:
            chunks[-1] += "\n\n" + buffer.strip()
        else:
            chunks.append(buffer.strip())
        buffer = ""

    for section in sections:
        if len(buffer) + len(section) > max_chars and buffer:
            flush()

        # Oversized section: split at single blank lines
        if len(section) > max_chars:
            if buffer:
                flush()
            sub_sections = re.split(r'\n\s*\n', section)
            for sub in sub_sections:
                if len(buffer) + len(sub) > max_chars and buffer:
                    flush()
                buffer += "\n\n" + sub if buffer else sub
            continue

        buffer += "\n\n" + section if buffer else section

    flush()
    return chunks


def _parse_pdf(lp: LiteParse, pdf_path: Path) -> str:
    """Extract spatial text from PDF, remove page numbers, join pages."""
    try:
        result = lp.parse(str(pdf_path))
    except Exception as e:
        print(f"  Warning: LiteParse failed for {pdf_path.name}: {e}")
        # Fallback to PyMuPDF
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            pages = [doc[i].get_text("text").strip() for i in range(len(doc))]
            doc.close()
            return "\n".join(p for p in pages if p)
        except Exception:
            return ""

    page_texts = []
    for page in result.pages:
        cleaned = _remove_page_numbers(page.text)
        # Strip trailing whitespace lines but keep internal layout
        cleaned = cleaned.rstrip()
        if cleaned.strip():
            page_texts.append(cleaned)

    return "\n".join(page_texts)


def create_chunks(pdf_dir: str, output_path: str):
    """Process all PDFs into spatial-layout chunks."""
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))

    lp = LiteParse()
    chunks = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        full_text = _parse_pdf(lp, pdf_path)
        if not full_text.strip():
            print(f"  Warning: No text extracted from {pdf_path.name}")
            continue

        title = _extract_title(full_text)
        source_name = title or pdf_path.stem

        sections = _split_by_blank_lines(full_text)

        for i, section_text in enumerate(sections, 1):
            # Extract first non-empty line as subtitle
            first_line = ""
            for line in section_text.split("\n"):
                line = line.strip()
                if line and len(line) > 3:
                    first_line = line[:60]
                    break
            # Use "title > subtitle" if subtitle differs from title
            if first_line and first_line != source_name and not first_line.startswith(source_name):
                source = f"{source_name} > {first_line}"
            else:
                source = source_name

            chunks.append({
                "id": f"{pdf_path.name}:{i}",
                "text": section_text,
                "source": source,
                "file": pdf_path.name,
                "page": i,
            })

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
