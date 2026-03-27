"""Process PDFs into parent/child chunks using LiteParse spatial extraction."""

import json
import re
import sys
from pathlib import Path

from liteparse import LiteParse

sys.path.insert(0, str(Path(__file__).parent))
from process_egov import get_egov_covered_pdfs


_PAGE_NUM_RE = re.compile(r"(?:^|\n)\s*-\s*\n\s*\d+\s*\n\s*-\s*(?:\n|$)", re.MULTILINE)
_PAGE_NUM_INLINE_RE = re.compile(r"^\s*-\s*\d+\s*-\s*$", re.MULTILINE)
_STANDARD_NO_RE = re.compile(r"((?:企業会計基準|適用指針|実務対応報告|会計基準|監査基準)[^\n]{0,20}?第\s*\d+\s*号)")
_OCR_JUNK_LINE_RE = re.compile(r"^[\s]*(?:[a-zA-Z]{1,3}[\s]+){1,}[a-zA-Z]{0,3}[\s]*$", re.MULTILINE)
_OVERLAP_CHARS = 100


def _remove_page_numbers(text: str) -> str:
    text = _PAGE_NUM_RE.sub("\n", text)
    text = _PAGE_NUM_INLINE_RE.sub("", text)
    return text


def _clean_ocr_artifacts(text: str) -> str:
    text = _OCR_JUNK_LINE_RE.sub("", text)
    text = re.sub(r"\n\s*\n\s*\n(\s*\n)*", "\n\n\n", text)
    return text


def _extract_title(text: str) -> str:
    std_patterns = [
        "企業会計基準", "適用指針", "実務対応報告", "監査基準", "中間監査", "期中レビュー",
        "内部統制", "サステナビリティ", "財務諸表", "連結", "減損", "退職給付", "リース",
        "税効果", "金融商品",
    ]
    for line in text.split("\n")[:30]:
        line = line.strip()
        if line and any(pattern in line for pattern in std_patterns):
            return line
    for line in text.split("\n")[:30]:
        line = line.strip()
        if line and len(line) >= 6 and not re.match(r"^(平成|令和|昭和|\d{4}\s*年)", line):
            return line
    return ""


def _infer_doc_type(title: str) -> str:
    for token in ("企業会計基準", "適用指針", "実務対応報告", "監査基準", "四半期", "連結", "財務諸表", "会社計算規則"):
        if token in title:
            return token
    return "PDF"


def _extract_standard_no(title: str) -> str:
    match = _STANDARD_NO_RE.search(title)
    return match.group(1).replace(" ", "") if match else ""


def _split_parent_sections(text: str, max_chars: int = 2000, min_chars: int = 200) -> list[str]:
    sections = re.split(r"\n\s*\n\s*\n", text)
    sections = [section.strip() for section in sections if section.strip()]
    parents: list[str] = []
    buffer = ""

    def flush():
        nonlocal buffer
        if not buffer.strip():
            buffer = ""
            return
        if len(buffer.strip()) >= min_chars:
            parents.append(buffer.strip())
        elif parents:
            parents[-1] += "\n\n" + buffer.strip()
        else:
            parents.append(buffer.strip())
        buffer = ""

    for section in sections:
        if len(section) > max_chars:
            if buffer:
                flush()
            sub_sections = re.split(r"\n\s*\n", section)
            for sub in sub_sections:
                if len(buffer) + len(sub) > max_chars and buffer:
                    flush()
                buffer += "\n\n" + sub if buffer else sub
            continue
        if len(buffer) + len(section) > max_chars and buffer:
            flush()
        buffer += "\n\n" + section if buffer else section

    flush()
    return parents


def _split_child_chunks(text: str, target_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if not paragraphs:
        return [text.strip()]

    children: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        segments = [paragraph]
        if len(paragraph) > target_chars * 1.3:
            step = max(target_chars - overlap_chars, 1)
            segments = [paragraph[i:i + target_chars] for i in range(0, len(paragraph), step)]
        for segment in segments:
            candidate = f"{buffer}\n\n{segment}".strip() if buffer else segment
            if len(candidate) <= target_chars * 1.2:
                buffer = candidate
            else:
                if buffer:
                    children.append(buffer.strip())
                    tail = buffer[-overlap_chars:].strip()
                    buffer = f"{tail}\n\n{segment}".strip() if tail else segment
                else:
                    children.append(segment.strip())
                    buffer = ""
    if buffer.strip():
        children.append(buffer.strip())
    return children


def _fitz_extract(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(str(pdf_path))
    pages = [doc[i].get_text("text").strip() for i in range(len(doc))]
    doc.close()
    return "\n".join(page for page in pages if page)


def _parse_pdf(lp: LiteParse, pdf_path: Path) -> str:
    try:
        result = lp.parse(str(pdf_path))
        page_texts = []
        for page in result.pages:
            if page.textItems and all(item.fontName == "OCR" for item in page.textItems):
                continue
            cleaned = _clean_ocr_artifacts(_remove_page_numbers(page.text)).rstrip()
            if cleaned.strip():
                page_texts.append(cleaned)
        text = "\n".join(page_texts)
        if text.strip():
            return text
    except Exception as e:
        print(f"  Warning: LiteParse failed for {pdf_path.name}: {e}")
    try:
        text = _fitz_extract(pdf_path)
        if text.strip():
            print(f"  Note: LiteParse empty, using fitz for {pdf_path.name}")
            return text
    except Exception:
        pass
    return ""


def create_chunks(pdf_dir: str, output_path: str):
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    egov_covered = get_egov_covered_pdfs()
    lp = LiteParse()
    chunks: list[dict] = []

    for pdf_path in pdf_files:
        if pdf_path.name in egov_covered:
            print(f"Skipping (covered by e-Gov XML): {pdf_path.name}")
            continue
        print(f"Processing: {pdf_path.name}")
        full_text = _parse_pdf(lp, pdf_path)
        if not full_text.strip():
            print(f"  Warning: No text extracted from {pdf_path.name}")
            continue

        doc_title = _extract_title(full_text) or pdf_path.stem
        doc_type = _infer_doc_type(doc_title)
        standard_no = _extract_standard_no(doc_title)
        parent_sections = _split_parent_sections(full_text)
        child_counter = 1
        file_chunks: list[dict] = []

        for parent_idx, section_text in enumerate(parent_sections, 1):
            first_line = ""
            for line in section_text.split("\n"):
                line = line.strip()
                if line and len(line) > 3:
                    first_line = line[:80]
                    break
            source = f"{doc_title} > {first_line}" if first_line and first_line != doc_title else doc_title
            parent_id = f"{pdf_path.name}:p{parent_idx}"
            parent_chunk = {
                "id": parent_id,
                "text": section_text,
                "source": source,
                "file": pdf_path.name,
                "page": parent_idx,
                "level": "parent",
                "parent_id": parent_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "standard_no": standard_no,
                "section_title": first_line,
                "section_path": source,
            }
            file_chunks.append(parent_chunk)

            for child_text in _split_child_chunks(section_text):
                file_chunks.append(
                    {
                        "id": f"{pdf_path.name}:c{child_counter}",
                        "text": child_text,
                        "source": source,
                        "file": pdf_path.name,
                        "page": parent_idx,
                        "level": "child",
                        "parent_id": parent_id,
                        "doc_title": doc_title,
                        "doc_type": doc_type,
                        "standard_no": standard_no,
                        "section_title": first_line,
                        "section_path": source,
                    }
                )
                child_counter += 1

        prev_tail = ""
        for chunk in file_chunks:
            if prev_tail:
                chunk["overlap_prefix"] = prev_tail
            prev_tail = chunk["text"][-_OVERLAP_CHARS:]

        # Skip files where all remaining chunks are garbage (single tiny chunk)
        if len(file_chunks) == 1 and len(file_chunks[0]["text"]) < 150:
            print(f"  Skipping (garbage-only file): {pdf_path.name}")
            continue
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
