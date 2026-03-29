"""Process PDFs into parent/child chunks using LiteParse spatial extraction."""

import json
import re
import sys
import unicodedata
from pathlib import Path

from liteparse import LiteParse

sys.path.insert(0, str(Path(__file__).parent))
try:
    from .process_egov import get_egov_covered_pdfs
    from .source_manifest import load_source_manifest, load_url_lookup, metadata_for_file
except ImportError:
    from process_egov import get_egov_covered_pdfs
    from source_manifest import load_source_manifest, load_url_lookup, metadata_for_file


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
        "企業会計基準", "適用指針", "実務対応報告", "移管指針", "監査基準", "中間監査", "期中レビュー",
        "内部統制", "サステナビリティ", "財務諸表", "連結", "減損", "退職給付", "リース",
        "税効果", "金融商品",
    ]
    _org_only = re.compile(r'^(企業会計基準委員会|日本公認会計士協会|金融庁|財務会計基準機構)\s*$')
    lines = [l.strip() for l in text.split("\n")[:150]]
    # Prefer lines that contain both a standard keyword AND a number (e.g. 第13号)
    for line in lines:
        if line and any(p in line for p in std_patterns) and re.search(r"第\s*\d+\s*号", line):
            return line
    # Fall back to first line with a standard keyword, skipping pure org-name lines
    for line in lines:
        if line and any(p in line for p in std_patterns) and not _org_only.match(line):
            return line
    # Final fallback: first non-trivial, non-org line
    for line in lines:
        if line and len(line) >= 6 and not _org_only.match(line) and not re.match(r"^(平成|令和|昭和|\d{4}\s*年)", line):
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


_PAGE_MARKER_RE = re.compile(r"\x01PAGE:(\d+)\x01\n?")


def _parse_pdf_with_pages(lp: LiteParse, pdf_path: Path) -> str:
    """Returns full text with embedded page markers \x01PAGE:N\x01 at each page boundary."""
    def _build(page_entries: list[tuple[int, str]]) -> str:
        parts = [f"\x01PAGE:{p}\x01\n{text}" for p, text in page_entries]
        return "\n".join(parts)

    try:
        result = lp.parse(str(pdf_path))
        page_entries: list[tuple[int, str]] = []
        for phys_page, page in enumerate(result.pages, 1):
            if page.textItems and all(item.fontName == "OCR" for item in page.textItems):
                continue
            cleaned = _clean_ocr_artifacts(_remove_page_numbers(page.text)).rstrip()
            if cleaned.strip():
                page_entries.append((phys_page, cleaned))
        if page_entries:
            return _build(page_entries)
    except Exception as e:
        print(f"  Warning: LiteParse failed for {pdf_path.name}: {e}")
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page_entries = []
        for i in range(len(doc)):
            pt = doc[i].get_text("text").strip()
            if pt:
                page_entries.append((i + 1, pt))
        doc.close()
        if page_entries:
            print(f"  Note: LiteParse empty, using fitz for {pdf_path.name}")
            return _build(page_entries)
    except Exception:
        pass
    return ""


def _extract_pdf_page(section_text: str) -> int:
    """Extract physical page number from the first page marker in the section."""
    m = _PAGE_MARKER_RE.search(section_text)
    return int(m.group(1)) if m else 1


def _strip_page_markers(text: str) -> str:
    """Remove all page markers from text."""
    return _PAGE_MARKER_RE.sub("", text)


_DATE_IN_FILENAME_RE = re.compile(r'(\d{8})')


def _date_from_filename(name: str) -> str:
    """Extract YYYYMMDD date string from filename, or '00000000' if none found."""
    m = _DATE_IN_FILENAME_RE.search(name)
    return m.group(1) if m else "00000000"


def _dedup_pdf_files(pdf_files: list[Path], lp: LiteParse) -> list[Path]:
    """Return only the newest file per doc_title, skipping older duplicates."""
    # Sort newest-first so the first file seen for each title wins
    sorted_files = sorted(pdf_files, key=lambda p: _date_from_filename(p.name), reverse=True)
    seen_titles: dict[str, Path] = {}
    kept: set[Path] = set()
    for pdf_path in sorted_files:
        try:
            text = _strip_page_markers(_parse_pdf_with_pages(lp, pdf_path))
        except Exception:
            kept.add(pdf_path)
            continue
        title = unicodedata.normalize("NFKC", re.sub(r"\s+", " ", _extract_title(text))).strip()
        if not title:
            kept.add(pdf_path)
            continue
        if title in seen_titles:
            newer = seen_titles[title]
            print(f"  Dedup: skipping {pdf_path.name} (superseded by {newer.name}, title: {title[:50]})")
        else:
            seen_titles[title] = pdf_path
            kept.add(pdf_path)
    return [p for p in pdf_files if p in kept]


def create_chunks(pdf_dir: str, output_path: str):
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    egov_covered = get_egov_covered_pdfs()
    lp = LiteParse()
    chunks: list[dict] = []
    data_dir = Path(output_path).resolve().parent
    manifest_sources = load_source_manifest(data_dir / "source_manifest.json")["sources"]
    url_lookup = load_url_lookup(data_dir / "pdf_sources.json")

    print("=== Deduplicating PDFs by title (keeping newest version) ===")
    pdf_files = _dedup_pdf_files([p for p in pdf_files if p.name not in egov_covered], lp)
    print(f"=== {len(pdf_files)} PDFs after deduplication ===\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        full_text = _parse_pdf_with_pages(lp, pdf_path)
        if not full_text.strip():
            print(f"  Warning: No text extracted from {pdf_path.name}")
            continue

        clean_full_text = _strip_page_markers(full_text)
        doc_title = _extract_title(clean_full_text) or pdf_path.stem
        doc_type = _infer_doc_type(doc_title)
        standard_no = _extract_standard_no(doc_title)
        source_meta = metadata_for_file(
            pdf_path.resolve(),
            data_dir=data_dir,
            manifest_sources=manifest_sources,
            url_lookup=url_lookup,
        )
        parent_sections = _split_parent_sections(full_text)
        child_counter = 1
        file_chunks: list[dict] = []
        current_pdf_page = 1  # tracks physical page as we walk sections in order

        for parent_idx, section_text in enumerate(parent_sections, 1):
            # If this section begins with a page marker, we've crossed a page boundary
            leading = section_text.lstrip("\n ")
            m_lead = _PAGE_MARKER_RE.match(leading)
            if m_lead:
                current_pdf_page = int(m_lead.group(1))
            pdf_page = current_pdf_page
            # After this section, inherit the LAST page marker seen (for next section)
            all_pages = [int(x) for x in _PAGE_MARKER_RE.findall(section_text)]
            if all_pages:
                current_pdf_page = all_pages[-1]
            clean_section = _strip_page_markers(section_text)
            first_line = ""
            for line in clean_section.split("\n"):
                line = line.strip()
                if line and len(line) > 3:
                    first_line = line[:80]
                    break
            source = f"{doc_title} > {first_line}" if first_line and first_line != doc_title else doc_title
            parent_id = f"{pdf_path.name}:p{parent_idx}"
            parent_chunk = {
                "id": parent_id,
                "text": clean_section,
                "source": source,
                "file": pdf_path.name,
                "page": parent_idx,
                "pdf_page": pdf_page,
                "level": "parent",
                "parent_id": parent_id,
                "doc_title": doc_title,
                "doc_type": doc_type,
                "standard_no": standard_no,
                "section_title": first_line,
                "section_path": source,
                **source_meta,
            }
            file_chunks.append(parent_chunk)

            for child_text in _split_child_chunks(clean_section):
                file_chunks.append(
                    {
                        "id": f"{pdf_path.name}:c{child_counter}",
                        "text": child_text,
                        "source": source,
                        "file": pdf_path.name,
                        "page": parent_idx,
                        "pdf_page": pdf_page,
                        "level": "child",
                        "parent_id": parent_id,
                        "doc_title": doc_title,
                        "doc_type": doc_type,
                        "standard_no": standard_no,
                        "section_title": first_line,
                        "section_path": source,
                        **source_meta,
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
