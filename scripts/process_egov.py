"""Process e-Gov XML law data into parent/child chunks and append to chunks.json."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


_OVERLAP_CHARS = 100


LAWS = {
    "338M50000040059": {
        "source": "財務諸表等の用語、様式及び作成方法に関する規則（財規）",
        "file": "egov_zaireg.xml",
        "supersedes_pdfs": ["reg_zaimuhyou_latest.pdf"],
    },
    "418M60000010013": {"source": "会社計算規則", "file": "egov_kaisha_keisan.xml", "supersedes_pdfs": []},
    "417AC0000000086": {"source": "会社法", "file": "egov_kaishaho.xml", "parts": ["第二編第五章"], "supersedes_pdfs": []},
    "351M50000040028": {
        "source": "連結財務諸表の用語、様式及び作成方法に関する規則（連結財規）",
        "file": "egov_renketsu.xml",
        "supersedes_pdfs": ["reg_renketsu_latest.pdf"],
    },
    "352M50000040038": {"source": "中間財務諸表等の用語、様式及び作成方法に関する規則（中間財規）", "file": "egov_chukan.xml", "supersedes_pdfs": []},
    "419M60000002064": {"source": "四半期連結財務諸表の用語、様式及び作成方法に関する規則（四半期財規）", "file": "egov_shihanki_zaireg.xml", "supersedes_pdfs": []},
    "348M50000040005": {"source": "企業内容等の開示に関する内閣府令（開示府令）", "file": "egov_kaiji_furei.xml", "supersedes_pdfs": []},
    "332M50000040012": {"source": "財務諸表等の監査証明に関する内閣府令（監査証明府令）", "file": "egov_kansa_furei.xml", "supersedes_pdfs": []},
    "419M60000002062": {"source": "財務計算に関する書類その他の情報の適正性を確保するための体制に関する内閣府令（内部統制府令）", "file": "egov_naibu_tosei_furei.xml", "supersedes_pdfs": []},
}


def get_egov_covered_pdfs() -> set[str]:
    return {pdf for info in LAWS.values() for pdf in info.get("supersedes_pdfs", [])}


def extract_text(elem) -> list[str]:
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.extend(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return parts


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


def download_law(law_id: str, cache_dir: Path) -> Path:
    import urllib.request

    cache_path = cache_dir / f"{law_id}.xml"
    if cache_path.exists():
        print(f"  Using cached: {cache_path}")
        return cache_path
    url = f"https://laws.e-gov.go.jp/api/1/lawdata/{law_id}"
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, str(cache_path))
    return cache_path


def _find_chapter(main_provision, chapter_path: str):
    parts = re.findall(r"(第[一二三四五六七八九十百]+(?:編|章|節))", chapter_path)
    elem = main_provision
    for part_title in parts:
        found = False
        for child in elem:
            for tag in ("Title", "PartTitle", "ChapterTitle", "SectionTitle"):
                title_elem = child.find(tag)
                if title_elem is not None and title_elem.text and part_title in title_elem.text:
                    elem = child
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  Warning: Could not find '{part_title}' in {chapter_path}")
            return None
    return elem


def _split_child_chunks(text: str, target_chars: int = 700, overlap_chars: int = 80) -> list[str]:
    if len(text) <= target_chars:
        return [text]
    step = max(target_chars - overlap_chars, 1)
    return [text[i:i + target_chars].strip() for i in range(0, len(text), step) if text[i:i + target_chars].strip()]


def process_law(xml_path: Path, source_name: str, file_name: str, parts: list[str] | None = None) -> list[dict]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    law = root.find(".//Law")
    law_body = law.find("LawBody")
    main_provision = law_body.find("MainProvision")

    if parts:
        articles = []
        for part_path in parts:
            section = _find_chapter(main_provision, part_path)
            if section is not None:
                articles.extend(section.findall(".//Article"))
    else:
        articles = main_provision.findall(".//Article")

    chunks: list[dict] = []
    page_counter = 1
    child_counter = 1
    buffer: list[tuple[str, str]] = []
    buffer_len = 0

    def flush():
        nonlocal page_counter, child_counter, buffer, buffer_len
        if not buffer:
            return
        section_titles = [title for title, _ in buffer if title]
        section_title = " / ".join(section_titles[:2]) if section_titles else source_name
        parent_text = "\n\n".join(text for _, text in buffer).strip()
        parent_id = f"{file_name}:p{page_counter}"
        source = f"{source_name} > {section_title}" if section_title != source_name else source_name
        parent = {
            "id": parent_id,
            "text": parent_text,
            "source": source,
            "file": file_name,
            "page": page_counter,
            "level": "parent",
            "parent_id": parent_id,
            "doc_title": source_name,
            "doc_type": "法令",
            "standard_no": "",
            "section_title": section_title,
            "section_path": source,
        }
        chunks.append(parent)
        for child_text in _split_child_chunks(parent_text):
            chunks.append(
                {
                    "id": f"{file_name}:c{child_counter}",
                    "text": child_text,
                    "source": source,
                    "file": file_name,
                    "page": page_counter,
                    "level": "child",
                    "parent_id": parent_id,
                    "doc_title": source_name,
                    "doc_type": "法令",
                    "standard_no": "",
                    "section_title": section_title,
                    "section_path": source,
                }
            )
            child_counter += 1
        page_counter += 1
        buffer = []
        buffer_len = 0

    for art in articles:
        article_title = clean_text("".join(extract_text(art.find("ArticleTitle")))) if art.find("ArticleTitle") is not None else ""
        art_text = clean_text("".join(extract_text(art)))
        if buffer_len + len(art_text) > 2200 and buffer:
            flush()
        buffer.append((article_title, art_text))
        buffer_len += len(art_text)

    flush()

    prev_tail = ""
    for chunk in chunks:
        if prev_tail:
            chunk["overlap_prefix"] = prev_tail
        prev_tail = chunk["text"][-_OVERLAP_CHARS:]

    return chunks


def main(chunks_path: str = "data/chunks.json"):
    cache_dir = Path("data/egov_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_file = Path(chunks_path)
    existing_chunks = json.loads(chunks_file.read_text(encoding="utf-8")) if chunks_file.exists() else []
    egov_files = {info["file"] for info in LAWS.values()}
    existing_chunks = [chunk for chunk in existing_chunks if chunk.get("file") not in egov_files]

    new_chunks: list[dict] = []
    for law_id, info in LAWS.items():
        print(f"Processing: {info['source']}")
        xml_path = download_law(law_id, cache_dir)
        chunks = process_law(xml_path, info["source"], info["file"], parts=info.get("parts"))
        new_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks")

    all_chunks = existing_chunks + new_chunks
    chunks_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total chunks: {len(all_chunks)} ({len(new_chunks)} from e-Gov)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    main(path)
