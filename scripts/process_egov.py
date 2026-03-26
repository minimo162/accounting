"""Process e-Gov XML law data into chunks and append to chunks.json.

Downloads law XML from e-Gov API and chunks by article groups (~2000 chars each).
Chunk ID format: "{source_name}:{page}" matching the file-based ID convention.
"""

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
        # PDFs containing the same law — these are skipped in process_pdfs.py
        "supersedes_pdfs": ["reg_zaimuhyou_latest.pdf"],
    },
    "351M50000040028": {
        "source": "連結財務諸表の用語、様式及び作成方法に関する規則（連結財規）",
        "file": "egov_renketsu.xml",
        "supersedes_pdfs": ["reg_renketsu_latest.pdf"],
    },
    "352M50000040038": {
        "source": "中間財務諸表等の用語、様式及び作成方法に関する規則（中間財規）",
        "file": "egov_chukan.xml",
        "supersedes_pdfs": [],
    },
}


def get_egov_covered_pdfs() -> set[str]:
    """e-Gov XML でカバーされるため PDF 処理をスキップすべきファイル名の集合を返す。"""
    return {pdf for info in LAWS.values() for pdf in info.get("supersedes_pdfs", [])}


def extract_text(elem) -> list[str]:
    """Recursively extract all text from an XML element."""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.extend(extract_text(child))
        if child.tail:
            parts.append(child.tail)
    return parts


def clean_text(text: str) -> str:
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def download_law(law_id: str, cache_dir: Path) -> Path:
    """Download law XML from e-Gov API (cached)."""
    import urllib.request
    cache_path = cache_dir / f"{law_id}.xml"
    if cache_path.exists():
        print(f"  Using cached: {cache_path}")
        return cache_path
    url = f"https://laws.e-gov.go.jp/api/1/lawdata/{law_id}"
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, str(cache_path))
    return cache_path


def process_law(xml_path: Path, source_name: str, file_name: str) -> list[dict]:
    """Parse law XML and create chunks grouped by articles (~2000 chars)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    law = root.find('.//Law')
    law_body = law.find('LawBody')
    main_provision = law_body.find('MainProvision')
    articles = main_provision.findall('.//Article')

    chunks = []
    page_counter = 1
    buffer_text = ""

    def flush():
        nonlocal buffer_text, page_counter
        if buffer_text.strip():
            chunks.append({
                "id": f"{file_name}:{page_counter}",
                "text": buffer_text.strip(),
                "source": f"{source_name} (p.{page_counter})",
                "file": file_name,
                "page": page_counter,
            })
            page_counter += 1
            buffer_text = ""

    for art in articles:
        art_text = clean_text(''.join(extract_text(art)))
        if len(buffer_text) + len(art_text) > 2000 and buffer_text:
            flush()
        buffer_text += "\n\n" + art_text if buffer_text else art_text

    flush()

    # Recent supplementary provisions
    for sp in law_body.findall('SupplProvision')[-3:]:
        sp_text = clean_text(''.join(extract_text(sp)))
        if len(sp_text) > 100:
            while sp_text:
                chunk_text = sp_text[:2000]
                sp_text = sp_text[2000:]
                chunks.append({
                    "id": f"{file_name}:{page_counter}",
                    "text": chunk_text.strip(),
                    "source": f"{source_name} 附則 (p.{page_counter})",
                    "file": file_name,
                    "page": page_counter,
                })
                page_counter += 1

    # Add overlap: prepend tail of previous chunk to each chunk
    prev_tail = ""
    for chunk in chunks:
        if prev_tail:
            chunk["text"] = prev_tail + "\n...\n" + chunk["text"]
        prev_tail = chunk["text"][-_OVERLAP_CHARS:]

    return chunks


def main(chunks_path: str = "data/chunks.json"):
    cache_dir = Path("data/egov_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load existing chunks
    chunks_file = Path(chunks_path)
    if chunks_file.exists():
        with open(chunks_file, encoding="utf-8") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    # Remove any previously generated e-Gov chunks
    egov_files = {info["file"] for info in LAWS.values()}
    existing_chunks = [c for c in existing_chunks if c.get("file") not in egov_files]

    # Process each law
    new_chunks = []
    for law_id, info in LAWS.items():
        print(f"Processing: {info['source']}")
        xml_path = download_law(law_id, cache_dir)
        chunks = process_law(xml_path, info["source"], info["file"])
        new_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks")

    all_chunks = existing_chunks + new_chunks
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Total chunks: {len(all_chunks)} ({len(new_chunks)} from e-Gov)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    main(path)
