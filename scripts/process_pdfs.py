"""Process downloaded PDFs into chunks.json for A-RAG indexing."""

import json
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Install PyMuPDF: uv add pymupdf")
    sys.exit(1)


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text.strip(),
            })
    doc.close()
    return pages


def create_chunks(pdf_dir: str, output_path: str):
    """Process all PDFs in a directory into chunks.json."""
    pdf_dir = Path(pdf_dir)
    chunks = []
    chunk_id = 0

    for pdf_path in sorted(pdf_dir.glob("**/*.pdf")):
        print(f"Processing: {pdf_path.name}")
        pages = extract_pages(pdf_path)

        for page in pages:
            chunks.append({
                "id": str(chunk_id),
                "text": page["text"],
                "source": f"{pdf_path.stem} (p.{page['page']})",
                "file": pdf_path.name,
                "page": page["page"],
            })
            chunk_id += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(chunks)} chunks from {len(list(pdf_dir.glob('**/*.pdf')))} PDFs")
    print(f"Output: {output}")


if __name__ == "__main__":
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/chunks.json"
    create_chunks(pdf_dir, output)
