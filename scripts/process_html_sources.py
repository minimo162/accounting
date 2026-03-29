"""Download and process HTML-based accounting standards into parent/child chunks."""

import argparse
import http.client
import json
import re
import socket
import signal
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
try:
    from .source_manifest import load_source_manifest, metadata_for_file, upsert_source_record
except ImportError:
    from source_manifest import load_source_manifest, metadata_for_file, upsert_source_record


HTML_SOURCES = [
    {
        "url": "https://www.ron.gr.jp/law/etc_txt/kigyokai.htm",
        "file": "html_kigyo_kaikei_gensoku.html",
        "source": "企業会計原則",
        "doc_type": "企業会計原則",
    },
    {
        "url": "https://www.ron.gr.jp/law/etc_txt/kikaichu.htm",
        "file": "html_kigyo_kaikei_chukai.html",
        "source": "企業会計原則注解",
        "doc_type": "注解",
    },
    {
        "url": "https://www.ipc.hokusei.ac.jp/~z00153/cost_accounting_standards.pdf",
        "file": "html_genka_keisan_kijun.pdf",
        "source": "原価計算基準",
        "doc_type": "原価計算基準",
        "is_pdf": True,
    },
    {
        "url": "https://www.fsa.go.jp/status/ifrs.html",
        "file": "html_fsa_ifrs.html",
        "source": "金融庁 IFRS 関連情報",
        "doc_type": "IFRS関連情報",
    },
    {
        "url": "https://www.fsa.go.jp/news/27/sonota/20160331-5.html",
        "file": "html_fsa_ifrs_disclosure_examples.html",
        "source": "IFRSに基づく連結財務諸表の開示例",
        "doc_type": "IFRS関連情報",
    },
    {
        "url": "https://www.fsa.go.jp/news/24/sonota/20130620-2.html",
        "file": "html_fsa_ifrs_policy.html",
        "source": "IFRSへの対応のあり方に関する当面の方針",
        "doc_type": "IFRS関連情報",
    },
    {
        "url": "https://www.jcci.or.jp/file/sangyo1/202403/documents-01.pdf",
        "file": "html_chusho_kaikei_youryo.pdf",
        "source": "中小企業の会計に関する基本要領",
        "doc_type": "中小企業会計",
        "is_pdf": True,
    },
]

_LEGACY_HTML_SOURCE_FILES = {"html_chusho_kaikei_youryo_about.html"}
_USER_AGENT = "Mozilla/5.0 (compatible; AccountingBot/1.0)"
_FETCH_RETRIES = 3
_FETCH_BACKOFF_SEC = 1.5
_FETCH_TIMEOUT_SEC = 30.0


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        if tag in ("br", "p", "div", "h1", "h2", "h3", "h4", "li", "tr", "dt", "dd"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        if tag in ("p", "div", "h1", "h2", "h3", "h4", "table", "ol", "ul"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


def _extract_pdf_text(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(str(pdf_path))
    pages = [doc[i].get_text("text").strip() for i in range(len(doc))]
    doc.close()
    return "\n\n".join(page for page in pages if page)


def _raise_fetch_timeout(signum, frame):
    raise TimeoutError(f"fetch exceeded {_FETCH_TIMEOUT_SEC:.0f}s")


def _download_with_retry(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, _FETCH_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_fetch_timeout)
            signal.setitimer(signal.ITIMER_REAL, _FETCH_TIMEOUT_SEC)
            try:
                with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SEC) as resp:
                    return resp.read()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.RemoteDisconnected,
            ConnectionResetError,
        ) as exc:
            last_error = exc
            if attempt >= _FETCH_RETRIES:
                break
            sleep_for = _FETCH_BACKOFF_SEC * attempt
            print(
                f"  Retry {attempt}/{_FETCH_RETRIES - 1} after fetch error: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error


def _split_parent_sections(text: str, max_chars: int = 2000) -> list[str]:
    sections = [section.strip() for section in re.split(r"\n\s*\n", text) if section.strip()]
    parents: list[str] = []
    buffer = ""
    for section in sections:
        candidate = f"{buffer}\n\n{section}".strip() if buffer else section
        if len(candidate) > max_chars and buffer:
            parents.append(buffer.strip())
            buffer = section
        else:
            buffer = candidate
    if buffer.strip():
        parents.append(buffer.strip())
    return parents


def _split_child_chunks(text: str, target_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    children: list[str] = []
    start = 0
    step = max(target_chars - overlap_chars, 1)
    while start < len(text):
        children.append(text[start:start + target_chars].strip())
        start += step
    return [child for child in children if child]


def _build_chunks(
    text: str,
    source_name: str,
    file_name: str,
    doc_type: str,
    source_meta: dict[str, str],
) -> list[dict]:
    parents = _split_parent_sections(text)
    chunks: list[dict] = []
    child_counter = 1
    for page, parent_text in enumerate(parents, start=1):
        first_line = next((line.strip()[:80] for line in parent_text.split("\n") if line.strip()), source_name)
        source = f"{source_name} > {first_line}" if first_line and first_line != source_name else source_name
        parent_id = f"{file_name}:p{page}"
        chunks.append(
            {
                "id": parent_id,
                "text": parent_text,
                "source": source,
                "file": file_name,
                "page": page,
                "level": "parent",
                "parent_id": parent_id,
                "doc_title": source_name,
                "doc_type": doc_type,
                "standard_no": "",
                "section_title": first_line,
                "section_path": source,
                **source_meta,
            }
        )
        for child_text in _split_child_chunks(parent_text):
            chunks.append(
                {
                    "id": f"{file_name}:c{child_counter}",
                    "text": child_text,
                    "source": source,
                    "file": file_name,
                    "page": page,
                    "level": "child",
                    "parent_id": parent_id,
                    "doc_title": source_name,
                    "doc_type": doc_type,
                    "standard_no": "",
                    "section_title": first_line,
                    "section_path": source,
                    **source_meta,
                }
            )
            child_counter += 1
    return chunks


def main(chunks_path: str = "data/chunks.json", *, refresh: bool = False):
    chunks_file = Path(chunks_path)
    data_dir = chunks_file.resolve().parent
    cache_dir = data_dir / "html_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = json.loads(chunks_file.read_text(encoding="utf-8")) if chunks_file.exists() else []
    html_files = {src["file"] for src in HTML_SOURCES} | _LEGACY_HTML_SOURCE_FILES
    existing_chunks = [chunk for chunk in existing_chunks if chunk.get("file") not in html_files]
    manifest_file = data_dir / "source_manifest.json"
    manifest_sources = load_source_manifest(manifest_file)["sources"]

    for legacy_file in _LEGACY_HTML_SOURCE_FILES:
        legacy_path = cache_dir / legacy_file
        if legacy_path.exists():
            legacy_path.unlink()

    new_chunks: list[dict] = []
    for src in HTML_SOURCES:
        cache_path = cache_dir / src["file"]
        if refresh or not cache_path.exists():
            print(f"Downloading: {src['url']}", flush=True)
            try:
                raw = _download_with_retry(src["url"])
            except Exception:
                if cache_path.exists():
                    print(f"  Warning: fetch failed, using cached copy: {cache_path.name}", flush=True)
                else:
                    raise
            else:
                if src.get("is_pdf"):
                    cache_path.write_bytes(raw)
                else:
                    html = raw.decode("utf-8", errors="replace")
                    cache_path.write_text(html, encoding="utf-8")
        record = upsert_source_record(
            manifest_file,
            file_path=cache_path.resolve(),
            data_dir=data_dir,
            url=src["url"],
            source_group="html_cache",
            kind="pdf" if src.get("is_pdf") else "html",
        )
        manifest_sources[record["path"]] = record
        if src.get("is_pdf"):
            text = _extract_pdf_text(cache_path)
        else:
            text = html_to_text(cache_path.read_text(encoding="utf-8"))
        source_meta = metadata_for_file(
            cache_path.resolve(),
            data_dir=data_dir,
            manifest_sources=manifest_sources,
        )
        built = _build_chunks(text, src["source"], src["file"], src["doc_type"], source_meta)
        new_chunks.extend(built)
        print(f"  {src['source']}: {len(built)} chunks", flush=True)

    all_chunks = existing_chunks + new_chunks
    chunks_file.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total chunks: {len(all_chunks)} ({len(new_chunks)} from HTML sources)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks_path", nargs="?", default="data/chunks.json")
    parser.add_argument("--refresh", action="store_true", help="Refresh cached HTML/PDF sources before chunking.")
    args = parser.parse_args()
    main(args.chunks_path, refresh=args.refresh)
