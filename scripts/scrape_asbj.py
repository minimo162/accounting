"""Scrape ASBJ website for accounting standard PDF links and download them."""

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx


BASE_URL = "https://www.asb-j.jp"

# ASBJ standard listing pages (actual URLs)
LISTING_PAGES = [
    "/jp/accounting_standards.html",           # 会計基準一覧（現行）
    "/jp/completed_accounting_standards.html",  # 会計基準一覧（公表済み）
]


async def fetch_page(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def extract_pdf_links(html: str, base_url: str) -> list[dict]:
    """Extract PDF links with context (surrounding text for title hints)."""
    pdf_pattern = re.compile(r'href=["\']([^"\']*\.pdf)["\']', re.IGNORECASE)
    links = []
    seen = set()

    for match in pdf_pattern.finditer(html):
        href = match.group(1)
        full_url = urljoin(base_url, href)
        if full_url not in seen:
            seen.add(full_url)
            filename = Path(href).name
            links.append({"url": full_url, "filename": filename})

    return links


async def download_pdf(client: httpx.AsyncClient, url: str, output_dir: Path, filename: str) -> bool:
    output_path = output_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skip (exists): {filename}")
        return True

    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  Downloaded: {filename} ({size_kb:.0f} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {filename} - {e}")
        return False


async def main(output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_pdfs: list[dict] = []

    async with httpx.AsyncClient(
        timeout=120,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AccountingBot/1.0)"},
    ) as client:
        for page_path in LISTING_PAGES:
            url = BASE_URL + page_path
            print(f"\nFetching: {url}")
            try:
                html = await fetch_page(client, url)
                pdfs = extract_pdf_links(html, url)
                print(f"  Found {len(pdfs)} PDF links")
                all_pdfs.extend(pdfs)
            except Exception as e:
                print(f"  Error: {e}")

        # Deduplicate
        seen_urls = set()
        unique_pdfs = []
        for pdf in all_pdfs:
            if pdf["url"] not in seen_urls:
                seen_urls.add(pdf["url"])
                unique_pdfs.append(pdf)

        print(f"\nTotal unique PDFs: {len(unique_pdfs)}")
        print(f"Downloading to: {output_path}")

        # Download in batches
        success = 0
        batch_size = 5
        for i in range(0, len(unique_pdfs), batch_size):
            batch = unique_pdfs[i : i + batch_size]
            results = await asyncio.gather(*[
                download_pdf(client, pdf["url"], output_path, pdf["filename"])
                for pdf in batch
            ])
            success += sum(results)

    print(f"\nDone! {success}/{len(unique_pdfs)} PDFs downloaded to {output_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    asyncio.run(main(output_dir))
