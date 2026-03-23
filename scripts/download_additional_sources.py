"""Download PDFs from additional accounting knowledge sources."""

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx


async def download_pdf(client: httpx.AsyncClient, url: str, output_path: Path) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  Skip (exists): {output_path.name}")
        return True
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        print(f"  Downloaded: {output_path.name} ({len(resp.content) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {output_path.name} - {e}")
        return False


async def scrape_and_download(client: httpx.AsyncClient, page_url: str, output_dir: Path) -> int:
    """Scrape a page for PDF links and download them."""
    try:
        resp = await client.get(page_url, follow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"  Error fetching {page_url}: {e}")
        return 0

    pdf_pattern = re.compile(r'href=["\']([^"\']*\.pdf)["\']', re.IGNORECASE)
    urls = set()
    for match in pdf_pattern.finditer(html):
        href = match.group(1)
        full_url = urljoin(page_url, href)
        urls.add(full_url)

    count = 0
    for url in urls:
        filename = Path(url).name
        if await download_pdf(client, url, output_dir / filename):
            count += 1
    return count


# ============================================================
# 1. 企業会計審議会 (Business Accounting Council) via FSA
# ============================================================
BAC_PDFS = [
    # 固定資産の減損に係る会計基準 (2002)
    ("https://www.fsa.go.jp/news/newsj/14/singi/f-20020809-1/f-20020809c.pdf", "bac_genson_kijun.pdf"),
    # 固定資産の減損に係る会計基準の適用指針 (企業会計審議会)
    ("https://www.fsa.go.jp/news/newsj/14/singi/f-20020809-1/f-20020809b.pdf", "bac_genson_iken.pdf"),
    # 退職給付に係る会計基準 (1998, 企業会計審議会)
    ("https://www.fsa.go.jp/p_mof/singikai/kaikei/tosin/1a909e.htm", "bac_taishoku_kijun.htm"),
    # 税効果会計に係る会計基準 (1998)
    ("https://www.fsa.go.jp/p_mof/singikai/kaikei/tosin/1a906e2.htm", "bac_zeikouka_kijun.htm"),
    # 外貨建取引等会計処理基準
    ("https://www.fsa.go.jp/p_mof/singikai/kaikei/tosin/1a903e2.htm", "bac_gaika_kijun.htm"),
    # 連結財務諸表に関する会計基準 (企業会計審議会原則)
    ("https://www.fsa.go.jp/p_mof/singikai/kaikei/tosin/1a905e.htm", "bac_renketsu_gensoku.htm"),
    # 研究開発費等に係る会計基準 (1998)
    ("https://www.fsa.go.jp/p_mof/singikai/kaikei/tosin/1a909e2.htm", "bac_kenkyu_kijun.htm"),
]

# FSA singi pages that may contain PDF links
BAC_PAGES = [
    "https://www.fsa.go.jp/singi/singi_kigyou/top.html",
]


# ============================================================
# 2. JICPA 実務指針
# ============================================================
JICPA_PDFS = [
    # 金融商品会計に関する実務指針
    ("https://jicpa.or.jp/specialized_field/publication/files/2-11-14-2-20150416.pdf", "jicpa_kinyushohin_jitsumu.pdf"),
    # 金融商品会計に関するQ&A
    ("https://jicpa.or.jp/specialized_field/publication/files/2-11-0-2-20150416.pdf", "jicpa_kinyushohin_qa.pdf"),
]

JICPA_PAGES = [
    "https://jicpa.or.jp/specialized_field/publication/practical_guidelines/",
]


# ============================================================
# 3. SSBJ (サステナビリティ基準委員会)
# ============================================================
SSBJ_PAGES = [
    "https://www.ssb-j.jp/jp/ssbj_standards.html",
    "https://www.ssb-j.jp/jp/ssbj_standards/2025-0305.html",
]


# ============================================================
# 4. 財務諸表等規則
# ============================================================
REGULATIONS_PDFS = [
    # 財務諸表等規則ガイドライン (金融庁)
    ("https://www.fsa.go.jp/common/law/kaiji/1.pdf", "reg_zaimuhyou_guideline.pdf"),
]

REGULATIONS_PAGES = [
    "https://www.fsa.go.jp/common/law/kaiji.html",
]


async def main():
    base_dir = Path("data/pdfs")

    async with httpx.AsyncClient(
        timeout=120,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AccountingBot/1.0)"},
    ) as client:
        total = 0

        # 1. BAC (企業会計審議会)
        print("\n=== 1. 企業会計審議会 (BAC) ===")
        bac_dir = base_dir / "bac"
        bac_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in BAC_PDFS:
            if await download_pdf(client, url, bac_dir / filename):
                total += 1
        for page in BAC_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, bac_dir)

        # 2. JICPA
        print("\n=== 2. JICPA 実務指針 ===")
        jicpa_dir = base_dir / "jicpa"
        jicpa_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in JICPA_PDFS:
            if await download_pdf(client, url, jicpa_dir / filename):
                total += 1
        for page in JICPA_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, jicpa_dir)

        # 3. SSBJ
        print("\n=== 3. SSBJ ===")
        ssbj_dir = base_dir / "ssbj"
        ssbj_dir.mkdir(parents=True, exist_ok=True)
        for page in SSBJ_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, ssbj_dir)

        # 4. 財務諸表等規則
        print("\n=== 4. 財務諸表等規則 ===")
        reg_dir = base_dir / "regulations"
        reg_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in REGULATIONS_PDFS:
            if await download_pdf(client, url, reg_dir / filename):
                total += 1
        for page in REGULATIONS_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, reg_dir)

        print(f"\n=== Total: {total} files downloaded ===")


if __name__ == "__main__":
    asyncio.run(main())
