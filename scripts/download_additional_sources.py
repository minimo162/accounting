"""Download PDFs from additional accounting knowledge sources."""

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx

sys.path.insert(0, str(Path(__file__).parent))
try:
    from .source_manifest import manifest_path, upsert_source_record
except ImportError:
    from source_manifest import manifest_path, upsert_source_record


async def download_pdf(
    client: httpx.AsyncClient,
    url: str,
    output_path: Path,
    *,
    data_dir: Path,
    manifest_file: Path,
) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        upsert_source_record(
            manifest_file,
            file_path=output_path.resolve(),
            data_dir=data_dir,
            url=url,
            source_group=output_path.resolve().relative_to(data_dir.resolve()).as_posix().split("/", 1)[0],
            kind="pdf",
        )
        print(f"  Skip (exists): {output_path.name}")
        return True
    try:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        upsert_source_record(
            manifest_file,
            file_path=output_path.resolve(),
            data_dir=data_dir,
            url=url,
            source_group=output_path.resolve().relative_to(data_dir.resolve()).as_posix().split("/", 1)[0],
            kind="pdf",
        )
        print(f"  Downloaded: {output_path.name} ({len(resp.content) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  Failed: {output_path.name} - {e}")
        return False


async def scrape_and_download(
    client: httpx.AsyncClient,
    page_url: str,
    output_dir: Path,
    *,
    data_dir: Path,
    manifest_file: Path,
) -> int:
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
        if await download_pdf(client, url, output_dir / filename, data_dir=data_dir, manifest_file=manifest_file):
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
    # 税効果会計に関する実務指針 (第6号・第10号・第11号 本文)
    ("https://jicpa.or.jp/specialized_field/files/6-11-6_10_11-2-20180219.pdf", "jicpa_zeikouka_jitsumu.pdf"),
    # 税効果会計に関する実務指針 (前書文)
    ("https://jicpa.or.jp/specialized_field/files/6-11-6_10_11-1-20180219.pdf", "jicpa_zeikouka_maegaki.pdf"),
    # 研究開発費及びソフトウェアの会計処理に関する実務指針 (第12号 本文)
    ("https://jicpa.or.jp/specialized_field/files/01138-003671.pdf", "jicpa_kenkyukaihatsu_jitsumu.pdf"),
    # 研究開発費及びソフトウェア 結論の背景
    ("https://jicpa.or.jp/specialized_field/files/01138-003673.pdf", "jicpa_kenkyukaihatsu_haikei.pdf"),
    # 研究開発費及びソフトウェア 設例による解説
    ("https://jicpa.or.jp/specialized_field/files/01138-003675.pdf", "jicpa_kenkyukaihatsu_setsurei.pdf"),
    # 退職給付会計に関する実務指針 (第13号 本文)
    ("https://jicpa.or.jp/specialized_field/files/01704-007137.pdf", "jicpa_taishokukyufu_jitsumu.pdf"),
    # 退職給付会計 結論の背景
    ("https://jicpa.or.jp/specialized_field/files/01704-007139.pdf", "jicpa_taishokukyufu_haikei.pdf"),
    # 退職給付会計 設例
    ("https://jicpa.or.jp/specialized_field/files/01704-007141.pdf", "jicpa_taishokukyufu_setsurei.pdf"),
]

JICPA_PAGES = [
    "https://jicpa.or.jp/specialized_field/publication/practical_guidelines/",
    "https://jicpa.or.jp/specialized_field/publication/research_report/",
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
    # 財務諸表等規則ガイドライン (最新版)
    ("https://www.fsa.go.jp/common/law/kaiji/zaiki.pdf", "reg_zaimuhyou_latest.pdf"),
    # 連結財務諸表規則ガイドライン (最新版)
    ("https://www.fsa.go.jp/common/law/kaiji/renketuzaiki.pdf", "reg_renketsu_latest.pdf"),
    # 中間財務諸表等規則ガイドライン
    ("https://www.fsa.go.jp/common/law/kaiji/05.pdf", "reg_chukan_guideline.pdf"),
    # 企業内容等開示ガイドライン (最新版)
    ("https://www.fsa.go.jp/common/law/kaiji/260220_kaiji.pdf", "reg_disclosure_guideline.pdf"),
    # 財務諸表等の監査証明に関する内閣府令ガイドライン
    ("https://www.fsa.go.jp/common/law/kaiji/kansa.pdf", "reg_kansa_guideline.pdf"),
]

REGULATIONS_PAGES = [
    "https://www.fsa.go.jp/common/law/kaiji.html",
]


async def main():
    base_dir = Path("data/pdfs")
    data_dir = base_dir.parent
    manifest_file = manifest_path(data_dir)

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
            if await download_pdf(client, url, bac_dir / filename, data_dir=data_dir, manifest_file=manifest_file):
                total += 1
        for page in BAC_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, bac_dir, data_dir=data_dir, manifest_file=manifest_file)

        # 2. JICPA
        print("\n=== 2. JICPA 実務指針 ===")
        jicpa_dir = base_dir / "jicpa"
        jicpa_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in JICPA_PDFS:
            if await download_pdf(client, url, jicpa_dir / filename, data_dir=data_dir, manifest_file=manifest_file):
                total += 1
        for page in JICPA_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, jicpa_dir, data_dir=data_dir, manifest_file=manifest_file)

        # 3. SSBJ
        print("\n=== 3. SSBJ ===")
        ssbj_dir = base_dir / "ssbj"
        ssbj_dir.mkdir(parents=True, exist_ok=True)
        for page in SSBJ_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, ssbj_dir, data_dir=data_dir, manifest_file=manifest_file)

        # 4. 財務諸表等規則
        print("\n=== 4. 財務諸表等規則 ===")
        reg_dir = base_dir / "regulations"
        reg_dir.mkdir(parents=True, exist_ok=True)
        for url, filename in REGULATIONS_PDFS:
            if await download_pdf(client, url, reg_dir / filename, data_dir=data_dir, manifest_file=manifest_file):
                total += 1
        for page in REGULATIONS_PAGES:
            print(f"  Scraping: {page}")
            total += await scrape_and_download(client, page, reg_dir, data_dir=data_dir, manifest_file=manifest_file)

        print(f"\n=== Total: {total} files downloaded ===")


if __name__ == "__main__":
    asyncio.run(main())
