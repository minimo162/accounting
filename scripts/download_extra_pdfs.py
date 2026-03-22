"""Download additional PDFs found by the research agent."""

import asyncio
from pathlib import Path
import httpx

EXTRA_URLS = [
    # 企業会計基準 (missing from initial scrape)
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kansoka_2015_1-1.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/20200331_01.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kabushikihoshu20210128_03.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ketsugou_6-1.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/20200331_03.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ketsugou2023pso_20240322_06.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kan_ren0720.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_37.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/fudosan-kaiji.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ketsugou_20190422.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_34.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/accounting-policies20200331_02.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/zeikouka20221028_02.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/chukan_20240322_02.pdf",
    # 適用指針
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/jikokabu2_3.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/20200331_12.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/zeikouka20221028_14.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/lease_20240913_16_20241101.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/stockop2-2.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/fukugo_s_20241101.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kan_ren0720_2_s_20220701.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_44.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/shihanki-s_13.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/spe-tanki_4.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/lease_20240913_24.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/zeikouka20221028_15_20241101.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kichu_20251016_23.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/nenjikaizen_20250311_08.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/lease_20240913_26.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_48.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/chukan_20240322_04.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/lease_20250423_02.pdf",
    # 実務対応報告
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/zaigai_2015_1.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/yusho_2018_02.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_52.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/ikan_20240701_53.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kichu_20251016_30.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/denshikirokuiten20220826_01.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kichu_20251016_31.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/kichu_20251016_32.pdf",
    "https://www.asb-j.jp/jp/wp-content/uploads/sites/4/zeikouka20260227_02.pdf",
]


async def main():
    output_dir = Path("data/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=120, verify=False, headers={
        "User-Agent": "Mozilla/5.0 (compatible; AccountingBot/1.0)"
    }) as client:
        new = 0
        for url in EXTRA_URLS:
            filename = Path(url).name
            path = output_dir / filename
            if path.exists() and path.stat().st_size > 0:
                continue
            try:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
                path.write_bytes(resp.content)
                print(f"Downloaded: {filename} ({len(resp.content)//1024} KB)")
                new += 1
            except Exception as e:
                print(f"Failed: {filename} - {e}")
        print(f"\nDownloaded {new} new PDFs")

if __name__ == "__main__":
    asyncio.run(main())
