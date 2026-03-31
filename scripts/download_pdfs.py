"""Synchronize known PDF sources into data/pdfs with manifest tracking."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

import httpx

try:
    from .source_manifest import (
        load_source_metadata_lookup,
        load_url_lookup,
        manifest_path,
        sha256_file,
        source_metadata_for_path,
        upsert_source_record,
    )
except ImportError:
    from source_manifest import (
        load_source_metadata_lookup,
        load_url_lookup,
        manifest_path,
        sha256_file,
        source_metadata_for_path,
        upsert_source_record,
    )


def _pdf_entries(source_map: dict[str, str]) -> list[tuple[str, str]]:
    entries = [
        (name, url)
        for name, url in source_map.items()
        if name.lower().endswith(".pdf")
    ]
    entries.sort(key=lambda item: item[0])
    return entries


async def sync_pdfs(
    output_dir: str = "data/pdfs",
    *,
    pdf_sources_path: str = "data/pdf_sources.json",
    source_metadata_path: str = "data/source_metadata.json",
    refresh_existing: bool = True,
    limit: int | None = None,
) -> dict[str, int]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_dir = output_path.parent
    sources = load_url_lookup(Path(pdf_sources_path))
    metadata_lookup = load_source_metadata_lookup(Path(source_metadata_path))
    entries = _pdf_entries(sources)
    if limit is not None:
        entries = entries[:limit]

    summary = {
        "checked": 0,
        "downloaded": 0,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
        "failed": 0,
    }
    manifest_file = manifest_path(data_dir)

    async with httpx.AsyncClient(
        timeout=180,
        verify=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AccountingBot/1.0)"},
    ) as client:
        for filename, url in entries:
            summary["checked"] += 1
            destination = output_path / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{summary['checked']}/{len(entries)}] {filename}")

            if destination.exists() and not refresh_existing:
                upsert_source_record(
                    manifest_file,
                    file_path=destination,
                    data_dir=data_dir,
                    url=url,
                    source_group="pdfs",
                    kind="pdf",
                    metadata=source_metadata_for_path(
                        destination,
                        data_dir=data_dir,
                        metadata_lookup=metadata_lookup,
                    ),
                )
                summary["skipped"] += 1
                print("  Skip (existing, refresh disabled)")
                continue

            try:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                payload = response.content
            except Exception as exc:
                summary["failed"] += 1
                print(f"  Failed: {exc}")
                continue

            existing_hash = sha256_file(destination) if destination.exists() else ""
            new_hash = hashlib.sha256(payload).hexdigest()
            if existing_hash == new_hash:
                summary["unchanged"] += 1
                print("  Unchanged")
            else:
                destination.write_bytes(payload)
                if existing_hash:
                    summary["updated"] += 1
                    print(f"  Updated ({len(payload) // 1024} KB)")
                else:
                    summary["downloaded"] += 1
                    print(f"  Downloaded ({len(payload) // 1024} KB)")

            upsert_source_record(
                manifest_file,
                file_path=destination,
                data_dir=data_dir,
                url=url,
                source_group="pdfs",
                kind="pdf",
                metadata=source_metadata_for_path(
                    destination,
                    data_dir=data_dir,
                    metadata_lookup=metadata_lookup,
                ),
            )

    print(
        "Summary:"
        f" checked={summary['checked']}"
        f" downloaded={summary['downloaded']}"
        f" updated={summary['updated']}"
        f" unchanged={summary['unchanged']}"
        f" skipped={summary['skipped']}"
        f" failed={summary['failed']}"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="data/pdfs")
    parser.add_argument("--pdf-sources", default="data/pdf_sources.json")
    parser.add_argument("--source-metadata", default="data/source_metadata.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not refresh existing files; only download missing PDFs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        sync_pdfs(
            args.output_dir,
            pdf_sources_path=args.pdf_sources,
            source_metadata_path=args.source_metadata,
            refresh_existing=not args.skip_existing,
            limit=args.limit,
        )
    )
