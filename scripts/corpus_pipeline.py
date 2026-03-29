"""Corpus update pipeline with diff summaries, snapshots, validation, and rollback."""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .download_pdfs import sync_pdfs
    from .process_egov import main as process_egov_chunks
    from .process_html_sources import main as process_html_chunks
    from .process_pdfs import create_chunks
    from .source_manifest import (
        load_source_manifest,
        load_url_lookup,
        manifest_path,
        save_source_manifest,
        scan_source_tree,
        summarize_source_diff,
    )
except ImportError:
    from download_pdfs import sync_pdfs
    from process_egov import main as process_egov_chunks
    from process_html_sources import main as process_html_chunks
    from process_pdfs import create_chunks
    from source_manifest import (
        load_source_manifest,
        load_url_lookup,
        manifest_path,
        save_source_manifest,
        scan_source_tree,
        summarize_source_diff,
    )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data_dir": data_dir,
        "chunks": data_dir / "chunks.json",
        "pdf_sources": data_dir / "pdf_sources.json",
        "manifest": manifest_path(data_dir),
        "index_dir": data_dir / "index",
        "pdf_dir": data_dir / "pdfs",
        "full_texts": data_dir / "full_texts.json",
        "runs_dir": data_dir / "pipeline_runs",
        "releases_dir": data_dir / "releases",
    }


def _searchable_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    children = [chunk for chunk in chunks if chunk.get("level") == "child"]
    return children or [chunk for chunk in chunks if chunk.get("level", "parent") == "parent"]


def validate_artifacts(
    *,
    data_dir: str = "data",
    chunks_path: str | None = None,
    index_dir: str | None = None,
) -> dict[str, Any]:
    paths = _default_paths(Path(data_dir))
    chunks_file = Path(chunks_path) if chunks_path else paths["chunks"]
    index_path = Path(index_dir) if index_dir else paths["index_dir"]
    manifest_file = paths["manifest"]

    errors: list[str] = []
    warnings: list[str] = []

    if not chunks_file.exists():
        return {"passed": False, "errors": [f"missing_chunks:{chunks_file}"], "warnings": []}

    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    if not chunks:
        errors.append("chunks_empty")

    ids: set[str] = set()
    child_parent_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("id", ""))
        if not chunk_id:
            errors.append("chunk_without_id")
            continue
        if chunk_id in ids:
            errors.append(f"duplicate_chunk_id:{chunk_id}")
        ids.add(chunk_id)
        if chunk.get("level") == "child":
            child_parent_ids.add(str(chunk.get("parent_id", "")))
        for key in ("source_path", "source_version", "source_fetched_at", "source_hash"):
            if not chunk.get(key):
                errors.append(f"missing_{key}:{chunk_id}")

    missing_parents = sorted(parent_id for parent_id in child_parent_ids if parent_id and parent_id not in ids)
    for parent_id in missing_parents:
        errors.append(f"missing_parent_chunk:{parent_id}")

    manifest_sources = load_source_manifest(manifest_file)["sources"]
    missing_manifest_paths = sorted(
        {
            str(chunk.get("source_path", ""))
            for chunk in chunks
            if chunk.get("source_path") and str(chunk.get("source_path")) not in manifest_sources
        }
    )
    for rel_path in missing_manifest_paths:
        errors.append(f"manifest_missing_source:{rel_path}")

    searchable = _searchable_chunks(chunks)
    meta_path = index_path / "sentence_meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as fh:
            meta = pickle.load(fh)
        index_chunk_ids = [str(chunk_id) for chunk_id in meta.get("sentence_to_chunk", [])]
        missing_from_chunks = sorted(chunk_id for chunk_id in index_chunk_ids if chunk_id not in ids)
        if missing_from_chunks:
            errors.append(f"index_missing_chunks:{missing_from_chunks[:5]}")
        if len(index_chunk_ids) != len(searchable):
            errors.append(
                f"index_searchable_count_mismatch:index={len(index_chunk_ids)} chunks={len(searchable)}"
            )
    else:
        warnings.append(f"missing_index_meta:{meta_path}")

    return {
        "passed": not errors,
        "chunk_count": len(chunks),
        "searchable_chunk_count": len(searchable),
        "errors": errors,
        "warnings": warnings,
    }


def snapshot_release(*, data_dir: str = "data", release_id: str, summary: dict[str, Any]) -> Path:
    paths = _default_paths(Path(data_dir))
    release_dir = paths["releases_dir"] / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "chunks.json": paths["chunks"],
        "pdf_sources.json": paths["pdf_sources"],
        "source_manifest.json": paths["manifest"],
        "full_texts.json": paths["full_texts"],
    }
    for name, src in artifacts.items():
        if src.exists():
            shutil.copy2(src, release_dir / name)

    index_dst = release_dir / "index"
    index_dst.mkdir(exist_ok=True)
    for name in ("sentence_index.npz", "sentence_meta.pkl", "sentence_index.pkl"):
        src = paths["index_dir"] / name
        if src.exists():
            shutil.copy2(src, index_dst / name)

    (release_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return release_dir


def rollback_release(*, data_dir: str = "data", release: str) -> Path:
    paths = _default_paths(Path(data_dir))
    release_path = Path(release)
    if not release_path.exists():
        release_path = paths["releases_dir"] / release
    if not release_path.exists():
        raise FileNotFoundError(f"release not found: {release}")

    for name in ("chunks.json", "pdf_sources.json", "source_manifest.json", "full_texts.json"):
        src = release_path / name
        if src.exists():
            shutil.copy2(src, paths["data_dir"] / name)

    index_src = release_path / "index"
    if index_src.exists():
        paths["index_dir"].mkdir(parents=True, exist_ok=True)
        for src in index_src.iterdir():
            if src.is_file():
                shutil.copy2(src, paths["index_dir"] / src.name)
    return release_path


def run_update_pipeline(
    *,
    data_dir: str = "data",
    refresh_pdfs: bool = True,
    refresh_html: bool = True,
    refresh_egov: bool = True,
    build_index_enabled: bool = True,
    build_full_texts_enabled: bool = False,
    snapshot: bool = True,
) -> dict[str, Any]:
    paths = _default_paths(Path(data_dir))
    before_manifest = load_source_manifest(paths["manifest"])["sources"]
    url_lookup = load_url_lookup(paths["pdf_sources"])
    run_id = _run_id()
    steps: dict[str, Any] = {}

    steps["sync_pdfs"] = asyncio.run(
        sync_pdfs(
            str(paths["pdf_dir"]),
            pdf_sources_path=str(paths["pdf_sources"]),
            refresh_existing=refresh_pdfs,
        )
    )
    create_chunks(str(paths["pdf_dir"]), str(paths["chunks"]))
    steps["process_pdfs"] = {"chunks_path": str(paths["chunks"])}

    process_html_chunks(str(paths["chunks"]), refresh=refresh_html)
    steps["process_html_sources"] = {"refresh": refresh_html}

    process_egov_chunks(str(paths["chunks"]), refresh=refresh_egov)
    steps["process_egov"] = {"refresh": refresh_egov}

    if build_full_texts_enabled:
        from process_full_texts import build_full_texts

        build_full_texts(str(paths["pdf_dir"]), str(paths["data_dir"] / "egov_cache"), str(paths["full_texts"]))
        steps["process_full_texts"] = {"output": str(paths["full_texts"])}

    if build_index_enabled:
        from build_index import build_index

        build_index(str(paths["chunks"]), str(paths["index_dir"]))
        steps["build_index"] = {"output_dir": str(paths["index_dir"])}

    current_manifest = load_source_manifest(paths["manifest"])["sources"]
    after_manifest = scan_source_tree(
        paths["data_dir"],
        base_sources=current_manifest,
        url_lookup=url_lookup,
    )
    save_source_manifest(paths["manifest"], after_manifest)
    diff_summary = summarize_source_diff(before_manifest, after_manifest)
    validation = validate_artifacts(data_dir=str(paths["data_dir"]))

    summary = {
        "run_id": run_id,
        "executed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_dir": str(paths["data_dir"]),
        "steps": steps,
        "diff_summary": diff_summary,
        "validation": validation,
    }

    run_dir = paths["runs_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["run_dir"] = str(run_dir)

    if snapshot:
        release_dir = snapshot_release(data_dir=str(paths["data_dir"]), release_id=run_id, summary=summary)
        summary["release_dir"] = str(release_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="Refresh sources, rebuild chunks, and optionally rebuild the index.")
    update.add_argument("--data-dir", default="data")
    update.add_argument("--skip-pdf-refresh", action="store_true")
    update.add_argument("--skip-html-refresh", action="store_true")
    update.add_argument("--skip-egov-refresh", action="store_true")
    update.add_argument("--skip-index", action="store_true")
    update.add_argument("--full-texts", action="store_true", help="Regenerate full_texts.json too.")
    update.add_argument("--no-snapshot", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate chunks, source manifest, and index integrity.")
    validate.add_argument("--data-dir", default="data")
    validate.add_argument("--chunks")
    validate.add_argument("--index-dir")

    rollback = subparsers.add_parser("rollback", help="Restore chunks and index artifacts from a saved release.")
    rollback.add_argument("release")
    rollback.add_argument("--data-dir", default="data")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "update":
        summary = run_update_pipeline(
            data_dir=args.data_dir,
            refresh_pdfs=not args.skip_pdf_refresh,
            refresh_html=not args.skip_html_refresh,
            refresh_egov=not args.skip_egov_refresh,
            build_index_enabled=not args.skip_index,
            build_full_texts_enabled=args.full_texts,
            snapshot=not args.no_snapshot,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["validation"]["passed"] else 1

    if args.command == "validate":
        summary = validate_artifacts(data_dir=args.data_dir, chunks_path=args.chunks, index_dir=args.index_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1

    if args.command == "rollback":
        release_path = rollback_release(data_dir=args.data_dir, release=args.release)
        print(f"Rolled back from: {release_path}")
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
