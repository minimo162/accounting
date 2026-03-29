"""Helpers for source manifests used by corpus update and rebuild jobs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
DEFAULT_MANIFEST_NAME = "source_manifest.json"
SOURCE_ROOTS = ("pdfs", "html_cache", "egov_cache")
_DATE_TOKEN_RE = re.compile(r"((?:19|20)\d{6})")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def iso_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".xml":
        return "xml"
    return suffix.lstrip(".") or "file"


def infer_source_version(name: str, source_hash: str = "") -> str:
    match = _DATE_TOKEN_RE.search(name)
    if match:
        return match.group(1)
    stem = Path(name).stem
    if stem.startswith("egov_"):
        return stem.removeprefix("egov_")
    if stem.startswith("html_"):
        return stem.removeprefix("html_")
    return source_hash[:12] if source_hash else stem


def manifest_path(data_dir: Path) -> Path:
    return data_dir / DEFAULT_MANIFEST_NAME


def load_source_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "updated_at": "", "sources": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "sources" not in payload or not isinstance(payload["sources"], dict):
        payload["sources"] = {}
    return payload


def save_source_manifest(path: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "version": MANIFEST_VERSION,
        "updated_at": utc_now_iso(),
        "sources": {key: sources[key] for key in sorted(sources)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_url_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.items()}


def relative_source_path(file_path: Path, data_dir: Path) -> str:
    return file_path.resolve().relative_to(data_dir.resolve()).as_posix()


def build_source_record(
    file_path: Path,
    *,
    data_dir: Path,
    existing_entry: dict[str, Any] | None = None,
    url: str = "",
    source_group: str = "",
    kind: str = "",
    checked_at: str | None = None,
) -> dict[str, Any]:
    stat = file_path.stat()
    source_hash = sha256_file(file_path)
    same_hash = existing_entry and existing_entry.get("source_hash") == source_hash
    rel_path = relative_source_path(file_path, data_dir)
    kind = kind or infer_source_kind(file_path)
    checked_at = checked_at or utc_now_iso()
    fetched_at = (
        str(existing_entry.get("fetched_at", ""))
        if same_hash and existing_entry and existing_entry.get("fetched_at")
        else iso_from_timestamp(stat.st_mtime)
    )
    version = (
        str(existing_entry.get("version", ""))
        if same_hash and existing_entry and existing_entry.get("version")
        else infer_source_version(file_path.name, source_hash)
    )
    return {
        "path": rel_path,
        "file_name": file_path.name,
        "kind": kind,
        "source_group": source_group or rel_path.split("/", 1)[0],
        "url": url or str((existing_entry or {}).get("url", "")),
        "version": version,
        "fetched_at": fetched_at,
        "checked_at": checked_at,
        "modified_at": iso_from_timestamp(stat.st_mtime),
        "file_size": stat.st_size,
        "source_hash": source_hash,
    }


def upsert_source_record(
    manifest_file: Path,
    *,
    file_path: Path,
    data_dir: Path,
    url: str = "",
    source_group: str = "",
    kind: str = "",
) -> dict[str, Any]:
    payload = load_source_manifest(manifest_file)
    sources = payload["sources"]
    rel_path = relative_source_path(file_path, data_dir)
    sources[rel_path] = build_source_record(
        file_path,
        data_dir=data_dir,
        existing_entry=sources.get(rel_path),
        url=url,
        source_group=source_group,
        kind=kind,
    )
    save_source_manifest(manifest_file, sources)
    return sources[rel_path]


def scan_source_tree(
    data_dir: Path,
    *,
    roots: tuple[str, ...] = SOURCE_ROOTS,
    base_sources: dict[str, dict[str, Any]] | None = None,
    url_lookup: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    base_sources = base_sources or {}
    url_lookup = url_lookup or {}
    for root_name in roots:
        root = data_dir / root_name
        if not root.exists():
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            rel_path = relative_source_path(file_path, data_dir)
            existing_entry = base_sources.get(rel_path, {})
            url = str(existing_entry.get("url", "")) or url_lookup.get(rel_path) or url_lookup.get(file_path.name, "")
            sources[rel_path] = build_source_record(
                file_path,
                data_dir=data_dir,
                existing_entry=existing_entry,
                url=url,
                source_group=root_name,
            )
    return sources


def summarize_source_diff(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed: list[dict[str, Any]] = []
    unchanged: list[str] = []

    for key in sorted(before_keys & after_keys):
        prev = before[key]
        curr = after[key]
        if prev.get("source_hash") == curr.get("source_hash"):
            unchanged.append(key)
            continue
        changed.append(
            {
                "path": key,
                "before_hash": str(prev.get("source_hash", ""))[:12],
                "after_hash": str(curr.get("source_hash", ""))[:12],
                "before_version": str(prev.get("version", "")),
                "after_version": str(curr.get("version", "")),
                "before_size": int(prev.get("file_size", 0)),
                "after_size": int(curr.get("file_size", 0)),
            }
        )

    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "unchanged_count": len(unchanged),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def metadata_for_file(
    file_path: Path,
    *,
    data_dir: Path,
    manifest_sources: dict[str, dict[str, Any]] | None = None,
    url_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest_sources = manifest_sources or {}
    url_lookup = url_lookup or {}
    rel_path = relative_source_path(file_path, data_dir)
    entry = manifest_sources.get(rel_path)
    if entry is None:
        entry = build_source_record(
            file_path,
            data_dir=data_dir,
            url=url_lookup.get(rel_path) or url_lookup.get(file_path.name, ""),
            source_group=rel_path.split("/", 1)[0],
        )
    return {
        "source_path": rel_path,
        "source_kind": str(entry.get("kind", infer_source_kind(file_path))),
        "source_group": str(entry.get("source_group", rel_path.split("/", 1)[0])),
        "source_url": str(entry.get("url", "")),
        "source_version": str(entry.get("version", infer_source_version(file_path.name))),
        "source_fetched_at": str(entry.get("fetched_at", "")),
        "source_checked_at": str(entry.get("checked_at", "")),
        "source_hash": str(entry.get("source_hash", "")),
    }
