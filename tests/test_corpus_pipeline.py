import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.corpus_pipeline import rollback_release, snapshot_release, validate_artifacts
from scripts import process_html_sources
from scripts.source_manifest import (
    build_source_record,
    metadata_for_file,
    save_source_manifest,
    scan_source_tree,
    summarize_source_diff,
)


class SourceManifestTests(unittest.TestCase):
    def test_build_source_record_preserves_fetched_at_when_hash_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            file_path = data_dir / "pdfs" / "sample.pdf"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"sample-pdf")

            existing = {
                "source_hash": "84c1b754fce58a3adc9fda2f8f84c1edcf7fdc9d8ac1b8e6d1724a1d4301c5a0",
                "fetched_at": "2025-03-28T00:00:00+00:00",
                "version": "20250328",
            }
            existing["source_hash"] = build_source_record(
                file_path,
                data_dir=data_dir,
                url="https://example.com/sample.pdf",
                source_group="pdfs",
                kind="pdf",
            )["source_hash"]

            record = build_source_record(
                file_path,
                data_dir=data_dir,
                existing_entry=existing,
                url="https://example.com/sample.pdf",
                source_group="pdfs",
                kind="pdf",
            )

            self.assertEqual(record["fetched_at"], "2025-03-28T00:00:00+00:00")
            self.assertEqual(record["version"], "20250328")
            self.assertEqual(record["url"], "https://example.com/sample.pdf")

    def test_scan_source_tree_and_diff_detect_source_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            (data_dir / "pdfs").mkdir(parents=True, exist_ok=True)
            (data_dir / "html_cache").mkdir(parents=True, exist_ok=True)

            pdf_a = data_dir / "pdfs" / "a_20250328.pdf"
            html_b = data_dir / "html_cache" / "b.html"
            pdf_a.write_bytes(b"old-a")
            html_b.write_text("<h1>old</h1>", encoding="utf-8")

            before = scan_source_tree(data_dir)

            pdf_a.write_bytes(b"new-a")
            html_b.unlink()
            pdf_c = data_dir / "pdfs" / "c_20250329.pdf"
            pdf_c.write_bytes(b"new-c")

            after = scan_source_tree(data_dir, base_sources=before)
            diff = summarize_source_diff(before, after)

            self.assertEqual(diff["changed_count"], 1)
            self.assertEqual(diff["removed_count"], 1)
            self.assertEqual(diff["added_count"], 1)
            self.assertEqual(diff["changed"][0]["path"], "pdfs/a_20250328.pdf")


class CorpusPipelineArtifactTests(unittest.TestCase):
    def _write_index_meta(self, index_dir: Path, chunk_id: str) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "sentences": ["child text"],
            "sentence_to_chunk": [chunk_id],
            "chunks": {chunk_id: {"id": chunk_id}},
        }
        with (index_dir / "sentence_meta.pkl").open("wb") as fh:
            pickle.dump(meta, fh)

    def test_validate_artifacts_passes_for_consistent_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            pdf_dir = data_dir / "pdfs"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            source_file = pdf_dir / "lease_20250328.pdf"
            source_file.write_bytes(b"pdf-bytes")

            manifest_sources = scan_source_tree(
                data_dir,
                url_lookup={"lease_20250328.pdf": "https://example.com/lease_20250328.pdf"},
            )
            save_source_manifest(data_dir / "source_manifest.json", manifest_sources)

            source_meta = metadata_for_file(
                source_file,
                data_dir=data_dir,
                manifest_sources=manifest_sources,
            )
            chunks = [
                {
                    "id": "lease_20250328.pdf:p1",
                    "text": "parent text",
                    "file": "lease_20250328.pdf",
                    "page": 1,
                    "level": "parent",
                    "parent_id": "lease_20250328.pdf:p1",
                    "source": "Lease",
                    "doc_title": "Lease",
                    "doc_type": "企業会計基準",
                    "standard_no": "企業会計基準第34号",
                    "section_title": "総則",
                    "section_path": "Lease > 総則",
                    **source_meta,
                },
                {
                    "id": "lease_20250328.pdf:c1",
                    "text": "child text",
                    "file": "lease_20250328.pdf",
                    "page": 1,
                    "level": "child",
                    "parent_id": "lease_20250328.pdf:p1",
                    "source": "Lease",
                    "doc_title": "Lease",
                    "doc_type": "企業会計基準",
                    "standard_no": "企業会計基準第34号",
                    "section_title": "総則",
                    "section_path": "Lease > 総則",
                    **source_meta,
                },
            ]
            (data_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_index_meta(data_dir / "index", "lease_20250328.pdf:c1")

            result = validate_artifacts(data_dir=str(data_dir))

            self.assertTrue(result["passed"])
            self.assertEqual(result["searchable_chunk_count"], 1)
            self.assertEqual(result["errors"], [])

    def test_snapshot_and_rollback_restore_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "index").mkdir(exist_ok=True)
            (data_dir / "chunks.json").write_text('{"version":1}', encoding="utf-8")
            (data_dir / "pdf_sources.json").write_text("{}", encoding="utf-8")
            (data_dir / "source_manifest.json").write_text('{"version":1,"sources":{}}', encoding="utf-8")
            (data_dir / "index" / "sentence_meta.pkl").write_bytes(b"meta")

            release_dir = snapshot_release(
                data_dir=str(data_dir),
                release_id="release-test",
                summary={"ok": True},
            )

            (data_dir / "chunks.json").write_text('{"version":2}', encoding="utf-8")
            rollback_release(data_dir=str(data_dir), release=str(release_dir))

            self.assertEqual((data_dir / "chunks.json").read_text(encoding="utf-8"), '{"version":1}')


class HtmlSourceProcessingTests(unittest.TestCase):
    def test_main_removes_legacy_html_source_chunks_and_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            cache_dir = data_dir / "html_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            legacy_cache = cache_dir / "html_chusho_kaikei_youryo_about.html"
            legacy_cache.write_text("<p>legacy</p>", encoding="utf-8")

            chunks = [
                {"id": "legacy:c1", "file": "html_chusho_kaikei_youryo_about.html", "text": "legacy"},
                {"id": "keep:c1", "file": "keep.pdf", "text": "keep"},
            ]
            chunks_path = data_dir / "chunks.json"
            chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

            with mock.patch.object(process_html_sources, "HTML_SOURCES", []):
                process_html_sources.main(str(chunks_path), refresh=False)

            updated = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.assertEqual(updated, [{"id": "keep:c1", "file": "keep.pdf", "text": "keep"}])
            self.assertFalse(legacy_cache.exists())


if __name__ == "__main__":
    unittest.main()
