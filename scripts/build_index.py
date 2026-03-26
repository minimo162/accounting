"""Build chunk-level embedding index using Gemini embeddings.

Supports incremental indexing with checksum-based change detection:
- Tracks per-file MD5 checksums of chunk IDs + text
- Detects new files AND modified files (e.g. after re-running process_pdfs.py)
- Removes stale embeddings for changed files before re-embedding
- Saves index after each file completes (crash-safe)

Embeds at chunk (page/article) level (~750 tokens avg), not sentence level.
This produces ~5K embeddings instead of ~130K, cutting build time from hours to minutes.
"""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding.gemini import GeminiEmbedder


def _compute_file_checksum(file_chunks: list[dict]) -> str:
    """チャンクのIDとテキストからMD5チェックサムを計算する。

    チャンク内容が変わればチェックサムも変わるため、再埋め込みが必要かを判定できる。
    """
    content = "||".join(
        f"{c['id']}\x00{c['text']}"
        for c in sorted(file_chunks, key=lambda c: c["id"])
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _chunk_file(chunk_id: str) -> str:
    """チャンクID 'filename.pdf:3' からファイル名部分を返す。"""
    return chunk_id.rsplit(":", 1)[0]


def _save_index(
    output_dir,
    chunks_text,
    all_embeddings_list,
    chunk_ids,
    chunk_map,
    indexed_files,
    model_name,
    file_chunk_checksums: dict[str, str],
):
    """Save current index state to disk."""
    npz_path = output_dir / "sentence_index.npz"
    meta_path = output_dir / "sentence_meta.pkl"

    embeddings_array = np.array(all_embeddings_list, dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_array = embeddings_array / norms

    np.savez_compressed(npz_path, embeddings=embeddings_array.astype(np.float16))

    meta = {
        "sentences": chunks_text,
        "sentence_to_chunk": chunk_ids,
        "chunks": chunk_map,
        "indexed_files": indexed_files,
        "model_name": model_name,
        "file_chunk_checksums": file_chunk_checksums,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    # Legacy pkl
    legacy_path = output_dir / "sentence_index.pkl"
    legacy = {
        "sentences": chunks_text,
        "embeddings": embeddings_array,
        "sentence_to_chunk": chunk_ids,
        "chunks": chunk_map,
        "model_name": model_name,
    }
    with open(legacy_path, "wb") as f:
        pickle.dump(legacy, f)

    return embeddings_array


def build_index(chunks_path: str, output_dir: str, api_key: str):
    """Build or incrementally update chunk-level embedding index."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "sentence_index.npz"
    meta_path = output_dir / "sentence_meta.pkl"

    # Load existing index if available
    all_texts: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_chunk_ids: list[str] = []
    indexed_files: set[str] = set()
    saved_checksums: dict[str, str] = {}

    if npz_path.exists() and meta_path.exists():
        print("Loading existing index...")
        data = np.load(str(npz_path))
        emb_array = data["embeddings"].astype(np.float32)
        all_embeddings = [emb_array[i] for i in range(len(emb_array))]
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        all_texts = meta["sentences"]
        all_chunk_ids = meta["sentence_to_chunk"]
        indexed_files = meta.get("indexed_files", set())
        saved_checksums = meta.get("file_chunk_checksums", {})
        print(f"  Existing: {len(all_texts)} chunks from {len(indexed_files)} files")

    # Build chunk_map from current chunks (always update)
    chunk_map = {c["id"]: c for c in chunks}

    # Group chunks by file
    chunks_by_file: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_file.setdefault(c["file"], []).append(c)

    # Compute current checksums
    current_checksums = {
        file_name: _compute_file_checksum(file_chunks)
        for file_name, file_chunks in chunks_by_file.items()
    }

    # Detect new files and files whose chunks have changed
    all_files = set(chunks_by_file.keys())
    files_to_reindex = sorted(
        f for f in all_files
        if f not in indexed_files or saved_checksums.get(f) != current_checksums[f]
    )

    # Remove stale embeddings for deleted files (before early return)
    deleted_files = [f for f in list(indexed_files) if f not in all_files]
    if deleted_files:
        print(f"Deleted files (removing from index): {len(deleted_files)}")
        keep = [
            (t, e, cid)
            for t, e, cid in zip(all_texts, all_embeddings, all_chunk_ids)
            if _chunk_file(cid) not in set(deleted_files)
        ]
        if keep:
            all_texts, all_embeddings, all_chunk_ids = zip(*keep)
            all_texts, all_embeddings, all_chunk_ids = list(all_texts), list(all_embeddings), list(all_chunk_ids)
        else:
            all_texts, all_embeddings, all_chunk_ids = [], [], []
        indexed_files -= set(deleted_files)
        for f in deleted_files:
            saved_checksums.pop(f, None)
        print(f"  Removed stale embeddings for {len(deleted_files)} deleted file(s)")

    if not files_to_reindex:
        print("All files already indexed and up-to-date. Nothing to do.")
        _save_index(
            output_dir, all_texts, all_embeddings, all_chunk_ids,
            chunk_map, indexed_files, "gemini-embedding-2-preview", current_checksums,
        )
        print("Updated chunk_map in metadata.")
        return

    # Count new vs changed
    new_files = [f for f in files_to_reindex if f not in indexed_files]
    changed_files = [f for f in files_to_reindex if f in indexed_files]
    if new_files:
        print(f"New files: {len(new_files)}")
    if changed_files:
        print(f"Changed files (will re-embed): {len(changed_files)}")

    # Remove stale embeddings for changed files
    files_to_remove = set(changed_files)
    if files_to_remove:
        keep = [
            (t, e, cid)
            for t, e, cid in zip(all_texts, all_embeddings, all_chunk_ids)
            if _chunk_file(cid) not in files_to_remove
        ]
        if keep:
            all_texts, all_embeddings, all_chunk_ids = zip(*keep)
            all_texts = list(all_texts)
            all_embeddings = list(all_embeddings)
            all_chunk_ids = list(all_chunk_ids)
        else:
            all_texts, all_embeddings, all_chunk_ids = [], [], []
        indexed_files -= files_to_remove
        print(f"  Removed stale embeddings for {len(files_to_remove)} file(s)")

    total_new_chunks = sum(len(chunks_by_file[f]) for f in files_to_reindex)
    print(f"Total chunks to embed: {total_new_chunks}")

    embedder = GeminiEmbedder(api_key=api_key, model="gemini-embedding-2-preview")
    embedded_count = 0

    for file_idx, file_name in enumerate(files_to_reindex):
        file_chunks = chunks_by_file[file_name]
        file_texts = [c["text"] for c in file_chunks]
        file_ids = [c["id"] for c in file_chunks]

        print(f"  [{file_idx+1}/{len(files_to_reindex)}] {file_name}: {len(file_chunks)} chunks...", end="", flush=True)

        file_embeddings = embedder.embed_batch(file_texts)

        all_texts.extend(file_texts)
        all_embeddings.extend(file_embeddings)
        all_chunk_ids.extend(file_ids)
        indexed_files.add(file_name)
        embedded_count += len(file_chunks)

        print(f" done ({embedded_count}/{total_new_chunks} total)")

        # Checkpoint: save after each file
        _save_index(
            output_dir, all_texts, all_embeddings, all_chunk_ids,
            chunk_map, indexed_files, embedder.model, current_checksums,
        )

    npz_size = (output_dir / "sentence_index.npz").stat().st_size / 1024 / 1024
    print(f"\nIndex complete:")
    print(f"  npz: {npz_size:.1f} MB")
    print(f"  Chunks indexed: {len(all_texts)}")
    print(f"  Embedding dim: {len(all_embeddings[0])}")
    print(f"  Indexed files: {len(indexed_files)}")


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    build_index(chunks_path, output_dir, api_key)
