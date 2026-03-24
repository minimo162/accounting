"""Build chunk-level embedding index using Gemini embeddings.

Supports incremental indexing with file-level checkpointing:
- Tracks which files have been embedded via indexed_files in metadata
- Only processes new files on subsequent runs
- Saves index after each file completes (crash-safe)

Embeds at chunk (page/article) level (~750 tokens avg), not sentence level.
This produces ~5K embeddings instead of ~130K, cutting build time from hours to minutes.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding.gemini import GeminiEmbedder


def _save_index(output_dir, chunks_text, all_embeddings_list, chunk_ids, chunk_map, indexed_files, model_name):
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
        print(f"  Existing: {len(all_texts)} chunks from {len(indexed_files)} files")

    # Build chunk_map from current chunks (always update)
    chunk_map = {c["id"]: c for c in chunks}

    # Group chunks by file
    chunks_by_file: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_file.setdefault(c["file"], []).append(c)

    # Determine new files
    all_files = set(chunks_by_file.keys())
    new_files = sorted(all_files - indexed_files)

    if not new_files:
        print("All files already indexed. Nothing to do.")
        _save_index(output_dir, all_texts, all_embeddings, all_chunk_ids, chunk_map, indexed_files, "gemini-embedding-2-preview")
        print("Updated chunk_map in metadata.")
        return

    total_new_chunks = sum(len(chunks_by_file[f]) for f in new_files)
    print(f"New files to index: {len(new_files)} ({total_new_chunks} chunks)")

    embedder = GeminiEmbedder(api_key=api_key, model="gemini-embedding-2-preview")
    embedded_count = 0

    for file_idx, file_name in enumerate(new_files):
        file_chunks = chunks_by_file[file_name]
        file_texts = [c["text"] for c in file_chunks]
        file_ids = [c["id"] for c in file_chunks]

        print(f"  [{file_idx+1}/{len(new_files)}] {file_name}: {len(file_chunks)} chunks...", end="", flush=True)

        # Embed all chunks for this file
        # embed_batch handles internal batching (50/call) and retries
        file_embeddings = embedder.embed_batch(file_texts)

        all_texts.extend(file_texts)
        all_embeddings.extend(file_embeddings)
        all_chunk_ids.extend(file_ids)
        indexed_files.add(file_name)
        embedded_count += len(file_chunks)

        print(f" done ({embedded_count}/{total_new_chunks} total)")

        # Checkpoint: save after each file
        _save_index(output_dir, all_texts, all_embeddings, all_chunk_ids, chunk_map, indexed_files, embedder.model)

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
