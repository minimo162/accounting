"""Build embedding index for searchable chunks with incremental reindex support."""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.config import EmbeddingConfig
from src.arag.retrieval import ChunkCorpus
from src.embedding import create_embedder


def _compute_file_checksum(file_chunks: list[dict]) -> str:
    content = "||".join(
        f"{c['id']}\x00{c['text']}\x00{c.get('context', '')}"
        for c in sorted(file_chunks, key=lambda c: c["id"])
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _embedding_text(chunk: dict) -> str:
    context = chunk.get("context", "")
    if context:
        return context + "\n\n" + chunk["text"]
    return chunk["text"]


def _chunk_file(chunk_id: str) -> str:
    return chunk_id.rsplit(":", 1)[0]


def _save_index(
    output_dir: Path,
    chunks_text: list[str],
    all_embeddings_list: list[np.ndarray],
    chunk_ids: list[str],
    chunk_map: dict[str, dict],
    indexed_files,
    model_name: str,
    file_chunk_checksums: dict[str, str],
):
    npz_path = output_dir / "sentence_index.npz"
    meta_path = output_dir / "sentence_meta.pkl"

    embeddings_array = np.array(all_embeddings_list, dtype=np.float32)
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

    legacy_path = output_dir / "sentence_index.pkl"
    with open(legacy_path, "wb") as f:
        pickle.dump({**meta, "embeddings": embeddings_array}, f)


def build_index(chunks_path: str, output_dir: str):
    chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    corpus = ChunkCorpus(chunks)
    searchable_chunks = corpus.searchable_chunks
    print(f"Loaded {len(chunks)} chunks ({len(searchable_chunks)} searchable)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "sentence_index.npz"
    meta_path = output_dir / "sentence_meta.pkl"

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
        indexed_files = set(meta.get("indexed_files", set()))
        saved_checksums = meta.get("file_chunk_checksums", {})
        print(f"  Existing: {len(all_texts)} chunks from {len(indexed_files)} files")

    chunk_map = {chunk["id"]: chunk for chunk in chunks}
    chunks_by_file: dict[str, list[dict]] = {}
    for chunk in searchable_chunks:
        chunks_by_file.setdefault(chunk["file"], []).append(chunk)

    current_checksums = {
        file_name: _compute_file_checksum(file_chunks)
        for file_name, file_chunks in chunks_by_file.items()
    }

    all_files = set(chunks_by_file.keys())
    files_to_reindex = sorted(
        file_name
        for file_name in all_files
        if file_name not in indexed_files or saved_checksums.get(file_name) != current_checksums[file_name]
    )

    deleted_files = [file_name for file_name in list(indexed_files) if file_name not in all_files]
    if deleted_files:
        keep = [
            (text, embedding, chunk_id)
            for text, embedding, chunk_id in zip(all_texts, all_embeddings, all_chunk_ids)
            if _chunk_file(chunk_id) not in set(deleted_files)
        ]
        if keep:
            all_texts, all_embeddings, all_chunk_ids = zip(*keep)
            all_texts, all_embeddings, all_chunk_ids = list(all_texts), list(all_embeddings), list(all_chunk_ids)
        else:
            all_texts, all_embeddings, all_chunk_ids = [], [], []
        indexed_files -= set(deleted_files)
        for file_name in deleted_files:
            saved_checksums.pop(file_name, None)

    if not files_to_reindex:
        print("All files already indexed and up-to-date. Nothing to do.")
        _save_index(output_dir, all_texts, all_embeddings, all_chunk_ids, chunk_map, indexed_files, EmbeddingConfig().model, current_checksums)
        return

    changed_files = [file_name for file_name in files_to_reindex if file_name in indexed_files]
    if changed_files:
        keep = [
            (text, embedding, chunk_id)
            for text, embedding, chunk_id in zip(all_texts, all_embeddings, all_chunk_ids)
            if _chunk_file(chunk_id) not in set(changed_files)
        ]
        if keep:
            all_texts, all_embeddings, all_chunk_ids = zip(*keep)
            all_texts, all_embeddings, all_chunk_ids = list(all_texts), list(all_embeddings), list(all_chunk_ids)
        else:
            all_texts, all_embeddings, all_chunk_ids = [], [], []
        indexed_files -= set(changed_files)

    embed_config = EmbeddingConfig()
    embedder = create_embedder(embed_config)
    total_new_chunks = sum(len(chunks_by_file[file_name]) for file_name in files_to_reindex)
    embedded_count = 0

    for file_idx, file_name in enumerate(files_to_reindex, start=1):
        file_chunks = chunks_by_file[file_name]
        file_texts = [_embedding_text(chunk) for chunk in file_chunks]
        file_ids = [chunk["id"] for chunk in file_chunks]
        print(f"  [{file_idx}/{len(files_to_reindex)}] {file_name}: {len(file_chunks)} chunks...", end="", flush=True)
        file_embeddings = embedder.embed_batch(file_texts)
        file_original_texts = [chunk["text"] for chunk in file_chunks]

        all_texts.extend(file_original_texts)
        all_embeddings.extend(file_embeddings)
        all_chunk_ids.extend(file_ids)
        indexed_files.add(file_name)
        embedded_count += len(file_chunks)
        print(f" done ({embedded_count}/{total_new_chunks} total)")

        _save_index(output_dir, all_texts, all_embeddings, all_chunk_ids, chunk_map, indexed_files, embedder.model, current_checksums)

    npz_size = (output_dir / "sentence_index.npz").stat().st_size / 1024 / 1024
    print("Index complete:")
    print(f"  npz: {npz_size:.1f} MB")
    print(f"  Searchable chunks indexed: {len(all_texts)}")
    print(f"  Embedding dim: {len(all_embeddings[0]) if all_embeddings else 0}")
    print(f"  Indexed files: {len(indexed_files)}")


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("Error: embedding API key not set")
        sys.exit(1)
    build_index(chunks_path, output_dir)
