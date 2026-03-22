"""Build sentence-level embedding index using Gemini embeddings."""

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.embedding.gemini import GeminiEmbedder

# Japanese-aware sentence splitting
_SENTENCE_RE = re.compile(r'(?<=[。．.！！\?？\n])\s*')


def split_sentences(text: str) -> list[str]:
    """Split Japanese text into sentences."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def build_index(chunks_path: str, output_dir: str, api_key: str):
    """Build sentence-level embedding index."""
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    # Split all chunks into sentences
    sentences = []
    sentence_to_chunk = []
    chunk_map = {}

    for chunk in chunks:
        chunk_map[chunk["id"]] = chunk
        sents = split_sentences(chunk["text"])
        for sent in sents:
            sentences.append(sent)
            sentence_to_chunk.append(chunk["id"])

    print(f"Split into {len(sentences)} sentences")

    # Embed all sentences
    embedder = GeminiEmbedder(api_key=api_key, model="gemini-embedding-2-preview")
    print("Embedding sentences with Gemini...")

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        embeddings = embedder.embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(sentences))}/{len(sentences)}")
        if i + batch_size < len(sentences):
            time.sleep(0.5)  # Rate limiting

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_array = embeddings_array / norms

    # Save index
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "sentences": sentences,
        "embeddings": embeddings_array,
        "sentence_to_chunk": sentence_to_chunk,
        "chunks": chunk_map,
        "model_name": embedder.model,
    }

    index_path = output_dir / "sentence_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(index, f)

    print(f"Index saved to {index_path}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Embedding dim: {embeddings_array.shape[1]}")
    print(f"  Index size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/index"
    api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    build_index(chunks_path, output_dir, api_key)
