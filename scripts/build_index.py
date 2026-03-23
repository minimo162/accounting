"""Build sentence-level embedding index using Gemini embeddings.

Features:
- Checkpoint saving every 5000 sentences (resumes from last checkpoint on restart)
- Retry with exponential backoff on API errors
- Progress tracking
"""

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

CHECKPOINT_INTERVAL = 5000  # Save checkpoint every N sentences
MAX_RETRIES = 5
RETRY_BASE_DELAY = 5  # seconds


def split_sentences(text: str) -> list[str]:
    """Split Japanese text into sentences."""
    sentences = _SENTENCE_RE.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def build_index(chunks_path: str, output_dir: str, api_key: str):
    """Build sentence-level embedding index with checkpointing."""
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

    total_sentences = len(sentences)
    print(f"Split into {total_sentences} sentences")

    # Check for checkpoint
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pkl"

    all_embeddings = []
    start_idx = 0

    if checkpoint_path.exists():
        print("Found checkpoint, resuming...")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        all_embeddings = checkpoint["embeddings"]
        start_idx = checkpoint["next_idx"]
        print(f"  Resuming from sentence {start_idx}/{total_sentences} ({len(all_embeddings)} embeddings loaded)")

    # Embed sentences
    embedder = GeminiEmbedder(api_key=api_key, model="gemini-embedding-2-preview")
    print("Embedding sentences with Gemini...")

    batch_size = 100
    for i in range(start_idx, total_sentences, batch_size):
        batch = sentences[i : i + batch_size]

        # Retry with exponential backoff
        for retry in range(MAX_RETRIES):
            try:
                embeddings = embedder.embed_batch(batch)
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** retry)
                    print(f"  Error at {i}: {e}. Retrying in {delay}s... (attempt {retry + 2}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    print(f"  FATAL: Failed after {MAX_RETRIES} retries at {i}: {e}")
                    # Save checkpoint before exiting
                    print(f"  Saving checkpoint at {i}...")
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump({"embeddings": all_embeddings, "next_idx": i}, f)
                    print(f"  Checkpoint saved. Re-run to resume.")
                    sys.exit(1)

        processed = min(i + batch_size, total_sentences)
        print(f"  Embedded {processed}/{total_sentences} ({processed * 100 // total_sentences}%)")

        # Save checkpoint periodically
        if (i - start_idx) > 0 and (i - start_idx) % CHECKPOINT_INTERVAL == 0:
            print(f"  Saving checkpoint at {processed}...")
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"embeddings": all_embeddings, "next_idx": processed}, f)

        if i + batch_size < total_sentences:
            time.sleep(0.5)  # Rate limiting

    print(f"Embedding complete: {len(all_embeddings)} vectors")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_array = embeddings_array / norms

    # Save index
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

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up")

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
