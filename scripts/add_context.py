"""Generate contextual prefixes for each chunk using Anthropic's Contextual Retrieval approach.

For each chunk, sends the full document + chunk to an LLM and generates a short context
description that situates the chunk within the document. This context is stored in the
`context` field of each chunk in chunks.json.

When chunks are later embedded (build_index.py), the context + text is used together,
producing embeddings that capture document-level meaning.

Usage:
    python scripts/add_context.py [chunks.json] [full_texts.json]

Requires CEREBRAS_API_KEY environment variable.
Supports incremental processing — skips chunks that already have a `context` field.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

# Cerebras config
_BASE_URL = "https://api.cerebras.ai/v1"
_MODEL = "gpt-oss-120b"

# Maximum document chars to include as context
# Cerebras gpt-oss-120b has 128K context — keep well within limit
_MAX_DOC_CHARS = 80_000

# Prompt template (Japanese, tailored for accounting standards)
_SYSTEM_PROMPT = """\
あなたは会計基準文書の検索精度を向上させるための文脈生成アシスタントです。"""

_USER_PROMPT_TEMPLATE = """\
<document>
{document}
</document>

上記の文書の中から、以下のチャンク（断片）が切り出されています。

<chunk>
{chunk}
</chunk>

このチャンクを文書全体の中に位置づける、簡潔な文脈説明文を日本語で生成してください。
検索時にこのチャンクが適切にヒットするよう、以下の情報を含めてください：
- この文書の正式名称（例：企業会計基準第X号「...」）
- このチャンクが扱っているトピック・条項
- 関連する上位概念やキーワード

回答は文脈説明文のみを出力し、それ以外は何も出力しないでください。3文以内で簡潔にまとめてください。"""


def _truncate_document(text: str, max_chars: int = _MAX_DOC_CHARS) -> str:
    """Truncate long documents, keeping beginning and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... 中略 ...]\n\n" + text[-half:]


def _build_document_map(
    full_texts: dict, chunks_by_file: dict[str, list[dict]]
) -> dict[str, str]:
    """Map each file to its full document text.

    Uses full_texts.json if available, otherwise reconstructs from chunks.
    """
    doc_map = {}
    for file_name in chunks_by_file:
        if file_name in full_texts:
            doc_map[file_name] = full_texts[file_name]["text"]
        else:
            # Reconstruct from chunks (e-Gov files etc.)
            file_chunks = sorted(chunks_by_file[file_name], key=lambda c: c["page"])
            doc_map[file_name] = "\n\n".join(c["text"] for c in file_chunks)
    return doc_map


def generate_contexts(
    chunks_path: str,
    full_texts_path: str,
    api_key: str,
    batch_save_interval: int = 20,
):
    """Generate contextual prefixes for all chunks missing a `context` field."""

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    with open(full_texts_path, encoding="utf-8") as f:
        full_texts = json.load(f)

    # Group chunks by file
    chunks_by_file: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_file.setdefault(c["file"], []).append(c)

    # Build document map
    doc_map = _build_document_map(full_texts, chunks_by_file)

    # Find chunks needing context
    pending = [c for c in chunks if not c.get("context")]
    if not pending:
        print("All chunks already have context. Nothing to do.")
        return

    print(f"Total chunks: {len(chunks)}")
    print(f"Already have context: {len(chunks) - len(pending)}")
    print(f"Need context: {len(pending)}")
    print(f"Model: {_MODEL} via Cerebras API")

    client = OpenAI(api_key=api_key, base_url=_BASE_URL)
    processed = 0
    errors = 0
    start_time = time.time()

    for i, chunk in enumerate(pending):
        file_name = chunk["file"]
        doc_text = _truncate_document(doc_map.get(file_name, ""))

        if not doc_text.strip():
            print(f"  Warning: No document text for {file_name}, skipping {chunk['id']}")
            errors += 1
            continue

        user_msg = _USER_PROMPT_TEMPLATE.format(
            document=doc_text,
            chunk=chunk["text"],
        )

        # Retry with exponential backoff
        context_text = None
        for retry in range(5):
            try:
                response = client.chat.completions.create(
                    model=_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                context_text = response.choices[0].message.content.strip()
                time.sleep(0.3)  # Proactive rate-limit throttle (~3.3 req/sec)
                break
            except Exception as e:
                delay = 3 * (2**retry)
                print(f"  Retry {retry + 1}/5 for {chunk['id']}: {e}. Waiting {delay}s...")
                time.sleep(delay)

        if context_text:
            chunk["context"] = context_text
            processed += 1
        else:
            print(f"  Failed after retries: {chunk['id']}")
            errors += 1

        # Progress with ETA
        if (i + 1) % 10 == 0 or i == len(pending) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta_mins = (len(pending) - i - 1) / rate / 60
            print(
                f"  [{i + 1}/{len(pending)}] processed={processed} errors={errors} "
                f"rate={rate:.1f}/s ETA={eta_mins:.0f}min"
            )

        # Periodic save
        if (i + 1) % batch_save_interval == 0:
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"  Checkpoint saved at {i + 1}")

    # Final save
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {processed} contexts generated, {errors} errors")
    print(f"Output: {chunks_path}")


if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"
    full_texts_path = sys.argv[2] if len(sys.argv) > 2 else "data/full_texts.json"
    api_key = os.getenv("CEREBRAS_API_KEY", "")

    if not api_key:
        print("Error: CEREBRAS_API_KEY environment variable not set")
        sys.exit(1)

    generate_contexts(chunks_path, full_texts_path, api_key)
