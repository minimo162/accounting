"""FastAPI backend for the accounting Q&A chat app."""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.arag.config import Config
from src.arag.agent import Agent
from src.arag.tools import KeywordSearchTool, SemanticSearchTool, ReadChunkTool, ToolRegistry
from src.embedding.gemini import GeminiEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="会計基準Q&A", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_agent: Agent | None = None

GCS_BUCKET = os.getenv("GCS_BUCKET", "jp-accounting-chat-data")
GCS_PREFIX = os.getenv("GCS_PREFIX", "index")


def _download_from_gcs(bucket_name: str, prefix: str, local_dir: Path):
    """Download index files from GCS if they don't exist locally."""
    from google.cloud import storage as gcs

    local_dir.mkdir(parents=True, exist_ok=True)
    index_dir = local_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = local_dir / "chunks.json"
    index_path = index_dir / "sentence_index.pkl"

    if chunks_path.exists() and index_path.exists():
        logger.info("Index files already exist locally, skipping download")
        return

    logger.info(f"Downloading index from gs://{bucket_name}/{prefix}/...")
    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    if not chunks_path.exists():
        blob = bucket.blob(f"{prefix}/chunks.json")
        blob.download_to_filename(str(chunks_path))
        logger.info(f"Downloaded chunks.json ({chunks_path.stat().st_size // 1024} KB)")

    if not index_path.exists():
        blob = bucket.blob(f"{prefix}/sentence_index.pkl")
        blob.download_to_filename(str(index_path))
        logger.info(f"Downloaded sentence_index.pkl ({index_path.stat().st_size // 1024 // 1024} MB)")


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = _init_agent()
    return _agent


def _init_agent() -> Agent:
    config = Config.from_env()
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    # Download from GCS if needed (for Cloud Run)
    if os.getenv("USE_GCS", "").lower() in ("1", "true", "yes"):
        _download_from_gcs(GCS_BUCKET, GCS_PREFIX, data_dir)

    # Load chunks
    chunks_path = data_dir / "chunks.json"
    with open(chunks_path) as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    # Init embedder
    embedder = GeminiEmbedder(
        api_key=config.embedding.api_key,
        model=config.embedding.model,
    )

    # Init tools
    registry = ToolRegistry()
    registry.register(KeywordSearchTool(chunks))
    registry.register(SemanticSearchTool(
        index_path=str(data_dir / "index" / "sentence_index.pkl"),
        embed_fn=embedder.embed_query,
    ))
    registry.register(ReadChunkTool(chunks))

    chunk_map = {c["id"]: c for c in chunks}
    return Agent(config=config, tools=registry, chunk_map=chunk_map)


class QuestionRequest(BaseModel):
    question: str


@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    """Non-streaming endpoint."""
    agent = get_agent()
    result = await agent.arun(req.question)
    return {
        "answer": result["answer"],
        "metadata": {
            "loops": result["loops"],
            "chunks_read_count": result.get("chunks_read_count", 0),
            "stop_reason": result["stop_reason"],
            "total_cost": result.get("total_cost", 0.0),
            "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
        },
    }


@app.post("/api/ask/stream")
async def ask_question_stream(req: QuestionRequest):
    """SSE streaming endpoint."""
    agent = get_agent()

    async def event_generator():
        async for event in agent.arun_stream(req.question):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Mount static files for frontend (built Svelte)
static_dir = Path("frontend/build")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
