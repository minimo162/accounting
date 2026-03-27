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

from src.arag.agent import Agent
from src.arag.config import Config, LLMConfig
from src.arag.llm import LLMClient
from src.arag.query_rewrite import QueryExpander
from src.arag.reranker import BaseReranker, HeuristicReranker, LLMReranker
from src.arag.retrieval import ChunkCorpus
from src.arag.tools import HybridSearchTool, KeywordSearchTool, ReadChunkTool, ReadDocumentTool, SemanticSearchTool, ToolRegistry
from src.embedding import create_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="会計基準Q&A", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Agent | None = None

GCS_BUCKET = os.getenv("GCS_BUCKET", "jp-accounting-chat-data")
GCS_PREFIX = os.getenv("GCS_PREFIX", "index")


def _download_from_gcs(bucket_name: str, prefix: str, local_dir: Path):
    from google.cloud import storage as gcs

    local_dir.mkdir(parents=True, exist_ok=True)
    index_dir = local_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = local_dir / "chunks.json"
    npz_path = index_dir / "sentence_index.npz"
    meta_path = index_dir / "sentence_meta.pkl"
    legacy_path = index_dir / "sentence_index.pkl"

    if chunks_path.exists() and npz_path.exists() and meta_path.exists():
        logger.info("Index files already exist locally, skipping download")
        return
    if chunks_path.exists() and legacy_path.exists():
        logger.info("Legacy index files already exist locally, skipping download")
        return

    logger.info(f"Downloading index from gs://{bucket_name}/{prefix}/...")
    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    if not chunks_path.exists():
        bucket.blob(f"{prefix}/chunks.json").download_to_filename(str(chunks_path))
        logger.info(f"Downloaded chunks.json ({chunks_path.stat().st_size // 1024} KB)")

    pdf_sources_path = local_dir / "pdf_sources.json"
    if not pdf_sources_path.exists():
        pdf_blob = bucket.blob(f"{prefix}/pdf_sources.json")
        if pdf_blob.exists():
            pdf_blob.download_to_filename(str(pdf_sources_path))
            logger.info("Downloaded pdf_sources.json")

    npz_blob = bucket.blob(f"{prefix}/sentence_index.npz")
    meta_blob = bucket.blob(f"{prefix}/sentence_meta.pkl")
    if npz_blob.exists() and meta_blob.exists():
        if not npz_path.exists():
            npz_blob.download_to_filename(str(npz_path))
        if not meta_path.exists():
            meta_blob.download_to_filename(str(meta_path))
    elif not legacy_path.exists():
        bucket.blob(f"{prefix}/sentence_index.pkl").download_to_filename(str(legacy_path))


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = _init_agent()
    return _agent


def _init_agent() -> Agent:
    config = Config.from_env()
    data_dir = Path(os.getenv("DATA_DIR", "data"))

    if os.getenv("USE_GCS", "").lower() in ("1", "true", "yes"):
        _download_from_gcs(GCS_BUCKET, GCS_PREFIX, data_dir)

    chunks = json.loads((data_dir / "chunks.json").read_text())
    logger.info(f"Loaded {len(chunks)} chunks")

    corpus = ChunkCorpus(chunks)
    embedder = create_embedder(config.embedding)
    llm_client = LLMClient(config.llm)

    registry = ToolRegistry()
    keyword_tool = KeywordSearchTool(corpus)
    semantic_tool = SemanticSearchTool(
        index_path=str(data_dir / "index" / "sentence_index.pkl"),
        embed_fn=embedder.embed_query,
        corpus=corpus,
    )
    if config.retrieval.reranker == "llm":
        rerank_llm = llm_client
        if config.llm.provider == "cerebras":
            rerank_llm = LLMClient(
                LLMConfig(
                    provider="cerebras",
                    model=os.getenv("RERANKER_LLM_MODEL", "llama3.1-8b"),
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    temperature=0.0,
                    max_tokens=256,
                )
            )
            logger.info("Using Cerebras-backed reranker model")
        reranker = LLMReranker(rerank_llm)
    elif config.retrieval.reranker == "none":
        reranker = BaseReranker()
    else:
        reranker = HeuristicReranker()
    registry.register(
        HybridSearchTool(
            semantic_tool=semantic_tool,
            keyword_tool=keyword_tool,
            query_expander=QueryExpander(config.retrieval, llm=llm_client),
            reranker=reranker,
            config=config.retrieval,
        )
    )
    registry.register(keyword_tool)
    registry.register(semantic_tool)
    registry.register(ReadChunkTool(corpus))
    registry.register(ReadDocumentTool(chunks))

    chunk_map = {chunk["id"]: chunk for chunk in chunks}
    pdf_sources = {}
    pdf_sources_path = data_dir / "pdf_sources.json"
    if pdf_sources_path.exists():
        pdf_sources = json.loads(pdf_sources_path.read_text())
        logger.info(f"Loaded {len(pdf_sources)} PDF source URLs")

    return Agent(config=config, tools=registry, chunk_map=chunk_map, pdf_sources=pdf_sources)


class ConversationTurn(BaseModel):
    question: str
    answer: str


class QuestionRequest(BaseModel):
    question: str
    history: list[ConversationTurn] = []


@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading agent and index...")
    get_agent()
    logger.info("Agent ready")


@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    agent = get_agent()
    result = await agent.arun(req.question, history=req.history)
    return {
        "answer": result["answer"],
        "metadata": {
            "loops": result["loops"],
            "chunks_read_count": result.get("cited_reference_count", 0),
            "read_chunk_count": result.get("read_chunk_count", 0),
            "stop_reason": result["stop_reason"],
            "total_cost": result.get("total_cost", 0.0),
            "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
        },
    }


@app.post("/api/ask/stream")
async def ask_question_stream(req: QuestionRequest):
    agent = get_agent()

    async def event_generator():
        try:
            async for event in agent.arun_stream(req.question, history=req.history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': f'エラーが発生しました: {str(e)}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


static_dir = Path("frontend/build")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
