"""FastAPI backend for the accounting Q&A chat app."""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.arag.agent import Agent
from src.arag.config import Config, LLMConfig
from src.arag.llm import LLMClient
from src.arag.observability import generate_request_id, question_sha1, request_context, structured_log
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


class UIEventRequest(BaseModel):
    event: str
    request_id: str | None = None
    reference_number: int | None = None
    reference_id: str | None = None
    question_length: int | None = None
    prior_answer_count: int | None = None
    reference_count: int | None = None
    uncertainty_present: bool | None = None
    developer_mode: bool | None = None


def _request_log_fields(
    req: QuestionRequest,
    request_id: str,
    path: str,
    *,
    stream: bool,
    elapsed_ms: float | None = None,
    result: dict | None = None,
    status_code: int = 200,
    error: Exception | None = None,
    monitor_case_id: str = "",
) -> dict:
    result = result or {}
    observability = dict(result.get("observability", {}))
    if "loops" in result and "loops" not in observability:
        observability["loops"] = result.get("loops", 0)
    if "stop_reason" in result and "stop_reason" not in observability:
        observability["stop_reason"] = result.get("stop_reason", "")
    if "total_retrieved_tokens" in result and "retrieved_tokens" not in observability:
        observability["retrieved_tokens"] = result.get("total_retrieved_tokens", 0)
    if "cited_reference_count" in result and "cited_reference_count" not in observability:
        observability["cited_reference_count"] = result.get("cited_reference_count", 0)
        observability["zero_reference"] = (result.get("cited_reference_count", 0) or 0) == 0
    if "read_chunk_count" in result and "read_chunk_count" not in observability:
        observability["read_chunk_count"] = result.get("read_chunk_count", 0)
    if "references" in result and "reference_count" not in observability:
        observability["reference_count"] = len(result.get("references", []))
    if "query_class" in result and "query_class" not in observability:
        observability["query_class"] = result.get("query_class")
    payload = {
        "request_id": request_id,
        "monitor_case_id": monitor_case_id or None,
        "path": path,
        "stream": stream,
        "status_code": status_code,
        "question_length": len(req.question),
        "question_sha1": question_sha1(req.question),
        "history_turns": len(req.history),
        **observability,
    }
    if elapsed_ms is not None:
        payload["elapsed_ms"] = round(elapsed_ms, 1)
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)
    return payload


def _response_metadata(request_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "query_class": result.get("query_class"),
        "loops": result.get("loops", 0),
        "chunks_read_count": result.get("cited_reference_count", 0),
        "read_chunk_count": result.get("read_chunk_count", 0),
        "stop_reason": result.get("stop_reason", ""),
        "total_cost": result.get("total_cost", 0.0),
        "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
        "evidence_coverage": result.get("evidence_coverage", {}),
        "uncertainty": result.get("uncertainty", {}),
    }


@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading agent and index...")
    get_agent()
    logger.info("Agent ready")


@app.post("/api/ask")
async def ask_question(req: QuestionRequest, request: Request):
    agent = get_agent()
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    monitor_case_id = request.headers.get("X-Monitor-Case-Id", "").strip()
    start = time.perf_counter()

    structured_log(
        logger,
        logging.INFO,
        "api_request_start",
        request_id=request_id,
        monitor_case_id=monitor_case_id or None,
        path="/api/ask",
        stream=False,
        question_length=len(req.question),
        question_sha1=question_sha1(req.question),
        history_turns=len(req.history),
    )

    try:
        with request_context(request_id, monitor_case_id=monitor_case_id):
            result = await agent.arun(req.question, history=req.history)

        elapsed_ms = (time.perf_counter() - start) * 1000
        structured_log(
            logger,
            logging.INFO,
            "api_request_complete",
            **_request_log_fields(
                req,
                request_id,
                "/api/ask",
                stream=False,
                elapsed_ms=elapsed_ms,
                result=result,
                monitor_case_id=monitor_case_id,
            ),
        )
        payload = {
            "answer": result["answer"],
            "references": result.get("references", []),
            "source_url_map": result.get("source_url_map", {}),
            "metadata": _response_metadata(request_id, result),
        }
        return JSONResponse(payload, headers={"X-Request-ID": request_id})
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        structured_log(
            logger,
            logging.ERROR,
            "api_request_error",
            **_request_log_fields(
                req,
                request_id,
                "/api/ask",
                stream=False,
                elapsed_ms=elapsed_ms,
                status_code=500,
                error=exc,
                monitor_case_id=monitor_case_id,
            ),
        )
        raise


@app.post("/api/ask/stream")
async def ask_question_stream(req: QuestionRequest, request: Request):
    agent = get_agent()
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    monitor_case_id = request.headers.get("X-Monitor-Case-Id", "").strip()
    start = time.perf_counter()

    structured_log(
        logger,
        logging.INFO,
        "api_request_start",
        request_id=request_id,
        monitor_case_id=monitor_case_id or None,
        path="/api/ask/stream",
        stream=True,
        question_length=len(req.question),
        question_sha1=question_sha1(req.question),
        history_turns=len(req.history),
    )

    async def event_generator():
        result_summary: dict | None = None
        try:
            with request_context(request_id, monitor_case_id=monitor_case_id):
                async for event in agent.arun_stream(req.question, history=req.history):
                    if event.get("type") == "done":
                        data = dict(event.get("data", {}))
                        data.setdefault("request_id", request_id)
                        event = {**event, "data": data}
                        result_summary = {
                            "loops": data.get("loops", 0),
                            "stop_reason": data.get("stop_reason", ""),
                            "total_retrieved_tokens": data.get("total_retrieved_tokens", 0),
                            "cited_reference_count": data.get("chunks_read_count", 0),
                            "read_chunk_count": data.get("read_chunk_count", 0),
                            "query_class": data.get("query_class"),
                            "evidence_coverage": data.get("evidence_coverage", {}),
                            "uncertainty": data.get("uncertainty", {}),
                            "observability": {
                                "request_id": request_id,
                                "monitor_case_id": monitor_case_id or None,
                                "query_class": data.get("query_class"),
                                "loops": data.get("loops", 0),
                                "stop_reason": data.get("stop_reason", ""),
                                "retrieved_tokens": data.get("total_retrieved_tokens", 0),
                                "cited_reference_count": data.get("chunks_read_count", 0),
                                "read_chunk_count": data.get("read_chunk_count", 0),
                                "zero_reference": (data.get("chunks_read_count", 0) or 0) == 0,
                            },
                        }
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"Stream error: {e}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            structured_log(
                logger,
                logging.ERROR,
                "api_request_error",
                **_request_log_fields(
                    req,
                    request_id,
                    "/api/ask/stream",
                    stream=True,
                    elapsed_ms=elapsed_ms,
                    status_code=500,
                    error=e,
                    result=result_summary,
                    monitor_case_id=monitor_case_id,
                ),
            )
            yield f"data: {json.dumps({'type': 'error', 'data': f'エラーが発生しました: {str(e)}'}, ensure_ascii=False)}\n\n"
            return
        elapsed_ms = (time.perf_counter() - start) * 1000
        structured_log(
            logger,
            logging.INFO,
            "api_request_complete",
            **_request_log_fields(
                req,
                request_id,
                "/api/ask/stream",
                stream=True,
                elapsed_ms=elapsed_ms,
                result=result_summary,
                monitor_case_id=monitor_case_id,
            ),
        )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id},
    )


@app.post("/api/ui-event")
async def record_ui_event(event: UIEventRequest, request: Request):
    structured_log(
        logger,
        logging.INFO,
        "ui_event",
        event_name=event.event,
        request_id=event.request_id or request.headers.get("X-Request-ID") or None,
        reference_number=event.reference_number,
        reference_id=event.reference_id,
        question_length=event.question_length,
        prior_answer_count=event.prior_answer_count,
        reference_count=event.reference_count,
        uncertainty_present=event.uncertainty_present,
        developer_mode=event.developer_mode,
        user_agent=request.headers.get("user-agent", "") or None,
    )
    return {"ok": True}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


static_dir = Path("frontend/build")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
