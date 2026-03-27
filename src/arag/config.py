"""Configuration management for the accounting RAG system."""

import os
import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = ""  # "cerebras", "gemini", or auto-detect
    model: str = "gpt-oss-120b"
    api_key: str = ""
    base_url: str = "https://api.cerebras.ai/v1"
    temperature: float = 0.0
    max_tokens: int = 16384

    def __post_init__(self):
        if not self.provider:
            self.provider = os.getenv("LLM_PROVIDER", "cerebras")
        if self.provider == "cerebras":
            if not self.api_key:
                self.api_key = os.getenv("CEREBRAS_API_KEY", "")
            self.base_url = "https://api.cerebras.ai/v1"
            if not self.model or "gemini" in self.model:
                self.model = "gpt-oss-120b"
        else:
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY", "")


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"
    model: str = "gemini-embedding-2-preview"
    api_key: str = ""
    base_url: str = ""
    batch_size: int = 16

    def __post_init__(self):
        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.getenv("GEMINI_API_KEY", "")
            else:
                self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if not self.base_url:
            self.base_url = os.getenv("EMBEDDING_BASE_URL", "")


@dataclass
class RetrievalConfig:
    semantic_top_k: int = 30
    keyword_top_k: int = 30
    final_top_k: int = 10
    rrf_k: int = 60
    child_chunk_chars: int = 800
    child_chunk_overlap: int = 120
    enable_query_expansion: bool = True
    enable_hyde: bool = False
    reranker: str = "heuristic"  # heuristic | llm | none
    rerank_top_n: int = 30
    expansion_max_variants: int = 3

    def __post_init__(self):
        self.semantic_top_k = int(os.getenv("RETRIEVAL_SEMANTIC_TOP_K", self.semantic_top_k))
        self.keyword_top_k = int(os.getenv("RETRIEVAL_KEYWORD_TOP_K", self.keyword_top_k))
        self.final_top_k = int(os.getenv("RETRIEVAL_FINAL_TOP_K", self.final_top_k))
        self.rrf_k = int(os.getenv("RETRIEVAL_RRF_K", self.rrf_k))
        self.enable_query_expansion = os.getenv("RETRIEVAL_ENABLE_QUERY_EXPANSION", str(self.enable_query_expansion)).lower() in {"1", "true", "yes"}
        self.enable_hyde = os.getenv("RETRIEVAL_ENABLE_HYDE", str(self.enable_hyde)).lower() in {"1", "true", "yes"}
        self.reranker = os.getenv("RETRIEVAL_RERANKER", self.reranker)
        self.rerank_top_n = int(os.getenv("RETRIEVAL_RERANK_TOP_N", self.rerank_top_n))
        self.expansion_max_variants = int(os.getenv("RETRIEVAL_EXPANSION_MAX_VARIANTS", self.expansion_max_variants))


@dataclass
class AgentConfig:
    max_loops: int = 25
    max_token_budget: int = 128000
    verbose: bool = False


@dataclass
class DataConfig:
    chunks_file: str = "data/chunks.json"
    index_dir: str = "data/index"
    gcs_bucket: str = ""
    gcs_prefix: str = "accounting-index"

    def __post_init__(self):
        if not self.gcs_bucket:
            self.gcs_bucket = os.getenv("GCS_BUCKET", "jp-accounting-chat-data")


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(
            llm=LLMConfig(**d.get("llm", {})),
            embedding=EmbeddingConfig(**d.get("embedding", {})),
            retrieval=RetrievalConfig(**d.get("retrieval", {})),
            agent=AgentConfig(**d.get("agent", {})),
            data=DataConfig(**d.get("data", {})),
        )
