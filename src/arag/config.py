"""Configuration management for the accounting RAG system."""

import os
import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = ""  # "deepseek", "cerebras", "gemini", or auto-detect
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 0

    def __post_init__(self):
        if not self.provider:
            self.provider = os.getenv("LLM_PROVIDER", "deepseek")
        self.provider = self.provider.lower()

        env_model = os.getenv("LLM_MODEL", "").strip()
        env_base_url = os.getenv("LLM_BASE_URL", "").strip()
        env_temperature = os.getenv("LLM_TEMPERATURE", "").strip()
        env_max_tokens = os.getenv("LLM_MAX_TOKENS", "").strip()

        if env_model:
            self.model = env_model
        if env_base_url:
            self.base_url = env_base_url
        if env_temperature:
            self.temperature = float(env_temperature)
        if env_max_tokens:
            self.max_tokens = int(env_max_tokens)

        if self.provider == "deepseek":
            if not self.api_key:
                self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
            if not self.base_url:
                self.base_url = "https://api.deepseek.com/v1"
            if not self.model or self.model == "gpt-oss-120b" or "gemini" in self.model.lower():
                self.model = "deepseek-chat"
            if self.max_tokens <= 0:
                self.max_tokens = 8192
        elif self.provider == "cerebras":
            if not self.api_key:
                self.api_key = os.getenv("CEREBRAS_API_KEY", "")
            if not self.base_url:
                self.base_url = "https://api.cerebras.ai/v1"
            if not self.model or "deepseek" in self.model.lower() or "gemini" in self.model.lower():
                self.model = "gpt-oss-120b"
            if self.max_tokens <= 0:
                self.max_tokens = 16384
        elif self.provider == "gemini":
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
            if not self.model or self.model == "gpt-oss-120b" or "deepseek" in self.model.lower():
                self.model = "gemini-2.5-flash"
            if self.max_tokens <= 0:
                self.max_tokens = 8192
        else:
            if not self.api_key:
                self.api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            if self.max_tokens <= 0:
                self.max_tokens = 16384


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
                self.api_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
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
    enable_llm_query_expansion: bool = True
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
        self.enable_llm_query_expansion = os.getenv("RETRIEVAL_ENABLE_LLM_QUERY_EXPANSION", str(self.enable_llm_query_expansion)).lower() in {"1", "true", "yes"}
        self.enable_hyde = os.getenv("RETRIEVAL_ENABLE_HYDE", str(self.enable_hyde)).lower() in {"1", "true", "yes"}
        self.reranker = os.getenv("RETRIEVAL_RERANKER", self.reranker).lower()
        self.rerank_top_n = int(os.getenv("RETRIEVAL_RERANK_TOP_N", self.rerank_top_n))
        self.expansion_max_variants = int(os.getenv("RETRIEVAL_EXPANSION_MAX_VARIANTS", self.expansion_max_variants))


@dataclass
class AgentConfig:
    max_loops: int = 25
    max_token_budget: int = 128000
    verbose: bool = False
    nudge_at_loop: int = 8
    wrap_up_after_searches: int = 0
    force_final_after_searches: int = 0
    force_final_after_reads: int = 0

    def __post_init__(self):
        self.max_loops = int(os.getenv("AGENT_MAX_LOOPS", self.max_loops))
        self.max_token_budget = int(os.getenv("AGENT_MAX_TOKEN_BUDGET", self.max_token_budget))
        self.verbose = os.getenv("AGENT_VERBOSE", str(self.verbose)).lower() in {"1", "true", "yes"}
        self.nudge_at_loop = int(os.getenv("AGENT_NUDGE_AT_LOOP", self.nudge_at_loop))
        self.wrap_up_after_searches = int(os.getenv("AGENT_WRAP_UP_AFTER_SEARCHES", self.wrap_up_after_searches))
        self.force_final_after_searches = int(os.getenv("AGENT_FORCE_FINAL_AFTER_SEARCHES", self.force_final_after_searches))
        self.force_final_after_reads = int(os.getenv("AGENT_FORCE_FINAL_AFTER_READS", self.force_final_after_reads))


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

    def __post_init__(self):
        self._apply_provider_tuning()

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

    def _apply_provider_tuning(self):
        if self.llm.provider != "deepseek":
            return

        if "RETRIEVAL_RERANKER" not in os.environ:
            self.retrieval.reranker = "heuristic"
        if "RETRIEVAL_ENABLE_LLM_QUERY_EXPANSION" not in os.environ:
            self.retrieval.enable_llm_query_expansion = False
        if "RETRIEVAL_EXPANSION_MAX_VARIANTS" not in os.environ:
            self.retrieval.expansion_max_variants = min(self.retrieval.expansion_max_variants, 2)
        if "RETRIEVAL_RERANK_TOP_N" not in os.environ:
            self.retrieval.rerank_top_n = min(self.retrieval.rerank_top_n, 12)
        if "RETRIEVAL_SEMANTIC_TOP_K" not in os.environ:
            self.retrieval.semantic_top_k = min(self.retrieval.semantic_top_k, 18)
        if "RETRIEVAL_KEYWORD_TOP_K" not in os.environ:
            self.retrieval.keyword_top_k = min(self.retrieval.keyword_top_k, 18)
        if "RETRIEVAL_FINAL_TOP_K" not in os.environ:
            self.retrieval.final_top_k = min(self.retrieval.final_top_k, 6)
        if "AGENT_NUDGE_AT_LOOP" not in os.environ:
            self.agent.nudge_at_loop = min(self.agent.nudge_at_loop, 4)
        if "AGENT_MAX_LOOPS" not in os.environ:
            self.agent.max_loops = min(self.agent.max_loops, 12)
        if "AGENT_WRAP_UP_AFTER_SEARCHES" not in os.environ:
            self.agent.wrap_up_after_searches = 2
        if "AGENT_FORCE_FINAL_AFTER_SEARCHES" not in os.environ:
            self.agent.force_final_after_searches = 3
        if "AGENT_FORCE_FINAL_AFTER_READS" not in os.environ:
            self.agent.force_final_after_reads = 3
