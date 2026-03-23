"""Configuration management for the accounting RAG system."""

import os
import json
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model: str = "gemini-3.1-flash-lite-preview"
    api_key: str = ""
    base_url: str = ""  # Not used for Gemini
    temperature: float = 0.0
    max_tokens: int = 16384

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")


@dataclass
class EmbeddingConfig:
    model: str = "gemini-embedding-2-preview"
    api_key: str = ""
    batch_size: int = 16

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("GEMINI_API_KEY", "")


@dataclass
class AgentConfig:
    max_loops: int = 15
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
            agent=AgentConfig(**d.get("agent", {})),
            data=DataConfig(**d.get("data", {})),
        )
