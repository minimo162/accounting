"""Embedding provider factory."""

from .base import BaseEmbedder
from .gemini import GeminiEmbedder
from .openai_compat import OpenAICompatibleEmbedder
from ..arag.config import EmbeddingConfig


def create_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    provider = config.provider.lower()
    if provider == "gemini":
        return GeminiEmbedder(api_key=config.api_key, model=config.model)
    if provider in {"openai", "openai_compat", "openai-compatible"}:
        return OpenAICompatibleEmbedder(
            api_key=config.api_key,
            model=config.model,
            base_url=config.base_url or None,
        )
    raise ValueError(f"Unsupported embedding provider: {config.provider}")
