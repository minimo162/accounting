"""OpenAI-compatible embedding provider."""

import logging

import numpy as np
from openai import OpenAI

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _extract_embedding(self, response) -> list[np.ndarray]:
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def embed_text(self, text: str) -> np.ndarray | None:
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return self._extract_embedding(response)[0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray | None:
        return self.embed_text(query)

    def embed_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[np.ndarray]:
        del task_type
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            return self._extract_embedding(response)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            sample = self.embed_text(texts[0]) if texts else None
            dim = len(sample) if sample is not None else 1536
            results: list[np.ndarray] = []
            for text in texts:
                emb = self.embed_text(text)
                results.append(emb if emb is not None else np.zeros(dim, dtype=np.float32))
            return results
