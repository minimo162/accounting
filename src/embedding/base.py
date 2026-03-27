"""Embedding provider interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract embedding provider."""

    model: str

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray | None:
        """Embed a single document string."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray | None:
        """Embed a single query string."""

    @abstractmethod
    def embed_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[np.ndarray]:
        """Embed a batch of texts."""
