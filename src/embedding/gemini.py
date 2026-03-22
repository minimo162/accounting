"""Gemini embedding client using google.genai SDK."""

import logging
import numpy as np
from google import genai

logger = logging.getLogger(__name__)


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-exp-03-07"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> np.ndarray | None:
        """Embed a single text string."""
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            return np.array(result.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def embed_query(self, query: str) -> np.ndarray | None:
        """Embed a query string."""
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=query,
                config={"task_type": "RETRIEVAL_QUERY"},
            )
            return np.array(result.embeddings[0].values, dtype=np.float32)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def embed_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[np.ndarray]:
        """Embed a batch of texts."""
        results = []
        # Process in smaller batches for API limits
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config={"task_type": task_type},
                )
                for emb in result.embeddings:
                    results.append(np.array(emb.values, dtype=np.float32))
            except Exception as e:
                logger.error(f"Batch embedding failed at {i}: {e}")
                # Try one by one as fallback
                for text in batch:
                    try:
                        r = self.client.models.embed_content(
                            model=self.model,
                            contents=text,
                            config={"task_type": task_type},
                        )
                        results.append(np.array(r.embeddings[0].values, dtype=np.float32))
                    except Exception as e2:
                        logger.error(f"Single embedding failed: {e2}")
                        results.append(np.zeros(3072, dtype=np.float32))
        return results
