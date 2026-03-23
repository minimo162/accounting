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
        """Embed a batch of texts with retry logic."""
        import time as _time

        results = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            success = False
            for retry in range(5):
                try:
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config={"task_type": task_type},
                    )
                    for emb in result.embeddings:
                        results.append(np.array(emb.values, dtype=np.float32))
                    success = True
                    break
                except Exception as e:
                    delay = 3 * (2 ** retry)
                    logger.warning(f"Batch embedding failed at {i} (retry {retry+1}/5): {e}. Waiting {delay}s...")
                    _time.sleep(delay)

            if not success:
                logger.error(f"Batch failed after 5 retries at {i}, falling back to individual")
                for text in batch:
                    for r_retry in range(3):
                        try:
                            r = self.client.models.embed_content(
                                model=self.model,
                                contents=text,
                                config={"task_type": task_type},
                            )
                            results.append(np.array(r.embeddings[0].values, dtype=np.float32))
                            break
                        except Exception as e2:
                            if r_retry < 2:
                                _time.sleep(3 * (2 ** r_retry))
                            else:
                                logger.error(f"Single embedding failed: {e2}")
                                results.append(np.zeros(3072, dtype=np.float32))
        return results
