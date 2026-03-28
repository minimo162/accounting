import os
import unittest
from unittest.mock import patch

from src.arag.config import Config, EmbeddingConfig, LLMConfig


class ConfigTests(unittest.TestCase):
    def test_llm_config_defaults_to_deepseek(self):
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig()

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(config.max_tokens, 8192)

    def test_llm_config_uses_cerebras_defaults_when_requested(self):
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "cerebras", "CEREBRAS_API_KEY": "test-cerebras-key"},
            clear=True,
        ):
            config = LLMConfig()

        self.assertEqual(config.provider, "cerebras")
        self.assertEqual(config.model, "gpt-oss-120b")
        self.assertEqual(config.base_url, "https://api.cerebras.ai/v1")
        self.assertEqual(config.max_tokens, 16384)
        self.assertEqual(config.api_key, "test-cerebras-key")

    def test_llm_config_honors_explicit_llm_overrides(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "LLM_MODEL": "deepseek-reasoner",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MAX_TOKENS": "4096",
                "LLM_TEMPERATURE": "0.2",
                "DEEPSEEK_API_KEY": "test-deepseek-key",
            },
            clear=True,
        ):
            config = LLMConfig()

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-reasoner")
        self.assertEqual(config.base_url, "https://example.com/v1")
        self.assertEqual(config.max_tokens, 4096)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.api_key, "test-deepseek-key")

    def test_embedding_config_accepts_google_api_key(self):
        with patch.dict(
            os.environ,
            {"GOOGLE_API_KEY": "test-google-key"},
            clear=True,
        ):
            config = EmbeddingConfig()

        self.assertEqual(config.api_key, "test-google-key")

    def test_config_applies_faster_retrieval_defaults_for_deepseek(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-deepseek-key",
            },
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.retrieval.reranker, "heuristic")
        self.assertFalse(config.retrieval.enable_llm_query_expansion)
        self.assertEqual(config.retrieval.expansion_max_variants, 2)
        self.assertEqual(config.retrieval.rerank_top_n, 12)
        self.assertEqual(config.retrieval.semantic_top_k, 18)
        self.assertEqual(config.retrieval.keyword_top_k, 18)
        self.assertEqual(config.agent.nudge_at_loop, 4)
        self.assertEqual(config.agent.max_loops, 12)
        self.assertEqual(config.agent.wrap_up_after_searches, 2)
        self.assertEqual(config.agent.force_final_after_searches, 3)
        self.assertEqual(config.agent.force_final_after_reads, 3)

    def test_config_honors_explicit_retrieval_overrides_for_deepseek(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-deepseek-key",
                "RETRIEVAL_RERANKER": "llm",
                "RETRIEVAL_ENABLE_LLM_QUERY_EXPANSION": "true",
                "RETRIEVAL_EXPANSION_MAX_VARIANTS": "4",
                "RETRIEVAL_RERANK_TOP_N": "24",
                "RETRIEVAL_SEMANTIC_TOP_K": "25",
                "RETRIEVAL_KEYWORD_TOP_K": "26",
            },
            clear=True,
        ):
            config = Config.from_env()

        self.assertEqual(config.retrieval.reranker, "llm")
        self.assertTrue(config.retrieval.enable_llm_query_expansion)
        self.assertEqual(config.retrieval.expansion_max_variants, 4)
        self.assertEqual(config.retrieval.rerank_top_n, 24)
        self.assertEqual(config.retrieval.semantic_top_k, 25)
        self.assertEqual(config.retrieval.keyword_top_k, 26)


if __name__ == "__main__":
    unittest.main()
