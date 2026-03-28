import unittest

from src.arag.config import RetrievalConfig
from src.arag.context import AgentContext
from src.arag.retrieval import SearchResult
from src.arag.tools.hybrid_search import HybridSearchTool


class FakeSemanticTool:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []
        self._corpus = self

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])

    def get_parent(self, chunk_id: str):
        return {"id": chunk_id, "text": "本文"}


class FakeKeywordTool:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search(self, keywords: list[str], top_k: int):
        key = tuple(keywords)
        self.calls.append((key, top_k))
        return self.results_by_query.get(key, [])


class FakeQueryExpander:
    def __init__(self, expansions, exact=False, hyde_doc=None):
        self.expansions = expansions
        self.exact = exact
        self.hyde_doc = hyde_doc

    def expand(self, query: str) -> list[str]:
        return list(self.expansions)

    def generate_hypothetical_document(self, query: str) -> str | None:
        return self.hyde_doc

    def is_exact_query(self, query: str) -> bool:
        return self.exact


class FakeReranker:
    def __init__(self, is_expensive=False):
        self.is_expensive = is_expensive
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        self.calls.append((query, results))
        return results


def make_result(parent_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=parent_id,
        parent_id=parent_id,
        score=score,
        source=f"{parent_id} source",
        snippet=f"{parent_id} snippet",
        text=f"{parent_id} text",
        metadata={"id": parent_id},
    )


class HybridSearchTests(unittest.TestCase):
    def test_search_uses_semantic_only_for_primary_non_exact_query(self):
        semantic = FakeSemanticTool({"リース 会計": [make_result("doc-a:p1", 0.9)]})
        keyword = FakeKeywordTool(
            {
                ("リース 会計",): [make_result("doc-a:p1", 0.8)],
                ("使用権資産",): [make_result("doc-b:p1", 0.7)],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(["リース 会計", "使用権資産"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        tool.search("リース 会計", top_k=5)

        self.assertEqual([query for query, _ in semantic.calls], ["リース 会計"])
        self.assertEqual([keywords for keywords, _ in keyword.calls], [("リース 会計",), ("使用権資産",)])

    def test_search_skips_semantic_and_expensive_rerank_for_exact_query_with_keyword_hits(self):
        semantic = FakeSemanticTool({})
        keyword = FakeKeywordTool(
            {
                ("企業会計基準第13号 第10項",): [
                    make_result("doc-a:p1", 0.95),
                    make_result("doc-b:p1", 0.85),
                ]
            }
        )
        reranker = FakeReranker(is_expensive=True)
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(["企業会計基準第13号 第10項"], exact=True),
            reranker=reranker,
            config=RetrievalConfig(rerank_top_n=10),
        )

        results, expansions, hyde_doc = tool.search("企業会計基準第13号 第10項", top_k=2)

        self.assertEqual([query for query, _ in semantic.calls], [])
        self.assertEqual([keywords for keywords, _ in keyword.calls], [("企業会計基準第13号 第10項",)])
        self.assertEqual(reranker.calls, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(expansions, ["企業会計基準第13号 第10項"])
        self.assertIsNone(hyde_doc)

    def test_execute_reuses_cached_results_for_same_query(self):
        semantic = FakeSemanticTool({"リース 会計": [make_result("doc-a:p1", 0.9)]})
        keyword = FakeKeywordTool({("リース 会計",): [make_result("doc-a:p1", 0.8)]})
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(["リース 会計"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )
        context = AgentContext()

        first_text, first_log = tool.execute(context, query="リース 会計", top_k=5)
        second_text, second_log = tool.execute(context, query="リース 会計", top_k=5)

        self.assertEqual(first_text, second_text)
        self.assertFalse(first_log.get("cached", False))
        self.assertTrue(second_log["cached"])
        self.assertEqual(len(semantic.calls), 1)
        self.assertEqual(len(keyword.calls), 1)


if __name__ == "__main__":
    unittest.main()
