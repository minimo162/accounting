import unittest

from src.arag.config import RetrievalConfig
from src.arag.context import AgentContext
from src.arag.query_rewrite import QueryProfile
from src.arag.retrieval import SearchResult
from src.arag.tools.hybrid_search import HybridSearchTool


class FakeSemanticTool:
    def __init__(self, results_by_query, aliases_by_file=None):
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []
        self._corpus = self
        self.aliases_by_file = aliases_by_file or {}

    def search(self, query: str, top_k: int):
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])

    def get_parent(self, chunk_id: str):
        return {"id": chunk_id, "text": "本文"}

    def get_document_aliases(self, file_name: str):
        return self.aliases_by_file.get(file_name, set())


class FakeKeywordTool:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def search(self, keywords: list[str], top_k: int):
        key = tuple(keywords)
        self.calls.append((key, top_k))
        return self.results_by_query.get(key, [])


class FakeQueryExpander:
    def __init__(self, expansions, exact=False, hyde_doc=None, profile=None):
        self.expansions = expansions
        self.exact = exact
        self.hyde_doc = hyde_doc
        self._profile = profile

    def expand(self, query: str) -> list[str]:
        return list(self.expansions)

    def generate_hypothetical_document(self, query: str) -> str | None:
        return self.hyde_doc

    def is_exact_query(self, query: str) -> bool:
        return self.exact

    def profile(self, query: str) -> QueryProfile:
        if self._profile is not None:
            return self._profile
        return QueryProfile(
            query=query,
            complexity="simple" if self.exact else "moderate",
            search_mode="keyword_first" if self.exact else "balanced",
            keywords=[],
        )


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
        metadata={"id": parent_id, "file": parent_id.split(":")[0]},
    )


class HybridSearchTests(unittest.TestCase):
    def test_query_expansion_can_prioritize_canonical_standard_title(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)

        expansions = expander.expand("リース会計基準の改正点を教えてください")

        self.assertEqual(
            expansions,
            [
                "リース会計基準の改正点を教えてください",
                "リースに関する会計基準 改正",
            ],
        )

    def test_query_profile_marks_exact_queries_as_simple_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        profile = QueryExpander.profile("企業会計基準第13号 第10項とは")

        self.assertEqual(profile.complexity, "simple")
        self.assertEqual(profile.search_mode, "keyword_first")

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

    def test_search_prefers_canonical_standard_variant_for_semantic_query(self):
        semantic = FakeSemanticTool({"リースに関する会計基準 改正": [make_result("doc-a:p1", 0.9)]})
        keyword = FakeKeywordTool(
            {
                ("リース会計基準の改正点を教えてください",): [make_result("doc-b:p1", 0.8)],
                ("リースに関する会計基準 改正",): [make_result("doc-a:p1", 0.85)],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(
                ["リース会計基準の改正点を教えてください", "リースに関する会計基準 改正"]
            ),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        tool.search("リース会計基準の改正点を教えてください", top_k=5)

        self.assertEqual([query for query, _ in semantic.calls], ["リースに関する会計基準 改正"])

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

    def test_change_query_penalizes_unchanged_sections(self):
        unchanged = SearchResult(
            chunk_id="old:p1",
            parent_id="old:p1",
            score=0.8,
            source="企業会計基準第13号 > 用語の定義及びリース取引の分類",
            snippet="改正前会計基準における定義を変更していない。",
            text="改正前会計基準における定義を変更していない。",
            metadata={"doc_type": "企業会計基準"},
        )
        changed = SearchResult(
            chunk_id="new:p1",
            parent_id="new:p1",
            score=0.7,
            source="企業会計基準第34号 > 本会計基準の公表",
            snippet="借手のすべてのリースについて資産及び負債を計上する。",
            text="借手のすべてのリースについて資産及び負債を計上する新たな取扱いを導入する。",
            metadata={"doc_type": "企業会計基準"},
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool({}),
            keyword_tool=FakeKeywordTool({}),
            query_expander=FakeQueryExpander(["リース会計基準 改正"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        ranked = tool._apply_change_intent_boosts("リース会計基準の改正点", [unchanged, changed])

        self.assertEqual([item.parent_id for item in ranked], ["new:p1", "old:p1"])

    def test_topic_alignment_boosts_exact_document_title_alias(self):
        standard = SearchResult(
            chunk_id="lease-main.pdf:p1",
            parent_id="lease-main.pdf:p1",
            score=0.6,
            source="企業会計基準第34号",
            snippet="本会計基準の公表",
            text="改正内容",
            metadata={"doc_type": "企業会計基準", "file": "lease-main.pdf"},
        )
        guidance = SearchResult(
            chunk_id="lease-guidance.pdf:p1",
            parent_id="lease-guidance.pdf:p1",
            score=0.9,
            source="企業会計基準適用指針第33号",
            snippet="適用指針",
            text="改正内容",
            metadata={"doc_type": "適用指針", "file": "lease-guidance.pdf"},
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool(
                {},
                aliases_by_file={
                    "lease-main.pdf": {"リースに関する会計基準"},
                    "lease-guidance.pdf": {"リースに関する会計基準の適用指針"},
                },
            ),
            keyword_tool=FakeKeywordTool({}),
            query_expander=FakeQueryExpander(["リース会計基準 改正", "リースに関する会計基準 改正"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        ranked = tool._apply_topic_alignment_boosts("リース会計基準の改正点", [guidance, standard])

        self.assertEqual([item.parent_id for item in ranked], ["lease-main.pdf:p1", "lease-guidance.pdf:p1"])

    def test_low_confidence_search_runs_corrective_query(self):
        guidance = SearchResult(
            chunk_id="lease-guidance.pdf:p1",
            parent_id="lease-guidance.pdf:p1",
            score=0.9,
            source="企業会計基準適用指針第33号",
            snippet="適用指針の補足説明",
            text="適用指針の補足説明",
            metadata={"doc_type": "適用指針", "file": "lease-guidance.pdf"},
        )
        standard = SearchResult(
            chunk_id="lease-main.pdf:p1",
            parent_id="lease-main.pdf:p1",
            score=0.7,
            source="企業会計基準第34号",
            snippet="借手のすべてのリースについて使用権資産及びリース負債を計上する。",
            text="借手のすべてのリースについて使用権資産及びリース負債を計上する。",
            metadata={"doc_type": "企業会計基準", "file": "lease-main.pdf"},
        )
        profile = QueryProfile(
            query="リース会計基準の改正点を教えてください",
            complexity="complex",
            search_mode="balanced",
            keywords=["リース", "改正"],
            canonical_focus_query="リースに関する会計基準 改正 使用権資産 リース負債",
            corrective_query="リースに関する会計基準 改正 使用権資産 リース負債",
        )
        semantic = FakeSemanticTool(
            {"リースに関する会計基準 改正 使用権資産 リース負債": [guidance]},
            aliases_by_file={
                "lease-main.pdf": {"リースに関する会計基準"},
                "lease-guidance.pdf": {"リースに関する会計基準の適用指針"},
            },
        )
        keyword = FakeKeywordTool(
            {
                ("リース会計基準の改正点を教えてください",): [guidance],
                ("リースに関する会計基準 改正 使用権資産 リース負債",): [standard],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(
                ["リース会計基準の改正点を教えてください"],
                profile=profile,
            ),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )
        context = AgentContext()

        text, log = tool.execute(context, query="リース会計基準の改正点を教えてください", top_k=3)

        self.assertEqual(log["corrective_query"], "リースに関する会計基準 改正 使用権資産 リース負債")
        self.assertIn(("リースに関する会計基準 改正 使用権資産 リース負債",), [keywords for keywords, _ in keyword.calls])
        self.assertIn("企業会計基準第34号", text)


if __name__ == "__main__":
    unittest.main()
