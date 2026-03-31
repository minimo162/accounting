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

    def exact_keyword_terms(self, query: str) -> list[str]:
        if self.exact:
            from src.arag.query_rewrite import QueryExpander

            return QueryExpander.exact_keyword_terms(query)
        return [query]

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

    def test_query_profile_treats_formal_title_as_exact_query(self):
        from src.arag.query_rewrite import QueryExpander

        profile = QueryExpander.profile("固定資産の減損に係る会計基準では減損の兆候をどう判断しますか")

        self.assertEqual(profile.search_mode, "keyword_first")

    def test_exact_keyword_terms_split_standard_and_article(self):
        from src.arag.query_rewrite import QueryExpander

        terms = QueryExpander.exact_keyword_terms("企業会計基準第13号 第10項では借手のリースをどのように扱いますか")

        self.assertEqual(terms, ["企業会計基準第13号", "第10項", "借手", "リース"])

    def test_exact_section_hit_matches_legacy_numeric_body_label(self):
        from src.arag.query_rewrite import QueryExpander

        hits = QueryExpander.count_exact_section_hits(["第10項"], ["10. 借手は通常の売買取引に準じて会計処理を行う。"])

        self.assertEqual(hits, 1)

    def test_split_exact_constraints_keeps_standard_number_as_doc_term(self):
        from src.arag.query_rewrite import QueryExpander

        doc_terms, section_terms = QueryExpander.split_exact_constraints(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか"
        )

        self.assertEqual(doc_terms, ["企業会計基準第13号"])
        self.assertEqual(section_terms, ["第10項"])

    def test_exact_keyword_term_sets_include_legacy_variants(self):
        from src.arag.query_rewrite import QueryExpander

        term_sets = QueryExpander.exact_keyword_term_sets(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか"
        )

        self.assertIn(["企業会計基準第13号", "第10項"], term_sets)
        self.assertIn(
            ["企業会計基準第13号", "企業会計基準第13 号", "第10項", "10項", "10.", "10．", "借手", "リース"],
            term_sets,
        )

    def test_query_expansion_adds_hedge_accounting_focus_terms(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)

        expansions = expander.expand("繰延ヘッジを適用するための主な要件を教えてください")
        profile = QueryExpander.profile("繰延ヘッジを適用するための主な要件を教えてください")

        self.assertEqual(
            expansions,
            [
                "繰延ヘッジを適用するための主な要件を教えてください",
                "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト 要件",
            ],
        )
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, expansions[1])

    def test_query_profile_marks_short_term_and_low_value_lease_as_simple_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "借手は短期リースや少額リースをどのように扱いますか"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "simple")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertEqual(profile.corrective_query, "借手 短期リース 少額リース")
        self.assertEqual(
            expansions,
            [
                "借手は短期リースや少額リースをどのように扱いますか",
                "借手 短期リース 少額リース",
            ],
        )

    def test_query_profile_marks_tax_rate_change_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "税効果会計で繰延税金資産・負債の計算に使う税率は何ですか。税率変更時の扱いも教えてください"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertEqual(profile.corrective_query, "税効果会計 税率 繰延税金資産 繰延税金負債 税率変更")
        self.assertEqual(
            expansions,
            [
                query,
                "税効果会計 税率 繰延税金資産 繰延税金負債 税率変更",
            ],
        )

    def test_query_profile_marks_principal_agent_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "収益認識基準における本人と代理人の区分はどう判断しますか"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "収益認識基準 本人 代理人 支配 総額 純額 判断 区分 比較")
        self.assertEqual(
            expansions,
            [
                query,
                "収益認識基準 本人 代理人 支配 総額 純額 判断 区分 比較",
            ],
        )

    def test_query_profile_keeps_exact_clause_query_simple(self):
        from src.arag.query_rewrite import QueryExpander

        query = "企業会計基準第13号第10項では借手のリースをどのように扱いますか"

        profile = QueryExpander.profile(query)

        self.assertEqual(profile.complexity, "simple")
        self.assertEqual(profile.search_mode, "keyword_first")

    def test_query_profile_marks_performance_obligation_identification_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "収益認識基準 履行義務 別個 保守サービス 値引き 契約 識別")
        self.assertEqual(
            expansions,
            [
                query,
                "収益認識基準 履行義務 別個 保守サービス 値引き 契約 識別",
            ],
        )

    def test_query_profile_marks_variable_consideration_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "収益認識基準における変動対価はどのように見積もり、いつ収益に反映しますか"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "収益認識基準 変動対価 見積り 制約 収益")
        self.assertEqual(
            expansions,
            [
                query,
                "収益認識基準 変動対価 見積り 制約 収益",
            ],
        )

    def test_query_profile_marks_contract_modification_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "収益認識基準で契約変更はどのように処理しますか。既存契約の継続か、新しい契約として扱うかの観点で教えてください"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertEqual(profile.corrective_query, "収益認識基準 契約変更 別個 既存 新しい契約 履行義務 取引価格")
        self.assertEqual(
            expansions,
            [
                query,
                "収益認識基準 契約変更 別個 既存 新しい契約 履行義務 取引価格",
            ],
        )

    def test_query_profile_marks_r_and_d_cost_treatment_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "研究開発費はどのように会計処理しますか。ソフトウェア開発費との違いにも触れてください"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "研究開発費 発生時 費用 ソフトウェア 資産 比較")
        self.assertEqual(
            expansions,
            [
                query,
                "研究開発費 発生時 費用 ソフトウェア 資産 比較",
            ],
        )

    def test_query_profile_marks_impairment_indication_as_moderate_keyword_first(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "固定資産の減損会計では、どのような場合に減損の兆候があると判断しますか"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "moderate")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "固定資産 減損 兆候 回収可能価額 使用価値 正味売却価額 判断 場合")
        self.assertEqual(
            expansions,
            [
                query,
                "固定資産 減損 兆候 回収可能価額 使用価値 正味売却価額 判断 場合",
            ],
        )

    def test_query_profile_prefers_keyword_first_for_complex_lease_revision_detail(self):
        from src.arag.query_rewrite import QueryExpander

        expander = QueryExpander(RetrievalConfig(expansion_max_variants=2), llm=None)
        query = "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください"

        profile = QueryExpander.profile(query)
        expansions = expander.expand(query)

        self.assertEqual(profile.complexity, "complex")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.detail_seeking)
        self.assertEqual(profile.corrective_query, "リースに関する会計基準 改正 借手 貸手 経過措置")
        self.assertEqual(
            expansions,
            [
                query,
                "リースに関する会計基準 改正 借手 貸手 経過措置",
            ],
        )

    def test_query_profile_detects_judgment_validation_query(self):
        from src.arag.query_rewrite import QueryExpander

        query = (
            "単体で計上される売却損（未実現損失）は連結調整で消去するべきものでないと判断しています。"
            "依拠: 連結財務諸表に関する会計基準 第36条。"
            "この理解が妥当か確認してください。"
        )

        profile = QueryExpander.profile(query)

        self.assertEqual(profile.complexity, "complex")
        self.assertEqual(profile.search_mode, "keyword_first")
        self.assertTrue(profile.verification_mode)
        self.assertTrue(profile.judgment_validation)
        self.assertEqual(
            profile.validation_claims,
            ["単体で計上される売却損（未実現損失）は連結調整で消去するべきものでないと判断しています"],
        )
        self.assertEqual(
            profile.verification_claims,
            [
                {
                    "claim": "単体で計上される売却損（未実現損失）は連結調整で消去するべきものでないと判断しています",
                    "cited_references": ["第36条", "連結財務諸表に関する会計基準"],
                    "doc_terms": ["連結財務諸表に関する会計基準"],
                    "section_terms": ["第36条"],
                    "search_query": "連結財務諸表に関する会計基準 第36条 単体 計上される売却損 未実現損失 連結調整",
                }
            ],
        )
        self.assertEqual(profile.cited_references, ["第36条", "連結財務諸表に関する会計基準"])
        self.assertEqual(
            profile.corrective_query,
            "第36条 連結財務諸表に関する会計基準 単体 計上される売却損 未実現損失 連結調整 消去するべき 判断",
        )

    def test_query_profile_does_not_mark_normal_explanatory_query_as_verification_mode(self):
        from src.arag.query_rewrite import QueryExpander

        query = "連結財務諸表に関する会計基準第36条の内容を教えてください"

        profile = QueryExpander.profile(query)

        self.assertFalse(profile.verification_mode)
        self.assertFalse(profile.judgment_validation)
        self.assertEqual(profile.verification_claims, [])

    def test_detail_seeking_keyword_first_prefers_corrective_query_over_canonical_focus(self):
        query = "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください"
        profile = QueryProfile(
            query=query,
            complexity="moderate",
            search_mode="keyword_first",
            keywords=["履行義務", "保守サービス", "値引き"],
            canonical_focus_query="収益認識基準 履行義務 別個",
            corrective_query="収益認識基準 履行義務 別個 保守サービス 値引き 契約 識別",
            detail_seeking=True,
            detail_terms=["識別"],
        )
        semantic = FakeSemanticTool({})
        keyword = FakeKeywordTool(
            {
                ("収益認識基準", "履行義務", "別個", "保守サービス", "値引き", "契約", "識別"): [
                    make_result("doc-a:p1", 0.9)
                ],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        results, _, _ = tool.search(query, top_k=5)

        self.assertEqual(
            keyword.calls[0][0],
            ("収益認識基準", "履行義務", "別個", "保守サービス", "値引き", "契約", "識別"),
        )
        self.assertEqual([item.parent_id for item in results], ["doc-a:p1"])

    def test_keyword_first_focus_query_uses_split_keywords(self):
        query = "収益認識基準における本人と代理人の区分はどう判断しますか"
        profile = QueryProfile(
            query=query,
            complexity="moderate",
            search_mode="keyword_first",
            keywords=["本人", "代理人", "収益認識基準", "区分"],
            canonical_focus_query="収益認識基準 本人 代理人 支配 総額 純額",
            corrective_query="収益認識基準 本人 代理人 支配 総額 純額",
        )
        semantic = FakeSemanticTool({})
        keyword = FakeKeywordTool(
            {
                ("収益認識基準", "本人", "代理人", "支配", "総額", "純額"): [make_result("doc-a:p1", 0.9)],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        results, _, _ = tool.search(query, top_k=5)

        self.assertEqual(keyword.calls[0][0], ("収益認識基準", "本人", "代理人", "支配", "総額", "純額"))
        self.assertEqual([item.parent_id for item in results], ["doc-a:p1"])

    def test_exact_query_uses_original_query_terms_instead_of_canonical_focus(self):
        query = "企業会計基準第13号第10項では借手のリースをどのように扱いますか"
        profile = QueryProfile(
            query=query,
            complexity="simple",
            search_mode="keyword_first",
            keywords=["企業会計基準第13号", "第10項", "借手", "リース"],
            canonical_focus_query="企業に関する会計基準 借手",
            corrective_query="企業に関する会計基準 借手",
        )
        semantic = FakeSemanticTool({})
        keyword = FakeKeywordTool(
            {
                ("企業会計基準第13号", "第10項", "借手", "リース"): [make_result("doc-a:p1", 0.9)],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], exact=True, profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        results, _, _ = tool.search(query, top_k=5)

        self.assertEqual(keyword.calls[0][0], ("企業会計基準第13号", "第10項"))
        self.assertIn(("企業会計基準第13号", "第10項", "借手", "リース"), [call[0] for call in keyword.calls])

    def test_judgment_validation_exact_query_uses_corrective_semantic_backfill(self):
        query = (
            "単体で計上される売却損（未実現損失）は連結調整で消去するべきものでないと判断しています。"
            "依拠: 連結財務諸表に関する会計基準 第36条。"
            "この理解が妥当か確認してください。"
        )
        corrective_query = "第36条 連結財務諸表に関する会計基準 単体 計上される売却損 未実現損失 連結調整 消去するべき 判断"
        profile = QueryProfile(
            query=query,
            complexity="complex",
            search_mode="keyword_first",
            keywords=["売却損", "未実現損失", "連結調整", "消去"],
            canonical_focus_query=corrective_query,
            corrective_query=corrective_query,
            judgment_validation=True,
            validation_claims=["単体で計上される売却損（未実現損失）は連結調整で消去するべきものでないと判断しています"],
            cited_references=["第36条", "連結財務諸表に関する会計基準"],
        )
        semantic = FakeSemanticTool({corrective_query: [make_result("doc-a:p36", 0.91)]})
        keyword = FakeKeywordTool({})
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], exact=True, profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        results, _, _ = tool.search(query, top_k=5)

        self.assertEqual(semantic.calls[0][0], corrective_query)
        self.assertEqual([item.parent_id for item in results], ["doc-a:p36"])

    def test_exact_query_uses_semantic_backfill_when_keyword_hits_lack_exact_evidence(self):
        query = "企業会計基準第13号第10項では借手のリースをどのように扱いますか"
        profile = QueryProfile(
            query=query,
            complexity="simple",
            search_mode="keyword_first",
            keywords=["企業会計基準第13号", "第10項", "借手", "リース"],
        )
        semantic = FakeSemanticTool(
            {
                query: [
                    SearchResult(
                        chunk_id="doc-sem:p2",
                        parent_id="doc-sem:p2",
                        score=0.92,
                        source="企業会計基準第13号 > 第10項",
                        snippet="10. 借手は通常の売買取引に準じて処理する。",
                        text="10. 借手は通常の売買取引に準じて処理する。",
                        metadata={"standard_no": "企業会計基準第13号", "section_title": "第10項"},
                    )
                ]
            }
        )
        keyword = FakeKeywordTool(
            {
                ("企業会計基準第13号", "第10項", "借手", "リース"): [make_result("doc-a:p1", 0.85)],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], exact=True, profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        results, _, _ = tool.search(query, top_k=5)

        self.assertEqual([called_query for called_query, _ in semantic.calls], [query])
        self.assertEqual(results[0].parent_id, "doc-sem:p2")

    def test_keyword_first_focus_query_adds_semantic_backfill_when_top_hit_is_too_narrow(self):
        query = "収益認識基準における本人と代理人の区分はどう判断しますか"
        profile = QueryProfile(
            query=query,
            complexity="moderate",
            search_mode="keyword_first",
            keywords=["本人", "代理人", "収益認識基準", "区分"],
            canonical_focus_query="収益認識基準 本人 代理人 支配 総額 純額",
            corrective_query="収益認識基準 本人 代理人 支配 総額 純額",
        )
        semantic = FakeSemanticTool({query: [make_result("doc-sem:p1", 0.9)]})
        keyword = FakeKeywordTool(
            {
                ("収益認識基準", "本人", "代理人", "支配", "総額", "純額"): [
                    SearchResult(
                        chunk_id="doc-key:c1",
                        parent_id="doc-key:p1",
                        score=0.8,
                        source="収益認識基準第29号 > 代理人として行動する場合の注記",
                        snippet="代理人として行動する場合の開示例",
                        text="代理人として行動する場合の注記例のみを示す。",
                        metadata={"id": "doc-key:p1", "file": "doc-key.pdf"},
                    )
                ],
            }
        )
        tool = HybridSearchTool(
            semantic_tool=semantic,
            keyword_tool=keyword,
            query_expander=FakeQueryExpander([query], profile=profile),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        tool.search(query, top_k=5)

        self.assertEqual([called_query for called_query, _ in semantic.calls], [query])

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
                ("企業会計基準第13号", "第10項"): [
                    SearchResult(
                        chunk_id="doc-a:p1",
                        parent_id="doc-a:p1",
                        score=0.95,
                        source="企業会計基準第13号 > 第10項",
                        snippet="第10項 借手は通常の売買取引に準じて処理する。",
                        text="第10項 借手は通常の売買取引に準じて処理する。",
                        metadata={"standard_no": "企業会計基準第13号", "section_title": "第10項"},
                    ),
                    SearchResult(
                        chunk_id="doc-b:p1",
                        parent_id="doc-b:p1",
                        score=0.85,
                        source="企業会計基準第13号 > 第10項",
                        snippet="第10項 オペレーティング・リース取引については通常の賃貸借取引に準じて処理する。",
                        text="第10項 オペレーティング・リース取引については通常の賃貸借取引に準じて処理する。",
                        metadata={"standard_no": "企業会計基準第13号", "section_title": "第10項"},
                    ),
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
        self.assertEqual(keyword.calls[0][0], ("企業会計基準第13号", "第10項"))
        self.assertEqual(reranker.calls, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(expansions, ["企業会計基準第13号 第10項"])
        self.assertIsNone(hyde_doc)

    def test_exact_constraint_filter_keeps_only_matching_standard_and_article(self):
        exact_match = SearchResult(
            chunk_id="std13:p10",
            parent_id="std13:p10",
            score=0.8,
            source="企業会計基準第13号 > 第10項",
            snippet="第10項 使用権資産を計上する。",
            text="第10項 使用権資産を計上する。",
            metadata={"doc_type": "企業会計基準", "standard_no": "企業会計基準第13号", "section_title": "第10項"},
        )
        wrong_standard = SearchResult(
            chunk_id="std20:p10",
            parent_id="std20:p10",
            score=1.2,
            source="実務対応報告第20号 > 第10項",
            snippet="第10項 注記を記載する。",
            text="第10項 注記を記載する。",
            metadata={"doc_type": "実務対応報告", "standard_no": "実務対応報告第20号", "section_title": "第10項"},
        )
        wrong_section = SearchResult(
            chunk_id="std13:p11",
            parent_id="std13:p11",
            score=1.1,
            source="企業会計基準第13号 > 第11項",
            snippet="第11項 別の取扱い。",
            text="第11項 別の取扱い。",
            metadata={"doc_type": "企業会計基準", "standard_no": "企業会計基準第13号", "section_title": "第11項"},
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool({}),
            keyword_tool=FakeKeywordTool({}),
            query_expander=FakeQueryExpander(["企業会計基準第13号 第10項"], exact=True),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        filtered = tool._filter_exact_mismatch_results(
            "企業会計基準第13号 第10項では借手のリースをどのように扱いますか",
            [wrong_standard, wrong_section, exact_match],
        )

        self.assertEqual([item.parent_id for item in filtered], ["std13:p10"])

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

    def test_search_filters_low_value_parent_results(self):
        toc = SearchResult(
            chunk_id="toc:p1",
            parent_id="toc:p1",
            score=0.9,
            source="移管指針第9号 > 目 次",
            snippet="目次",
            text="目 次\nヘッジ会計の適用要件\nリスク管理方針文書の記載事項",
            metadata={"doc_type": "適用指針"},
        )
        actual = SearchResult(
            chunk_id="real:p30",
            parent_id="real:p30",
            score=0.7,
            source="移管指針第9号 > （ヘッジ取引開始時（事前テスト））",
            snippet="正式な文書による明確化",
            text="企業はヘッジ取引開始時に正式な文書によって明確にしなければならない。",
            metadata={"doc_type": "適用指針"},
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool({}),
            keyword_tool=FakeKeywordTool({}),
            query_expander=FakeQueryExpander(["ヘッジ会計 要件"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        ranked = tool._filter_low_value_parent_results([toc, actual])

        self.assertEqual([item.parent_id for item in ranked], ["real:p30"])

    def test_focus_term_boosts_prioritize_core_hedge_requirement_sections(self):
        peripheral = SearchResult(
            chunk_id="qa:p24",
            parent_id="qa:p24",
            score=0.8,
            source="移管指針第12号 > Q&A",
            snippet="金利スワップをヘッジ取引として扱う場合のQ&A。",
            text="Q&Aの断片。ヘッジ会計という語はあるが、有効性や正式な文書には触れない。",
            metadata={"doc_type": "適用指針"},
        )
        core = SearchResult(
            chunk_id="guidance:p30",
            parent_id="guidance:p30",
            score=0.6,
            source="移管指針第9号 > （ヘッジ取引開始時（事前テスト））",
            snippet="正式な文書によりヘッジ手段とヘッジ対象を明確にする。",
            text="ヘッジ会計の適用要件として、正式な文書と有効性の事前テストが求められる。",
            metadata={"doc_type": "適用指針", "section_title": "ヘッジ会計の適用要件"},
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool({}),
            keyword_tool=FakeKeywordTool({}),
            query_expander=FakeQueryExpander(["繰延ヘッジ 要件"]),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )

        ranked = tool._apply_focus_term_boosts("繰延ヘッジを適用するための主な要件を教えてください", [peripheral, core])

        self.assertEqual([item.parent_id for item in ranked], ["guidance:p30", "qa:p24"])

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

    def test_execute_sets_current_search_query_to_effective_focus_query(self):
        profile = QueryProfile(
            query="繰延ヘッジを適用するための主な要件を教えてください",
            complexity="simple",
            search_mode="keyword_first",
            keywords=["繰延ヘッジ", "要件"],
            canonical_focus_query="ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
            corrective_query="ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
        )
        keyword = FakeKeywordTool(
            {
                (
                    "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
                ): [make_result("doc-a:p1", 0.9)]
            }
        )
        tool = HybridSearchTool(
            semantic_tool=FakeSemanticTool({}),
            keyword_tool=keyword,
            query_expander=FakeQueryExpander(
                ["繰延ヘッジを適用するための主な要件を教えてください"],
                profile=profile,
            ),
            reranker=FakeReranker(),
            config=RetrievalConfig(),
        )
        context = AgentContext()

        _, log = tool.execute(context, query=profile.query, top_k=3)

        self.assertEqual(
            context.current_search_query,
            "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
        )
        self.assertEqual(log["effective_query"], context.current_search_query)


if __name__ == "__main__":
    unittest.main()
