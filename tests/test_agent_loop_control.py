import asyncio
import json
import unittest

from src.arag.agent import Agent
from src.arag.config import AgentConfig, Config, DataConfig, EmbeddingConfig, LLMConfig, RetrievalConfig
from src.arag.context import AgentContext
from src.arag.observability import request_context
from src.arag.tools.base import BaseTool
from src.arag.tools.registry import ToolRegistry


class AgentLoopControlTests(unittest.TestCase):
    class FakeLLM:
        def __init__(self):
            self.messages = None

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=None):
            self.messages = messages
            return {"message": {"content": "final answer"}, "cost": 0.0}

        def count_message_tokens(self, messages):
            return 0

    class CoverageReviewLLM:
        def __init__(self):
            self.messages = None

        def count_message_tokens(self, messages):
            return 0

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=None):
            self.messages = messages
            return {
                "message": {
                    "content": "## 結論\n- 借手は使用権資産を計上します[lease-main.pdf:p10]。"
                },
                "cost": 0.0,
            }

    class FakeTools:
        def get_schemas(self, context=None):
            return []

    class DummyTool(BaseTool):
        def __init__(self, name: str):
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        def get_schema(self):
            return {"name": self._name, "parameters": {"type": "object", "properties": {}}}

        def execute(self, context, **kwargs):
            return "", {}

    class NaturalAnswerLLM:
        def count_message_tokens(self, messages):
            return 0

        async def achat(self, messages, tools=None, temperature=0.0, max_tokens=None):
            return {"message": {"content": "## 結論\n- 借手は使用権資産を計上します[lease-main.pdf:p10]。"}, "cost": 0.0}

    def make_agent(self) -> Agent:
        config = Config(
            llm=LLMConfig(provider="deepseek", api_key="test-key"),
            embedding=EmbeddingConfig(),
            retrieval=RetrievalConfig(),
            agent=AgentConfig(
                max_loops=12,
                nudge_at_loop=4,
                wrap_up_after_searches=2,
                force_final_after_searches=3,
                force_final_after_reads=3,
            ),
            data=DataConfig(),
        )
        agent = Agent.__new__(Agent)
        agent.config = config
        agent.max_loops = config.agent.max_loops
        agent.max_token_budget = config.agent.max_token_budget
        agent.verbose = config.agent.verbose
        agent.nudge_at_loop = config.agent.nudge_at_loop
        agent.wrap_up_after_searches = config.agent.wrap_up_after_searches
        agent.force_final_after_searches = config.agent.force_final_after_searches
        agent.force_final_after_reads = config.agent.force_final_after_reads
        agent.chunk_map = {}
        agent.pdf_sources = {}
        return agent

    def test_should_nudge_after_two_searches_and_reads(self):
        agent = self.make_agent()
        context = AgentContext()
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)

        self.assertTrue(agent._should_nudge_wrap_up(context))

    def test_should_force_after_three_searches_and_reads(self):
        agent = self.make_agent()
        context = AgentContext()
        for _ in range(3):
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)

        self.assertTrue(agent._should_force_wrap_up(context))

    def test_does_not_force_before_read_budget(self):
        agent = self.make_agent()
        context = AgentContext()
        for _ in range(3):
            context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)

        self.assertFalse(agent._should_force_wrap_up(context))

    def test_search_budget_counts_keyword_and_semantic_searches(self):
        agent = self.make_agent()
        context = AgentContext()
        context.add_retrieval_log("keyword_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("semantic_search", 10)
        context.add_retrieval_log("read_chunk", 10)

        self.assertTrue(agent._should_nudge_wrap_up(context))

    def test_tool_registry_keeps_all_search_tools_before_hybrid_search(self):
        registry = ToolRegistry()
        for tool_name in ("hybrid_search", "keyword_search", "semantic_search", "read_chunk"):
            registry.register(self.DummyTool(tool_name))
        context = AgentContext()

        names = [entry["function"]["name"] for entry in registry.get_schemas(context)]

        self.assertEqual(names, ["hybrid_search", "keyword_search", "semantic_search", "read_chunk"])

    def test_tool_registry_hides_standalone_search_tools_after_hybrid_search(self):
        registry = ToolRegistry()
        for tool_name in ("hybrid_search", "keyword_search", "semantic_search", "read_chunk", "read_document"):
            registry.register(self.DummyTool(tool_name))
        context = AgentContext()
        context.add_retrieval_log("hybrid_search", 10)

        names = [entry["function"]["name"] for entry in registry.get_schemas(context)]

        self.assertEqual(names, ["hybrid_search", "read_chunk", "read_document"])

    def test_tool_registry_hides_read_document_for_exact_queries(self):
        registry = ToolRegistry()
        for tool_name in ("hybrid_search", "keyword_search", "semantic_search", "read_chunk", "read_document"):
            registry.register(self.DummyTool(tool_name))
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項では借手のリースをどのように扱いますか", {"complexity": "simple"})

        names = [entry["function"]["name"] for entry in registry.get_schemas(context)]

        self.assertEqual(names, ["hybrid_search", "keyword_search", "semantic_search", "read_chunk"])

    def test_tool_registry_allows_read_document_after_exact_shortfall(self):
        registry = ToolRegistry()
        for tool_name in ("hybrid_search", "keyword_search", "semantic_search", "read_chunk", "read_document"):
            registry.register(self.DummyTool(tool_name))
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項では借手のリースをどのように扱いますか", {"complexity": "simple"})
        context.add_search_entry({"exact_shortfall": True})
        context.add_retrieval_log("hybrid_search", 10, metadata={"exact_shortfall": True})

        names = [entry["function"]["name"] for entry in registry.get_schemas(context)]

        self.assertEqual(names, ["read_chunk", "read_document"])

    def test_tool_registry_hides_hybrid_search_after_first_exact_query_search(self):
        registry = ToolRegistry()
        for tool_name in ("hybrid_search", "keyword_search", "semantic_search", "read_chunk", "read_document"):
            registry.register(self.DummyTool(tool_name))
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項では借手のリースをどのように扱いますか", {"complexity": "simple"})
        context.add_retrieval_log("hybrid_search", 10)

        names = [entry["function"]["name"] for entry in registry.get_schemas(context)]

        self.assertEqual(names, ["read_chunk"])

    def test_simple_query_uses_tighter_wrap_up_budget(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項とは", {"complexity": "simple"})
        context.add_retrieval_log("keyword_search", 10)
        context.add_retrieval_log("read_chunk", 10)

        self.assertTrue(agent._should_nudge_wrap_up(context))

    def test_simple_query_forces_after_two_searches_and_reads(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項とは", {"complexity": "simple"})
        for _ in range(2):
            context.add_retrieval_log("keyword_search", 10)
            context.add_retrieval_log("read_chunk", 10)

        self.assertTrue(agent._should_force_wrap_up(context))

    def test_high_confidence_evidence_reduces_force_budget_for_complex_query(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question("リース会計基準の改正点を教えてください", {"complexity": "complex"})
        for _ in range(2):
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.add_search_entry({"confidence": 0.81})
        context.add_search_entry({"confidence": 0.76})
        for idx in range(4):
            context.set_evidence_note(f"chunk-{idx}", "evidence")

        self.assertTrue(agent._should_force_wrap_up(context))

    def test_build_cross_reference_message_prompts_additional_search(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question("企業結合後の会計処理を確認したいです", {"complexity": "moderate"})
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "ketsugou.pdf:p12",
            "未実現損益の消去については企業会計基準第22号第36条参照。",
            source="企業結合に関する会計基準 > 取得後の会計処理",
        )

        message = agent._build_cross_reference_message(context)

        self.assertIsNotNone(message)
        self.assertIn("企業会計基準第22号 第36条", message["content"])
        self.assertEqual(context.cross_reference_nudges_sent, 1)

    def test_build_cross_reference_message_stops_after_two_nudges(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question("企業結合後の会計処理を確認したいです", {"complexity": "moderate"})
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "ketsugou.pdf:p12",
            "未実現損益の消去については企業会計基準第22号第36条参照。",
            source="企業結合に関する会計基準 > 取得後の会計処理",
        )
        context.cross_reference_nudges_sent = 2

        message = agent._build_cross_reference_message(context)

        self.assertIsNone(message)

    def test_context_initializes_verification_results(self):
        context = AgentContext()
        context.set_question(
            "この判断は妥当ですか",
            {
                "complexity": "complex",
                "verification_mode": True,
                "verification_claims": [
                    {
                        "claim": "再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しない",
                        "search_query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                        "doc_terms": ["企業会計基準第22号"],
                        "section_terms": ["第36条"],
                        "cited_references": ["企業会計基準第22号", "第36条"],
                        "target_transaction": "土地譲渡 / 未実現損失",
                    }
                ],
            },
        )

        self.assertEqual(len(context.verification_results), 1)
        self.assertEqual(context.verification_results[0]["judgment"], "△要注意")
        self.assertEqual(context.verification_results[0]["status"], "pending")

    def test_context_marks_verification_claim_supported_when_exact_evidence_is_read(self):
        context = AgentContext()
        context.set_question(
            "この判断は妥当ですか",
            {
                "complexity": "complex",
                "verification_mode": True,
                "verification_claims": [
                    {
                        "claim": "再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しない",
                        "search_query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                        "doc_terms": ["企業会計基準第22号"],
                        "section_terms": ["第36条"],
                        "cited_references": ["企業会計基準第22号", "第36条"],
                        "target_transaction": "土地譲渡 / 未実現損失",
                    }
                ],
            },
        )
        context.add_search_entry(
            {
                "query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                "effective_query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                "chunk_ids": ["consol.pdf:p36"],
            }
        )
        context.set_evidence_note(
            "consol.pdf:p36",
            "第36条では、土地譲渡による未実現損失の扱いを定めている。",
            source="企業会計基準第22号 > 第36条",
        )

        self.assertEqual(context.verification_results[0]["judgment"], "○適切")
        self.assertEqual(context.verification_results[0]["status"], "supported")
        self.assertEqual(context.verification_results[0]["evidence_chunk_ids"], ["consol.pdf:p36"])

    def test_context_tracks_evidence_slot_coverage(self):
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {"complexity": "complex"},
        )

        context.set_evidence_note(
            "lease-main.pdf:p10",
            "借手は使用権資産とリース負債を計上する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )
        context.set_evidence_note(
            "lease-main.pdf:p11",
            "適用初年度の経過措置が定められている。",
            source="企業会計基準第34号 > 経過措置",
        )

        coverage = context.get_evidence_coverage()

        self.assertEqual(coverage["covered_slots"], ["借手", "経過措置"])
        self.assertIn("貸手", coverage["uncovered_slots"])
        self.assertIn("関連基準", coverage["uncovered_slots"])

    def test_related_standard_alias_counts_toward_related_standard_slot(self):
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {"complexity": "complex"},
        )

        context.set_evidence_note(
            "lease-main.pdf:p20",
            "使用権資産の減価償却や減損の扱いについて、固定資産の減損に係る会計基準との関係が示されている。",
            source="企業会計基準第34号 > 固定資産の減損に係る会計基準との関係",
        )

        self.assertIn("関連基準", context.get_evidence_coverage()["covered_slots"])

    def test_revenue_questions_do_not_infer_related_standard_slot_from_standard_name(self):
        context = AgentContext()
        context.set_question(
            "収益認識基準における本人と代理人の区分はどう判断しますか",
            {"complexity": "moderate"},
        )

        self.assertEqual(context.evidence_slots, ["本人", "代理人"])

    def test_hedge_requirement_question_infers_focus_slots_from_corrective_query(self):
        context = AgentContext()
        context.set_question(
            "繰延ヘッジを適用するための主な要件を教えてください",
            {
                "complexity": "moderate",
                "detail_seeking": True,
                "canonical_focus_query": "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
                "corrective_query": "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト 要件",
            },
        )

        self.assertEqual(
            context.evidence_slots,
            ["ヘッジ", "有効性", "文書化", "事前", "事後"],
        )

    def test_arun_emits_structured_completion_log(self):
        agent = self.make_agent()
        agent.llm = self.NaturalAnswerLLM()
        agent.tools = self.FakeTools()
        agent.chunk_map = {
            "lease-main.pdf:p10": {
                "id": "lease-main.pdf:p10",
                "source": "企業会計基準第34号 > 借手の会計処理",
                "text": "借手は使用権資産を計上する。",
            }
        }

        with request_context("req-agent-test", monitor_case_id="lease_case"):
            with self.assertLogs("src.arag.agent", level="INFO") as logs:
                result = asyncio.run(agent.arun("借手は何を認識しますか", history=[]))

        self.assertEqual(result["request_id"], "req-agent-test")
        structured = []
        for entry in logs.output:
            _, _, message = entry.partition("src.arag.agent:")
            payload = json.loads(message.strip())
            if payload.get("event") == "agent_run_complete":
                structured.append(payload)
        self.assertEqual(len(structured), 1)
        self.assertEqual(structured[0]["request_id"], "req-agent-test")
        self.assertEqual(structured[0]["monitor_case_id"], "lease_case")
        self.assertEqual(structured[0]["loops"], 1)
        self.assertEqual(structured[0]["reference_count"], 1)
        self.assertFalse(structured[0]["zero_reference"])

    def test_retry_natural_answer_when_moderate_query_has_too_few_citations(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "固定資産の減損会計では、どのような場合に減損の兆候があると判断しますか",
            {"complexity": "moderate", "search_mode": "keyword_first", "keywords": ["減損", "兆候"]},
        )
        context.mark_chunk_read("impairment:p1")
        context.mark_chunk_read("impairment:p2")
        context.set_evidence_note("impairment:p1", "減損の兆候の例示がある。", source="減損会計基準 > 兆候")
        context.set_evidence_note("impairment:p2", "回収可能価額は使用価値と正味売却価額のいずれか高い方。", source="減損会計基準 > 回収可能価額")

        self.assertTrue(
            agent._should_retry_natural_answer({"cited_reference_count": 1}, context)
        )

    def test_do_not_retry_natural_answer_for_simple_exact_query(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手"]},
        )
        context.mark_chunk_read("lease:p10")
        context.mark_chunk_read("lease:p11")
        context.set_evidence_note("lease:p10", "借手は通常の売買取引に準じて処理する。", source="企業会計基準第13号 > 第10項")
        context.set_evidence_note("lease:p11", "通常の賃貸借取引に係る方法に準じた会計処理。", source="企業会計基準第13号 > 第10項")

        self.assertFalse(
            agent._should_retry_natural_answer({"cited_reference_count": 1}, context)
        )

    def test_retry_natural_answer_for_exact_query_without_exact_evidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手"]},
        )
        context.mark_chunk_read("lease:p11")
        context.set_evidence_note("lease:p11", "借手の一般的な説明。", source="企業会計基準第13号 > 借手")

        self.assertTrue(
            agent._should_retry_natural_answer({"cited_reference_count": 1}, context)
        )

    def test_force_stop_reason_returns_evidence_sufficient_before_budget(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "借手は短期リースや少額リースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手", "短期リース", "少額リース"]},
        )
        context.add_retrieval_log("keyword_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_search_entry({"confidence": 0.82})
        context.set_evidence_note(
            "lease-main.pdf:p25",
            "借手は短期リース及び少額リースについて使用権資産とリース負債を計上しないことができる。",
            source="企業会計基準第34号 > 借手の会計処理",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_exact_query_is_not_sufficient_without_exact_evidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手", "リース"]},
        )
        context.add_retrieval_log("hybrid_search", 10, metadata={"exact_shortfall": True})
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note("lease:p11", "借手の一般的な説明。", source="企業会計基準第13号 > 借手")

        self.assertFalse(agent._has_sufficient_evidence(context))

    def test_exact_query_with_read_document_exact_evidence_is_sufficient(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手", "リース"]},
        )
        context.add_retrieval_log("hybrid_search", 10, metadata={"exact_shortfall": True})
        context.add_retrieval_log("read_document", 10)
        context.set_evidence_note(
            "lease:p10",
            "10. 借手は、通常の売買取引に係る方法に準じて会計処理を行う。",
            source="企業会計基準第13号 > 借手側",
        )

        self.assertTrue(agent._has_sufficient_evidence(context))

    def test_hedge_requirement_query_requires_effectiveness_slot_before_stop(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "繰延ヘッジを適用するための主な要件を教えてください",
            {
                "complexity": "moderate",
                "search_mode": "keyword_first",
                "detail_seeking": True,
                "canonical_focus_query": "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト",
                "corrective_query": "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト 要件",
            },
        )
        for _ in range(2):
            context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note("hedge:p1", "正式な文書によりヘッジ取引を明確にする。", source="移管指針第9号 > 文書化")
        context.set_evidence_note("hedge:p2", "ヘッジ取引開始時に事前テストを行う。事後テストも必要である。", source="移管指針第9号 > 有効性")
        context.evidence_slot_hits["有効性"].clear()

        self.assertFalse(agent._has_sufficient_evidence(context))

    def test_exact_gap_message_guides_read_document_and_blocks_general_fallback(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["企業会計基準第13号", "第10項"]},
        )
        context.add_search_entry({"exact_shortfall": True})
        context.add_retrieval_log("hybrid_search", 10, metadata={"exact_shortfall": True})

        message = agent._build_exact_gap_message(context)

        self.assertIsNotNone(message)
        self.assertIn("read_document", message["content"])
        self.assertIn("一般論", message["content"])

    def test_exact_evidence_message_surfaces_clause_key_terms(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["企業会計基準第13号", "第10項"]},
        )
        context.set_evidence_note(
            "lease:p10",
            "10. 借手は、通常の売買取引に係る方法に準じた会計処理により、リース資産及びリース債務を計上する。",
            source="企業会計基準第13号 > 借手側",
        )

        message = agent._build_exact_evidence_message(context)

        self.assertIsNotNone(message)
        self.assertIn("通常の売買処理", message["content"])

    def test_keyword_first_query_can_stop_with_slot_coverage_even_without_high_confidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "企業会計基準第13号第10項では借手のリースをどのように扱いますか",
            {"complexity": "simple", "search_mode": "keyword_first", "keywords": ["借手", "リース"]},
        )
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_search_entry({"confidence": 0.22})
        context.set_evidence_note(
            "lease-old.pdf:p10",
            "借手はファイナンス・リース取引について通常の売買取引に準じた処理を行う。",
            source="企業会計基準第13号 > 第10項",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_full_slot_coverage_can_stop_without_high_confidence_for_moderate_query(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "税効果会計で繰延税金資産・負債の計算に使う税率は何ですか。税率変更時の扱いも教えてください",
            {
                "complexity": "moderate",
                "search_mode": "balanced",
                "keywords": ["税効果会計", "繰延税金資産", "繰延税金負債", "税率", "税率変更"],
            },
        )
        for _ in range(2):
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        for confidence in (0.31, 0.34):
            context.add_search_entry({"confidence": confidence, "chunk_ids": ["tax:p1", "tax:p2"]})
        context.set_evidence_note(
            "tax:p1",
            "繰延税金資産は回収見込期の税率で計算する。",
            source="税効果会計 > 繰延税金資産",
        )
        context.set_evidence_note(
            "tax:p2",
            "繰延税金負債も回収又は支払が見込まれる期の税率で計算する。",
            source="税効果会計 > 繰延税金負債",
        )
        context.set_evidence_note(
            "tax:p3",
            "税率は決算日に成立している税法に基づき、税率変更時には見直す。",
            source="税効果会計 > 税率変更",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_keyword_first_moderate_query_can_stop_with_one_example_slot_missing(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください",
            {
                "complexity": "moderate",
                "search_mode": "keyword_first",
                "keywords": ["履行義務", "保守サービス", "値引き"],
            },
        )
        query = "収益認識基準 履行義務 別個 値引き"
        context.add_search_entry(
            {
                "confidence": 0.49,
                "query": query,
                "effective_query": query,
                "chunk_ids": ["rev:p1", "rev:p2", "rev:p3"],
            }
        )
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "rev:p1",
            "履行義務は顧客との契約で約束した別個の財又はサービスごとに識別する。",
            source="収益認識基準 > 履行義務",
        )
        context.set_evidence_note(
            "rev:p2",
            "値引きは原則として履行義務に比例配分する。",
            source="収益認識基準 > 値引きの配分",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_keyword_first_moderate_query_can_stop_with_one_example_slot_missing_at_mid_confidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください",
            {
                "complexity": "moderate",
                "search_mode": "keyword_first",
                "keywords": ["履行義務", "保守サービス", "値引き"],
            },
        )
        query = "収益認識基準 履行義務 別個 値引き"
        context.add_search_entry(
            {
                "confidence": 0.41,
                "query": query,
                "effective_query": query,
                "chunk_ids": ["rev:p1", "rev:p2", "rev:p3"],
            }
        )
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "rev:p1",
            "履行義務は顧客との契約で約束した別個の財又はサービスごとに識別する。",
            source="収益認識基準 > 履行義務",
        )
        context.set_evidence_note(
            "rev:p2",
            "値引きは原則として履行義務に比例配分する。",
            source="収益認識基準 > 値引きの配分",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_detail_keyword_first_query_can_stop_with_detail_rich_notes_at_lower_confidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください",
            {
                "complexity": "moderate",
                "search_mode": "keyword_first",
                "detail_seeking": True,
                "keywords": ["履行義務", "保守サービス", "値引き"],
            },
        )
        query = "収益認識基準 履行義務 別個 保守サービス 値引き 契約 識別"
        context.add_search_entry(
            {
                "confidence": 0.36,
                "query": query,
                "effective_query": query,
                "chunk_ids": ["rev:p1", "rev:p2", "rev:p3"],
                "exact_shortfall": False,
            }
        )
        context.add_retrieval_log("hybrid_search", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "rev:p1",
            "履行義務は顧客との契約で約束した財又はサービスを識別し、別個の便益を提供できるかや、他の約束と統合して提供されるかを判断する。",
            source="収益認識基準 > 履行義務の識別",
        )
        context.set_evidence_note(
            "rev:p2",
            "保守サービスが単独で便益を提供できる場合は別個の履行義務となり、値引きは履行義務の識別後に取引価格の配分で検討する。",
            source="収益認識基準 > 契約の分解",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_maintenance_contract_note_counts_toward_maintenance_service_slot(self):
        context = AgentContext()
        context.set_question(
            "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください",
            {
                "complexity": "moderate",
                "search_mode": "keyword_first",
                "keywords": ["履行義務", "保守サービス", "値引き"],
            },
        )
        context.set_evidence_note(
            "rev:p3",
            "保守契約が単独で便益を提供できる場合には別個の履行義務として扱う。",
            source="収益認識基準 > 契約の分解",
        )

        self.assertIn("保守サービス", context.get_evidence_coverage()["covered_slots"])

    def test_complex_keyword_first_query_can_stop_with_required_slot_coverage_after_single_search(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {
                "complexity": "complex",
                "search_mode": "keyword_first",
                "keywords": ["借手", "貸手", "経過措置", "関連基準"],
            },
        )
        query = "リースに関する会計基準 改正 借手 貸手 経過措置"
        context.add_search_entry(
            {
                "confidence": 0.58,
                "query": query,
                "effective_query": query,
                "chunk_ids": ["lease:p10", "lease:p11", "lease:p12"],
            }
        )
        context.add_retrieval_log("hybrid_search", 10)
        for _ in range(3):
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "lease:p10",
            "借手は使用権資産とリース負債を計上する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )
        context.set_evidence_note(
            "lease:p11",
            "貸手は現行基準の分類を維持する。",
            source="企業会計基準第34号 > 貸手の会計処理",
        )
        context.set_evidence_note(
            "lease:p12",
            "適用初年度の経過措置が定められている。",
            source="企業会計基準第34号 > 経過措置",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_stagnating_search_results_can_stop_when_required_slot_coverage_is_met(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "税効果会計で繰延税金資産・負債の計算に使う税率は何ですか。税率変更時の扱いも教えてください",
            {
                "complexity": "moderate",
                "search_mode": "balanced",
                "keywords": ["税効果会計", "繰延税金資産", "繰延税金負債", "税率", "税率変更"],
            },
        )
        for confidence, query in (
            (0.28, "税効果会計 税率"),
            (0.33, "税効果会計 税率変更"),
            (0.35, "税効果会計 税率変更"),
        ):
            context.add_search_entry(
                {
                    "confidence": confidence,
                    "query": query,
                    "effective_query": query,
                    "chunk_ids": ["tax:p1", "tax:p2", "tax:p3"],
                }
            )
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "tax:p1",
            "繰延税金資産は回収見込期の税率で計算する。",
            source="税効果会計 > 繰延税金資産",
        )
        context.set_evidence_note(
            "tax:p2",
            "繰延税金負債も回収又は支払が見込まれる期の税率で計算する。",
            source="税効果会計 > 繰延税金負債",
        )
        context.set_evidence_note(
            "tax:p3",
            "税率変更時には決算日に成立している税法ベースで見直す。",
            source="税効果会計 > 税率変更",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_complex_query_can_stop_with_required_slot_coverage_at_moderate_confidence(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "税効果会計で繰延税金資産・負債の計算に使う税率は何ですか。税率変更時の扱いも教えてください",
            {
                "complexity": "complex",
                "search_mode": "balanced",
                "keywords": ["税効果会計", "繰延税金資産", "繰延税金負債", "税率", "税率変更"],
            },
        )
        for confidence, query in (
            (0.45, "税効果会計 税率"),
            (0.46, "税効果会計 税率変更"),
            (0.49, "税効果会計 税率変更 再計算"),
        ):
            context.add_search_entry(
                {
                    "confidence": confidence,
                    "query": query,
                    "effective_query": query,
                    "chunk_ids": ["tax:p1", "tax:p2", "tax:p3"],
                }
            )
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "tax:p1",
            "繰延税金資産は回収見込期の税率で計算する。",
            source="税効果会計 > 繰延税金資産",
        )
        context.set_evidence_note(
            "tax:p2",
            "繰延税金負債も回収又は支払が見込まれる期の税率で計算する。",
            source="税効果会計 > 繰延税金負債",
        )
        context.set_evidence_note(
            "tax:p3",
            "税率変更時には決算日に成立している税法ベースで見直す。",
            source="税効果会計 > 税率変更",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_complex_query_can_stop_when_only_one_slot_remains_after_two_searches(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {
                "complexity": "complex",
                "search_mode": "balanced",
                "keywords": ["借手", "貸手", "経過措置", "関連基準"],
            },
        )
        for confidence, query in (
            (0.72, "リース会計基準 改正 借手 貸手 経過措置"),
            (0.68, "リース会計基準 改正 借手 貸手 経過措置"),
        ):
            context.add_search_entry(
                {
                    "confidence": confidence,
                    "query": query,
                    "effective_query": query,
                    "chunk_ids": ["lease:p10", "lease:p11", "lease:p12"],
                }
            )
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "lease:p10",
            "借手は使用権資産とリース負債を計上する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )
        context.set_evidence_note(
            "lease:p11",
            "貸手は現行基準の分類を維持する。",
            source="企業会計基準第34号 > 貸手の会計処理",
        )
        context.set_evidence_note(
            "lease:p12",
            "適用初年度の経過措置が定められている。",
            source="企業会計基準第34号 > 経過措置",
        )

        self.assertEqual(agent._force_stop_reason(context), "evidence_sufficient")

    def test_force_stop_reason_falls_back_to_retrieval_budget_when_slots_remain_uncovered(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {"complexity": "complex", "keywords": ["借手", "貸手", "経過措置", "関連基準"]},
        )
        for confidence in (0.82, 0.78, 0.75):
            context.add_search_entry({"confidence": confidence})
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "lease-main.pdf:p10",
            "借手は使用権資産とリース負債を計上する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )
        context.set_evidence_note(
            "lease-main.pdf:p11",
            "適用初年度の経過措置が定められている。",
            source="企業会計基準第34号 > 経過措置",
        )

        self.assertEqual(agent._force_stop_reason(context), "retrieval_budget")

    def test_coverage_gap_message_mentions_uncovered_slots_and_read_document(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください",
            {"complexity": "complex"},
        )
        for _ in range(2):
            context.add_retrieval_log("hybrid_search", 10)
            context.add_retrieval_log("read_chunk", 10)
        context.set_evidence_note(
            "lease-main.pdf:p10",
            "借手は使用権資産とリース負債を計上する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )

        message = agent._build_coverage_gap_message(context)

        self.assertIsNotNone(message)
        self.assertIn("貸手", message["content"])
        self.assertIn("関連基準", message["content"])
        self.assertIn("read_document", message["content"])

    def test_force_final_answer_includes_evidence_notes(self):
        agent = self.make_agent()
        agent.llm = self.FakeLLM()
        agent.chunk_map = {
            "lease-main.pdf:p10": {"source": "企業会計基準第34号 > 借手の会計処理"},
            "lease-main.pdf:p11": {"source": "企業会計基準第34号 > 契約条件の変更"},
        }
        context = AgentContext()
        context.set_question("リース会計基準の改正点と経過措置を教えてください", {"complexity": "complex"})
        context.add_searched_chunks(["lease-main.pdf:p10", "lease-main.pdf:p11"])
        context.set_evidence_note("lease-main.pdf:p10", "借手は使用権資産とリース負債を計上する。[抜粋]")
        context.set_evidence_note("lease-main.pdf:p11", "適用初年度の経過措置が定められている。")

        answer, _ = agent._force_final_answer([{"role": "user", "content": "question"}], context)

        self.assertEqual(answer, "final answer")
        self.assertIsNotNone(agent.llm.messages)
        note_message = next(
            message for message in agent.llm.messages
            if "## 重要箇所メモ" in str(message.get("content", ""))
        )
        self.assertEqual(note_message["role"], "user")
        self.assertIn("## 充足済み論点", note_message["content"])
        self.assertIn("## 重要箇所メモ", note_message["content"])
        self.assertIn("lease-main.pdf:p10", note_message["content"])
        self.assertIn("借手は使用権資産とリース負債を計上する。", note_message["content"])
        self.assertNotIn("[抜粋]", note_message["content"])
        self.assertIn("経過措置", note_message["content"])
        self.assertTrue(any(
            "【詳細回答ルール】" in str(message.get("content", ""))
            for message in agent.llm.messages
        ))

    def test_force_final_answer_includes_verification_structure_messages(self):
        agent = self.make_agent()
        agent.llm = self.FakeLLM()
        context = AgentContext()
        context.set_question(
            "この判断は妥当ですか",
            {
                "complexity": "complex",
                "verification_mode": True,
                "verification_claims": [
                    {
                        "claim": "再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しない",
                        "search_query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                        "doc_terms": ["企業会計基準第22号"],
                        "section_terms": ["第36条"],
                        "cited_references": ["企業会計基準第22号", "第36条"],
                        "target_transaction": "土地譲渡 / 未実現損失",
                    }
                ],
            },
        )
        context.add_search_entry(
            {
                "query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                "effective_query": "企業会計基準第22号 第36条 土地譲渡 未実現損失",
                "chunk_ids": ["consol.pdf:p36"],
            }
        )
        context.set_evidence_note(
            "consol.pdf:p36",
            "第36条では、土地譲渡による未実現損失の扱いを定めている。",
            source="企業会計基準第22号 > 第36条",
        )

        agent._force_final_answer([{"role": "user", "content": "question"}], context)

        self.assertTrue(any(
            "## 主張別照合メモ" in str(message.get("content", ""))
            for message in agent.llm.messages
        ))
        self.assertTrue(any(
            "【判断検証の最終回答ルール】" in str(message.get("content", ""))
            and "## 主張要約" in str(message.get("content", ""))
            and "## 照合結果" in str(message.get("content", ""))
            and "## 追加考慮事項" in str(message.get("content", ""))
            and "## 参照" in str(message.get("content", ""))
            for message in agent.llm.messages
        ))

    def test_verification_mode_uses_twelve_loop_cap(self):
        agent = self.make_agent()
        agent.max_loops = 20
        context = AgentContext()
        context.set_question(
            "この判断は妥当ですか",
            {
                "complexity": "complex",
                "verification_mode": True,
                "verification_claims": [{"claim": "テスト主張"}],
            },
        )

        self.assertEqual(agent._loop_budget(context), 12)

    def test_retry_natural_answer_when_verification_answer_missing_sections(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "この判断は妥当ですか",
            {
                "complexity": "complex",
                "verification_mode": True,
                "verification_claims": [{"claim": "テスト主張"}],
            },
        )
        result = {
            "answer": "主張は概ね妥当です[1]。",
            "cited_reference_count": 1,
        }

        self.assertTrue(agent._should_retry_natural_answer(result, context))

    def test_force_final_answer_prefers_slot_snippets_over_full_notes(self):
        agent = self.make_agent()
        agent.llm = self.FakeLLM()
        agent.chunk_map = {
            "rev:p1": {"source": "収益認識基準 > 変動対価"},
            "rev:p2": {"source": "収益認識基準 > 契約変更"},
        }
        context = AgentContext()
        context.set_question(
            "収益認識基準における変動対価はどのように見積もり、いつ収益に反映しますか",
            {"complexity": "moderate", "keywords": ["変動対価", "見積り"]},
        )
        context.add_searched_chunks(["rev:p1", "rev:p2"])
        context.set_evidence_note(
            "rev:p1",
            "変動対価は見積り、重要な戻入れを生じさせない可能性が高い範囲で収益に含める。補足の長い説明文が続く。",
            source="収益認識基準 > 変動対価",
        )
        context.set_evidence_note(
            "rev:p2",
            "契約変更は別個の契約として扱う場合がある。補足の長い説明文が続く。",
            source="収益認識基準 > 契約変更",
        )

        agent._force_final_answer([{"role": "user", "content": "question"}], context)

        note_message = next(
            message for message in agent.llm.messages
            if "## 重要箇所メモ" in str(message.get("content", ""))
        )
        self.assertIn("- 変動対価 | rev:p1 | 収益認識基準 > 変動対価", note_message["content"])
        self.assertIn("変動対価は見積り、重要な戻入れを生じさせない可能性が高い範囲で収益に含める。", note_message["content"])
        self.assertNotIn("補足の長い説明文が続く。", note_message["content"])

    def test_force_final_answer_keeps_adjacent_detail_sentences_for_detail_seeking_question(self):
        agent = self.make_agent()
        agent.llm = self.FakeLLM()
        agent.chunk_map = {
            "lease:p10": {"source": "企業会計基準第34号 > 借手の会計処理"},
        }
        context = AgentContext()
        context.set_question(
            "借手と貸手の会計処理の違いを詳しく教えてください",
            {"complexity": "complex", "keywords": ["借手", "貸手"], "detail_seeking": True},
        )
        context.add_searched_chunks(["lease:p10"])
        context.set_evidence_note(
            "lease:p10",
            "借手は使用権資産とリース負債を計上する。"
            "ただし、短期リース又は少額リースは費用処理を選択できる。"
            "貸手は現行の分類を維持する。",
            source="企業会計基準第34号 > 借手の会計処理",
        )

        agent._force_final_answer([{"role": "user", "content": "question"}], context)

        note_message = next(
            message for message in agent.llm.messages
            if "## 重要箇所メモ" in str(message.get("content", ""))
        )
        self.assertIn("ただし、短期リース又は少額リースは費用処理を選択できる。", note_message["content"])

    def test_retry_natural_answer_when_detail_seeking_answer_has_too_few_cited_lines(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "借手と貸手の会計処理の違いを詳しく教えてください",
            {"complexity": "complex", "keywords": ["借手", "貸手"], "detail_seeking": True},
        )
        context.mark_chunk_read("lease:p10")
        context.mark_chunk_read("lease:p11")
        context.set_evidence_note("lease:p10", "借手は使用権資産を計上する。", source="企業会計基準第34号 > 借手の会計処理")
        context.set_evidence_note("lease:p11", "貸手はリース債権を計上する。", source="企業会計基準第34号 > 貸手の会計処理")

        result = {
            "answer": "## 結論\n- 借手は使用権資産を計上します[1]。",
            "cited_reference_count": 1,
        }

        self.assertTrue(agent._should_retry_natural_answer(result, context))

    def test_do_not_retry_natural_answer_when_detail_seeking_answer_has_enough_cited_lines(self):
        agent = self.make_agent()
        context = AgentContext()
        context.set_question(
            "借手と貸手の会計処理の違いを詳しく教えてください",
            {"complexity": "complex", "keywords": ["借手", "貸手"], "detail_seeking": True},
        )
        context.mark_chunk_read("lease:p10")
        context.mark_chunk_read("lease:p11")
        context.set_evidence_note("lease:p10", "借手は使用権資産を計上する。", source="企業会計基準第34号 > 借手の会計処理")
        context.set_evidence_note("lease:p11", "貸手はリース債権を計上する。", source="企業会計基準第34号 > 貸手の会計処理")

        result = {
            "answer": (
                "## 結論\n"
                "- 借手は使用権資産とリース負債を計上し、短期リース等は例外処理を選択できます[1]。\n"
                "- 貸手は現行の分類を維持し、借手とは会計処理の軸が異なります[2]。"
            ),
            "cited_reference_count": 2,
        }

        self.assertFalse(agent._should_retry_natural_answer(result, context))

    def test_run_inserts_one_shot_coverage_review_before_forced_wrap_up(self):
        agent = self.make_agent()
        agent.llm = self.CoverageReviewLLM()
        agent.tools = self.FakeTools()
        agent.chunk_map = {
            "lease-main.pdf:p10": {
                "id": "lease-main.pdf:p10",
                "parent_id": "lease-main.pdf:p10",
                "text": "借手は使用権資産を計上する。",
                "source": "企業会計基準第34号 > 借手の会計処理",
                "file": "lease-main.pdf",
                "pdf_page": 10,
            }
        }
        agent.pdf_sources = {"lease-main.pdf": "https://example.com/lease-main.pdf"}

        def seed_context(context: AgentContext, question: str):
            context.set_question(
                question,
                {"complexity": "complex", "keywords": ["借手", "貸手"]},
            )
            for _ in range(3):
                context.add_search_entry({"confidence": 0.82})
                context.add_retrieval_log("hybrid_search", 10)
                context.add_retrieval_log("read_chunk", 10)
            context.set_evidence_note(
                "lease-main.pdf:p10",
                "借手は使用権資産を計上する。",
                source="企業会計基準第34号 > 借手の会計処理",
            )

        agent._seed_context = seed_context

        result = agent.run("借手と貸手の会計処理を教えてください")

        self.assertEqual(result["stop_reason"], "natural")
        self.assertIsNotNone(agent.llm.messages)
        self.assertTrue(any(
            "最終回答の直前です。論点カバレッジを 1 回だけ再点検してください。" in str(message.get("content", ""))
            for message in agent.llm.messages
        ))
        self.assertIn("貸手", result["answer"])
        self.assertIn("不十分", result["answer"])


if __name__ == "__main__":
    unittest.main()
