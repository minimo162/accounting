import unittest

from src.arag.agent import Agent
from src.arag.config import AgentConfig, Config, DataConfig, EmbeddingConfig, LLMConfig, RetrievalConfig
from src.arag.context import AgentContext


class AgentLoopControlTests(unittest.TestCase):
    class FakeLLM:
        def __init__(self):
            self.messages = None

        def chat(self, messages, tools=None, temperature=0.0, max_tokens=None):
            self.messages = messages
            return {"message": {"content": "final answer"}, "cost": 0.0}

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
        note_message = agent.llm.messages[-2]
        self.assertEqual(note_message["role"], "user")
        self.assertIn("## 充足済み論点", note_message["content"])
        self.assertIn("## 重要箇所メモ", note_message["content"])
        self.assertIn("lease-main.pdf:p10", note_message["content"])
        self.assertIn("借手は使用権資産とリース負債を計上する。", note_message["content"])
        self.assertNotIn("[抜粋]", note_message["content"])
        self.assertIn("経過措置", note_message["content"])


if __name__ == "__main__":
    unittest.main()
