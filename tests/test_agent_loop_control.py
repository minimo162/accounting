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
        self.assertIn("## 重要箇所メモ", note_message["content"])
        self.assertIn("lease-main.pdf:p10", note_message["content"])
        self.assertIn("借手は使用権資産とリース負債を計上する。", note_message["content"])
        self.assertNotIn("[抜粋]", note_message["content"])
        self.assertIn("経過措置", note_message["content"])


if __name__ == "__main__":
    unittest.main()
