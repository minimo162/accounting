import unittest

from src.arag.agent import Agent
from src.arag.config import AgentConfig, Config, DataConfig, EmbeddingConfig, LLMConfig, RetrievalConfig
from src.arag.context import AgentContext


class AgentLoopControlTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
