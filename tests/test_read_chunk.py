import unittest

from src.arag.context import AgentContext
from src.arag.retrieval import ChunkCorpus
from src.arag.tools.read_chunk import ReadChunkTool


class ReadChunkToolTests(unittest.TestCase):
    def test_read_chunk_extracts_query_relevant_evidence(self):
        chunks = [
            {
                "id": "lease-main.pdf:p10",
                "parent_id": "lease-main.pdf:p10",
                "level": "parent",
                "file": "lease-main.pdf",
                "source": "企業会計基準第34号 > 借手の会計処理",
                "text": (
                    "リースに関する会計基準。\n"
                    "総論として本会計基準の目的を示す。\n"
                    "借手は原則として使用権資産及びリース負債を計上する。\n"
                    "借手はリース負債を現在価値で測定する。\n"
                    + ("ここから先は用語の一般説明が続く。\n" * 120)
                    + "委員会の開催経緯を説明する付随情報。\n"
                    + "参考資料や周辺論点の説明。\n"
                    + "さらに周辺的な記述が長く続く。"
                ),
            }
        ]
        corpus = ChunkCorpus(chunks)
        tool = ReadChunkTool(corpus)
        context = AgentContext()
        context.set_question(
            "リース会計基準の改正点を教えてください",
            {"complexity": "simple"},
        )
        context.set_current_search_query("リースに関する会計基準 改正 使用権資産 リース負債")

        text, log = tool.execute(context, chunk_ids=["lease-main.pdf:p10"])

        self.assertIn("使用権資産及びリース負債を計上する。", text)
        self.assertIn("[抜粋]", text)
        self.assertNotIn("さらに周辺的な記述が長く続く。", text)
        self.assertGreater(log["retrieved_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
