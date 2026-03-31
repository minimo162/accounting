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

    def test_read_chunk_uses_slot_terms_to_keep_detail_examples(self):
        chunks = [
            {
                "id": "revenue.pdf:p20",
                "parent_id": "revenue.pdf:p20",
                "level": "parent",
                "file": "revenue.pdf",
                "source": "企業会計基準第29号 > 履行義務の識別",
                "text": (
                    "履行義務は契約における約束を基礎として識別する。\n"
                    "顧客に移転する財又はサービスが別個であるかを判断する。\n"
                    + ("一般論の説明が続く。\n" * 80)
                    + "保守サービスが機器販売と別個である場合は、別個の履行義務として区分する。\n"
                    "値引きが複数の約束に関連する場合は、取引価格の配分との関係も検討する。\n"
                    + ("周辺的な補足説明が続く。\n" * 40)
                ),
            }
        ]
        corpus = ChunkCorpus(chunks)
        tool = ReadChunkTool(corpus)
        context = AgentContext()
        context.set_question(
            "収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください",
            {"complexity": "moderate", "detail_seeking": True},
        )
        context.set_current_search_query("収益認識基準 履行義務 別個")

        text, _ = tool.execute(context, chunk_ids=["revenue.pdf:p20"])

        self.assertIn("保守サービスが機器販売と別個である場合", text)
        self.assertIn("値引きが複数の約束に関連する場合", text)

    def test_read_chunk_compacts_requirement_listing_queries(self):
        chunks = [
            {
                "id": "hedge.pdf:p8",
                "parent_id": "hedge.pdf:p8",
                "level": "parent",
                "file": "hedge.pdf",
                "source": "金融商品会計実務指針 > ヘッジ会計の適用要件",
                "text": (
                    "ヘッジ会計の適用要件を説明する。\n"
                    "正式な文書によりヘッジ対象、ヘッジ手段、ヘッジ対象リスクを明確にする。\n"
                    "事前テストにより有効性が高い見込みであることを確認する。\n"
                    "事後テストにより継続して有効性を評価する。\n"
                    "リスク管理方針に基づいてヘッジを実行する。\n"
                    + ("周辺的な背景説明が続く。\n" * 120)
                    + "ヘッジ会計を採用しない場合の補足説明。\n"
                ),
            }
        ]
        corpus = ChunkCorpus(chunks)
        tool = ReadChunkTool(corpus)
        context = AgentContext()
        context.set_question(
            "繰延ヘッジを適用するための主な要件を教えてください",
            {"complexity": "moderate", "detail_seeking": True},
        )
        context.set_current_search_query(
            "ヘッジ会計 繰延ヘッジ ヘッジ会計の適用要件 正式な文書 有効性 事前テスト 事後テスト 要件"
        )

        text, _ = tool.execute(context, chunk_ids=["hedge.pdf:p8"])

        self.assertIn("正式な文書によりヘッジ対象", text)
        self.assertIn("事前テスト", text)
        self.assertIn("事後テスト", text)
        self.assertNotIn("ヘッジ会計を採用しない場合の補足説明。", text)


if __name__ == "__main__":
    unittest.main()
