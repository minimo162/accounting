import unittest

from src.arag.context import AgentContext
from src.arag.tools.read_document import ReadDocumentTool


class ReadDocumentToolTests(unittest.TestCase):
    def _make_chunks(self) -> list[dict]:
        long_text = "A" * 60_000
        return [
            {
                "id": "sample.pdf:1",
                "file": "sample.pdf",
                "source": "サンプル基準 > 総則",
                "text": long_text[:30_000],
            },
            {
                "id": "sample.pdf:2",
                "file": "sample.pdf",
                "source": "サンプル基準 > 各論",
                "text": long_text[30_000:],
            },
        ]

    def test_read_document_uses_complex_budget(self):
        tool = ReadDocumentTool(self._make_chunks())
        context = AgentContext()
        context.set_question("複合論点の詳細を詳しく教えてください", {"complexity": "complex"})

        text, log = tool.execute(context, name="sample.pdf")

        self.assertEqual(log["max_chars"], 24_000)
        self.assertIn("[読取範囲: 0〜24,000文字目]", text)
        self.assertIn('read_document(name="sample.pdf", offset=24000)', text)

    def test_read_document_reduces_budget_after_multiple_read_chunk_calls(self):
        tool = ReadDocumentTool(self._make_chunks())
        context = AgentContext()
        context.set_question("複合論点の詳細を詳しく教えてください", {"complexity": "complex"})
        context.mark_chunk_read("sample.pdf:p1", 100)
        context.mark_chunk_read("sample.pdf:p2", 100)

        text, log = tool.execute(context, name="sample.pdf")

        self.assertEqual(log["max_chars"], 18_000)
        self.assertIn("[読取範囲: 0〜18,000文字目]", text)
        self.assertIn('read_document(name="sample.pdf", offset=18000)', text)


if __name__ == "__main__":
    unittest.main()
