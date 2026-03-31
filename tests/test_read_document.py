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

    def _make_exact_chunks(self) -> list[dict]:
        return [
            {
                "id": "lease.pdf:1",
                "file": "lease.pdf",
                "source": "企業会計基準第13号 > 総則",
                "text": "1. 総則",
            },
            {
                "id": "lease.pdf:2",
                "file": "lease.pdf",
                "source": "企業会計基準第13号 > 借手側",
                "text": "10. 借手は、通常の売買取引に係る方法に準じて会計処理を行う。15. オペレーティング・リース取引については、通常の賃貸借取引に係る方法に準じて会計処理を行う。",
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

    def test_read_document_registers_exact_evidence_from_legacy_section_label(self):
        tool = ReadDocumentTool(self._make_exact_chunks())
        context = AgentContext()
        context.set_question("企業会計基準第13号第10項では借手のリースをどのように扱いますか", {"complexity": "simple"})

        _, _ = tool.execute(context, name="lease.pdf")

        self.assertIn("lease.pdf:2", context.evidence_notes)
        self.assertIn("10. 借手は", context.evidence_notes["lease.pdf:2"])
        self.assertIn("lease.pdf:2", context.exact_evidence_chunk_ids)


if __name__ == "__main__":
    unittest.main()
