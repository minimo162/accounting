"""Download accounting standard PDFs from ASBJ using browser-use."""

import asyncio
import os
import re
import sys
from pathlib import Path

from browser_use import Agent as BrowserAgent
from langchain_openai import ChatOpenAI


async def download_pdfs(output_dir: str):
    """Use browser-use to navigate ASBJ and download PDFs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use Cerebras-compatible or any available LLM for browser agent
    llm = ChatOpenAI(
        model="gpt-oss-120b",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url="https://api.cerebras.ai/v1",
    )

    task = """
    https://www.asb-j.jp/jp/accounting_standards/ にアクセスして、
    以下の会計基準文書のPDFリンクを全て収集してください：

    1. 企業会計基準（全て）
    2. 企業会計基準適用指針（全て）
    3. 実務対応報告（全て）

    各PDFのURLとタイトルをリストとして出力してください。
    フォーマット: タイトル | URL
    """

    agent = BrowserAgent(task=task, llm=llm)
    result = await agent.run()

    # Parse result and save PDF URLs
    print("Browser agent result:")
    print(result)
    return result


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/pdfs"
    asyncio.run(download_pdfs(output_dir))
