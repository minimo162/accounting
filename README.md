# 会計基準 Q&A

日本の会計基準についてAIが検索・回答するチャットアプリ。

## アーキテクチャ

- **A-RAG (Agentic RAG)**: ReActループで `hybrid_search` / `read_chunk` を中心に回答生成
- **Retrieval**: dense + BM25 + RRF + heuristic/LLM rerank + optional query expansion/HyDE
- **Chunking**: parent/child chunking。検索は child、表示は parent
- **LLM**: Cerebras (gpt-oss-120b) / Gemini フォールバック
- **Embedding**: provider 抽象化済み。既定は Gemini、OpenAI-compatible embeddings にも対応
- **Frontend**: SvelteKit + SSE ストリーミング
- **Backend**: FastAPI (Python)
- **インフラ**: Google Cloud Run + GCS

## データソース

| ソース | 内容 |
|--------|------|
| ASBJ | 企業会計基準、適用指針、実務対応報告 |
| SSBJ | サステナビリティ開示基準 (ユニバーサル基準、一般開示基準、気候関連開示基準) |
| 企業会計審議会 | 固定資産の減損、退職給付、税効果、外貨換算、連結、研究開発費 等 |
| JICPA | 金融商品会計実務指針、税効果実務指針、研究開発費/ソフトウェア実務指針、退職給付実務指針 等 |
| e-Gov法令 | 財規、連結財規、中間財規、四半期財規、会社計算規則、会社法(計算等)、開示府令、監査証明府令、内部統制府令 |
| HTML/PDF | 企業会計原則(本文＋注解)、原価計算基準、財務諸表等規則ガイドライン 等 |

- チャンク数: 4,218
- Embedding次元: 3,072 (float16圧縮)

## プロジェクト構成

```
src/
  api/main.py          # FastAPI エントリポイント (startup事前ロード)
  arag/
    agent.py           # ReActエージェントループ (最大15ループ、128Kトークン予算)
    llm.py             # LLMクライアント (Cerebras/Gemini、リトライ付き)
    prompt.py          # システムプロンプト (質問複雑度に応じた検索戦略)
    config.py          # 設定管理
    context.py         # エージェント実行コンテキスト (チャンク読取追跡、トークン集計)
    retrieval.py         # ChunkCorpus / BM25 / RRF
    query_rewrite.py     # query expansion / HyDE
    reranker.py          # heuristic / LLM reranker
    tools/
      hybrid_search.py    # dense + keyword + rerank の統合検索
      semantic_search.py  # dense retrieval
      keyword_search.py   # BM25 / exact-hit 検索
      read_chunk.py       # parent chunk 全文読み取り (隣接展開)
      filters.py          # チャンク品質フィルタ (表紙・目次除外、名簿・短文タグ)
      registry.py         # ツール登録
  embedding/
    factory.py         # 埋め込み provider ファクトリ
    gemini.py          # Gemini Embedder
    openai_compat.py   # OpenAI-compatible Embedder
frontend/              # SvelteKit SPA (ストリーミング回答、参照リンク、モバイル対応)
scripts/
  build_index.py       # searchable child chunk の embedding インデックス構築
  process_pdfs.py      # PDF → parent/child chunk 変換
  process_egov.py      # e-Gov法令API → parent/child chunk 変換
  process_html_sources.py  # HTML/PDFソース → parent/child chunk 変換
  eval_retrieval.py    # Recall/MRR/nDCG 評価と候補出力
  download_additional_sources.py  # JICPA・企業会計審議会等のPDFダウンロード
  update_chunk_sources.py  # pdf_sources.json 更新
  deploy.sh            # Cloud Run デプロイ
data/
  chunks.json          # チャンクデータ
  pdf_sources.json     # ファイル名→ソースURL マッピング (172件)
  index/
    sentence_index.npz   # Embedding (float16圧縮)
    sentence_meta.pkl    # メタデータ (文、チャンクマッピング)
```

## セットアップ

```bash
# 依存関係インストール
uv sync

# フロントエンドビルド
cd frontend && npm ci && npm run build && cd ..

# 環境変数
export GEMINI_API_KEY="your-api-key"
export CEREBRAS_API_KEY="your-api-key"
# OpenAI-compatible embeddings を使う場合
export EMBEDDING_API_KEY="your-api-key"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
```

## インデックス構築

```bash
# PDFからチャンク生成
python scripts/process_pdfs.py

# e-Gov法令からチャンク生成
python scripts/process_egov.py

# HTML/PDFソースからチャンク生成
python scripts/process_html_sources.py

# Embeddingインデックス構築 (チェックポイント付き、中断再開可能)
python scripts/build_index.py

# retrieval 評価候補の出力
python scripts/eval_retrieval.py --write-candidates
```

## ローカル実行

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## デプロイ

```bash
# Cloud Run にデプロイ (Cloud Build)
bash scripts/deploy.sh

# GCS にインデックスをアップロード
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/
gsutil cp data/pdf_sources.json gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
```

## コールドスタート最適化

- `@app.on_event("startup")` で事前ロード (リクエスト前にインデックスをメモリに展開)
- Embedding を float16 + npz圧縮で保存 (53%削減)
- GCS からの npz/meta 優先ダウンロード
- API コールにリトライロジック (3回、指数バックオフ)
