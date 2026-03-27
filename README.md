# 会計基準 Q&A

日本の会計基準について、出典付きで検索・回答するチャットアプリです。

- Backend: FastAPI + Agentic RAG
- Frontend: SvelteKit + SSE ストリーミング
- LLM: Cerebras `gpt-oss-120b` を既定利用、必要に応じて Gemini
- Retrieval: dense + BM25 + RRF + heuristic/LLM rerank + optional query expansion / HyDE
- Chunking: parent/child chunking。検索は child、表示と引用は parent
- Infra: Cloud Run + GCS

## アプリ概要

質問を受けると、エージェントが複数の検索ツールを使い分けて根拠を集め、最終回答を生成します。

1. `hybrid_search` で dense + keyword の統合検索
2. 必要に応じて `semantic_search` / `keyword_search` を個別利用
3. `read_chunk` で parent chunk を精読
4. `read_document` で文書全体を確認
5. 収集した根拠をもとに回答と参照情報を返却

## データソース

| ソース | 内容 |
|--------|------|
| ASBJ | 企業会計基準、適用指針、実務対応報告 |
| SSBJ | サステナビリティ開示基準 |
| 企業会計審議会 | 減損、退職給付、税効果、外貨換算、連結、研究開発費など |
| JICPA | 各種実務指針 |
| e-Gov法令 | 財規、連結財規、中間財規、四半期財規、会社計算規則、会社法関連など |
| HTML/PDF | 企業会計原則、原価計算基準、財務諸表等規則ガイドラインなど |

主な生成物:

- `data/chunks.json`: parent / child chunk を含む検索用データ
- `data/pdf_sources.json`: 出典 URL マッピング
- `data/index/sentence_index.npz`: searchable child chunk の埋め込み配列
- `data/index/sentence_meta.pkl`: chunk 対応やメタデータ

補足:

- `sentence_index.*` という名前ですが、実態は searchable child chunk の index です
- `read_document` は `chunks.json` から全文を再構築するため、`full_texts.json` は必須ではありません

## リポジトリ構成

```text
src/
  api/main.py              FastAPI エントリポイント
  arag/                    エージェント、検索、リライト、rerank
  arag/tools/              hybrid_search / read_chunk / read_document など
  embedding/               Gemini / OpenAI-compatible embedding provider
frontend/
  src/routes/+page.svelte  チャット UI
scripts/
  process_pdfs.py          PDF を parent/child chunk 化
  process_egov.py          e-Gov XML を parent/child chunk 化
  process_html_sources.py  HTML/PDF ソースを parent/child chunk 化
  process_full_texts.py    補助用の全文テキスト生成
  add_context.py           chunk に追加文脈を付与
  build_index.py           searchable chunk の埋め込み index 構築
  eval_retrieval.py        retrieval 評価
  deploy.sh                Cloud Run デプロイ
data/
  chunks.json
  pdf_sources.json
  index/
```

## セットアップ

前提:

- Python `3.12`
- Node.js `22` 系
- `uv`

依存関係のインストール:

```bash
uv sync
cd frontend
npm ci
cd ..
```

環境変数:

```bash
export GEMINI_API_KEY="your-api-key"
export CEREBRAS_API_KEY="your-api-key"

# OpenAI-compatible embeddings を使う場合
export EMBEDDING_API_KEY="your-api-key"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"

# 任意
export LLM_PROVIDER="cerebras"   # or gemini
export USE_GCS="false"           # ローカル data/ を使う場合
export DATA_DIR="data"
```

## ローカル開発

### Backend のみ起動

フロント開発時の既定 API 先は `http://localhost:8000` です。

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend を含めて起動

別ターミナルで:

```bash
cd frontend
npm run dev
```

本番相当の静的配信を確認したい場合:

```bash
cd frontend
npm run build
cd ..
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## データ更新フロー

最小構成の流れ:

```bash
# 1. PDF / HTML / e-Gov から chunk を生成
python scripts/process_pdfs.py
python scripts/process_html_sources.py
python scripts/process_egov.py

# 2. 必要なら全文テキストや追加コンテキストを生成
python scripts/process_full_texts.py
python scripts/add_context.py

# 3. searchable chunk の index を構築
python scripts/build_index.py
```

補足:

- `build_index.py` はチェックポイント付きで差分更新に対応しています
- retrieval 候補を評価したい場合は `python scripts/eval_retrieval.py --write-candidates` を使います
- Cloud Run では `USE_GCS=true` の場合、起動時に GCS から `chunks.json` と index 一式を取得します

## デプロイ

Cloud Run へのデプロイ:

```bash
bash scripts/deploy.sh
```

インデックスを GCS にアップロードする例:

```bash
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/
gsutil cp data/pdf_sources.json gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
```

## 実装メモ

- `src/api/main.py` で startup 時にエージェントとインデックスを preload
- `src/arag/agent.py` は最大 25 ループ、128K token budget で ReAct 実行
- `src/arag/tools/hybrid_search.py` で dense / keyword / rerank を統合
- embedding provider は Gemini と OpenAI-compatible backend を切り替え可能
- 埋め込みは `npz + float16` で圧縮保存し、起動コストを抑制
