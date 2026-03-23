# 会計基準 Q&A

日本の会計基準についてAIが検索・回答するチャットアプリ。

## アーキテクチャ

- **A-RAG (Agentic RAG)**: ReActループでセマンティック検索・キーワード検索・チャンク読み取りを組み合わせて回答を生成
- **LLM**: Gemini (gemini-3.1-flash-lite-preview)
- **Embedding**: Gemini (gemini-embedding-2-preview / 3072次元)
- **Frontend**: SvelteKit + SSE ストリーミング
- **Backend**: FastAPI (Python)
- **インフラ**: Google Cloud Run + GCS

## データソース

| ソース | 内容 |
|--------|------|
| ASBJ | 企業会計基準、適用指針、実務対応報告 |
| SSBJ | サステナビリティ開示基準 (ユニバーサル基準、一般開示基準、気候関連開示基準) |
| 会計基準審議会 | 固定資産の減損に係る会計基準 等 |
| 財務諸表等規則 | 関連する規則の引用 |

- チャンク数: 4,424
- 文数 (sentence-level index): 124,466
- Embedding次元: 3,072 (float16圧縮)

## プロジェクト構成

```
src/
  api/main.py          # FastAPI エントリポイント (startup事前ロード)
  arag/
    agent.py           # ReActエージェントループ
    llm.py             # Gemini LLMクライアント (リトライ付き)
    prompt.py          # システムプロンプト
    config.py          # 設定管理
    tools/
      semantic_search.py  # Gemini embedding による意味検索 (npz/float16対応)
      keyword_search.py   # キーワード検索
      read_chunk.py       # チャンク全文読み取り
  embedding/
    gemini.py          # Gemini Embedder (バッチ処理・リトライ)
frontend/              # SvelteKit SPA
scripts/
  build_index.py       # 文レベルembeddingインデックス構築 (チェックポイント付き)
  process_pdfs.py      # PDF → チャンク変換
  deploy.sh            # Cloud Run デプロイ
  download_additional_sources.py  # 追加ソースダウンロード
data/
  chunks.json          # チャンクデータ
  index/
    sentence_index.npz   # Embedding (float16圧縮, ~670MB)
    sentence_meta.pkl    # メタデータ (文、チャンクマッピング)
    sentence_index.pkl   # レガシー形式 (float32, ~1.5GB)
```

## セットアップ

```bash
# 依存関係インストール
uv sync

# フロントエンドビルド
cd frontend && npm ci && npm run build && cd ..

# 環境変数
export GEMINI_API_KEY="your-api-key"
```

## インデックス構築

```bash
# PDFからチャンク生成
python scripts/process_pdfs.py

# Embeddingインデックス構築 (チェックポイント付き、中断再開可能)
python scripts/build_index.py
```

## ローカル実行

```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## デプロイ

```bash
# Cloud Run にデプロイ (Cloud Build + GCR)
bash scripts/deploy.sh

# GCS にインデックスをアップロード
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
```

## コールドスタート最適化

- `@app.on_event("startup")` で事前ロード (リクエスト前にインデックスをメモリに展開)
- Embedding を float16 + npz圧縮で保存 (1,482MB → 694MB, 53%削減)
- GCS からの npz/meta 優先ダウンロード
- Gemini API コールにリトライロジック (3回、指数バックオフ)
