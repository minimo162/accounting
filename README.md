# 会計基準 Q&A

日本の会計基準について、出典付きで検索・回答するチャットアプリです。

- Backend: FastAPI + Agentic RAG
- Frontend: SvelteKit + SSE ストリーミング
- LLM: DeepSeek V3.2（API model: `deepseek-chat`）を既定利用、必要に応じて Gemini / Cerebras
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
- 実行時の `semantic_search` は `data/index/sentence_index.pkl` を読み込みます。`build_index.py` は互換性のため `npz` / `meta.pkl` とあわせて `pkl` も出力します

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
  eval_answers.py          回答品質・速度評価
  deploy.sh                Cloud Run デプロイ
eval/
  answer_eval_set.jsonl    回答評価ケース
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
export DEEPSEEK_API_KEY="your-api-key"
export GEMINI_API_KEY="your-api-key"
export CEREBRAS_API_KEY="your-api-key"  # Cerebras に戻す場合のみ

# OpenAI-compatible embeddings を使う場合
export EMBEDDING_API_KEY="your-api-key"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"

# 任意
export LLM_PROVIDER="deepseek"   # or gemini / cerebras
export LLM_MODEL="deepseek-chat" # DeepSeek-V3.2 の非 thinking モード
export LLM_BASE_URL="https://api.deepseek.com/v1"
export USE_GCS="false"           # ローカル data/ を使う場合
export DATA_DIR="data"
export GCS_BUCKET="jp-accounting-chat-data"
export GCS_PREFIX="index"
```

補足:

- LLM 推論は `LLM_PROVIDER` に従って DeepSeek / Gemini / Cerebras を切り替えます
- DeepSeek V3.2 は公式 API 上では `deepseek-chat` として公開されているため、既定値はその model ID を使っています
- 埋め込みは常に Gemini を使うため、インデックス構築や `semantic_search` には `GEMINI_API_KEY` が必要です
- `USE_GCS=true` の場合、起動時に `chunks.json` / `pdf_sources.json` / index 一式を GCS から取得します

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
- 回答品質と速度の回帰を見たい場合は `python scripts/eval_answers.py` と `eval/answer_eval_set.jsonl` を使います
- Cloud Run では `USE_GCS=true` の場合、起動時に GCS から `chunks.json` と index 一式を取得します
- `add_context.py` を実行して `context` が変わった場合も、`build_index.py` の checksum 判定で再埋め込み対象になります

更新判断の目安:

- PDF ソースを追加・差し替えた: `process_pdfs.py` のあとに `build_index.py`
- e-Gov 法令キャッシュを更新した: `process_egov.py` のあとに `build_index.py`
- 検索精度改善のため `context` を付与し直した: `add_context.py` のあとに `build_index.py`
- `read_document` 用の補助テキストだけ更新したい: `process_full_texts.py` のみでも可

重複回避のルール:

- 一部法令は PDF ではなく e-Gov XML を正本として扱います
- その対象 PDF は `process_pdfs.py` 側でスキップされる前提なので、法令系ソースを追加する際は `process_egov.py` の `LAWS` 定義との重複を確認してください

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

デプロイ時の前提:

- `scripts/deploy.sh` には `PROJECT_ID` `REGION` `SERVICE_NAME` `REPO` が固定値で入っています
- 実行環境で `gcloud auth login` と `gcloud config set project jp-accounting-chat` が済んでいる必要があります
- Cloud Run 上では `DEEPSEEK_API_KEY` を別途設定しておく必要があります。`deploy.sh` では非シークレット設定のみ更新します
- 別環境へ出す場合は、先に `scripts/deploy.sh` の定数と GCS 設定を見直してください

## 評価

retrieval のオフライン評価:

```bash
uv run python scripts/eval_retrieval.py --queries data/eval_queries.jsonl --mode hybrid
```

回答品質と速度の評価:

```bash
# ローカル agent に対して実行
uv run python scripts/eval_answers.py \
  --backend local \
  --cases eval/answer_eval_set.jsonl \
  --format markdown \
  --output eval/reports/local_answer_eval.md

# デプロイ済み API に対して実行
uv run python scripts/eval_answers.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --cases eval/answer_eval_set.jsonl \
  --format markdown \
  --output eval/reports/prod_answer_eval.md \
  --fail-on-fail
```

`eval/answer_eval_set.jsonl` の主な項目:

- `must_include_all`: 必ず含めたい語
- `must_include_any`: どれか 1 つ以上含めたい語
- `must_exclude`: 出してほしくない語
- `min_citations`: 本文中の最低参照数
- `max_loops`: 許容 loop 数の上限
- `max_latency_sec`: 応答時間の上限
- `max_retrieved_tokens`: retrieval で読んだ token 数の上限

運用メモ:

- しきい値は「理想値」ではなく、まず現行ベースラインを継続監視できる値に合わせています
- retrieval を改善して baseline が下がったら、`eval/answer_eval_set.jsonl` の `max_loops` と `max_retrieved_tokens` を一緒に引き締めます

GitHub Actions:

- `.github/workflows/answer-eval.yml` は `workflow_dispatch` と週次 schedule で実行します
- repository variable `ACCOUNTING_QA_API_URL` を設定すると、デプロイ済み API に対して `scripts/eval_answers.py` を走らせ、Markdown レポートを artifact に保存します
- workflow 側はまず計測レポートの蓄積を優先し、常時失敗にはしません。quality gate にしたい場合は手元や別 workflow で `--fail-on-fail` を付けます

## トラブルシュート

- `Error: GEMINI_API_KEY environment variable not set`
  - `build_index.py` 実行時に発生します。埋め込み構築には Gemini API キーが必須です
- 起動時に index が見つからない
  - ローカルなら `data/chunks.json` と `data/index/sentence_index.pkl` を確認してください
  - Cloud Run なら `USE_GCS=true` と `GCS_BUCKET` `GCS_PREFIX` の組み合わせを確認してください
- フロントから API に接続できない
  - 開発時のフロント既定 API は `http://localhost:8000` です。backend を別ポートで起動している場合は合わせてください

## 実装メモ

- `src/api/main.py` で startup 時にエージェントとインデックスを preload
- `src/arag/agent.py` は最大 25 ループ、128K token budget で ReAct 実行
- `src/arag/tools/hybrid_search.py` で dense / keyword / rerank を統合
- embedding provider は Gemini と OpenAI-compatible backend を切り替え可能
- 埋め込みは `npz + float16` で圧縮保存し、起動コストを抑制

## 引き継ぎ（2026-03-28）

### 直近セッションで行った主な変更

#### `scripts/process_pdfs.py`

- **物理ページ番号の正確な紐付け**: 全チャンクに `pdf_page` フィールドを付与。ページ境界マーカー `\x01PAGE:N\x01` をフルテキストに埋め込み、セクション分割時に `current_pdf_page` を順追跡することで 1 ページずれ問題を解消
- **タイトル抽出の改善 (`_extract_title`)**:
  - `第N号` を含む行を最優先（機関名より先に返す）
  - `移管指針` をキーワードに追加
  - 機関名のみの行（`企業会計基準委員会` 等）を全ループでスキップ
  - 検索行数を 50 → 150 に拡大
- **自動重複除去 (`_dedup_pdf_files`)**:
  - ファイル名から YYYYMMDD を抽出して新しい順にソート
  - 同一 `doc_title`（NFKC 正規化済み）の旧バージョンを自動スキップ
  - 199 → 148 PDF（51 ファイル削除）、17,088 → 11,226 チャンクに削減

#### `src/arag/agent.py`

- **`pdf_page` を使った `#page=N` アンカー付き URL 生成**: ソースカードのリンクが正確な PDF ページに飛ぶように
- **`source_url_map` にページアンカー付き URL を格納**: `第N号` キーに対して `base_url#page=N` を保存
- **重複引用の排除**: `_get_referenced_chunks` で `parent_id` による dedup を追加（親+子チャンクが同時に表示されていた問題を修正）
- **`_strip_chunk_refs` の拡張**: `[-21]` `[-23]` 形式の孤立した角括弧参照も除去
- **`ToolRegistry.execute` の引数名変更**: `name` → `tool_name`（LLM が `name=` 引数付きで `read_document` を呼ぶと TypeError になっていたバグを修正）

#### `frontend/src/routes/+page.svelte`

- **文中インラインリンクを廃止**: ページ推定の精度に限界があるためリンクを削除。ソースカードのみで引用ナビゲーションを提供
- **ソースカード表示名の修正**: `source` フィールドの `>` 以降（セクション見出し）を切り捨て、文書名のみ表示
- **コピーボタン（⎘）追加**: PDF 内 Ctrl+F 用にチャンクテキストをクリップボードにコピー
- **`copiedRefId` を `$state()` に変更**: Svelte 5 での reactivity 警告を修正

### 現在の既知課題

- `lease_20240913_06.pdf`（第35号）、`lease_20240913_07.pdf`（第36号）、`lease_20240913_08.pdf`（第18号）、`lease_20240913_12.pdf`（第29号）、`lease_20240913_43.pdf`（第26号）はリース基準と同日公表された他の基準で、対応する新版が index に存在しないため残っている。将来的に新版が追加されれば自動的に除去される
- `process_pdfs.py` のみ変更した場合、`build_index.py` の再実行（embedding 再構築）が必要。現在 GCS 上の `sentence_index.*` は古い chunks に対応しており、chunks の追加・削除があると検索精度に影響する可能性がある（今回の変更で大幅にチャンク数が減ったため、**index 再構築を推奨**）

### index 再構築手順

```bash
python scripts/build_index.py
gcloud storage cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gcloud storage cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
gcloud storage cp data/index/sentence_index.pkl gs://jp-accounting-chat-data/index/
bash scripts/deploy.sh
```
