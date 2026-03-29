# 会計基準 Q&A

日本の会計基準について、出典付きで検索・回答するチャットアプリです。

- Backend: FastAPI + Agentic RAG
- Frontend: SvelteKit + SSE ストリーミング
- LLM: DeepSeek V3.2（API model: `deepseek-chat`）を既定利用、必要に応じて Gemini / Cerebras
- Retrieval: dense + BM25 + RRF + heuristic/LLM rerank + optional query expansion / HyDE
- Chunking: parent/child chunking。検索は child、表示と引用は parent
- Infra: Cloud Run + GCS

運用手順は [docs/operations_runbook.md](/workspace/accounting/accounting/docs/operations_runbook.md) を参照してください。通常デプロイ、corpus 更新、quality gate fail、secret rotation、Cloud Run rollback、誤引用対応、週次レビュー基準を 1 か所にまとめています。

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
  corpus_pipeline.py       corpus 更新 / validate / rollback の統合ジョブ
  process_pdfs.py          PDF を parent/child chunk 化
  process_egov.py          e-Gov XML を parent/child chunk 化
  process_html_sources.py  HTML/PDF ソースを parent/child chunk 化
  process_full_texts.py    補助用の全文テキスト生成
  add_context.py           chunk に追加文脈を付与
  build_index.py           searchable chunk の埋め込み index 構築
  source_manifest.py       source hash / fetched_at / version の manifest 管理
  eval_retrieval.py        retrieval 評価
  eval_answers.py          回答品質・速度評価
  monitor_answer_eval.py   代表質問の定期監視
  deploy.sh                Cloud Run デプロイ
eval/
  answer_eval_set.jsonl    回答評価ケース
data/
  chunks.json
  pdf_sources.json
  source_manifest.json
  pipeline_runs/
  releases/
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

推奨フロー:

```bash
uv run python scripts/corpus_pipeline.py update
```

このジョブは次をまとめて実行します。

- `data/pdf_sources.json` を使った PDF sync
- `process_pdfs.py` / `process_html_sources.py --refresh` / `process_egov.py --refresh`
- `build_index.py`
- `source_manifest.json` の更新
- `pipeline_runs/<run_id>/summary.json` の出力
- `releases/<run_id>/` への snapshot 作成

個別実行したい場合の最小構成:

```bash
# 1. PDF / HTML / e-Gov から chunk を生成
uv run python scripts/download_pdfs.py
uv run python scripts/process_pdfs.py
uv run python scripts/process_html_sources.py --refresh
uv run python scripts/process_egov.py --refresh

# 2. 必要なら全文テキストや追加コンテキストを生成
uv run python scripts/process_full_texts.py
uv run python scripts/add_context.py

# 3. searchable chunk の index を構築
uv run python scripts/build_index.py
```

補足:

- `source_manifest.json` には各 source file の `version` `fetched_at` `source_hash` `checked_at` を保存します
- 再構築後の `chunks.json` には `source_version` `source_fetched_at` `source_hash` `source_url` を埋め込みます
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

integrity check:

```bash
uv run python scripts/corpus_pipeline.py validate
```

この validate は少なくとも次を確認します。

- `chunks.json` に source metadata が入っていること
- child chunk の `parent_id` が壊れていないこと
- `source_manifest.json` と `chunks.json` の source path が一致すること
- `sentence_meta.pkl` の chunk IDs が現在の `chunks.json` と整合すること

重複回避のルール:

- 一部法令は PDF ではなく e-Gov XML を正本として扱います
- その対象 PDF は `process_pdfs.py` 側でスキップされる前提なので、法令系ソースを追加する際は `process_egov.py` の `LAWS` 定義との重複を確認してください

## デプロイ

Cloud Run へのデプロイ:

```bash
bash scripts/deploy.sh
```

`scripts/deploy.sh` は既定で次の順に実行します。

1. pre-deploy local subset eval (`gate_mode=hard`, 既定 3 ケース)
2. Cloud Run deploy
3. post-deploy production API eval (`gate_mode=hard`)

レポートは既定で `eval/reports/deploy/` に出ます。hard gate を一時的に bypass する必要がある場合だけ、理由を残して実行します。

```bash
EVAL_OVERRIDE_REASON="temporary rollout for logging-only check" bash scripts/deploy.sh
```

主な env:

- `RUN_PREDEPLOY_LOCAL_EVAL=0`: local subset gate をスキップ
- `RUN_POSTDEPLOY_API_EVAL=0`: post-deploy API gate をスキップ
- `EVAL_GATE_MODE=hard|strict|none`: gate の厳しさ
- `EVAL_LOCAL_LIMIT=3`: pre-deploy のケース数
- `EVAL_PROD_LIMIT=5`: post-deploy のケース数を絞るときに使う
- `EVAL_OVERRIDE_REASON=...`: manual override 理由をレポートに残す
- `EVAL_REPORT_DIR=...`: レポート出力先

デプロイ前チェック:

```bash
uv run python scripts/corpus_pipeline.py validate
uv run pytest tests/test_corpus_pipeline.py tests/test_agent_loop_control.py tests/test_agent_references.py
```

インデックスを GCS にアップロードする例:

```bash
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/
gsutil cp data/pdf_sources.json gs://jp-accounting-chat-data/index/
gsutil cp data/source_manifest.json gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
```

版管理付きで置く例:

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
gsutil cp data/pdf_sources.json gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
gsutil cp data/source_manifest.json gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
gsutil cp data/index/sentence_index.pkl gs://jp-accounting-chat-data/index/releases/${RUN_ID}/
```

デプロイ時の前提:

- `scripts/deploy.sh` には `PROJECT_ID` `REGION` `SERVICE_NAME` `REPO` が固定値で入っています
- 実行環境で `gcloud auth login` と `gcloud config set project jp-accounting-chat` が済んでいる必要があります
- Cloud Run 上では `DEEPSEEK_API_KEY` を別途設定しておく必要があります。`deploy.sh` では非シークレット設定のみ更新します
- 別環境へ出す場合は、先に `scripts/deploy.sh` の定数と GCS 設定を見直してください

rollback:

```bash
uv run python scripts/corpus_pipeline.py rollback <run_id>
```

`data/releases/<run_id>/` に保存された `chunks.json` `pdf_sources.json` `source_manifest.json` `index/*` を現在の `data/` に戻します。Cloud Run / GCS を巻き戻す場合は、同じ `<run_id>` の release artifact を再アップロードしてください。

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
  --gate-mode hard \
  --json-output eval/reports/prod_answer_eval_gate.json \
  --markdown-output eval/reports/prod_answer_eval_gate.md
```

gate mode:

- `none`: レポートのみ。exit code では block しない
- `hard`: 引用整合と重大品質 fail のみ block
- `strict`: fail reason が 1 件でもあれば block

`hard` で block するのは次です。

- `missing_all`
- `must_exclude`
- `citations<...`
- `missing_inline_citations`
- `uncited_lines>...`
- `reference_alignment_mismatch`
- `missing_reference_urls`

manual override が必要な場合は、理由を残して実行します。

```bash
uv run python scripts/eval_answers.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --cases eval/answer_eval_set.jsonl \
  --gate-mode hard \
  --override-reason "known latency regression during index rebuild" \
  --markdown-output eval/reports/prod_answer_eval_gate.md
```

`eval/answer_eval_set.jsonl` の主な項目:

- `must_include_all`: 必ず含めたい語
- `must_include_any`: どれか 1 つ以上含めたい語
- `must_exclude`: 出してほしくない語
- `min_citations`: 本文中の最低参照数
- `max_loops`: 許容 loop 数の上限
- `max_latency_sec`: 応答時間の上限
- `max_retrieved_tokens`: retrieval で読んだ token 数の上限
- `max_uncited_lines`: 見出し以外で引用番号が付いていない行の許容数。通常は `0`
- `allowed_stop_reasons`: 許容する停止理由。`max_loops` で終わった回答を quality gate で落としたいときに使います
- `require_inline_citations`: `[1]` 形式の本文引用を必須にするか。既定は `true`
- `require_reference_alignment`: 本文の最大引用番号と `references` 件数の一致を必須にするか。既定は `true`
- `require_reference_urls`: 参照カードに URL があることを必須にするか。既定は `true`

運用メモ:

- しきい値は「理想値」ではなく、まず現行ベースラインを継続監視できる値に合わせています
- retrieval を改善して baseline が下がったら、`eval/answer_eval_set.jsonl` の `max_loops` と `max_retrieved_tokens` を一緒に引き締めます
- baseline を更新するときは、少なくとも 2 回連続で full eval を回し、pass rate 100%、`max_uncited_lines=0` 維持、参照整合エラー 0 件を確認してから閾値を下げてください
- `answer_eval` は本文品質だけでなく、無引用文、参照番号不整合、参照 URL 欠落も落とします。`/api/ask` は評価用に `references` と `source_url_map` も返します

本番監視向けの代表質問チェック:

```bash
uv run python scripts/monitor_answer_eval.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --limit 3 \
  --output eval/reports/prod_monitor.json \
  --fail-on-threshold
```

このスクリプトは PRD の目標値を既定値として使います。

- `avg_latency_sec <= 35`
- `p95_latency_sec <= 60`
- `avg_loops <= 5`

必要なら `--avg-latency-threshold` `--p95-latency-threshold` `--avg-loops-threshold` で上書きできます。API 実行時は `X-Monitor-Case-Id` を付けるので、Cloud Logging 上で代表質問だけを絞って確認できます。

## Observability

`/api/ask` と `/api/ask/stream` は request ごとに `X-Request-ID` を返し、回答 metadata にも `request_id` を含めます。frontend は既定で利用者向けの簡潔な summary chips を表示し、`開発表示` に切り替えると request id / stop reason / retrieved tokens / 実行 step を確認できます。回答 metadata には `evidence_coverage` と `uncertainty` も含めるので、未確定の論点を UI でも明示できます。

Cloud Logging 向けの structured log は JSON 1 行で出力します。主な項目:

- `request_id`
- `monitor_case_id`
- `query_class`
- `search_mode`
- `loops`
- `tool_call_count`
- `read_chunk_count`
- `retrieved_tokens`
- `stop_reason`
- `reference_count`
- `cited_reference_count`
- `zero_reference`
- `exploration_signature`
- `elapsed_ms`

`src/arag/agent.py` は retrieval 経路まで含む詳細ログ、`src/api/main.py` は HTTP request 単位の完了/失敗ログを出します。加えて `/api/ui-event` は `question_submitted` `followup_submitted` `answer_rendered` `reference_opened` を structured log に出すので、参照カード click-through と再質問率を Cloud Logging で集計できます。Cloud Run デプロイでは `APP_ENV=prod` `OBS_SERVICE=accounting-qa` と Cloud Run labels を設定するよう `scripts/deploy.sh` を更新しています。

GitHub Actions:

- `.github/workflows/answer-eval.yml` は `workflow_dispatch` と週次 schedule で実行します
- `workflow_dispatch` では `gate_mode` `local_limit` `override_reason` を指定できます
- repository variable `ACCOUNTING_QA_API_URL` を設定すると、deployed API に対する production gate も実行します
- local subset gate は `eval/reports/local_answer_eval_gate.{md,json}`、production gate は `eval/reports/prod_answer_eval_gate.{md,json}` を artifact に保存します
- manual override を使う場合は `override_reason` を空にしないでください。理由は report と `GITHUB_STEP_SUMMARY` に残ります
- repository variable / secret の設定:
  - `ACCOUNTING_QA_API_URL`: デプロイ済み API の `/api/ask` ベース URL
  - `DEEPSEEK_API_KEY`: local eval や deploy 時に使う LLM key
- レポートでは `Gate passed` `Blocking case IDs` `Reference alignment failures` `Missing reference URL failures` を優先確認してください

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
