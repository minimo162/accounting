# 運用 Runbook

会計基準 Q&A の通常運用、データ更新、品質 gate、障害対応、週次レビューの標準手順です。  
対象は `accounting-qa` の Cloud Run / GCS / answer eval 運用です。

## 1. 前提チェック

作業前に次を確認します。

- `uv sync` と `frontend/npm ci` が完了している
- `gcloud auth login` と `gcloud config set project jp-accounting-chat` が済んでいる
- ローカル eval を回す場合は `DEEPSEEK_API_KEY` を設定済み
- index 再構築を回す場合は `GEMINI_API_KEY` または対応する embedding provider の key を設定済み

最低限の確認コマンド:

```bash
uv --version
gcloud auth list
gcloud config get-value project
```

`gcloud` が `PATH` にいない環境では、必要に応じて `GCLOUD_BIN=/absolute/path/to/gcloud` を付けて `scripts/deploy.sh` を実行します。

## 2. 通常デプロイ

通常デプロイは次の順です。

1. データと index の整合を確認する
2. 回帰テストを実行する
3. `scripts/deploy.sh` を実行する
4. pre/post deploy の answer eval report を確認する

標準コマンド:

```bash
uv run python scripts/corpus_pipeline.py validate
uv run pytest tests/test_corpus_pipeline.py tests/test_agent_loop_control.py tests/test_agent_references.py tests/test_api_reference_contract.py
cd frontend && npm run check && cd ..
bash scripts/deploy.sh
```

確認ポイント:

- `eval/reports/deploy/predeploy_local_answer_eval.{md,json}`
- `eval/reports/deploy/postdeploy_prod_answer_eval.{md,json}`
- Cloud Run revision と service URL
- hard gate の `Blocking case IDs`

manual override は例外対応です。使う条件は次の両方を満たすときだけです。

- hard fail の原因が利用者影響ではなく、時間制約つきの運用上の都合である
- rollback 条件、担当者、修正期限を issue または ticket に残せる

例:

```bash
EVAL_OVERRIDE_REASON="temporary rollout for logging-only check; revert if prod gate degrades" bash scripts/deploy.sh
```

## 3. Corpus 更新と Index 再構築

ソース更新、chunk 再生成、index 再構築、manifest 更新、snapshot 作成は `corpus_pipeline.py update` を使います。

```bash
uv run python scripts/corpus_pipeline.py update
uv run python scripts/corpus_pipeline.py validate
```

このジョブは次を行います。

- PDF sync
- `process_pdfs.py`
- `process_html_sources.py --refresh`
- `process_egov.py --refresh`
- `build_index.py`
- `source_manifest.json` 更新
- `data/pipeline_runs/<run_id>/summary.json` 出力
- `data/releases/<run_id>/` snapshot 作成

更新後は GCS へ artifact を上げてから deploy します。

```bash
gsutil cp data/chunks.json gs://jp-accounting-chat-data/index/
gsutil cp data/pdf_sources.json gs://jp-accounting-chat-data/index/
gsutil cp data/source_manifest.json gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.npz gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_meta.pkl gs://jp-accounting-chat-data/index/
gsutil cp data/index/sentence_index.pkl gs://jp-accounting-chat-data/index/
```

注意:

- `data/releases/` は `update` を一度も回していない workspace には存在しません
- `validate` で `missing_source_*` や `index_searchable_count_mismatch` が大量に出る場合、現在の `data/` が旧形式または stale index の可能性が高いです。先に `update` を実行してください
- Cloud Run revision rollback だけでは GCS 上の index は戻りません。data 問題は artifact rollback と再アップロードが別途必要です

## 4. Quality Gate Fail 対応

まず fail が hard か soft かを確認します。

- hard fail: `missing_all` `must_exclude` `citations<...` `missing_inline_citations` `uncited_lines>...` `reference_alignment_mismatch` `missing_reference_urls`
- soft fail: `loops` `latency` `retrieved_tokens` などの性能劣化

初動:

1. report から `Blocking case IDs` と `failure_reasons` を確認する
2. 単一ケースで再実行する
3. `request_id` と `monitor_case_id` を使って Cloud Logging を引く
4. corpus / retrieval / answer synthesis / UI のどこで壊れたかを切り分ける

単一ケースの再実行:

```bash
uv run python scripts/eval_answers.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --cases eval/answer_eval_set.jsonl \
  --case-id lease_revision_detail \
  --gate-mode hard \
  --markdown-output eval/reports/debug_lease_revision_detail.md \
  --json-output eval/reports/debug_lease_revision_detail.json
```

Cloud Logging で見る軸:

- `request_id`
- `monitor_case_id`
- `event=api_request_complete`
- `event=agent_run_complete`
- `event=ui_event`

代表的な切り分け:

- `missing_all`: retrieval miss、coverage repair miss、質問分解 miss を優先確認
- `reference_alignment_mismatch`: answer 中の `[N]` と `references` 件数のズレ。引用生成 or sanitize を確認
- `missing_reference_urls`: `pdf_sources.json` と reference card 生成を確認
- `loops` / `retrieved_tokens`: query rewrite、hybrid search、coverage gate、excerpt selection を確認

意図的に gate fail を再現して運用手順を試す例:

```bash
tmp=$(mktemp)
printf '%s\n' \
  '{"case_id":"intentional_fail","question":"借手は短期リースや少額リースをどのように扱いますか","must_include_all":["存在しない論点"],"min_citations":1,"max_loops":10,"max_latency_sec":120,"max_retrieved_tokens":20000,"max_uncited_lines":0,"allowed_stop_reasons":["natural","retrieval_budget","evidence_sufficient"]}' \
  > "$tmp"

uv run python scripts/eval_answers.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --cases "$tmp" \
  --gate-mode hard \
  --markdown-output eval/reports/intentional_fail_runbook.md \
  --json-output eval/reports/intentional_fail_runbook.json

rm -f "$tmp"
```

この例では `missing_all=['存在しない論点']` が hard fail になります。

## 5. Secret 更新

現在の Cloud Run deploy は `DEEPSEEK_API_KEY=deepseek-api-key:latest` を参照します。DeepSeek key をローテーションする場合:

1. 新しい key を Secret Manager に追加
2. deploy を再実行して最新 version を反映
3. post-deploy gate を確認

```bash
printf '%s' "$DEEPSEEK_API_KEY" | gcloud secrets versions add deepseek-api-key --data-file=-
bash scripts/deploy.sh
```

ローカル開発用の key は `.env` または shell export を更新します。

注意:

- `scripts/deploy.sh` は `DEEPSEEK_API_KEY` secret しか更新しません
- provider を Gemini などへ切り替える場合は Cloud Run env vars も合わせて見直してください
- key 更新後は最低でも単一ケースの API eval を 1 回回して 401/403 が出ないことを確認します

## 6. Cloud Run Rollback

rollback は 2 種類あります。

- code / config rollback: Cloud Run revision を戻す
- data rollback: GCS / local artifact を戻す

### 6-1. Revision rollback

直近 revision を確認:

```bash
gcloud run revisions list \
  --service accounting-qa \
  --region asia-northeast1 \
  --project jp-accounting-chat \
  --limit 10
```

前 revision に 100% 戻す:

```bash
gcloud run services update-traffic accounting-qa \
  --region asia-northeast1 \
  --project jp-accounting-chat \
  --to-revisions=accounting-qa-00176-jsb=100
```

確認:

```bash
gcloud run services describe accounting-qa \
  --region asia-northeast1 \
  --project jp-accounting-chat \
  --format='value(status.latestReadyRevisionName,status.url)'
```

### 6-2. Data rollback

artifact snapshot がある場合:

```bash
uv run python scripts/corpus_pipeline.py rollback <run_id>
```

戻した後は restored artifact を GCS に再アップロードし、Cloud Run を再 deploy します。

重要:

- revision rollback は GCS 上の `chunks.json` / index を戻しません
- data rollback は Cloud Run revision を戻しません
- corpus 問題では両方必要になることがあります

## 7. 誤引用・誤回答報告の対応

まず利用者から次を回収します。

- 質問文
- 回答本文
- `request_id`
- 参照カード URL またはスクリーンショット
- 期待していた根拠

対応手順:

1. 同じ質問を `/api/ask` で再実行して再現する
2. `request_id` で `api_request_complete` と `agent_run_complete` を確認する
3. `references` と `source_url_map` が正しいかを見る
4. `read_chunk` に誤った親チャンクが混ざっていないか確認する
5. corpus 側の source metadata / `pdf_page` / section title を確認する

分類の目安:

- corpus artifact 問題: chunk metadata、`pdf_page`、`pdf_sources.json` が誤っている
- retrieval 問題: 関係ない chunk を読んでいる
- synthesis 問題: 根拠はあるが answer が取り違えている
- UI 問題: answer / references は正しいが、カード表示やリンク先が壊れている
- scope 外質問: corpus に十分な根拠がないのに断定してしまった

再発防止:

- unit test を追加する
- 必要なら `eval/answer_eval_set.jsonl` にケースを追加する
- representative question なら `monitor_answer_eval.py` でも追う
- root cause と再発防止策を issue / PR に残す

## 8. 週次品質レビュー

週次レビューでは最低限次を見ます。

- `answer_eval` full run の pass rate
- hard fail 件数
- soft fail の増減
- `avg_latency_sec`
- `p95_latency_sec`
- `avg_loops`
- `avg_retrieved_tokens`
- `reference_alignment_failures`
- `missing_reference_url_failures`
- `zero_reference` 件数
- `reference_opened / answer_rendered`
- `followup_submitted / question_submitted`

代表質問の定期監視:

```bash
uv run python scripts/monitor_answer_eval.py \
  --backend api \
  --api-url https://accounting-qa-xdt66erlqa-an.a.run.app \
  --limit 3 \
  --output eval/reports/prod_monitor.json \
  --fail-on-threshold
```

baseline 更新条件:

- full eval を少なくとも 2 回連続で実行する
- pass rate 100%
- `max_uncited_lines=0` 維持
- 参照整合エラー 0 件
- 参照 URL 欠落 0 件
- 改善が一時的な揺れでないと判断できる

その条件を満たしたら、`eval/answer_eval_set.jsonl` の `max_loops` と `max_retrieved_tokens` を一緒に引き締めます。

評価ケース追加基準:

- 利用者影響があった質問
- 長文・複数論点で coverage が落ちやすい質問
- exact query / clause 指定のような precision が必要な質問
- 既知の latency / exploration regression を起こした質問
- UI 上の不確実性表示や citation alignment に影響する質問

## 9. Unsupported / Corpus 外質問の Expected Behavior

現行の対象範囲は日本基準中心です。IFRS 専用論点や組織内文書は、corpus に明示的に追加されていない限り対象外です。

期待する挙動:

- corpus に根拠がない論点は断定しない
- 「今回確認できた根拠では不十分」と明示する
- 部分的に根拠がある論点だけを引用付きで回答する
- UI では `uncertainty` と参照カードで、どこまで確認できたかを分かるようにする

してはいけない挙動:

- corpus にない基準や社内規程を推測で補う
- 引用なしで結論を断定する
- 関連しそうな別基準を根拠として誤用する

将来拡張時の扱い:

- IFRS を本格対応するなら source ingest、eval case、monitor case を JGAAP と分けて追加する
- 組織内文書を統合するなら access control、source labeling、out-of-scope 文言を先に定義する
- domain を増やすときは `doc_type` と eval set の coverage を先に拡張する

## 10. Hallucination 発生時の再発防止

hallucination を確認したら、その場の修正だけで閉じません。

1. 再現質問を保存する
2. failing eval case を追加するか、既存 case を tighten する
3. root cause を `corpus / retrieval / synthesis / UI` のいずれかに分類する
4. unit test または E2E test を追加する
5. 修正後に単一ケースと full eval を回す
6. 必要なら週次レビューの monitor 対象に昇格する

## 11. Dry Run Checklist

runbook 自体のドライランでは最低限次を実施します。

```bash
bash -n scripts/deploy.sh
uv run pytest tests/test_corpus_pipeline.py
uv run pytest tests/test_api_reference_contract.py tests/test_agent_references.py
uv run python scripts/monitor_answer_eval.py --backend api --api-url https://accounting-qa-xdt66erlqa-an.a.run.app --limit 1 --output eval/reports/prod_monitor_runbook_dryrun.json
```

この checklist で確認したいこと:

- deploy script の shell syntax が壊れていない
- rollback / validate の基礎挙動が test で担保されている
- monitor script が report を出せる
- answer eval fail を意図的に再現できる
