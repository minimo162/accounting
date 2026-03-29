#!/bin/bash
set -euo pipefail

PROJECT_ID="jp-accounting-chat"
REGION="asia-northeast1"
SERVICE_NAME="accounting-qa"
REPO="accounting-qa"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}"
GCLOUD_BIN="${GCLOUD_BIN:-gcloud}"
RUN_PREDEPLOY_LOCAL_EVAL="${RUN_PREDEPLOY_LOCAL_EVAL:-1}"
RUN_POSTDEPLOY_API_EVAL="${RUN_POSTDEPLOY_API_EVAL:-1}"
EVAL_CASES="${EVAL_CASES:-eval/answer_eval_set.jsonl}"
EVAL_GATE_MODE="${EVAL_GATE_MODE:-hard}"
EVAL_LOCAL_LIMIT="${EVAL_LOCAL_LIMIT:-3}"
EVAL_PROD_LIMIT="${EVAL_PROD_LIMIT:-}"
EVAL_OVERRIDE_REASON="${EVAL_OVERRIDE_REASON:-}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-eval/reports/deploy}"

if ! command -v "${GCLOUD_BIN}" >/dev/null 2>&1; then
  if [ -x "${HOME}/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD_BIN="${HOME}/google-cloud-sdk/bin/gcloud"
  else
    echo "gcloud not found" >&2
    exit 1
  fi
fi

run_answer_eval() {
  local backend="$1"
  local output_prefix="$2"
  local api_url="${3:-}"
  local limit="${4:-}"
  local -a cmd=(
    uv run python scripts/eval_answers.py
    --backend "${backend}"
    --cases "${EVAL_CASES}"
    --gate-mode "${EVAL_GATE_MODE}"
    --format markdown
    --json-output "${EVAL_REPORT_DIR}/${output_prefix}.json"
    --markdown-output "${EVAL_REPORT_DIR}/${output_prefix}.md"
  )

  if [ -n "${api_url}" ]; then
    cmd+=(--api-url "${api_url}")
  fi
  if [ -n "${limit}" ]; then
    cmd+=(--limit "${limit}")
  fi
  if [ -n "${EVAL_OVERRIDE_REASON}" ]; then
    cmd+=(--override-reason "${EVAL_OVERRIDE_REASON}")
  fi

  "${cmd[@]}"
}

echo "=== Building and deploying to Cloud Run ==="

mkdir -p "${EVAL_REPORT_DIR}"

if [ "${RUN_PREDEPLOY_LOCAL_EVAL}" = "1" ]; then
  echo "=== Running pre-deploy local answer eval gate (${EVAL_GATE_MODE}) ==="
  if [ -n "${EVAL_OVERRIDE_REASON}" ]; then
    echo "Gate override reason: ${EVAL_OVERRIDE_REASON}"
  fi
  run_answer_eval "local" "predeploy_local_answer_eval" "" "${EVAL_LOCAL_LIMIT}"
fi

# Configure docker auth
"${GCLOUD_BIN}" auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null

# Build and push Docker image using Cloud Build
echo "Building Docker image..."
"${GCLOUD_BIN}" builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" --region "${REGION}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
"${GCLOUD_BIN}" run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --labels "app=accounting-qa,component=api,env=prod" \
  --update-env-vars "LLM_PROVIDER=deepseek,LLM_MODEL=deepseek-chat,LLM_BASE_URL=https://api.deepseek.com/v1,USE_GCS=true,GCS_BUCKET=jp-accounting-chat-data,GCS_PREFIX=index,RETRIEVAL_RERANKER=heuristic,RETRIEVAL_ENABLE_LLM_QUERY_EXPANSION=false,RETRIEVAL_SEMANTIC_TOP_K=18,RETRIEVAL_KEYWORD_TOP_K=18,RETRIEVAL_RERANK_TOP_N=12,RETRIEVAL_EXPANSION_MAX_VARIANTS=2,APP_ENV=prod,OBS_SERVICE=accounting-qa" \
  --remove-env-vars "RERANKER_LLM_MODEL" \
  --update-secrets "DEEPSEEK_API_KEY=deepseek-api-key:latest" \
  --min-instances 0 \
  --max-instances 10

echo "=== Deployment complete ==="
echo "Service URL:"
SERVICE_URL="$("${GCLOUD_BIN}" run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)')"
echo "${SERVICE_URL}"

if [ "${RUN_POSTDEPLOY_API_EVAL}" = "1" ]; then
  echo "=== Running post-deploy production answer eval gate (${EVAL_GATE_MODE}) ==="
  run_answer_eval "api" "postdeploy_prod_answer_eval" "${SERVICE_URL}" "${EVAL_PROD_LIMIT}"
fi

echo "Answer eval reports:"
echo "- ${EVAL_REPORT_DIR}/predeploy_local_answer_eval.md"
echo "- ${EVAL_REPORT_DIR}/predeploy_local_answer_eval.json"
if [ "${RUN_POSTDEPLOY_API_EVAL}" = "1" ]; then
  echo "- ${EVAL_REPORT_DIR}/postdeploy_prod_answer_eval.md"
  echo "- ${EVAL_REPORT_DIR}/postdeploy_prod_answer_eval.json"
fi
