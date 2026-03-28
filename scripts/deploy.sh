#!/bin/bash
set -euo pipefail

PROJECT_ID="jp-accounting-chat"
REGION="asia-northeast1"
SERVICE_NAME="accounting-qa"
REPO="accounting-qa"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}"
GCLOUD_BIN="${GCLOUD_BIN:-gcloud}"

if ! command -v "${GCLOUD_BIN}" >/dev/null 2>&1; then
  if [ -x "${HOME}/google-cloud-sdk/bin/gcloud" ]; then
    GCLOUD_BIN="${HOME}/google-cloud-sdk/bin/gcloud"
  else
    echo "gcloud not found" >&2
    exit 1
  fi
fi

echo "=== Building and deploying to Cloud Run ==="

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
  --update-env-vars "LLM_PROVIDER=deepseek,LLM_MODEL=deepseek-chat,LLM_BASE_URL=https://api.deepseek.com/v1,USE_GCS=true,GCS_BUCKET=jp-accounting-chat-data,GCS_PREFIX=index,RETRIEVAL_RERANKER=heuristic,RETRIEVAL_ENABLE_LLM_QUERY_EXPANSION=false,RETRIEVAL_SEMANTIC_TOP_K=18,RETRIEVAL_KEYWORD_TOP_K=18,RETRIEVAL_RERANK_TOP_N=12,RETRIEVAL_EXPANSION_MAX_VARIANTS=2" \
  --remove-env-vars "RERANKER_LLM_MODEL" \
  --update-secrets "DEEPSEEK_API_KEY=deepseek-api-key:latest" \
  --min-instances 0 \
  --max-instances 10

echo "=== Deployment complete ==="
echo "Service URL:"
"${GCLOUD_BIN}" run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)'
