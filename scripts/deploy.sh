#!/bin/bash
set -euo pipefail

PROJECT_ID="jp-accounting-chat"
REGION="asia-northeast1"
SERVICE_NAME="accounting-qa"
REPO="accounting-qa"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE_NAME}"

echo "=== Building and deploying to Cloud Run ==="

# Configure docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet 2>/dev/null

# Build and push Docker image using Cloud Build
echo "Building Docker image..."
gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" --region "${REGION}"

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "CEREBRAS_API_KEY=${CEREBRAS_API_KEY},GEMINI_API_KEY=${GEMINI_API_KEY},USE_GCS=true,GCS_BUCKET=jp-accounting-chat-data,GCS_PREFIX=index" \
  --min-instances 0 \
  --max-instances 10

echo "=== Deployment complete ==="
echo "Service URL:"
gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format 'value(status.url)'
