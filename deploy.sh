#!/usr/bin/env bash
# Deploy text classification model to GCP Cloud Run
# Usage: ./deploy.sh <GCP_PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh <GCP_PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE_NAME="cc-text-classifier"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Detection of the virtual environment python
PYTHON_CMD="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
fi

echo "=== Step 1: Convert model to ONNX ==="
$PYTHON_CMD src/export_onnx.py

echo ""
echo "=== Step 2: Configure Docker Authentication ==="
gcloud auth configure-docker gcr.io --quiet

echo ""
echo "=== Step 3: Build Docker image (targeting linux/amd64) ==="
docker build --platform linux/amd64 -t "${SERVICE_NAME}" .

echo ""
echo "=== Check Model Size ==="
ls -lh models/model.onnx

echo ""
echo "=== Step 4: Tag and push to GCR ==="
docker tag "${SERVICE_NAME}" "${IMAGE}"
docker push "${IMAGE}"

echo ""
echo "=== Step 5: Deploy to Cloud Run ==="
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --region "${REGION}" \
  --memory 1Gi \
  --allow-unauthenticated

echo ""
echo "=== Done! ==="
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format "value(status.url)")
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test with:"
echo "  curl -X POST ${SERVICE_URL}/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"Convolutional Neural Networks for Image Recognition\"}'"
