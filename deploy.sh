#!/usr/bin/env bash
set -e

# ==============================================================================
# Deployment Script for Gemma 2 Embedding on Fireworks AI
# Optimizes for cost-effectiveness (On-Demand H100 with Scale-to-Zero)
# ==============================================================================

FIRECTL_BIN="$HOME/.local/firectl/firectl"
if ! command -v firectl &>/dev/null; then
    if [ -f "$FIRECTL_BIN" ]; then
        FIRECTL="$FIRECTL_BIN"
    else
        echo "Error: firectl not found. Please install via: brew tap fw-ai/firectl && brew install firectl"
        exit 1
    fi
else
    FIRECTL="firectl"
fi

echo "=== Fireworks AI Gemma 2 Embedding Deployment ==="
echo "Using CLI: $FIRECTL"

ACCOUNT_ID=$($FIRECTL account get -o json 2>/dev/null | grep -o '"id": *"[^"]*"' | head -1 | cut -d'"' -f4 || true)
if [ -z "$ACCOUNT_ID" ]; then
    ACCOUNT_ID=$(grep -E "^account_id" ~/.fireworks/settings.ini 2>/dev/null | awk -F= '{print $2}' | tr -d ' ' || true)
fi

if [ -z "$ACCOUNT_ID" ]; then
    echo "Error: Could not determine Fireworks account ID. Run '$FIRECTL auth login' first."
    exit 1
fi
echo "Account ID: $ACCOUNT_ID"

# Model Target Options:
# 1. BAAI/bge-multilingual-gemma2 (9.24B parameters - the full size Gemma 2 text embedding model)
# 2. google/embeddinggemma-2 (744M parameters - Google DeepMind multimodal embedding model)
MODEL_ID="${1:-gemma2-embedding-9b}"
SOURCE_PATH="${2:-}"
HF_URL="${3:-https://huggingface.co/BAAI/bge-multilingual-gemma2}"

echo "Model ID:         $MODEL_ID"
echo "Hugging Face Ref: $HF_URL"

if [ -z "$SOURCE_PATH" ]; then
    cat << 'EOF'

[!] No source path provided.
To register a custom model with Fireworks AI, model weights must be stored in:
  1. A local directory containing config.json and .safetensors weights
  2. An S3 bucket: s3://my-bucket/path/to/bge-multilingual-gemma2/
  3. A GCS bucket: gs://my-bucket/path/to/bge-multilingual-gemma2/

Usage:
  ./deploy.sh <model_id> <source_path> [huggingface_url]

Example:
  ./deploy.sh gemma2-embed s3://my-company-models/bge-multilingual-gemma2 https://huggingface.co/BAAI/bge-multilingual-gemma2

EOF
    exit 1
fi

echo ""
echo "--> Step 1: Registering Model on Fireworks..."
$FIRECTL model create "$MODEL_ID" "$SOURCE_PATH" \
    --embedding \
    --hugging-face-url "$HF_URL" \
    --description "Gemma 2 Full Size Embedding Model"

echo ""
echo "--> Step 2: Creating Cost-Effective On-Demand Deployment..."
echo "    - Accelerator: 1x NVIDIA_H100_80GB ($8.00/hr)"
echo "    - Scale to Zero: Enabled (10 min idle timeout)"
echo "    - Min Replicas: 0 | Max Replicas: 1"

$FIRECTL deployment create "accounts/$ACCOUNT_ID/models/$MODEL_ID" \
    --accelerator-type NVIDIA_H100_80GB \
    --accelerator-count 1 \
    --accept-shapeless-risk \
    --min-replica-count 0 \
    --max-replica-count 1 \
    --scale-to-zero-window 10m \
    --wait

echo ""
echo "=== Deployment Completed Successfully! ==="
echo "Endpoint Model ID: accounts/$ACCOUNT_ID/models/$MODEL_ID"
echo "Run benchmark test via:"
echo "  python benchmark.py --model accounts/$ACCOUNT_ID/models/$MODEL_ID --batch-sizes '1,16,64,128,256'"
