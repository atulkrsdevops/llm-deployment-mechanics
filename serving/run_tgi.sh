#!/bin/bash
# Serve the same quantized model with Hugging Face TGI, via Docker.
#
# Setup (on the GPU instance):
#   Install Docker + NVIDIA Container Toolkit if not already present
#   (Deep Learning AMI usually has both)
#
# Usage:
#   ./run_tgi.sh
#
# Then test with:
#   curl http://localhost:8000/generate \
#     -X POST \
#     -H "Content-Type: application/json" \
#     -d '{"inputs": "Explain quantization in one sentence:", "parameters": {"max_new_tokens": 50}}'

set -e

MODEL_PATH="${1:-$(pwd)/../quantization/models/mistral-7b-awq}"
VOLUME="$HOME/tgi-data"
mkdir -p "$VOLUME"

echo "Starting TGI server with model: $MODEL_PATH"

docker run --gpus all \
  --shm-size 1g \
  -p 8000:80 \
  -v "$MODEL_PATH":/data/model \
  -v "$VOLUME":/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id /data/model \
  --quantize awq \
  --max-total-tokens 4096 \
  2>&1 | tee tgi_server.log
