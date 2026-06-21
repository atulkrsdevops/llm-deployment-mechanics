#!/bin/bash
# Serve the AWQ-quantized model with vLLM's OpenAI-compatible API server.
#
# Setup (on the GPU instance):
#   pip install vllm
#
# Usage:
#   ./run_vllm.sh
#
# Then test with:
#   curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model": "mistral-7b-awq", "prompt": "Explain quantization in one sentence:", "max_tokens": 50}'

set -e

MODEL_PATH="${1:-../quantization/models/mistral-7b-awq}"

echo "Starting vLLM server with model: $MODEL_PATH"
echo "Logs piping to vllm_server.log - watch GPU memory with: watch -n1 nvidia-smi"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --quantization awq \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  --served-model-name mistral-7b-awq \
  2>&1 | tee vllm_server.log
