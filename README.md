# LLM Deployment Mechanics

Hands-on exploration of how LLMs actually get served in production: quantization for compression, and vLLM for high-throughput inference. AWS Terraform infra is included but untested (blocked by GPU spot quota); the actual deployment ran on a RunPod A40.

## Why this project exists

Most GenAI portfolios stop at "I built a RAG app." This one covers the layer underneath: how a 14GB model gets shrunk to fit cheap hardware, and how a serving engine squeezes high throughput out of a single GPU. This is the infrastructure layer of MLOps — the part that turns a model into a product.

## What's in here

| Folder | Status | Contents |
|---|---|---|
| `terraform/` | ⚠️ Untested | Provisions a g5.xlarge spot GPU instance + security group. Blocked by AWS's default zero GPU spot quota on new accounts — never actually ran |
| `quantization/` | ✅ Working (AWQ) | Scripts to quantize a base model with AWQ and GPTQ. AWQ produced the results below; GPTQ was attempted but blocked by a flaky `auto-gptq` build and deprioritized |
| `serving/` | ✅ Working (vLLM) | Launch scripts for vLLM and TGI. vLLM is what's actually been deployed and tested; `run_tgi.sh` and the FastAPI gateway (`app.py`) are scaffolded but untested |
| `benchmark/` | ✅ Working | Load-testing script — produced the throughput/latency numbers below |
| `docs/` | ✅ Working | Benchmark results and lessons learned. Architecture diagram not yet done |

## What's actually been run end-to-end

1. Quantized Mistral-7B-Instruct-v0.2 to AWQ 4-bit on a GPU pod
2. Backed up the quantized model to Hugging Face Hub
3. Deployed it with vLLM via the official `vllm/vllm-openai` Docker image on a RunPod A40
4. Hit the live OpenAI-compatible API endpoint with a concurrent load test and recorded real throughput/latency

## Not yet done

- GPTQ quantization comparison (AWQ vs GPTQ)
- TGI serving comparison (vLLM vs TGI head-to-head)
- AWS Terraform deployment actually running end-to-end (code written, blocked on GPU spot quota approval)
- FastAPI gateway tested against a live backend
- CloudWatch metrics integration
- Architecture diagram

## How to reproduce

1. **Quantize:**
   ```
   cd quantization
   pip install -r requirements.txt
   python quantize_awq.py
   ```
2. **Serve** (recommended: official Docker image, not manual pip install — see Lessons Learned for why):
   ```
   docker run --gpus all -p 8000:8000 --ipc=host \
     vllm/vllm-openai:latest \
     --model <your-quantized-model-path-or-hf-repo> \
     --quantization awq --dtype float16 --max-model-len 4096 \
     --gpu-memory-utilization 0.85 --served-model-name mistral-7b-awq
   ```
3. **Benchmark:**
   ```
   cd benchmark
   python load_test.py --backend vllm --concurrency 10 --total-requests 100
   ```

## Model used

Default: `mistralai/Mistral-7B-Instruct-v0.2` (swap in `quantization/config.py`). FP16 needs ~14GB VRAM; AWQ 4-bit needs ~4GB — this is the gap measured below.

## Results

- Quantized Mistral-7B-Instruct-v0.2 with AWQ 4-bit: **14GB → 3.9GB (72% size reduction)**
- Served via vLLM on a single A40 GPU, achieving **139 tokens/sec** at concurrency=10
- p50 latency 6.99s, p95 latency 7.15s, p99 latency 7.19s — tight spread even under concurrent load

Quantized model published: https://huggingface.co/atulkrs/mistral-7b-awq

## Lessons Learned

Deploying vLLM hit a string of real-world infra issues worth documenting:

- AWS spot quota defaults to 0 for GPU instances on new accounts — required a Service Quotas request, and a separate AMI naming change broke the original Terraform AMI filter
- Driver/CUDA version mismatches between pip-installed PyTorch and the host GPU driver caused repeated import failures
- Resolved by switching from a manual pip install to the official `vllm/vllm-openai` Docker image, which ships a pre-tested, matched dependency set
- AWQ quantization requires `--dtype float16` explicitly; `--dtype auto` can resolve to an unsupported dtype
- vLLM's official Docker image auto-generates an API key (`VLLM_API_KEY`) and enforces it on all requests by default

## Stack

RunPod (A40, actual deployment) · AWS EC2 + Terraform (g5.xlarge spot, untested) · vLLM · AutoAWQ · Hugging Face Hub · Docker · Hugging Face TGI (planned) · AutoGPTQ (planned) · FastAPI (scaffolded) · CloudWatch (planned)
