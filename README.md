# LLM Deployment Mechanics

Hands-on exploration of how LLMs actually get served in production: quantization for compression, and vLLM / TGI for high-throughput inference. Built on a single AWS g5.xlarge (A10G, 24GB VRAM) spot instance.

## Why this project exists

Most GenAI portfolios stop at "I built a RAG app." This one covers the layer underneath: how a 14GB model gets shrunk to fit cheap hardware, and how serving engines squeeze high throughput out of a single GPU. This is the infrastructure layer of MLOps — the part that turns a model into a product.

## What's in here

| Folder | Contents |
|---|---|
| `terraform/` | Provisions the g5.xlarge spot GPU instance + security group |
| `quantization/` | Scripts to quantize a base model with AWQ and GPTQ, with before/after measurements |
| `serving/` | Launch scripts for vLLM and TGI serving the same quantized model |
| `benchmark/` | Load-testing script (concurrent requests) + results comparing vLLM vs TGI |
| `docs/` | Architecture diagram, benchmark results, lessons learned |

## Workflow

1. `cd terraform && terraform apply` — stand up the GPU instance
2. `quantization/quantize_awq.py` — quantize the base model, record VRAM before/after
3. `serving/run_vllm.sh` — serve the quantized model via vLLM's OpenAI-compatible API
4. `serving/run_tgi.sh` — serve the same model via TGI
5. `benchmark/load_test.py` — hit both endpoints under concurrent load, log p50/p95 latency + throughput
6. Fill in `docs/benchmark_results.md` with your actual numbers
7. `terraform destroy` when done (spot GPU instances are billed by the second — don't forget this step)

## Model used

Default: `mistralai/Mistral-7B-Instruct-v0.2` (swap in `quantization/config.py`). FP16 needs ~14GB VRAM; AWQ 4-bit needs ~5GB — this is the gap we're measuring.

## Results

- Quantized Mistral-7B-Instruct-v0.2 with AWQ 4-bit: **14GB → 3.9GB (72% size reduction)**
- Served via vLLM on a single A40 GPU, achieving **139 tokens/sec** at concurrency=10
- p95 latency held at 7.15s even under concurrent load

Quantized model published: https://huggingface.co/atulkrs/mistral-7b-awq

## Lessons Learned

Deploying vLLM hit a string of real-world infra issues worth documenting:
- AWS spot quota defaults to 0 for GPU instances on new accounts — required a Service Quotas request
- Driver/CUDA version mismatches between pip-installed PyTorch and the host GPU driver caused repeated import failures
- Resolved by switching from a manual pip install to the official `vllm/vllm-openai` Docker image, which ships a pre-tested, matched dependency set
- AWQ quantization requires `--dtype float16` explicitly; `--dtype auto` can resolve to an unsupported dtype
## Stack

AWS EC2 (g5.xlarge, spot) · Terraform · vLLM · Hugging Face TGI · AutoAWQ · AutoGPTQ · FastAPI · CloudWatch
