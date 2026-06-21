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

## Results (fill in after running)

See `docs/benchmark_results.md` for throughput/latency tables and the cost-per-1k-tokens comparison.

## Stack

AWS EC2 (g5.xlarge, spot) · Terraform · vLLM · Hugging Face TGI · AutoAWQ · AutoGPTQ · FastAPI · CloudWatch
