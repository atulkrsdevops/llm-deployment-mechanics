# Benchmark Results

Fill this in after running the quantization scripts and load tests. This is the table that does the talking in interviews.

## Quantization: VRAM and size reduction

| Method | FP16 VRAM | Quantized VRAM | Reduction | Quantization Time |
|---|---|---|---|---|
| AWQ 4-bit  | 14 GB | 3.9 GB | 72.1 % | _ s |
| GPTQ 4-bit | _ GB | _ GB | _ % | _ s |

*Source: `quantization/awq_results.json`, `quantization/gptq_results.json`*

## Serving: vLLM vs TGI throughput and latency

Same quantized model, same g5.xlarge instance, same load pattern.

| Engine | Concurrency | Req/sec | Tokens/sec | p50 latency | p95 latency | p99 latency |
|---|---|---|---|---|---|---|
| vLLM | 10 | 1.39 | 139.43 | 6.99s | 7.15s | 7.19s |
| TGI  | 10 | _ | _ | _ s | _ s | _ s |

*Source: `benchmark/results_vllm.json`, `benchmark/results_tgi.json`*

## Cost-per-1k-tokens estimate

g5.xlarge spot price at time of test: $___/hr

| Engine | Tokens/sec | Tokens/hr | Cost per 1k tokens |
|---|---|---|---|
| vLLM | _ | _ | $___ |
| TGI  | _ | _ | $___ |

## Observations

- Which engine handled concurrency better, and why (PagedAttention vs TGI's continuous batching)?
- Did quantization meaningfully hurt output quality? (Spot-check a few generations manually)
- What broke, and how did you fix it? (driver mismatches, OOM at high concurrency, etc. — this is often the most interesting interview content)

## Architecture

(Add a simple diagram here: client → FastAPI gateway → vLLM/TGI → quantized model on GPU)
