"""
Concurrent load test against a running vLLM or TGI server. Measures
throughput (requests/sec, tokens/sec) and latency percentiles (p50/p95/p99).

Usage:
    python load_test.py --backend vllm --concurrency 10 --total-requests 100
    python load_test.py --backend tgi  --concurrency 10 --total-requests 100

Requires: pip install httpx numpy
"""
import argparse
import asyncio
import json
import time

import httpx
import numpy as np

PROMPTS = [
    "Explain the difference between quantization and pruning.",
    "What is PagedAttention and why does it matter for LLM serving?",
    "Write a short summary of how Kubernetes schedules pods.",
    "Describe the tradeoffs between AWQ and GPTQ quantization.",
    "What does GPU memory utilization mean in the context of vLLM?",
]


async def single_request(client: httpx.AsyncClient, backend: str, prompt: str, max_tokens: int):
    t0 = time.time()
    if backend == "vllm":
        resp = await client.post(
            "http://localhost:8000/v1/completions",
            json={"model": "mistral-7b-awq", "prompt": prompt, "max_tokens": max_tokens},
            timeout=60,
        )
        data = resp.json()
        tokens_out = data.get("usage", {}).get("completion_tokens", max_tokens)
    else:
        resp = await client.post(
            "http://localhost:8000/generate",
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}},
            timeout=60,
        )
        tokens_out = max_tokens  # TGI doesn't always return token counts in this field

    latency = time.time() - t0
    return latency, tokens_out


async def run_load_test(backend: str, concurrency: int, total_requests: int, max_tokens: int):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    tokens_total = 0

    async def worker(i: int):
        nonlocal tokens_total
        async with sem:
            async with httpx.AsyncClient() as client:
                prompt = PROMPTS[i % len(PROMPTS)]
                latency, tokens = await single_request(client, backend, prompt, max_tokens)
                latencies.append(latency)
                tokens_total += tokens

    t0 = time.time()
    await asyncio.gather(*[worker(i) for i in range(total_requests)])
    wall_time = time.time() - t0

    latencies_arr = np.array(latencies)
    results = {
        "backend": backend,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_time_sec": round(wall_time, 2),
        "requests_per_sec": round(total_requests / wall_time, 2),
        "tokens_per_sec": round(tokens_total / wall_time, 2),
        "latency_p50_sec": round(float(np.percentile(latencies_arr, 50)), 3),
        "latency_p95_sec": round(float(np.percentile(latencies_arr, 95)), 3),
        "latency_p99_sec": round(float(np.percentile(latencies_arr, 99)), 3),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "tgi"], required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--total-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    results = asyncio.run(
        run_load_test(args.backend, args.concurrency, args.total_requests, args.max_tokens)
    )
    print(json.dumps(results, indent=2))

    with open(f"results_{args.backend}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results_{args.backend}.json - copy into docs/benchmark_results.md")


if __name__ == "__main__":
    main()
