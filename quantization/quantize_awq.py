"""
Quantize a base model to 4-bit using AWQ, and record VRAM/size before vs after.

Run on the GPU instance:
    python quantize_awq.py

Requires: pip install autoawq transformers accelerate
"""
import json
import time

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from config import BASE_MODEL, QUANT_OUTPUT_DIR_AWQ, CALIBRATION_SAMPLES


def get_gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9


def main():
    print(f"Loading base model: {BASE_MODEL}")
    t0 = time.time()
    model = AutoAWQForCausalLM.from_pretrained(BASE_MODEL, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    load_time = time.time() - t0
    mem_before = get_gpu_mem_gb()
    print(f"FP16 model loaded in {load_time:.1f}s, using {mem_before:.2f} GB VRAM")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print(f"Quantizing with calibration samples={CALIBRATION_SAMPLES} ...")
    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    quant_time = time.time() - t0
    mem_after = get_gpu_mem_gb()

    model.save_quantized(QUANT_OUTPUT_DIR_AWQ)
    tokenizer.save_pretrained(QUANT_OUTPUT_DIR_AWQ)

    results = {
        "model": BASE_MODEL,
        "method": "AWQ 4-bit",
        "fp16_vram_gb": round(mem_before, 2),
        "quantized_vram_gb": round(mem_after, 2),
        "vram_reduction_pct": round((1 - mem_after / mem_before) * 100, 1),
        "quantization_time_sec": round(quant_time, 1),
        "output_dir": QUANT_OUTPUT_DIR_AWQ,
    }

    print(json.dumps(results, indent=2))
    with open("awq_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Quantized model saved to {QUANT_OUTPUT_DIR_AWQ}")
    print("Copy these numbers into docs/benchmark_results.md")


if __name__ == "__main__":
    main()
