"""
Quantize the same base model to 4-bit using GPTQ, for a second comparison point
against AWQ. Different algorithm, similar goal - useful to be able to speak to
the tradeoffs between the two in an interview.

Run on the GPU instance:
    python quantize_gptq.py

Requires: pip install auto-gptq transformers accelerate optimum
"""
import json
import time

import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from config import BASE_MODEL, QUANT_OUTPUT_DIR_GPTQ, CALIBRATION_DATASET, CALIBRATION_SAMPLES


def get_gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / 1e9


def load_calibration_examples(tokenizer, n_samples: int):
    from datasets import load_dataset

    ds = load_dataset(CALIBRATION_DATASET, "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100][:n_samples]
    return [tokenizer(t, return_tensors="pt") for t in texts]


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"Loading base model: {BASE_MODEL}")
    t0 = time.time()

    quant_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(BASE_MODEL, quant_config)
    load_time = time.time() - t0
    mem_before = get_gpu_mem_gb()
    print(f"FP16 model loaded in {load_time:.1f}s, using {mem_before:.2f} GB VRAM")

    print(f"Preparing {CALIBRATION_SAMPLES} calibration examples ...")
    examples = load_calibration_examples(tokenizer, CALIBRATION_SAMPLES)

    print("Quantizing with GPTQ ...")
    t0 = time.time()
    model.quantize(examples)
    quant_time = time.time() - t0
    mem_after = get_gpu_mem_gb()

    model.save_quantized(QUANT_OUTPUT_DIR_GPTQ)
    tokenizer.save_pretrained(QUANT_OUTPUT_DIR_GPTQ)

    results = {
        "model": BASE_MODEL,
        "method": "GPTQ 4-bit",
        "fp16_vram_gb": round(mem_before, 2),
        "quantized_vram_gb": round(mem_after, 2),
        "vram_reduction_pct": round((1 - mem_after / mem_before) * 100, 1),
        "quantization_time_sec": round(quant_time, 1),
        "output_dir": QUANT_OUTPUT_DIR_GPTQ,
    }

    print(json.dumps(results, indent=2))
    with open("gptq_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Quantized model saved to {QUANT_OUTPUT_DIR_GPTQ}")
    print("Copy these numbers into docs/benchmark_results.md")


if __name__ == "__main__":
    main()
