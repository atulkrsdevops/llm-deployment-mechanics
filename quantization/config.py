"""Shared config for quantization scripts."""

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
QUANT_OUTPUT_DIR_AWQ = "./models/mistral-7b-awq"
QUANT_OUTPUT_DIR_GPTQ = "./models/mistral-7b-gptq"

# Calibration data for quantization - small representative sample is enough
CALIBRATION_DATASET = "wikitext"
CALIBRATION_SAMPLES = 128
