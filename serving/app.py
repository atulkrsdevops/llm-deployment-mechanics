"""
Thin FastAPI wrapper in front of vLLM or TGI, so you have a single consistent
API surface regardless of which serving engine is running underneath -
and a place to bolt on auth, logging, and CloudWatch metrics later.

Run:
    BACKEND=vllm uvicorn app:app --host 0.0.0.0 --port 9000
    BACKEND=tgi  uvicorn app:app --host 0.0.0.0 --port 9000
"""
import os
import time

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

BACKEND = os.environ.get("BACKEND", "vllm")  # "vllm" or "tgi"
BACKEND_URL = "http://localhost:8000"

app = FastAPI(title="LLM Deployment Mechanics - Inference Gateway")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.get("/health")
async def health():
    return {"backend": BACKEND, "status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        if BACKEND == "vllm":
            resp = await client.post(
                f"{BACKEND_URL}/v1/completions",
                json={
                    "model": "mistral-7b-awq",
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens,
                },
            )
            text = resp.json()["choices"][0]["text"]
        else:  # tgi
            resp = await client.post(
                f"{BACKEND_URL}/generate",
                json={
                    "inputs": req.prompt,
                    "parameters": {"max_new_tokens": req.max_tokens},
                },
            )
            text = resp.json()["generated_text"]

    latency_ms = round((time.time() - t0) * 1000, 1)
    return {"backend": BACKEND, "text": text, "latency_ms": latency_ms}
