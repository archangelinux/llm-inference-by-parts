"""end-to-end correctness gate for the serving layer: 20 concurrent HTTP requests,
each streamed over SSE, must every one produce exactly its solo greedy text --
no interleaving, no cross-contamination, no drops.

Spawns the real server (uvicorn subprocess) so the whole stack is under test:
HTTP -> handler -> inbox -> engine loop -> scheduler -> outbox -> SSE.
Slow (~1min: two model loads + 20 requests through 4 slots). Run with pytest or plain python."""
import asyncio
import json
import subprocess
import sys
import time

import httpx
import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig, DTYPE
from engine.model import GPT

PORT = 8400  # not 8000: don't collide with a dev server that's already running
BASE = f"http://127.0.0.1:{PORT}"
N_CONCURRENT = 20
N_NEW = 25  # keep the run short; concurrency is what's under test, not length

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()

prompts = [
    "Hello",
    "The meaning of life is",
    "In 2019, OpenAI released a language model that",
    "def fibonacci(n):",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the capital of France?\nA: Paris.\nQ: What is the capital of Japan?\nA:",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    ("Alice gave the book to Bob because he had asked her politely. "
     "Later that afternoon, Bob returned it to"),
]

_solo_cache = {}

def solo_text(prompt):
    """the oracle: solo greedy continuation, decoded (server streams decoded tokens)"""
    if prompt not in _solo_cache:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out = model.generate(ids, max_new_tokens=N_NEW)
        _solo_cache[prompt] = tok.decode(out[0, -N_NEW:])
    return _solo_cache[prompt]


async def one_request(client, prompt):
    """what a real client does: POST, read the SSE stream, join the tokens"""
    text = ""
    async with client.stream("POST", f"{BASE}/generate",
                             json={"prompt": prompt, "max_new_tokens": N_NEW}) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                text += ev["token"]
                if ev["done"]:
                    break
    return text


async def fire_all():
    async with httpx.AsyncClient(timeout=120) as client:
        # 20 requests = the 8 prompts cycled; all launched in the same instant
        jobs = [one_request(client, prompts[i % len(prompts)]) for i in range(N_CONCURRENT)]
        return await asyncio.gather(*jobs)


def test_concurrent_requests():
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90  # model load takes a while
        while time.time() < deadline:
            try:
                httpx.get(BASE + "/", timeout=2)
                break
            except httpx.TransportError:
                time.sleep(1)
        else:
            raise RuntimeError("server never came up")

        results = asyncio.run(fire_all())

        for i, text in enumerate(results):
            p = prompts[i % len(prompts)]
            expected = solo_text(p)
            print(f"[{i:2d}] {p[:35]!r:37s} {'ok' if text == expected else 'MISMATCH'}")
            assert text == expected, (
                f"request {i} ({p[:40]!r}) diverged:\n"
                f"  served:   {text!r}\n"
                f"  expected: {expected!r}"
            )
        print(f"{N_CONCURRENT} concurrent requests: all match solo")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    test_concurrent_requests()
