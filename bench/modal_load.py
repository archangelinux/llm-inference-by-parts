#serving under load on a NVIDIA A10G: the same bench/load.py protocol run in a Modal container

"""Spawns the server (uvicorn, server.app) inside the container, fires the
same concurrency levels {1,4,8,16} at it over loopback, reports req/s, delivered
tok/s, TTFT, p50/p95. Loopback = no real network between client and server, so
this measures the serving stack + scheduler utilization on fast hardware, not
internet latency."""

#usage: modal run bench/modal_load.py  writes bench/modal_load_results.json locally


import json
from pathlib import Path

import modal

app = modal.App("llm-inference-load")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers", "fastapi", "uvicorn", "httpx")
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("server", remote_path="/root/server"))

LEVELS = [1, 4, 8, 16]
WAVES = 3
N_NEW = 25

PROMPTS = [
    "Hello",
    "The meaning of life is",
    "In 2019, OpenAI released a language model that",
    "def fibonacci(n):",
    "1, 1, 2, 3, 5, 8, 13,",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]


@app.function(gpu="A10G", image=image, timeout=1800)
def load_bench():
    import asyncio
    import statistics
    import subprocess
    import sys
    import time

    import httpx

    BASE = "http://127.0.0.1:8000"

    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app",
                               "--port", "8000"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            httpx.get(BASE + "/", timeout=2)
            break
        except httpx.TransportError:
            time.sleep(1)
    else:
        raise RuntimeError("server never came up")

    async def one_request(client, prompt):
        t0 = time.perf_counter()
        ttft = None
        async with client.stream("POST", f"{BASE}/generate",
                                 json={"prompt": prompt, "max_new_tokens": N_NEW}) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    if json.loads(line[6:])["done"]:
                        break
        return ttft, time.perf_counter() - t0

    async def bench_level(client, level):
        ttfts, lats, wave_rps = [], [], []
        for _ in range(WAVES):
            t0 = time.perf_counter()
            results = await asyncio.gather(*[
                one_request(client, PROMPTS[i % len(PROMPTS)]) for i in range(level)])
            wave_rps.append(level / (time.perf_counter() - t0))
            ttfts += [r[0] for r in results]
            lats += [r[1] for r in results]
        q = statistics.quantiles(lats, n=100)
        rps = statistics.mean(wave_rps)
        return {"concurrency": level, "req_per_s": round(rps, 2),
                "delivered_tok_s": round(rps * N_NEW, 1),
                "ttft_s": round(statistics.median(ttfts), 3),
                "p50_s": round(q[49], 3), "p95_s": round(q[94], 3)}

    async def main():
        async with httpx.AsyncClient(timeout=300) as client:
            await bench_level(client, 2)  # warmup
            rows = []
            for level in LEVELS:
                row = await bench_level(client, level)
                rows.append(row)
                print(f"c={row['concurrency']:3d}  req/s={row['req_per_s']:6.2f}  "
                      f"delivered={row['delivered_tok_s']:7.1f} tok/s  "
                      f"ttft={row['ttft_s']:6.3f}s  p50={row['p50_s']:6.3f}s  p95={row['p95_s']:6.3f}s")
            return rows

    try:
        return asyncio.run(main())
    finally:
        server.terminate()


@app.local_entrypoint()
def main():
    rows = load_bench.remote()
    out = Path(__file__).parent / "modal_load_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")
