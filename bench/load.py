#load test the serving layer: req/s, latency percentiles, time-to-first-token vs concurrency

"""Async load bench against the real server (spawned as a subprocess).

- per concurrency level: fire `level` simultaneous streaming requests, 3 waves,
  measure per-request TTFT (start -> first token) and latency (start -> done)
- req/s = completed requests / wave wall-clock, averaged over waves
- p99 with this few samples is really just max -- reported anyway for the format
- expect: TTFT grows with concurrency (queueing for slots + frontier);
  req/s grows past n_slots only modestly (excess requests just queue)

Usage:
  python bench/load.py
"""

import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

PORT = 8401
BASE = f"http://127.0.0.1:{PORT}"
LEVELS = [1, 4, 8, 16]
WAVES = 3
N_NEW = 25

RESULTS_FILE = Path(__file__).parent / "load_results.json"

PROMPTS = [
    "Hello",
    "The meaning of life is",
    "In 2019, OpenAI released a language model that",
    "def fibonacci(n):",
    "1, 1, 2, 3, 5, 8, 13,",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
]

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
    return ttft, time.perf_counter() - t0  # (ttft, latency)

async def bench_level(client, level):
    ttfts, lats, wave_rps = [], [], []
    for _ in range(WAVES):
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, PROMPTS[i % len(PROMPTS)]) for i in range(level)])
        wall = time.perf_counter() - t0
        wave_rps.append(level / wall)
        ttfts += [r[0] for r in results]
        lats += [r[1] for r in results]
    q = statistics.quantiles(lats, n=100)  # q[49]=p50, q[94]=p95, q[98]=p99
    return {"concurrency": level, "req_per_s": round(statistics.mean(wave_rps), 2),
            "ttft_s": round(statistics.median(ttfts), 3),
            "p50_s": round(q[49], 3), "p95_s": round(q[94], 3), "p99_s": round(q[98], 3)}

async def main():
    async with httpx.AsyncClient(timeout=300) as client:
        await bench_level(client, 2)  # warmup: shapes compile, caches allocate
        rows = []
        for level in LEVELS:
            row = await bench_level(client, level)
            rows.append(row)
            print(f"c={row['concurrency']:3d}  req/s={row['req_per_s']:6.2f}  "
                  f"ttft={row['ttft_s']:6.3f}s  p50={row['p50_s']:6.3f}s  "
                  f"p95={row['p95_s']:6.3f}s  p99={row['p99_s']:6.3f}s")
        return rows

if __name__ == "__main__":
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                httpx.get(BASE + "/", timeout=2)
                break
            except httpx.TransportError:
                time.sleep(1)
        else:
            sys.exit("server never came up")

        rows = asyncio.run(main())
        RESULTS_FILE.write_text(json.dumps(rows, indent=2))
        print(f"wrote {RESULTS_FILE}")
    finally:
        server.terminate()
        server.wait()
