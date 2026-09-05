#the throughput/latency tradeoff of static batching

"""Decode throughput + per-request latency vs batch size, greedy, fixed-length prompts

- one row per batch size in batch_results.jsonl; a rerun replaces the file
- same prompt repeated b times: speed depends on lengths only, not content
- eos_id=None so every row runs all N_NEW steps -> constant work per run
- expect: throughput climbs steeply then flattens (hardware saturating); latency rises monotonically; ratio between them is what batching "costs"

"""

import json
import time
from pathlib import Path

import torch

from engine.config import DEVICE, GPTConfig, sync
from engine.model import GPT

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
PROMPT_LEN = 16
N_NEW = 50
N_RUNS = 5 #median matters more than mean on a fanless machine?

RESULTS_FILE = Path(__file__).parent / "batch_results.jsonl"

model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()


if __name__ == "__main__":
    prompt = torch.full((1, PROMPT_LEN), 464, device=DEVICE)  # 464 = " The"

    rows = []
    for b in BATCH_SIZES:
        all_ids = [prompt] * b
        model.generate_batch(all_ids, max_new_tokens=5)  # warmup per size: mps compiles kernels per shape
        runs = []  # (throughput tok/s, latency s) per repeat
        for _ in range(N_RUNS):
            sync()
            t0 = time.perf_counter()
            model.generate_batch(all_ids, max_new_tokens=N_NEW)
            sync()
            wall = time.perf_counter() - t0
            runs.append((round(b * N_NEW / wall, 2), round(wall, 3)))
        tput = [r[0] for r in runs]
        lat = [r[1] for r in runs]
        rows.append({"device": DEVICE, "prompt_len": PROMPT_LEN, "n_new": N_NEW,
                     "batch_size": b, "tok_per_sec": tput, "latency_s": lat})
        print(f"b={b:3d}  tok/s: {tput}  latency(s): {lat}")

    with open(RESULTS_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[device: {DEVICE}] wrote {RESULTS_FILE}")
