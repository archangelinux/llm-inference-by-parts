#how many tokens per second can my model generate, and how does that degrade as the prompt gets longer

"""Decode throughput, single stream (batch 1), greedy.

- results.jsonl holds one set of rows (16/128/512) per mechanism, grouped in PATHS order
- a run re-times only the mechanisms you name (default: all) and replaces just their rows; every other mechanism's rows stay pinned
- for accurate numbers, time one mechanism per invocation: a heavy path (like naive at 512) heats and throttles the GPU, contaminating whatever is timed after it in the same run

Usage:
  python bench/run.py            # re-time every mechanism, with a COOLDOWN_S pause between them
  python bench/run.py cached     # re-time only "cached" on a fresh GPU, keep other rows (most accurate)
"""

import json
import sys
import time
from pathlib import Path

import torch

from engine.config import DEVICE, GPTConfig
from engine.model import GPT

PROMPT_LENGTHS = [16, 128, 512] #32 fold range in input size
#actual work grows faster than 32x
#MLP and projection layers are proportional to t (each token processed independently) bu attention compares every token against every earlier token so t^2
N_NEW = 50 #new tokens per measurement, long enough to average out per-step jitter
N_RUNS = 3 #repeats just long enough to spot variance
COOLDOWN_S = 60 #pause between mechanisms so a heavy one doesn't heat/throttle the GPU for the next

RESULTS_FILE = Path(__file__).parent / "results.jsonl"

model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval() #load the model once at module level, not timed; eval() is no op without batchnorm and dropout etc. since not training

def sync():  #make sure GPU work is actually finished before reading the clock
    if DEVICE == "mps":
        torch.mps.synchronize()

@torch.no_grad()
def naive_generate(ids, n_new):
    #pinned baseline: the pre-KV-cache greedy loop
    for _ in range(n_new):
        logits, _ = model(ids) #no cache passed
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids

def cached_generate(ids, n_new):
    return model.generate(ids, n_new) #greedy (do_sample=False), preallocated KV cache

# name -> generation fn
PATHS = {
    "naive": naive_generate,
    "cached": cached_generate,
}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(PATHS) #which mechanisms to (re)time this invocation; default all
    unknown = [n for n in chosen if n not in PATHS]
    if unknown:
        sys.exit(f"unknown mechanism(s) {unknown}; choices: {list(PATHS)}")

    # speed doesn't depend on prompt content, only length -> just repeat one token here
    prompt_ids = torch.full((1, max(PROMPT_LENGTHS)), 464, device=DEVICE)  # 464 = " The"

    for name in chosen:
        PATHS[name](prompt_ids[:, :16], 5)  # warmup (first calls pay one-time compile/alloc cost)

    new_rows = {} #name -> its 3 records (one per prompt length)
    for j, name in enumerate(chosen): #one mechanism at a time -> records group as sets of 16/128/512
        if j > 0: #only between mechanisms, not before the first
            print(f"cooling down {COOLDOWN_S}s before {name}...")
            time.sleep(COOLDOWN_S)
        gen = PATHS[name]
        new_rows[name] = []
        for L in PROMPT_LENGTHS:
            runs = []
            for _ in range(N_RUNS):
                sync() #drain any leftover GPU work
                t0 = time.perf_counter() #start clock; perf_counter is monotonic and high res
                gen(prompt_ids[:, :L], N_NEW)
                sync() #wait until all N_NEW tokens actually finished
                runs.append(round(N_NEW / (time.perf_counter() - t0), 2))
            #one flat record per (label, prompt_len): easy to filter/plot later
            new_rows[name].append({"device": DEVICE, "n_new": N_NEW, "label": name,
                                   "prompt_len": L, "tok_per_sec": runs})
            print(f"{name:6s}  prompt_len={L:4d}  tok/s: {runs}") #show runs individually, notice short workloads are noisy

    #merge: rows of mechanisms not re-timed this invocation are carried over untouched
    old_rows = {}
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            r = json.loads(line)
            old_rows.setdefault(r["label"], []).append(r)

    with open(RESULTS_FILE, "w") as f:
        for name in PATHS: #write in PATHS order so each mechanism's set stays grouped
            for r in new_rows.get(name, old_rows.get(name, [])):
                f.write(json.dumps(r) + "\n")
    print(f"[device: {DEVICE}] updated {', '.join(chosen)} in {RESULTS_FILE}")
