#how many tokens per second can my model generate, and how does that degrade as the prompt gets longer

"""Baseline decode throughput: single stream (batch 1), greedy, no KV cache.
tok/s = new tokens / wall time. Every step re-runs the full sequence, so
longer prompts should be visibly slower — that's the point of the baseline."""
import json
import time
import torch
from engine.config import DEVICE, GPTConfig
from engine.model import GPT

PROMPT_LENGTHS = [16, 128, 512] #32 fold range in input size
#actual work grows faster than 32x
#MLP and projection layers are proportional to t (each token processed independently) bu attention compares every token against every earlier token so t^2
N_NEW = 50 #new tokens per measurement, long enough to average out per-step jitter
N_RUNS = 3 #repeats just long enough to spot variance 

model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval() #load the model once at module level, not timed; eval() is no op without batchnorm and dropout etc. since not training

def sync():  #make sure GPU work is actually finished before reading the clock
    if DEVICE == "mps":
        torch.mps.synchronize()

def generate(ids, n_new):
    with torch.inference_mode(): #skip building the auto graph from backprop
        for _ in range(n_new): #same greedy loop as in tests/test_logits.py
            next_id = model(ids)[:, -1, :].argmax(dim=-1, keepdim=True) #last position logit shape(1, 50257)
            ids = torch.cat([ids, next_id], dim=1)
    return ids

# speed doesn't depend on prompt content, only length -> just repeat one token here
prompt_ids = torch.full((1, max(PROMPT_LENGTHS)), 464, device=DEVICE)  # 464 = " The"

generate(prompt_ids[:, :16], 5)  # warmup (first calls pay one-time compile/alloc cost)

results = {"device": DEVICE, "n_new_tokens": N_NEW, "tok_per_sec": {}}
for L in PROMPT_LENGTHS:
    runs = []
    for _ in range(N_RUNS):
        sync() #drain any leftover GPU work
        t0 = time.perf_counter() #start clock; perf_counter is mnotonic and high res
        generate(prompt_ids[:, :L], N_NEW) #generate 50 tokens
        sync() #wait until all 50 finished
        runs.append(round(N_NEW / (time.perf_counter() - t0), 2)) #stop clock and record all 3 rates (short workloads end up being more noisy, so show runs individually instead of averageing out)
    results["tok_per_sec"][L] = runs
    print(f"prompt_len={L:4d}  tok/s: {runs}")

with open("bench/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"[device: {DEVICE}] saved bench/results.json")
