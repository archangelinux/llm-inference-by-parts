#continuous vs static batching under a staggered workload - the scheduler's payoff, measured

"""Per-request completion time: Engine (continuous) vs sequential generate_batch groups (static).

- same 8 requests both ways, eos_id="." so finishes stagger naturally (2..50 tokens)
- static: FIFO groups of N_SLOTS; a group runs until its LAST member finishes, and every
  request in it completes only when the group returns; later groups wait their full turn
- continuous: Engine with N_SLOTS slots; a finished request completes the step it finishes,
  and its slot is refilled from the queue immediately
- metric is completion time per request (start -> done), NOT total throughput: continuous
  batching wins on waiting, not on raw tok/s (expect total makespan roughly similar)

"""

import time
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig
from engine.model import GPT
from engine.scheduler import Engine, Request

N_SLOTS = 3
N_NEW = 50
EOS = 13  # "."
N_RUNS = 3

RESULTS_FILE = Path(__file__).parent / "continuous_results.jsonl"

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()

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
all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]

def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()

def static_run():
    """FIFO groups of N_SLOTS through generate_batch; completion = when your group returns."""
    done_at = [0.0] * len(prompts)
    sync()
    t0 = time.perf_counter()
    for g in range(0, len(prompts), N_SLOTS):
        group = list(range(g, min(g + N_SLOTS, len(prompts))))
        model.generate_batch([all_ids[i] for i in group], max_new_tokens=N_NEW, eos_id=EOS)
        sync()
        now = time.perf_counter() - t0  # everyone in the group completes together
        for i in group:
            done_at[i] = now
    return done_at

def continuous_run():
    """all submitted up front; completion = the step a request's done flag flips."""
    engine = Engine(model=model, n_slots=N_SLOTS, max_len=512)
    reqs = [Request(prompt_ids=ids, max_new_tokens=N_NEW, eos_id=EOS) for ids in all_ids]
    for r in reqs:
        engine.submit(r)
    done_at = [None] * len(prompts)
    sync()
    t0 = time.perf_counter()
    while engine.waiting or any(r is not None for r in engine.running):
        engine.step()
        sync()
        now = time.perf_counter() - t0
        for i, r in enumerate(reqs):
            if r.done and done_at[i] is None:
                done_at[i] = now
    return done_at

if __name__ == "__main__":
    static_run(); continuous_run()  # warmup both paths (mps compiles per shape)

    stat = [static_run() for _ in range(N_RUNS)]
    cont = [continuous_run() for _ in range(N_RUNS)]
    stat_med = [sorted(r[i] for r in stat)[N_RUNS // 2] for i in range(len(prompts))]
    cont_med = [sorted(r[i] for r in cont)[N_RUNS // 2] for i in range(len(prompts))]

    print(f"\n{'prompt':44s} {'static':>8s} {'contin':>8s}   completion time (s, median of {N_RUNS})")
    for i, p in enumerate(prompts):
        print(f"[{i}] {p[:40]!r:42s} {stat_med[i]:8.2f} {cont_med[i]:8.2f}")
    print(f"\n{'mean':46s} {sum(stat_med)/len(stat_med):8.2f} {sum(cont_med)/len(cont_med):8.2f}")
    print(f"{'makespan (last finisher)':46s} {max(stat_med):8.2f} {max(cont_med):8.2f}")

    import json
    with open(RESULTS_FILE, "w") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"device": DEVICE, "n_slots": N_SLOTS, "n_new": N_NEW,
                                "prompt": p[:40], "static_s": stat_med[i],
                                "continuous_s": cont_med[i]}) + "\n")
    print(f"[device: {DEVICE}] n_slots={N_SLOTS}, wrote {RESULTS_FILE}")
