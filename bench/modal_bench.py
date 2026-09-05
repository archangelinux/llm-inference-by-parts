# benchmarks on NVIDIA A10G via Modal
    #adds the bandwidth sanity check
    #per-config warmup, N_RUNS=5, medians, one config at a time

#naive vs cached: prompt len {16,128,512}, 50 new tokens greedy , b=1
#batch sweep: b in {1..128}, prompt 16, 50 new tokens 
#bandwidth check   measured b=1 decode ms/token vs the theoretical floor:
                    #every decode step must stream all ~0.5GB of fp32 weights, and
                    #an A10G moves ~600 GB/s -> floor ~0.83 ms/token. How many
                    #multiples of the floor we actually take = how much overhead
                    #(kernel launches, Python dispatch) sits on top of physics.


# usage: modal run bench/modal_bench.py; writes bench/modal_results.json locally
import json
from pathlib import Path

import modal

app = modal.App("llm-inference-bench")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers")
         .add_local_dir("engine", remote_path="/root/engine"))

A10G_BANDWIDTH_GB_S = 600  # spec sheet: 600 GB/s GDDR6

PROMPT_LENS = [16, 128, 512]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
N_NEW = 50
N_RUNS = 5

@app.function(gpu="A10G", image=image, timeout=1800)
def bench():
    import statistics
    import time
    import torch
    from engine.config import DEVICE, GPTConfig, sync
    from engine.model import GPT

    assert DEVICE == "cuda", f"expected cuda, got {DEVICE}"
    model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()
    results = {"device": torch.cuda.get_device_name(0), "n_new": N_NEW, "n_runs": N_RUNS}

    def timed(fn, tokens):
        #median tok/s over N_RUNS, after one warmup call"""
        fn()
        runs = []
        for _ in range(N_RUNS):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            runs.append(tokens / (time.perf_counter() - t0))
        return round(statistics.median(runs), 1)

    @torch.no_grad()
    def naive(ids, n_new):
        for _ in range(n_new):
            logits, _ = model(ids)
            ids = torch.cat([ids, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)

    # naive vs cached, one config at a time
    results["naive"], results["cached"] = {}, {}
    for L in PROMPT_LENS:
        ids = torch.full((1, L), 464, device=DEVICE)
        results["naive"][L] = timed(lambda: naive(ids, N_NEW), N_NEW)
        results["cached"][L] = timed(lambda: model.generate(ids, N_NEW), N_NEW)
        print(f"prompt {L:4d}: naive {results['naive'][L]:8.1f}  cached {results['cached'][L]:8.1f} tok/s")

    # batch sweep
    prompt = torch.full((1, 16), 464, device=DEVICE)
    results["batched"] = {}
    for b in BATCH_SIZES:
        all_ids = [prompt] * b
        results["batched"][b] = timed(lambda: model.generate_batch(all_ids, N_NEW), b * N_NEW)
        print(f"b={b:4d}: {results['batched'][b]:10.1f} tok/s total "
              f"({results['batched'][b] / b:7.1f} per row)")

    # continuous vs static under a staggered workload (mirrors bench/continuous.py:
    # same 8 requests, eos="." so finishes stagger; metric = per-request completion time)
    from transformers import GPT2Tokenizer

    from engine.scheduler import Engine, Request
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    prompts = ["Hello", "The meaning of life is",
               "In 2019, OpenAI released a language model that", "def fibonacci(n):",
               "1, 1, 2, 3, 5, 8, 13,",
               "Q: What is the capital of France?\nA: Paris.\nQ: What is the capital of Japan?\nA:",
               "The quick brown fox jumps over the lazy dog. The quick brown fox",
               "Alice gave the book to Bob because he had asked her politely. Later that afternoon, Bob returned it to"]
    all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]
    N_SLOTS, EOS = 3, 13

    def static_run():
        done_at, t0 = [0.0] * len(prompts), time.perf_counter()
        for g in range(0, len(prompts), N_SLOTS):
            group = list(range(g, min(g + N_SLOTS, len(prompts))))
            model.generate_batch([all_ids[i] for i in group], max_new_tokens=N_NEW, eos_id=EOS)
            sync()
            for i in group:
                done_at[i] = time.perf_counter() - t0
        return done_at

    def continuous_run():
        engine = Engine(model=model, n_slots=N_SLOTS, max_len=512)
        reqs = [Request(prompt_ids=ids, max_new_tokens=N_NEW, eos_id=EOS) for ids in all_ids]
        for r in reqs:
            engine.submit(r)
        done_at, t0 = [None] * len(prompts), time.perf_counter()
        while engine.waiting or any(r is not None for r in engine.running):
            engine.step()
            sync()
            now = time.perf_counter() - t0
            for i, r in enumerate(reqs):
                if r.done and done_at[i] is None:
                    done_at[i] = now
        return done_at

    static_run(); continuous_run()  # warmup
    stat = [static_run() for _ in range(N_RUNS)]
    cont = [continuous_run() for _ in range(N_RUNS)]
    med = lambda runs, i: sorted(r[i] for r in runs)[N_RUNS // 2]
    stat_med = [med(stat, i) for i in range(len(prompts))]
    cont_med = [med(cont, i) for i in range(len(prompts))]
    results["continuous_vs_static"] = {
        "static_mean_s": round(sum(stat_med) / len(stat_med), 3),
        "continuous_mean_s": round(sum(cont_med) / len(cont_med), 3),
        "static_makespan_s": round(max(stat_med), 3),
        "continuous_makespan_s": round(max(cont_med), 3),
    }
    print(f"continuous vs static: mean {results['continuous_vs_static']['continuous_mean_s']}s "
          f"vs {results['continuous_vs_static']['static_mean_s']}s")

    # bandwidth sanity check (kipply-style): decode step floor = weight bytes / bandwidth
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    floor_ms = weight_bytes / (A10G_BANDWIDTH_GB_S * 1e9) * 1e3
    measured_ms = 1e3 / results["cached"][16]  # b=1, short-context decode
    results["bandwidth_check"] = {
        "weight_gb": round(weight_bytes / 1e9, 3),
        "floor_ms_per_token": round(floor_ms, 3),
        "measured_ms_per_token": round(measured_ms, 3),
        "multiple_of_floor": round(measured_ms / floor_ms, 1),
    }
    print(f"bandwidth check: floor {floor_ms:.2f} ms/tok, measured {measured_ms:.2f} "
          f"({measured_ms / floor_ms:.1f}x the physics)")
    return results


@app.local_entrypoint()
def main():
    results = bench.remote()
    out = Path(__file__).parent / "modal_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
