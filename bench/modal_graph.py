#cuda-graph decode on an A10G: engine decode throughput, eager vs graphed
    #both models x {fp32, fp16, int8-torch, int8-kernel} x {1, 4} slots; 50 new tokens per request
    #eager = the decode forward launched op by op from python (~200-1000 launches/token)
    #graphed = the same forward captured once and replayed per step

# usage: modal run bench/modal_graph.py; writes bench/modal_graph_results.json
import json
from pathlib import Path

import modal

app = modal.App("llm-inference-graph-bench")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers")
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels"))

N_NEW = 50
N_RUNS = 5


@app.function(gpu="A10G", image=image, timeout=1800)
def bench():
    import statistics
    import time

    import torch
    from transformers import AutoTokenizer
    from engine.config import DEVICE, GPTConfig, QwenConfig, sync
    from engine.load import HF_ID
    from engine.quant import QuantLinear, quantize_model
    from engine.scheduler import Engine, Request

    def build(name, variant):
        dtype = torch.float32 if variant == "fp32" else torch.float16
        if name == "qwen":
            from engine.qwen import Qwen
            model = Qwen.from_pretrained(QwenConfig()).to(DEVICE, dtype).eval()
        else:
            from engine.model import GPT
            model = GPT.from_pretrained(GPTConfig()).to(DEVICE, dtype).eval()
        if variant.startswith("int8"):
            model = quantize_model(model)
            for m in model.modules():
                if isinstance(m, QuantLinear):
                    m.use_kernel = variant == "int8-kernel"  #int8-torch: dequantize-then-matmul, the slow reference path
        return model

    results = {"device": torch.cuda.get_device_name(0), "n_new": N_NEW, "n_runs": N_RUNS, "runs": []}
    prompt_ids = torch.full((1, 16), 464, device=DEVICE)

    def decode_time(model, n_slots, use_graph):
        #one engine per measurement; first run() captures (if graphed) and warms; then time N_RUNS batches
        e = Engine(model=model, n_slots=n_slots, max_len=128, use_graph=use_graph)
        def batch():
            for _ in range(n_slots):
                e.submit(Request(prompt_ids=prompt_ids, max_new_tokens=N_NEW))
            sync(); t0 = time.perf_counter(); e.run(); sync()
            return time.perf_counter() - t0
        batch()  # warmup / capture
        return statistics.median(batch() for _ in range(N_RUNS))

    for name in ["gpt2", "qwen"]:
        for variant in ["fp32", "fp16", "int8-torch", "int8-kernel"]:
            model = build(name, variant)
            for n_slots in [1, 4]:
                row = {"model": name, "variant": variant, "n_slots": n_slots}
                for use_graph in [False, True]:
                    t = decode_time(model, n_slots, use_graph)
                    key = "graphed" if use_graph else "eager"
                    row[key + "_tok_s"] = round(n_slots * N_NEW / t, 1)
                    row[key + "_ms_per_step"] = round(1e3 * t / N_NEW, 3)
                row["speedup"] = round(row["eager_ms_per_step"] / row["graphed_ms_per_step"], 2)
                results["runs"].append(row)
                print(f"{name:5s} {variant:12s} slots={n_slots}  eager {row['eager_ms_per_step']:6.2f} ms/step  "
                      f"graphed {row['graphed_ms_per_step']:6.2f} ms/step  {row['speedup']:.2f}x  "
                      f"({row['eager_tok_s']:.0f} -> {row['graphed_tok_s']:.0f} tok/s)")
            del model
            torch.cuda.empty_cache()
    return results


@app.local_entrypoint()
def main():
    results = bench.remote()
    out = Path(__file__).parent / "modal_graph_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
