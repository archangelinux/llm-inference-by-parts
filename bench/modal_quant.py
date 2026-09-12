#bench on an A10G: fp32 / fp16 / int8-torch / int8-kernel
    #per variant: b=1 decode tok/s, weight bytes per step, % of the 600 GB/s
    #bandwidth floor achieved end to end
    #plus a per-op microbench of one c_attn-shaped matmul, where the kernel's
    #own bandwidth shows without the ~5ms/step python dispatch overhead

# usage: modal run bench/modal_quant.py; writes bench/modal_quant_results.json
import json
from pathlib import Path

import modal

app = modal.App("llm-inference-quant-bench")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "transformers")
         .add_local_dir("engine", remote_path="/root/engine")
         .add_local_dir("kernels", remote_path="/root/kernels"))

A10G_BANDWIDTH_GB_S = 600
N_NEW = 50
N_RUNS = 5


@app.function(gpu="A10G", image=image, timeout=1800)
def bench():
    import statistics
    import time

    import torch
    import torch.nn.functional as F
    from engine.config import DEVICE, GPTConfig, sync
    from engine.model import GPT
    from engine.quant import QuantLinear, quantize_model

    assert DEVICE == "cuda"
    results = {"device": torch.cuda.get_device_name(0), "n_new": N_NEW, "n_runs": N_RUNS}

    def timed(fn, tokens):
        fn()  # warmup
        runs = []
        for _ in range(N_RUNS):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            runs.append(tokens / (time.perf_counter() - t0))
        return statistics.median(runs)

    def weight_bytes(model):
        #everything a decode step must read: remaining fp params + int8 q + scales
        b = sum(p.numel() * p.element_size() for p in model.parameters())
        for m in model.modules():
            if isinstance(m, QuantLinear):
                b += m.q.numel() + m.scale.numel() * m.scale.element_size()
        return b

    def build(name):
        model = GPT.from_pretrained(GPTConfig()).to(DEVICE, torch.float32 if name == "fp32" else torch.float16).eval()
        if name.startswith("int8"):
            model = quantize_model(model)
            for m in model.modules():
                if isinstance(m, QuantLinear):
                    m.use_kernel = name == "int8-kernel"
        return model

    ids = torch.full((1, 16), 464, device=DEVICE)
    results["variants"] = {}
    for name in ["fp32", "fp16", "int8-torch", "int8-kernel"]:
        model = build(name)
        tok_s = timed(lambda: model.generate(ids, N_NEW), N_NEW)
        wb = weight_bytes(model)
        ms = 1e3 / tok_s
        floor_ms = wb / (A10G_BANDWIDTH_GB_S * 1e9) * 1e3
        results["variants"][name] = {
            "decode_tok_s": round(tok_s, 1),
            "ms_per_token": round(ms, 3),
            "weight_gb_per_step": round(wb / 1e9, 3),
            "floor_ms": round(floor_ms, 3),
            "pct_of_bandwidth": round(100 * floor_ms / ms, 1),
        }
        print(f"{name:12s} {tok_s:7.1f} tok/s  {ms:.2f} ms/tok  reads {wb/1e9:.3f} GB  "
              f"floor {floor_ms:.2f} ms  -> {100*floor_ms/ms:.1f}% of bandwidth")
        del model
        torch.cuda.empty_cache()

    #per-op microbench: c_attn-shaped ops (2304x768), M=1 decode shape.
    #each approach cycles through N_COPIES separate weight tensors inside the
    #captured graph -- replaying one weight 100x would serve it from the 6MB L2
    #cache and report impossible >100%-of-DRAM numbers; real decode cycles 48
    #matrices per token and gets no such reuse
    N_COPIES = 8  # 8 x 3.5MB fp16 (or 1.77MB int8) copies > 6MB L2
    model = build("int8-kernel")
    layer = model.transformer.h[0].attn.c_attn
    qs = [layer.q.clone() for _ in range(N_COPIES)]
    scales = [layer.scale.clone() for _ in range(N_COPIES)]
    w16s = [(q.to(torch.float16) * sc) for q, sc in zip(qs, scales)]
    x = torch.randn(1, 1, 768, device=DEVICE, dtype=torch.float16)

    def op_time(fn, reps=100, iters=30):
        #cuda-graph replay: capture `reps` launches once, time replays. a plain
        #python loop times the ~20us/launch cpu dispatch, not the kernel (which
        #is ~5us); replay runs the batch gpu-side with no per-launch cpu cost
        for j in range(10):
            fn(j % N_COPIES)
        sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for j in range(reps):
                fn(j % N_COPIES)  #cycle weight copies so reads hit DRAM, not L2
        g.replay()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        sync()
        return (time.perf_counter() - t0) / iters / reps

    N, K = layer.q.shape
    import triton

    from kernels.dequant_matmul import dequant_matmul_kernel
    s_flats = [sc.squeeze(1) for sc in scales]
    x2 = x.reshape(1, K)
    kout = torch.empty((1, N), device=DEVICE, dtype=torch.float16)
    def kernel_op(i):
        q = qs[i]
        grid = (triton.cdiv(1, 16), triton.cdiv(N, 32))
        dequant_matmul_kernel[grid](x2, q, s_flats[i], kout,
                                    x2.stride(0), x2.stride(1),
                                    q.stride(1), q.stride(0),
                                    kout.stride(0), kout.stride(1),
                                    1, N, K, BLOCK_M=16, BLOCK_N=32, BLOCK_K=256,
                                    num_warps=4)
    ops = {
        #bytes = what the op reads+writes beyond the tiny x/out (dominated by weights)
        "fp16_linear": (lambda i: F.linear(x, w16s[i], layer.bias), 2 * N * K),
        "int8_slow": (lambda i: F.linear(x, qs[i].to(x.dtype) * scales[i], layer.bias), N * K + 2 * N * K * 2),
        "int8_kernel": (kernel_op, N * K),
    }
    results["c_attn_op"] = {}
    for name, (fn, nbytes) in ops.items():
        t = op_time(fn)
        gbs = nbytes / t / 1e9
        results["c_attn_op"][name] = {
            "us": round(t * 1e6, 1),
            "bytes_moved": nbytes,
            "achieved_gb_s": round(gbs, 1),
            "pct_of_bandwidth": round(100 * gbs / A10G_BANDWIDTH_GB_S, 1),
        }
        print(f"c_attn op {name:12s} {t*1e6:7.1f} us  moves {nbytes/1e6:.2f} MB  "
              f"-> {gbs:.0f} GB/s ({100*gbs/A10G_BANDWIDTH_GB_S:.1f}% of peak)")
    return results


@app.local_entrypoint()
def main():
    results = bench.remote()
    out = Path(__file__).parent / "modal_quant_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
