#the clean instrument: capture 100 kernel launches in a cuda graph, time replays.
    #replay has no per-launch cpu cost, so this is pure gpu execution time --
    #the number the wall-clock and event sweeps couldn't see past the ~20us
    #python launch overhead
# usage: modal run kernels/dev.py --name sweep_graph

import time

import torch
import torch.nn.functional as F

from engine.quant import quantize_weight
from kernels.dequant_matmul import dequant_matmul_kernel
import triton

REPS = 100  # launches captured per graph


def graph_time(fn, iters=30):
    #capture REPS launches once, then time replays
    for _ in range(10):
        fn()  # warmup + let triton finish compiling before capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPS):
            fn()
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters / REPS * 1e6  # us per op


def main():
    assert torch.cuda.is_available()
    for (N, K) in [(2304, 768), (768, 3072)]:
        M = 1
        W = torch.randn((N, K), device="cuda", dtype=torch.float16)
        q, s = quantize_weight(W)
        s_flat = s.squeeze(1)
        x = torch.randn((M, K), device="cuda", dtype=torch.float16)
        out = torch.empty((M, N), device="cuda", dtype=torch.float16)
        nbytes = N * K

        t = graph_time(lambda: F.linear(x, W))
        print(f"({N}x{K}) fp16 F.linear : {t:6.1f} us  {2*nbytes/t/1e3:6.0f} GB/s (reads 2x bytes)")

        for BN, BK, warps in [(32, 64, 4), (32, 128, 8), (32, 256, 4), (64, 64, 4), (64, 256, 8)]:
            grid = (triton.cdiv(M, 16), triton.cdiv(N, BN))
            def run():
                dequant_matmul_kernel[grid](x, q, s_flat, out,
                                            x.stride(0), x.stride(1),
                                            q.stride(1), q.stride(0),
                                            out.stride(0), out.stride(1),
                                            M, N, K, BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK,
                                            num_warps=warps)
            t = graph_time(run)
            print(f"({N}x{K}) BN={BN:4d} BK={BK:4d} w={warps}: {t:6.1f} us  {nbytes/t/1e3:6.0f} GB/s "
                  f"({100*nbytes/t/1e3/600:.0f}% of peak)")
