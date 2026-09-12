#measure the kernel's GPU execution time with cuda events (excludes the python
    #launch gap that dominated the wall-clock sweep) at the decode shape M=1
# usage: modal run kernels/dev.py --name sweep_gpu

import torch
import torch.nn.functional as F

from engine.quant import quantize_weight
from kernels.dequant_matmul import dequant_matmul_kernel
import triton


def gpu_time(fn, iters=100):
    #per-launch cuda events: elapsed time between events is GPU execution only
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        total += a.elapsed_time(b)  # ms
    return total / iters * 1e3  # us


def main():
    assert torch.cuda.is_available()
    for (N, K) in [(2304, 768), (768, 3072)]:  # c_attn and mlp.c_proj shapes
        M = 1
        W = torch.randn((N, K), device="cuda", dtype=torch.float16)
        q, s = quantize_weight(W)
        s_flat = s.squeeze(1)
        x = torch.randn((M, K), device="cuda", dtype=torch.float16)
        out = torch.empty((M, N), device="cuda", dtype=torch.float16)
        nbytes = N * K

        t = gpu_time(lambda: F.linear(x, W))
        print(f"({N}x{K}) fp16 F.linear : {t:6.1f} us  {2*nbytes/t/1e3:6.0f} GB/s (reads 2x bytes)")

        for BN, BK, warps in [(32, 64, 4), (32, 256, 4), (64, 128, 4), (128, 256, 2), (128, 256, 8)]:
            grid = (triton.cdiv(M, 16), triton.cdiv(N, BN))
            def run():
                dequant_matmul_kernel[grid](x, q, s_flat, out,
                                            x.stride(0), x.stride(1),
                                            q.stride(1), q.stride(0),
                                            out.stride(0), out.stride(1),
                                            M, N, K, BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK,
                                            num_warps=warps)
            t = gpu_time(run)
            print(f"({N}x{K}) BN={BN:4d} BK={BK:4d} w={warps}: {t:6.1f} us  {nbytes/t/1e3:6.0f} GB/s "
                  f"({100*nbytes/t/1e3/600:.0f}% of peak)")
