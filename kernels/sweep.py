#config sweep for dequant_matmul_kernel at the decode shape (M=1, c_attn 2304x768)
    #times every (BLOCK_N, BLOCK_K, num_warps) combo per-op, vs the fp16 cuBLAS baseline
# usage: modal run kernels/dev.py --name sweep

import time

import torch
import torch.nn.functional as F

from engine.quant import quantize_weight
from kernels.dequant_matmul import dequant_matmul_kernel
import triton


def op_time(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    assert torch.cuda.is_available()
    N, K, M = 2304, 768, 1
    W = torch.randn((N, K), device="cuda", dtype=torch.float16)
    q, s = quantize_weight(W)
    s_flat = s.squeeze(1)
    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    bytes_moved = N * K  # int8 weight read dominates

    base = op_time(lambda: F.linear(x, W))
    print(f"baseline fp16 F.linear: {base*1e6:7.1f} us ({2*N*K/base/1e9:.0f} GB/s on 2x bytes)")

    rows = []
    for BN in [32, 64, 128]:
        for BK in [64, 128, 256]:
            for warps in [2, 4, 8]:
                grid = (triton.cdiv(M, 16), triton.cdiv(N, BN))
                def run():
                    dequant_matmul_kernel[grid](x, q, s_flat, out,
                                                x.stride(0), x.stride(1),
                                                q.stride(1), q.stride(0),
                                                out.stride(0), out.stride(1),
                                                M, N, K, BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK,
                                                num_warps=warps)
                t = op_time(run)
                rows.append((t, BN, BK, warps))
                print(f"BN={BN:4d} BK={BK:4d} warps={warps}  {t*1e6:7.1f} us  {bytes_moved/t/1e9:6.0f} GB/s")

    rows.sort()
    t, BN, BK, warps = rows[0]
    print(f"\nbest: BN={BN} BK={BK} warps={warps}  {t*1e6:.1f} us  {bytes_moved/t/1e9:.0f} GB/s "
          f"({100*bytes_moved/t/1e9/600:.1f}% of peak) vs fp16 baseline {base*1e6:.1f} us")
