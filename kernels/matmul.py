# GPU kernel to perform A @ B shape (M, K) @ (K, N) = (M, N)
# in vector_add.py, a copy = a single run of the output vector; here we want each copy to own a rectangular tile of the output matrix
import torch
import triton
import triton.language as tl

@triton.jit 
def matmul_kernel(a_ptr, b_ptr, out_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_om, stride_on, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N:tl.constexpr, BLOCK_K:tl.constexpr): 
    pid_m = tl.program_id(0) #unique id for each copy
    pid_n = tl.program_id(1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M) #rows
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N) #cols
    mask_out = (offset_m[:,None] < M) & (offset_n[None,:] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offset_k = k + tl.arange(0, BLOCK_K)
        mask_a =  (offset_m[:,None] < M) & (offset_k[None,:] < K)
        mask_b =  (offset_k[:,None] < K) & (offset_n[None,:] < N)
        a = tl.load(a_ptr + offset_m[:,None]*stride_am + offset_k[None,:]*stride_ak, mask = mask_a)
        b = tl.load(b_ptr + offset_k[:,None]*stride_bk + offset_n[None,:]*stride_bn, mask = mask_b)
        acc += tl.dot(a, b)
    tl.store(out_ptr + offset_m[:,None]*stride_om + offset_n[None,:]*stride_on, acc, mask = mask_out)


def main():
    assert torch.cuda.is_available()
    M, K, N = 256, 512, 128
    A = torch.rand((M, K), device="cuda", dtype=torch.float16)
    B = torch.rand((K, N), device="cuda", dtype=torch.float16)
    out = torch.empty((M, N), device="cuda", dtype=torch.float32) #acc (and out) is fp32
    BM, BN, BK = 32, 32, 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN)) 
    matmul_kernel[grid](A, B, out,
                        A.stride(0), A.stride(1), # stride(0) = jump one row, stride(1) = jump one col
                        B.stride(0), B.stride(1),    
                        out.stride(0), out.stride(1),
                        M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    ref = A.float() @ B.float()
    #not 0.0 like vector_add: loads/tl.dot is fp16, round differently than the ref which is fp32
    #~0.05 on ~128 is healthy; ~100 means a stride bug
    print((out - ref).abs().max().item())
