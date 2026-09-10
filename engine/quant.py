import torch
import torch.nn as nn
import torch.nn.functional as F
try: 
    import triton
    import triton.language as tl
except ImportError: triton = None

#per channel quantization
def quantize_weight(w): #->(q, scale)
    row_maxes = w.abs().max(dim=1, keepdim=True).values #(w.shape[0], 1)
    scale = (row_maxes/127).clamp(min=torch.finfo(w.dtype).tiny) # .tiny is the smallest positive normal value, vs .min is negative
    q = (w/scale).round().to(torch.int8)
    return (q, scale)


class QuantLinear(nn.Module):
    def __init__(self, linear): #built from an existing Linear
        super().__init__()
        q, s = quantize_weight(linear.weight)
        self.bias = linear.bias # nn.Parameter autoregistered
        #need to register tensors with pytorch
        self.register_buffer("q", q)
        self.register_buffer("scale", s)
        
    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16:
            return self._forward_kernel(x)
        w = self.q.to(x.dtype)* self.scale #slow path for mps/cpu
        return F.linear(x, w, self.bias) #x @ w.T + bias
    
    def _forward_kernel(self, x):
        from kernels.dequant_matmul import dequant_matmul_kernel
        N = self.q.shape[0]
        K = x.shape[-1]   # 768, always the last dim
        x2 = x.reshape(-1, K)              # (4, 10, 768) -> (40, 768)
        M = x2.shape[0]                    # 40 
        BM, BN, BK = 16, 64, 64 
        s = self.scale.squeeze(1)
        #w = self.q.to(x.dtype)* self.scale
        w = torch.empty((M, N), device=x.device, dtype=x.dtype) #acc (and out) is fp32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN)) 
        dequant_matmul_kernel[grid](x, self.q, s, w,
                        x2.stride(0), x2.stride(1), # stride(0) = jump one row, stride(1) = jump one col
                        self.q.stride(1), self.q.stride(0),   #swap for B to be (K, N)   
                        w.stride(0), w.stride(1),
                        M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        return (w + self.bias).reshape(*x.shape[:-1], N)   #(40, 2304) -> (4, 10, 2304)


def quantize_model(model):
    for block in model.transformer.h: #for each hiddne block
        block.attn.c_attn = QuantLinear(block.attn.c_attn)
        block.attn.c_proj = QuantLinear(block.attn.c_proj)
        block.mlp.c_fc = QuantLinear(block.mlp.c_fc)
        block.mlp.c_proj = QuantLinear(block.mlp.c_proj)
    return model
