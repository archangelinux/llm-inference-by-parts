# GPU kernel to perform x + y for two tensors 
#torch's x + y already does this at the same speed
#this is to illustrate/learn the plumbing common to every kernel (program ids, offsets, mask, load/store) on the smallest possible example 
#this will later translate over to the fused dequant-matmul kernel, which is this same plumbing with the matmul and int8 dequant between the load and the store.

import torch
import triton
import triton.language as tl

@triton.jit #marking for GPU compilation; tensors arrive as pointers (index 0 memory address) 
#block is the chunk size thats fixed at compile time
#GPU runs thousands of copies of this block-sized program at the same time, one copy per chunk
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr): 
    pid = tl.program_id(0) #unique id for each copy
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n #vector of n booleans, to mask out what was added from ceiling div
    x = tl.load(x_ptr + offset, mask = mask)
    y = tl.load(y_ptr + offset, mask = mask)
    tl.store(out_ptr + offset, x + y, mask = mask)


def main():
    assert torch.cuda.is_available()
    #build inputs on cuda
    n = 10000
    x = torch.rand(n, device = "cuda");
    y = torch.rand(n, device = "cuda");
    out = torch.empty_like(x) #mem allocation
    #run the kernel into an output tensor
    #grid is the number of copies the kernel will run
    grid = (triton.cdiv(n, 1024),) #ceiling division, =10 for this example. tuple because grids can be multidim
    add_kernel[grid](x, y, out, n, BLOCK = 1024) 
    #reference = the same math in plain torch
    #max abs err vs the reference
    print((out - (x + y)).abs().max().item())
