#main() is what kernels/dev.py runs on the A10G

import torch
# import triton
# import triton.language as tl


# @triton.jit
# def my_kernel(...):        the kernel itself: runs once per program-id, on one tile
#     ...


def main():
    assert torch.cuda.is_available()
    # 1) build inputs on cuda
    # 2) run the kernel into an output tensor
    # 3) reference = the same math in plain torch
    # 4) print max abs err vs the reference, and assert it's within tolerance
    # 5) optional: time both (torch.cuda.synchronize around timers, median of runs)
    print("template: nothing to run")
