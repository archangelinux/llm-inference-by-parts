import torch
from dataclasses import dataclass

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# GPT-2 124M
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True