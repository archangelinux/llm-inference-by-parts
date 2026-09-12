import torch
from dataclasses import dataclass
import os

DTYPE = torch.float16 if os.environ.get("DTYPE")=="fp16" else torch.float32 #replaces model.half()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def sync():
    """block until all queued device work is done -- call before reading a timer"""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

# GPT-2 124M
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True

@dataclass
class QwenConfig:
    block_size: int = 32768 #max_pos_embds, RoPE makes long context cheaper
    vocab_size: int = 151936 #3x bigger
    n_layer: int = 28 
    n_head: int = 16
    n_embd: int = 1024 #hidden_size
    bias: bool = False

    n_kv_head: int = 8 #GQA number - 16 q-heads share 8 k/v-heads, 2 qs per kv
    head_dim: int = 128 # 1024/16 = 64, but heads are 128 wide; q projects to 16×128 = 2048, k/v to 8×128 = 1024, o_proj maps 2048 → 1024

    intermediate_size: int = 3072 #MLP width (3x hidden, isntead of 4x hidden in gpt)
    rms_norm_eps: float = 1e-06
    rope_theta:float = 1000000.0 # base frequency, constant in angle formula
    #eos_token_id = 151643