import torch
import torch.nn as nn
from torch.nn import functional as funct
import math

from engine.config import DEVICE, GPTConfig

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) #token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd) #positional embeddings

    def forward(self, ids): #ids from HF tokenizer
        device = ids.device
        b, t = ids.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device) # row numbers, shape t
        tok_emb = self.wte(ids) # (b, t, n_embd)
        # e.g. tensor of shape (1, 3, 768):
        # [[ [row 15496's 768 floats],
        #    [row 16432's 768 floats],
        #    [row   995's 768 floats] ]]
        pos_emb = self.wpe(pos) # (t, n_embd); shared across batches of same t, content independent, learned for GPT2 (doesnt have to be learned)
        # e.g. tensor of shape (3, 768):
        # [ [row 0's 768 floats],
        #   [row 1's 768 floats],
        #   [row 2's 768 floats] ]
        return tok_emb + pos_emb

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(in_features = config.n_embd, out_features = config.n_embd*3, bias = config.bias)
        #batch and split all 3 instead of:
        #self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        # self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(in_features = config.n_embd, out_features = config.n_embd, bias = config.bias)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

         # lower triangular causal mask -> only attend to the left in the input sequence
        self.register_buffer("tril_mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, t, c = x.shape
        head_size = c // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, head_size).transpose(1, 2) 
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2) 
        v = v.view(b, t, self.n_head, head_size).transpose(1, 2) 

        #implementation from attention is all you need paper
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril_mask[:,:,:t,:t] == 0, float('-inf')) #for decoder arch
        att = funct.softmax(att, dim=-1)
        y = att @ v # (b, n_head, t, t) x (b, n_head, t, head_size) -> (b, n_head, t, head_size)
        y = y.transpose(1, 2).contiguous().view(b, t, c) #re-assemble all head outputs side by side

        return self.c_proj(y)

#feed forward = FFN = multi layer perceptron; need time to “think” about the gathered data before calculating the logits
class MLP(nn.Module): 
    def __init__(self, config):
        super().__init__()
        # *4 lets the layer compute richer per-token functions than it could at native width
        self.c_fc = nn.Linear(in_features= config.n_embd, out_features = config.n_embd * 4, bias = config.bias) #convolution, fully-connected 
        self.c_proj = nn.Linear(in_features= config.n_embd * 4, out_features = config.n_embd, bias = config.bias)
        self.gelu = nn.GELU(approximate = "tanh") #fast approx, GPT2

    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #add to input --> residual/skip connection
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            embd = Embedding(config), #could get rid of this module and merge in to match HF naming 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #hidden blocks
            ln_f = nn.LayerNorm(config.n_embd) #final
        )
        )
        self.lm_head = nn.Linear(in_features = config.n_embd, out_features = config.vocab_size, bias = False)
        self.lm_head.weight = self.transformer.embd.wte.weight

    def forward(self, ids):
        x = self.transformer.embd(ids)
        for block in self.transformer.h:
            x = block(x)
        x = self.lm_head(self.transformer.ln_f(x))
        return x

if __name__ == "__main__":
    cfg = GPTConfig()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE)
    x = Embedding(cfg).to(DEVICE)(ids)
    print("embedding:", x.shape)
    x = CausalSelfAttention(cfg).to(DEVICE)(x)
    print("attention:", x.shape)
    x = MLP(cfg).to(DEVICE)(x)
    print("mlp:", x.shape)
    x = Block(cfg).to(DEVICE)(x)
    print("block: ", x.shape)
    x = GPT(cfg).to(DEVICE)(ids)
    print("gpt:", x.shape)