import torch
import torch.nn as nn
from torch.nn import functional as funct
import math

from engine.config import DEVICE, GPTConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer 

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
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

    def forward(self, x, kv_cache = None): #kv_cache: (k_past, v_past) 
        b, t, c = x.shape #(b, t, n_embd)
        head_size = c // self.n_head # n_embd//n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, head_size).transpose(1, 2) #unpacks into seperate heads, transpose to move heads up from to process all heads in parallel as seperate batch elements
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
        self.config = config

    def forward(self, ids):
        x = self.transformer.embd(ids)
        for block in self.transformer.h:
            x = block(x)
        x = self.lm_head(self.transformer.ln_f(x))
        return x

    @classmethod
    def from_pretrained(cls, config):
        model = cls(config)
        sd = model.state_dict()
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        with torch.no_grad():
            for k in sd_hf.keys():
                if k in ("transformer.wte.weight", "transformer.wpe.weight"):
                    my_k = "transformer.embd." + k.removeprefix("transformer.")
                else:
                    my_k = k

                if any(k.endswith(w) for w in transposed):
                    # Conv1D weights to transpose for Linear
                    assert sd_hf[k].shape[::-1] == sd[my_k].shape
                    sd[my_k].copy_(sd_hf[k].t())
                else:
                    assert sd_hf[k].shape == sd[my_k].shape
                    sd[my_k].copy_(sd_hf[k])

        print(f"loaded {len(sd_hf)} tensors")
        return model

    @torch.no_grad() # inference only -> don't build the autograd graph (faster, less memory)
    def generate(self, ids, max_new_tokens, do_sample=False, temperature=1.0, top_k=None):
        """Autoregressive decoding: predict one token, append it, repeat.
        ids: (b, t) prompt token ids -> returns (b, t + max_new_tokens)."""
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens (wpe dimensions); a longer sequence would index past the embedding table and crash
            ids_cond = ids if ids.size(1) <= self.config.block_size else ids[:, -self.config.block_size:]
            logits = self(ids_cond) # (b, t, vocab_size): a score for "what comes next" at every position
            # we only want the prediction after the last token; temperature rescales confidence: <1 exaggerates the gap between high and low scores, >1 shrinks it (1.0 = untouched)
            logits = logits[:, -1, :] / temperature # (b, vocab_size)
            if not do_sample:
                # greedy: always take the single highest-scoring token. deterministic,
                # same output every run -- what the tests/fixtures rely on
                next_id = logits.argmax(dim=-1, keepdim=True) # (b, 1)
            else:
                # sampling: pick the next token at random, weighted by the model's confidence
                if top_k is not None:
                    # keep only the k best-scoring tokens: find the kth-best score per row, set everything below it to -inf (-> probability 0 after softmax), random draw can never land on a garbage tail token
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # softmax turns raw scores into probabilities (all positive, sum to 1)
                probs = funct.softmax(logits, dim=-1)
                # multinomial = weighted dice roll: token with prob 0.3 is drawn 30% of the time
                next_id = torch.multinomial(probs, num_samples=1) # (b, 1)
            ids = torch.cat((ids, next_id), dim=1) # append -> next iteration sees it as context

        return ids



if __name__ == "__main__":
    cfg = GPTConfig()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE) # (b, t) = (2, 8) simulated
    x = Embedding(cfg).to(DEVICE)(ids) # (b, t, n_embd)
    print("embedding:", x.shape)
    x = CausalSelfAttention(cfg).to(DEVICE)(x) # (b, t, n_embd)
    print("attention:", x.shape)
    x = MLP(cfg).to(DEVICE)(x)
    print("mlp:", x.shape)
    x = Block(cfg).to(DEVICE)(x)
    print("block: ", x.shape)
    x = GPT(cfg).to(DEVICE)(ids)
    print("gpt:", x.shape)


    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()
    ids = tok("The meaning of life is", return_tensors="pt").input_ids.to(DEVICE) #(b, t) = (1, 5); or t the sequence length is the number of subword tokens. b is batch size = number of input strings
    print(tok.decode(model.generate(ids, max_new_tokens=20)[0]))