import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from engine.config import DEVICE, GPTConfig, DTYPE
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PAD_TOKEN = 0 #filler id for pad slots; arbitrary (mask makes it weightless), module-level so tests can import it

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) #token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd) #positional embeddings

    def forward(self, ids, past_length = 0, pos_ids = None): #ids from HF tokenizer, pos_ids (b, t) dtype long are per-slot wpe row numbers for batching
        device = ids.device
        t = ids.shape[1]
        tok_emb = self.wte(ids) # (b, t, n_embd)
        # e.g. tensor of shape (1, 3, 768):
        # [[ [row 15496's 768 floats],
        #    [row 16432's 768 floats],
        #    [row   995's 768 floats] ]]
        if pos_ids is None:
            pos = torch.arange(past_length, past_length+t, dtype=torch.long, device=device) # row numbers, shape t, t is the 
            pos_emb = self.wpe(pos) # (t, n_embd); shared across batches of same t, content independent, learned for GPT2 (doesnt have to be learned)
        # e.g. tensor of shape (3, 768):
        # [ [row 0's 768 floats],
        #   [row 1's 768 floats],
        #   [row 2's 768 floats] ]
        else:
            pos_emb = self.wpe(pos_ids)

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

    #kv_cache: (k_past, v_past, past_len) - full-size buffers preallocated by generate(); only the first past_len positions are used
    #attn_mask: (b, total_t) 
    def forward(self, x, kv_cache = None, attn_mask = None): 
        b, t, c = x.shape #(b, t, n_embd)
        head_size = c // self.n_head # n_embd//n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        #unpacks into seperate heads, transpose to move heads up front to process all heads in parallel as seperate batch elements
        q = q.view(b, t, self.n_head, head_size).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_size).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_size).transpose(1, 2)

        #v1 concat cache: correct but allocates a new tensor + copies the whole cache every token
        #kv_cache was (k_past, v_past), growing by 1 along dim -2 each step:
        # if kv_cache is not None:
        #     k_past, v_past = kv_cache
        #     k = torch.concat((k_past, k), dim=-2)
        #     v = torch.concat((v_past, v), dim=-2)
        # new_kv_cache = (k, v)

        #v2 preallocated: k_past is always max_len long - past tokens + empty slots - so past_len (not .shape) 
        if kv_cache is not None:
            k_past, v_past, past_len = kv_cache
            total = past_len + t
            k_past[:, :, past_len:total] = k
            v_past[:, :, past_len:total] = v
            #attend over the filled prefix only; slicing returns a view, not a copy
            k = k_past[:, :, :total]
            v = v_past[:, :, :total]
            new_kv_cache = (k_past, v_past, total) #same buffers, counter advanced by t
        else:
            new_kv_cache = None #plain forward (tests/logit checks) builds no cache

        #dynamic mask slicing
        current_t = q.size(-2)
        total_t = k.size(-2)

        #implementation from attention is all you need paper
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) #(b, nh, current_t, total_t)
        #att = att.masked_fill(self.tril_mask[:,:,:t,:t] == 0, float('-inf')) #for decoder arch
        
        if attn_mask is not None:
            #apply no-attention to the mask for batching
            # the Nones without the colon is the same thing as .unsqueeze(1).unsqueeze(1) to add dimensions of 1 at position 1 to line up with nh, current_t dimensions in att
            att = att.masked_fill(attn_mask[:, None, None, :] == 0, torch.finfo(att.dtype).min) #not -inf to avoid padding all -inf --> softmax to NaN

        att = att.masked_fill(self.tril_mask[:, :, total_t - current_t : total_t, :total_t] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1)

        y = att @ v # (b, n_head, t, t) x (b, n_head, t, head_size) -> (b, n_head, t, head_size)
        y = y.transpose(1, 2).contiguous().view(b, current_t, c) #re-assemble all head outputs side by side

        return self.c_proj(y), new_kv_cache

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

    def forward(self, x, kv_cache= None, attn_mask = None):
        attn_out, next_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache, attn_mask = attn_mask) 
        x = x + attn_out #add to input --> residual/skip connection
        x = x + self.mlp(self.ln_2(x)) 
        return x, next_kv_cache

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

    def forward(self, ids, kv_past=None, attn_mask=None):
        if kv_past is not None:
            past_length = kv_past[0][2]  # layer 0's past_len counter:
        else:
            past_length = 0


        # build pos_ids left-padding for batching
        if attn_mask is None:
            x = self.transformer.embd(ids, past_length=past_length, pos_ids = None)
        else:
            #slot position = number of 1s strictly to its left in its row of the attn_mask
            #inclusive cumulative sum , minus 1 and clamp negative to get exclusive
            pos_ids = (torch.cumsum(attn_mask, dim=-1) - 1).clamp(min=0) # (b, total_t)
            pos_ids = pos_ids[:, -ids.size(1):]  # keep last t columns: mask covers cache+input, embd only embeds input
            x = self.transformer.embd(ids, past_length=past_length, pos_ids=pos_ids)


        new_kv = [] #per layer cache tensors
        for i, block in enumerate(self.transformer.h):
            layer_past = kv_past[i] if kv_past is not None else None
            x, layer_cache = block(x, kv_cache=layer_past, attn_mask=attn_mask)
            new_kv.append(layer_cache)
        x = self.lm_head(self.transformer.ln_f(x))
        return x, new_kv

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
        b = ids.shape[0]
        head_size = self.config.n_embd // self.config.n_head
        w = self.lm_head.weight #any weight works here, just borrowing its device/dtype precision for the buffers
        # preallocation: one (b, nh, max_len, hs) K buffer + V buffer per layer
        # written into slice-by-slice each step. max_len = the most positions we could ever hold: prompt + new tokens, capped at block_size (the model can't attend past that)
        max_len = min(ids.shape[1] + max_new_tokens, self.config.block_size)
        kv_cache = [
            (torch.empty(b, self.config.n_head, max_len, head_size, device=w.device, dtype=w.dtype), #this way buffer allocations can just follow the weights and both be fp16
             torch.empty(b, self.config.n_head, max_len, head_size, device=w.device, dtype=w.dtype),
             0) # (k_past, v_past, past_len): past_len counts how many positions are filled
            for _ in range(self.config.n_layer)
        ]

        for i in range(max_new_tokens):
            if i == 0:
                # prefill: run the whole prompt through once to fill the cache (cropped to the last block_size tokens if the prompt alone is overlong)
                ids_cond = ids[:, -self.config.block_size:]
            else:
                # decode: cache holds k/v for every earlier position, so only the newest token needs a forward pass
                ids_cond = ids[:, [-1]]
                # v1 overflow handling: reset-and-reprefill -> exact (fresh positions 0..1023), but re-runs a full block_size prefill on EVERY step past the limit:
                # if ids.size(1) > self.config.block_size:
                #     ids_cond = ids[:, -self.config.block_size:]
                #     kv_cache = None
                if kv_cache[0][2] == max_len:
                    # SLIDING: cache is full (only happens once we hit block_size) -> drop oldest position: roll shifts everything one slot left, the stale
                    # copy left in the last slot gets overwritten by this step's write.
                    # approximation, not identical to re-running the last block_size tokens from scratch: cached k/v keep the absolute positions they were computed with, and every new token reuses wpe row block_size-1
                    for k_past, v_past, _ in kv_cache:
                        k_past.copy_(torch.roll(k_past, shifts=-1, dims=2))
                        v_past.copy_(torch.roll(v_past, shifts=-1, dims=2))
                    kv_cache = [(k, v, max_len - 1) for k, v, _ in kv_cache]

            logits, kv_cache = self(ids_cond, kv_past = kv_cache) # (b, t, vocab_size): a score for "what comes next" at every position
            # we only want the prediction after the last token; temperature rescales confidence: <1 exaggerates the gap between high and low scores, >1 shrinks it (1.0 = untouched)
            logits = logits[:, -1, :] / temperature # (b, vocab_size)
            if not do_sample:
                # greedy: always take the single highest-scoring token. deterministic for testing
                next_id = logits.argmax(dim=-1, keepdim=True) # (b, 1)
            else:
                # sampling: pick the next token at random, weighted by the model's confidence
                if top_k is not None:
                    # keep only the k best-scoring tokens: find the kth-best score per row, set everything below it to -inf (-> probability 0 after softmax), random draw can never land on a garbage tail token
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # softmax turns raw scores into probabilities (all positive, sum to 1)
                probs = F.softmax(logits, dim=-1)
                # multinomial = weighted dice roll
                next_id = torch.multinomial(probs, num_samples=1) # (b, 1)
            ids = torch.cat((ids, next_id), dim=1) # append -> next iteration sees it as context

        return ids

    @torch.no_grad()
    def generate_batch(self, all_ids, max_new_tokens, eos_id=None): #all_ids is a python list of b tensors
        t_max = max(ids.shape[1] for ids in all_ids)
        b = len(all_ids)
        head_size = self.config.n_embd // self.config.n_head
        w = self.lm_head.weight

        # preset to all pads
        ids  = torch.full((b, t_max), PAD_TOKEN, dtype=torch.long, device=w.device) 
        mask = torch.zeros((b, t_max), dtype=torch.long, device=w.device) 

        for i, seq in enumerate(all_ids):
            ids[i, -seq.shape[1]:] = seq[0] #overwrite the right end of row i with real ids
            mask[i, -seq.shape[1]:] = 1 #mark those slots as used

        max_len = max_new_tokens + t_max
        assert(max_len <= self.config.block_size)

        kv_cache = [
            (torch.empty(b, self.config.n_head, max_len, head_size, device=w.device, dtype=w.dtype),
            torch.empty(b, self.config.n_head, max_len, head_size, device=w.device, dtype=w.dtype),
                0) for _ in range(self.config.n_layer)
        ]

        #prefill
        logits, kv_cache = self(ids, kv_past=kv_cache, attn_mask=mask) #forward
        completed = torch.zeros(b, dtype = torch.bool, device=w.device) #track which sequences have been completed

        for i in range(max_new_tokens):
            logits = logits[:, -1, :] 
            next_id = logits.argmax(dim=-1, keepdim=True) # (b, 1)
            if eos_id is not None:
                next_id[completed] = PAD_TOKEN #bool tensor indexing, sets position with True in completed to the pad
                completed = completed.masked_fill(next_id.squeeze(1) == eos_id , True) #like a "stop generating stuff" switch takes effect next iteration
            ids = torch.cat((ids, next_id), dim=1)
            mask = torch.cat((mask, torch.ones((b, 1), dtype=torch.long, device=w.device)), dim = 1)
            if i < max_new_tokens - 1: #final forward is never read
                logits, kv_cache = self(next_id, kv_past=kv_cache, attn_mask=mask)
            if completed.all(): 
                missing = (t_max + max_new_tokens) - ids.size(1) 
                ids = F.pad(ids, (0, missing), value=PAD_TOKEN)
                break
        return ids #(b, t_max + max_new_tokens)


if __name__ == "__main__":
    cfg = GPTConfig()
    ids = torch.randint(0, cfg.vocab_size, (2, 8), device=DEVICE) # (b, t) = (2, 8) simulated
    x = Embedding(cfg).to(DEVICE)(ids) # (b, t, n_embd)
    print("embedding:", x.shape)
    x, _ = CausalSelfAttention(cfg).to(DEVICE)(x) # (b, t, n_embd); returns (y, kv_cache) now -> unpack, cache is None here
    print("attention:", x.shape)
    x = MLP(cfg).to(DEVICE)(x)
    print("mlp:", x.shape)
    x, _ = Block(cfg).to(DEVICE)(x)
    print("block: ", x.shape)
    x, _ = GPT(cfg).to(DEVICE)(ids)
    print("gpt:", x.shape)

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()
    ids = tok("The meaning of life is", return_tensors="pt").input_ids.to(DEVICE) #(b, t) = (1, 5); or t the sequence length is the number of subword tokens. b is batch size = number of input strings
    print(tok.decode(model.generate(ids, max_new_tokens=20)[0]))