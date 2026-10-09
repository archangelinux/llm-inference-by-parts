# llm-inference-by-parts, Part 1

Below is a guide on the process, and my learnings from building a from-scratch transformer inference engine and serving layer in PyTorch (GPT-2 + Qwen3) with RoPE, GQA, KV caching, continuous batching, int8 quantization on a Triton kernel, and graph decode, analyzed each time I added a new part.

I hope this can be a helpful reference to others who want somewhere to start, and for making the murkier concepts, albeit of a canonical project, a little easier to internalize.

## What is inference?

A language model's life has two parts. **Training** sets the model's weights (its numbers) by showing it huge amounts of text. **Inference** is using the trained model: the weights stay fixed, and the model generates a response to a new prompt.

The model doesn't write the response all at once. It splits the prompt into tokens (words or pieces of words) and runs them through the model, which gives a score for every possible next token. It picks one, adds it to the end, and runs the model again:

```
"The meaning of life is"         → model → " to"
"The meaning of life is to"      → model → " be"
"The meaning of life is to be"   → model → " happy"
```

This repeats until the response is finished, so a 100-token response takes 100 runs through the model. Inference optimization is about making each run faster and cheaper, and about running many users' requests on the same GPU at once.

An **inference engine** is the code that does this. It loads the weights, runs the model, picks each next token, and manages the GPU memory and the batching of requests. In this project that's `engine/`.

A **serving layer** connects the engine to users. When you send a message in ChatGPT or Claude, it goes to one of many GPU servers, and each server is generating responses for many people at once. The serving layer takes in requests, sends each one to a server with room for it, and streams the response back as it's generated (usually a few tokens at a time), so you can start reading before the model has finished. In this project that's `server/`: one server in front of one engine, where the engine's scheduler decides which requests run together.

---

In retrospect, basic inference optimization almost naturally chains into an intuitive implementation order, since most techniques exist to fix a specific bottleneck that a previous one might have exposed. I think that makes it easy to build knowledge quickly without getting overwhelmed and thinking you need to figure out everything all at once, or even plan things out too much.

Even when it unfolds "in a different order", you have enough context to judge and implement with intent. E.g. I implemented my int8 quantization before CUDA graphing which resulted in practically no speedup. But that's one way to learn why CUDA graphing is important (to remove the launch overhead), so you then go about making up for it without having lost the plot.

I found it helpful to implement the more naive approaches first to better understand how and why the alternatives I'd end with perform better, and what and by how much each mechanism optimizes.

This project does not introduce any new architecture and was intended for learning through experimentally dissecting inference mechanisms. Every stage is checked against HuggingFace's `transformers` implementation of the same models, [gpt2](https://huggingface.co/openai-community/gpt2) and [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base): my engine has to generate the same tokens theirs does. The reference model is loaded with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")          # or "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
```

`tests/make_fixtures.py` runs 8 prompts through it and saves the logits and 50-token greedy outputs, and the tests compare my engine against those.

Hardware: Macbook Air M1, 2020 (MPS), NVIDIA A10G (via Modal)

Repo [README.md](https://github.com/archangelinux/llm-inference#readme) for metrics / findings. Below is a full writeup of the process, which you can also find on my site.

### Resources

There are many resources/papers dedicated to the different mechanisms implemented (some partially or simplified) in this project linked below:

- Attention Is All You Need: https://arxiv.org/pdf/1706.03762
- Efficiently Scaling Transformer Inference: https://arxiv.org/pdf/2211.05102
- ORCA (iteration-level scheduling): https://www.usenix.org/system/files/osdi22-yu.pdf
- LLM.int8() (W8A8 and activation outliers): https://arxiv.org/abs/2208.07339
- nanoGPT model.py: https://github.com/karpathy/nanoGPT/blob/master/model.py
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- Qwen3-0.6B-Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base

For the modern port:

- Gating / GLU variants: https://arxiv.org/pdf/2002.05202
- RMSNorm: https://arxiv.org/pdf/1910.07467
- RoPE: https://arxiv.org/pdf/2104.09864

---

## 0. Prerequisites

- Watch Karpathy's **"Let's build GPT: from scratch"** (~4 hrs). This is a good calibration check as well as a reference check. I took notes and didn't code along the first time. I rewatched it halfway through my project to code it out and review some concepts, which turned out to be super worthwhile and elucidating.
- Environment: Python 3.11+, `torch`, `transformers` (for weight/tokenizer downloads and correctness checks ONLY), `numpy`, `matplotlib`, `fastapi`, `uvicorn`, `httpx`.

  ```bash
  pip install torch transformers numpy matplotlib fastapi uvicorn httpx pytest
  ```

- Download GPT-2 (124M) weights via HuggingFace. Confirm you can run the reference model and generate text. Every correctness test compares against its output.
- Repo structure:

  ```
  llm-inference-by-parts/
  ├── engine/     # model code: attention, mlp, norms, generation, scheduler
  ├── server/     # FastAPI layer
  ├── bench/      # benchmark scripts + saved results (JSON) + charts
  ├── kernels/    # Triton kernels (added in stage 5)
  ├── eval/       # quality gates for lossy precision (added in stage 5)
  ├── tests/      # correctness checks vs HuggingFace
  └── README.md   # roadmap and bench results
  ```

- Later add eval for checking quality.
- Start a log/notes for questions you answer along the way, but also to keep unresolved questions for later to stop yourself from rabbit holing too deep (BFS > DFS this project?).

---

## 1. Building the engine

**Goal:** a GPT-2 forward pass, written from scratch, that loads HuggingFace's pretrained weights and generates coherent text. The structure follows Karpathy's nanoGPT.

### Implementation

1. **Tokenizer:** use HF's GPT-2 tokenizer. It turns text into a list of token ids (numbers), and turns ids back into text. Writing a tokenizer isn't part of this project.
2. **Embeddings:** the model can't do math on token ids directly, so each id is swapped for a vector of 768 numbers. GPT-2 has two lookup tables for this:
   - `wte` has one vector per token in the vocabulary, and says which token it is.
   - `wpe` has one vector per position (0 to 1023), and says where the token is.
   - Add the two vectors together, and that sum is what goes into the first block.
   - Notes on my code: modularizing this wasn't exactly necessary and introduced an intermediate module in terms of matching namings with two of HF's parameters. Code is still pretty clean though and this was a good option to break things down step by step.
3. **Attention:** this is where tokens get information from each other.
   - One linear layer (`c_attn`) turns each token's vector into three vectors: a query (q), a key (k) and a value (v).
   - These are split into 12 heads, so 12 separate attention patterns run side by side.
   - Each token compares its q with the k of every earlier token to get a score, and takes a weighted average of their v's using those scores.
   - A token can only look at earlier tokens, never later ones (the causal mask).
   - A second linear layer (`c_proj`) combines the 12 heads' outputs back into one vector.
4. **MLP (multilayer perceptron):** a small feed-forward network inside every transformer block, right after attention. Attention moves information between tokens. The MLP then processes each token's vector on its own, with no information passing between tokens.
   - It's two linear layers: the first makes the vector 4x wider (768 → 3072), and the second brings it back to 768.
   - Between them is GELU, an activation function. The wider middle layer gives the model more room to transform each token before it's compressed back to 768.

   <details>
   <summary><b>What are GELU and ReLU?</b></summary>

   They're activation functions: a function applied to every number between the MLP's two linear layers. Without one, two linear layers in a row add up to a single linear layer, and the MLP could only learn straight-line relationships.

   - **ReLU** is `max(0, x)`: positive numbers pass through unchanged, and negative numbers become 0. This makes a sharp corner at 0.
   - **GELU** is `x × Φ(x)`, where Φ(x) is the probability that a random value from a standard normal distribution is less than x. Large positive x passes through almost unchanged, large negative x becomes almost 0, and values near 0 are partly kept. The curve is smooth, with no corner.

   | x | -2 | -0.5 | 0.5 | 2 |
   |---|---|---|---|---|
   | ReLU | 0 | 0 | 0.5 | 2 |
   | GELU | -0.05 | -0.15 | 0.35 | 1.95 |

   The smooth curve makes training more stable, and GELU is the standard choice in GPT and BERT. GPT-2 uses a faster approximation: `nn.GELU(approximate="tanh")`.
   </details>
5. **Block:** LayerNorm → attention → add the block's input back (the residual) → LayerNorm → MLP → add the input back again. GPT-2 stacks 12 of these.
6. **Final LayerNorm and LM head:** the LM head turns each position's vector into a score for every token in the vocabulary. The highest score is the model's guess for the next token. In GPT-2, the LM head reuses `wte`'s weights instead of having its own.
7. **Load the weights:** copy every tensor from HF's GPT-2 into the matching module in your model.
   - Four of HF's GPT-2 weights (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) are stored as (in, out), but `nn.Linear` stores weights as (out, in). Transpose those four when you copy them. nanoGPT does the same thing.
8. **Greedy decoding loop:** run the whole sequence through the model, take the highest-scoring token at the last position, add it to the end, and repeat. This is deliberately slow: every step reruns the whole sequence from scratch.
9. **Sampling:** add temperature and top-k as options for more varied text. Keep greedy as the default, because the tests need the same output every time.

   <details>
   <summary><b>What are greedy decoding, temperature and top-k?</b></summary>

   The model outputs a logit (a score) for every token in the vocabulary. Softmax turns the logits into probabilities that add up to 1. There are two ways to choose the next token from them:

   - **Greedy:** always take the most likely token. The same prompt always gives the same output, which is why the tests use it. Its downside is that text tends to repeat itself.
   - **Sampling:** pick a token at random, weighted by the probabilities. A token with probability 0.6 is picked about 60% of the time.

   Two settings change the probabilities before sampling:

   - **Temperature:** divide every logit by the temperature before softmax. Below 1, the most likely tokens get even more likely. Above 1, the probabilities get more even, and the text gets more random.
   - **Top-k:** keep only the k most likely tokens and set the rest to 0, so a very unlikely token can never be picked.

   Example with three tokens whose logits are 2.0, 1.0 and 0.5:

   | | token A | token B | token C |
   |---|---|---|---|
   | temperature 1 | 0.63 | 0.23 | 0.14 |
   | temperature 0.5 | 0.84 | 0.11 | 0.04 |
   | temperature 2 | 0.48 | 0.29 | 0.23 |
   | temperature 1, top-k 2 | 0.73 | 0.27 | 0 |

   Greedy picks A every time. All of this is in `engine/sampling.py`.
   </details>

### Shapes

Weights are learned during training and don't change during inference. Activations are the values computed from them for a given input:

```
WEIGHTS (fixed):    wte         wpe          c_attn (×12 layers)       lm_head
                     │           │                  │                     │
                     ▼           ▼                  ▼                     ▼
ACTIVATIONS:      tok_emb  +  pos_emb  =  x ───► q, k, v ───► attn ─► ... ─► logits
                                                 │  └──┴── k and v get stored in the cache (stage 2)
                                                 └──────── q is used once and thrown away
```

Notation: `B` = batch size (number of prompts), `T` = sequence length (number of subword tokens), `C` = `n_embd` = 768, `nh` = `n_head` = 12, `hs` = `head_size` = 64.

**1) IDs:** calling the GPT-2 tokenizer with `return_tensors="pt"` (PyTorch) on a string returns a dictionary with 2 main keys:

```python
{
  "input_ids":      tensor([[...]]),  # (B, T) = (num of strings, num of subword tokens)
  "attention_mask": tensor([[...]]),  # (B, T), 1 = real token, 0 = padding
}
```

**2) Embeddings**

```python
wte = nn.Embedding(vocab_size, n_embd)   # one row per vocab token, n_embd floats each
wpe = nn.Embedding(block_size, n_embd)   # one row per position in the context window
```

`self.wte(ids)` is a lookup, not a calculation. It's the same as indexing `self.wte.weight[ids]`. `ids` has shape (B, T) with integers between 0 and vocab_size-1, so the output has shape (B, T, C): each token id is replaced by its 768-number row.

For example, `ids = [[464, 3616]]` with shape (1, 2) gives `[[weight[464], weight[3616]]]` with shape (1, 2, 768).

`wpe` works the same way, except it's indexed by position (0..block_size-1). `block_size` (1024) is the longest context the model was trained on. It's used again later to build the causal attention mask.

`tok_emb` is (B, T, C) and `pos_emb` is (T, C). PyTorch **broadcasting** adds the same positional embeddings to every sequence in the batch.

**3) Causal self-attention**

```
x                                   (B, T, C)
c_attn(x)                           (B, T, 3C)      split into q, k, v, each (B, T, C)
q.view(B, T, nh, hs)                (B, T, nh, hs)
  .transpose(1, 2)                  (B, nh, T, hs)  heads move to dim 1 so all 12 are computed at once
q @ k.transpose(-2, -1) / sqrt(hs)  (B, nh, T, T)   score of every query against every key
causal mask + softmax over last dim (B, nh, T, T)   each row sums to 1; future positions get 0
att @ v                             (B, nh, T, hs)
.transpose(1, 2).view(B, T, C)      (B, T, C)       the 12 head outputs concatenated back to 768
c_proj                              (B, T, C)
```

The causal mask is a lower-triangular `tril` matrix of shape (block_size, block_size). Scores above the diagonal are set to `-inf` before softmax, which makes their weight 0. Token i can only attend to tokens 0..i.

**4) MLP**

```
c_fc(x)     (B, T, 4C)   project up by a factor of 4 (768 -> 3072)
gelu        (B, T, 4C)   applied to each number separately
c_proj      (B, T, C)    back down
```

**5) Transformer block:** `ln_1(x)` normalizes each token's 768 numbers to mean 0 and variance 1, then applies a learned scale and shift. Shape stays (B, T, C). Then attention, residual add, `ln_2`, MLP, residual add. The shape is (B, T, C) going in and coming out, which is why 12 blocks can be stacked.

**6) Full GPT:** after the 12 blocks and the final `ln_f`:

```
lm_head(x)   (B, T, vocab_size)
```

`lm_head` is an `nn.Linear`. PyTorch's linear layer computes `y = x @ W.T + b`. `W` is stored as (out_features, in_features), so `lm_head.weight` has shape (vocab_size, n_embd), and the multiply is

```
(B, T, n_embd) @ (n_embd, vocab_size) = (B, T, vocab_size)
```

Each position gets a score (logit) for every token in the vocab. For generation we only need the last position, `logits[:, -1, :]` of shape (B, vocab_size), and greedy decoding takes the argmax.

`lm_head.weight` and `wte.weight` are the same tensor. `GPT.__init__` assigns one to the other (weight tying). Both are (vocab_size, n_embd). The model scores its final hidden vector against every token's embedding row.

### Testing

Run the same prompts through your model and HF's. The logits should match to about 1e-3 in fp32, and greedy generation should produce exactly the same tokens.

### Benchmark

After benching: notice that shorter lengths exhibit slightly more variance over the runs vs longer lengths

—

At this point, I've implemented and benchmarked a simple forward pass based off of karpathy's build gpt/nanogpt tutorial. This was great for learning the tensor and attention math and mechanics, as well as the basic architectural layout. My model is mathematically complete, but it is **highly inefficient and static**. It can process a single prompt in a single chunk, but it cannot scale.

From here, through adding and benchmarking llm inference optimization like caching, batching, scheduling, adaptive compute, etc, we can better understand how these more modern components, and the mechanics behind them, actually affect performance.

---

## 2. KV caching

**Goal:** stop recomputing values that were already computed.

Without a cache, `generate` runs the entire sequence through all 12 blocks for every new token. Only the last position is new. Every earlier position has the same tokens, mask, k and v as on the previous step, and gets computed again anyway. The KV cache stores k and v from previous steps. Each step computes q, k and v for the new token only, and attends the new q against the stored k and v.

This is correct because attention is causal. At every layer, token 7's k and v depend only on tokens 0..7. A new token 8 can't change them, so they never need recomputing.

<details>
<summary><b>Why k and v and not q?</b></summary>

Attention wants to answer: "How much does my current word (Query) care about all past words (Keys) to extract their meanings (Values)?" In the generation step:

The Query (Q): Represents only the brand-new token just generated. You want to see how this new token links back to history. You never need past queries for your current query. Once past tokens finish querying their respective histories in previous loop steps, their Query isn't used anymore.

The Keys and Values (K and V): Represent the entire conversation history database. Every time a new query token enters the layer, it needs to look at the keys of all previous tokens to calculate **attention scores**, and then multiply those scores by **all previous values**, to get the new token's attention output.

If you don't cache K and V, you force the model to recompute those historical database matrices from scratch on every single token step.
</details>

### Implementation

The generation loop now has 2 phases with different shapes:

- **Prefill** (step 0): the whole prompt goes through once. k and v are computed as usual and written into the cache.
- **Decode** (every later step): only the newest token goes in. The model computes q, k, v for that one token, writes its k and v into the cache, attends over everything in the cache, and predicts the next token.

```
prefill:  "The meaning of life is"  ──►  t = 5   cache columns 0-4 filled
decode 1: " to"                     ──►  t = 1   column 5 filled, attends over 0-5
decode 2: " be"                     ──►  t = 1   column 6 filled, attends over 0-6
```

Prefill does math on many tokens for each read of the weights. Decode reads every weight to process one token, so its speed depends mostly on how fast memory can be read.

1. **Start with concat:** each layer returns `(k, v)`, and the next step does `torch.cat((k_past, k), dim=-2)`. It's correct, but it allocates a new tensor and copies the whole cache for every token.
2. **Then preallocate:** in `generate`, allocate `min(prompt + max_new_tokens, block_size)` columns per layer once, and write each step into a slice: `k_past[:, :, past_len:total] = k`. Attention reads `k_past[:, :, :total]`, which is a view, not a copy.
   - The buffer is now bigger than what's been written, so each layer's cache needs a counter: `(k_past, v_past, past_len)`. Read the position from the counter, not from `k_past.size(-2)`.
   - Allocate in `generate`, not in attention, because only `generate` knows the final length.
   - Create the buffers with the weights' device and dtype (`lm_head.weight.device`, `.dtype`), so the cache switches to fp16 when the weights do in stage 5.
3. **Return the cache from `forward`** as `(logits, cache)`. Anything that calls `forward` directly needs updating. Code that goes through `generate()` doesn't.

The cache costs memory: for GPT-2, 12 layers × 2 (k and v) × 12 heads × 64 floats = 18,432 floats per token.

#### When the sequence outgrows block_size

GPT-2 has 1024 position rows, so the cache is full at 1024 tokens. There are two options:

- **Reset and re-prefill (exact):** delete the cache and run the last 1024 tokens through the model again. Every step after the limit costs a full 1024-token prefill.
- **Slide (approximate):** shift every cache buffer left by one column, dropping the oldest token, and keep going. Constant cost per step.

Sliding is approximate because each cached k and v already includes its token's position. `Embedding.forward` adds `pos_emb` into x at the start, and x goes through `ln_1` and `c_attn` to produce k and v. Example with `block_size = 4`: tokens A B C D fill the cache, then E arrives:

```
re-prefill:  B@0  C@1  D@2  E@3     positions 0..3, same as training
slide:       B@1  C@2  D@3  E@3     D and E both have position 3
```

After sliding, the cached positions start at 1 instead of 0, and every new token gets position 3 because the counter can't go past the last row. Positions are part of the q·k dot products, so the scores differ slightly from a fresh computation. I chose sliding because it only happens past 1024 tokens, the output degrades gradually, and re-prefilling every step is expensive. Leave a comment in `generate` saying it's approximate.

With RoPE (stage 6), scores depend only on the distance between tokens, so sliding doesn't have this problem.

### Shapes

```
x                         (B, 1, C)
q, k, v                   (B, nh, 1, hs)
k_past, v_past            (B, nh, past_len, hs)
k after write             (B, nh, past_len + 1, hs)      cache grows along dim -2 (the T dim)
q @ k.T                   (B, nh, 1, past_len + 1)       how much the one new token cares about each earlier token
att @ v                   (B, nh, 1, hs)
```

For the causal mask, the new token is the last row of the full triangle. With the 5-token prompt above, the first decode step has `total_t = 5 + 1 = 6`, so the mask slice is `tril[5:6, :6]`: 1 row, 6 columns, all ones. In code this is `tril_mask[:, :, total_t - current_t : total_t, :total_t]`, which gives the full triangle for prefill (`current_t = total_t`) and the last row for decode (`current_t = 1`).

### Testing

- Cached and uncached greedy generation must produce identical tokens. This catches off-by-one errors in positions and mask slicing.
- Run `test_greedy` again. `test_logits` won't catch cache bugs, because it doesn't use the cache.

### Benchmark

- Add the cached `generate` to `bench/run.py` next to the naive loop and run both at {16, 128, 512}.
- Keep the naive loop as a function in `bench/run.py`, not just as old numbers, so it can be re-measured next to every new version.

After benching: Throughput is now nearly flat across prompt lengths. That flatness is the signature of a working KV cache — the per-token cost no longer depends much on how long the context is, because each step processes one token instead of re-processing all of them.

Without the cache, each step runs all t tokens through the model and computes t × t attention scores. With it, each step runs 1 token and computes 1 × t scores.

#### Getting reliable numbers

These apply to every later stage.

**Storing results**

- Store one record per (mechanism, prompt length), with the mechanism's name in the record. Don't overwrite a baseline you can't re-measure.
- Keep a dict of name → function (`PATHS` in `bench/run.py`), so adding a mechanism is two lines and the timing loop doesn't change.

**Why numbers vary on a laptop**

- Heat: a long heavy run heats the chip, the chip slows itself down, and the next measurement is slower.
- Shared GPU: the M1 GPU also draws the screen and browser.
- Warm-up: the first call compiles GPU shaders and grows the memory allocator, so it's slower than the rest.

**How to get reliable numbers**

- Time one mechanism per process, with the machine idle and a cooldown between heavy runs.
- Run an untimed warm-up for each shape before timing.
- Call `sync()` before starting and before stopping the timer. GPU calls return before the GPU finishes, so without it you measure how long it took to queue the work.
- Keep every run's number, not just the average. The spread shows how much to trust the median.
- Compare how numbers change across prompt lengths, not single values. The shape of the curve stays the same even when individual numbers vary.
- Compare numbers from the same run when possible. When comparing across runs, re-measure the baseline too.

---

## 3. Static and continuous batching

**Goal:** run multiple sequences of different lengths in a single forward pass

After kv cache, we introduce batching. "Batching" isn't actually "new" per se, as we already have generate() that runs shape (b, t). What we are introducing here is how to handle batches of different length sequences, which will be solved with padding, since tensors must be rectangular

- Along with padding, comes masking the padding out of attention
- Note that since we want to access the last position's logits with `[:, -1, :]` this would mean all last logits need to line up and be in the last (-1th) column, which implies padding must be on the left side.
- With left padding, comes an adjustment of the starting position ids

### Implementation

#### Static batching

`generate_batch` takes N prompts, left-pads them to the length of the longest, builds a mask, and runs prefill and decode on the whole batch with one KV cache.

```
prompts: "Hello" (1 tok), "The meaning of life is" (5 tok), "Today is a" (3 tok)

ids                        mask           pos_ids = (cumsum(mask) - 1).clamp(min=0)
[PAD PAD PAD PAD Hello]    [0 0 0 0 1]    [0 0 0 0 0]
[The mea of  life is  ]    [1 1 1 1 1]    [0 1 2 3 4]
[PAD PAD Today is  a  ]    [0 0 1 1 1]    [0 0 0 1 2]
```

1. **Add an `attn_mask` argument to `GPT.forward`:** attention takes a (B, total_t) mask. `attn_mask[:, None, None, :]` reshapes it to (B, 1, 1, total_t) so it applies to every head and every query row. `None` inside an index adds a dimension of size 1, the same as `unsqueeze`. Set masked scores to `torch.finfo(dtype).min`, not `-inf`: a padding row is fully masked, and a row of all `-inf` gives NaN from softmax.
2. **Compute position ids from the mask:** a token's position is the number of real tokens before it in its row: the cumulative sum of the mask minus 1, clamped at 0 for the pads. During decode the mask covers the cache plus the new token, but only the new token is embedded, so keep the last t columns of `pos_ids`.
3. **Handle EOS at different times:** track finished rows with a `completed` bool tensor. After a row produces EOS, feed it `PAD_TOKEN` until every row is done or `max_new_tokens` is reached. Right-pad the output to (B, t_max + max_new_tokens).

The problem with static batching: it takes N requests, pads them, runs the whole generation, and returns all N when the longest one finishes. Short requests wait for the longest one, and a request that arrives after the batch starts waits for the whole batch.

There are two ways to batch prompts of different lengths:

1. **Pad to a rectangle (this project):** pad every prompt to the same length and mask the pads. Some compute is spent on pads, but it's simple to build.
2. **Flatten (vLLM/TGI):** no padding. All tokens go into one 1-D sequence, with extra tensors (positions, `cu_seqlens`) that record where each sequence starts. No compute is spent on pads, but attention needs a custom kernel (FlashAttention-varlen).

#### Continuous batching

Between decode steps, remove finished requests from the batch and put waiting requests in the free rows. This is the iteration-level scheduling from the ORCA paper, without vLLM's paged attention and chunked prefill.

`Engine` (`engine/scheduler.py`) holds the model, a fixed number of slots (rows), a KV cache of shape (n_slots, nh, max_len, hs) per layer shared by all slots, a mask of shape (n_slots, max_len), and the queue of waiting requests. Each call to `step()` does four things:

1. **Evict** finished requests: zero the row's mask, set its next token to PAD, mark the slot free.
2. **Admit** waiting requests into free slots.
3. Run **one decode step** for all slots at once.
4. **Pick, harvest, detect** for each running slot: pick the next token from the logits, append it to the request's output, check for EOS or the token limit.

```
        ┌───────────────────────── one step() ───────────────────────┐
 waiting│  EVICT      ADMIT            one batched     PICK/HARVEST/ │
 queue ─┼► finished ─► waiting ──────► decode forward ─► DETECT      ├─► next step()
        │  requests    requests,       (all slots,      per slot     │
        │  free slot   prefill each    one column)                   │
        └────────────────────────────────────────────────────────────┘
```

**Problem 1:** a newly admitted request has nothing in the cache. The other rows have different prompt lengths and have generated different numbers of tokens. Where does the new request's cache go?

**Solution:** all rows share one **frontier**, the column the next decode step writes to. Write a new request's prompt so that it ends at the frontier: columns `[frontier - P, frontier)`, where P is the prompt length. Everything to its left in that row is masked out, including the k/v left by the slot's previous request. Positions still come out right because they're computed from the mask, not from column numbers.

**Problem 2:** a new request needs its whole prompt run through the model (t = P), while the running rows only need their latest token (t = 1).

**Solution:** run them separately. On admission, run a prefill for the new request alone, with ids of shape (1, P), writing k/v into its slot's columns. After that, it decodes one token per step with the rest of the batch.

The mask decides which columns each row attends to. Any column layout works as long as each row's real tokens are in order and correctly marked in the mask.

#### Walkthrough

3 slots, `max_len = 10`. A (3-token prompt, max 4 new tokens) and B (2-token prompt, max 5 new tokens) are waiting at the first `step()`. C (2-token prompt, max 5 new tokens) arrives during step 1.

**Step 1:** no requests are running, so evict does nothing. Admit:

- Slot 0 is free and the batch is empty, so A sets the frontier: `frontier = P = 3`, the column right after its prompt. Its prefill writes columns 0-2 (counter = frontier − P = 0).
- Slot 1: B has P = 2 ≤ frontier = 3, so it's admitted. Its prefill writes columns 1-2 (counter = 3 − 2 = 1).
- Slot 2: the queue is empty. The row's mask stays all zeros and its next token is PAD.

Every model call is followed by pick + harvest + detect. A's prefill logits give `a3`, which goes into A's output and into `next_id[0]`. `next_id` holds the token each slot will feed into the next decode step. If the first pick is EOS, the request finishes without decoding.

```
col:          0    1    2    3    4    5    6    7    8    9
slot 0 (A):   a0   a1   a2   .                                   next_id = a3
slot 1 (B):   ·    b0   b1   .                                   next_id = b2
slot 2:       ·    ·    ·    ·                                   next_id = PAD
                             ▲ frontier = 3
```

Then set column 3 of the mask to 1 for every running row, and run one decode forward for all 3 slots with `next_id = [[a3], [b2], [PAD]]`. It writes column 3 in every row. Slot 2's write is masked out. The logits give `a4` and `b3`. `frontier += 1`.

**Step 2:** C arrives. Slot 2 is free and P = 2 ≤ frontier = 4, so C's prefill writes columns 2-3. Then one decode at column 4 for all three.

```
col:          0    1    2    3    4    5    6    7    8    9
slot 0 (A):   a0   a1   a2   a3   a4
slot 1 (B):   ·    b0   b1   b2   b3
slot 2 (C):   ·    ·    c0   c1   c2
                                  ▲ decode writes column 4, frontier -> 5
```

**Step 3:** decode at column 5. A picks `a6`, its 4th new token, so A is marked done.

**Step 4:** evict A. Nothing is waiting, so slot 0 stays empty. B and C decode at column 6. B reaches its limit.

**Steps 5-6:** evict B. C decodes at column 7 and reaches its limit. Evict C. Nothing is running or waiting, so `run()` returns.

#### Design choices

**When to record output tokens:** either slice a request's tokens out of the ids tensor when it finishes (option 1), or append each running request's new token to its `output_ids` every step (option 2). I chose option 2.

- With option 1, you have to calculate which columns of a reused row belong to the current request. The row also holds tokens from the slot's previous request, and that calculation is easy to get off by one.
- With option 2, the output is complete when the request finishes, so eviction only frees the slot.
- With option 2, a request's output so far can be read at any step, which the server uses to stream tokens (stage 4).

**When a waiting prompt is longer than the frontier:** say D has P = 6 and the frontier is 4. D's prompt would need to start at column −2.

1. **D waits (implemented):** admission requires `P ≤ frontier`. Otherwise D stays first in the queue. The frontier grows by 1 every step, so D waits at most P − frontier steps. If the batch empties first, the frontier resets and D is admitted immediately.
2. **Shift the running rows right:** also correct, because positions come from the mask. But it copies every layer's k/v buffers on each such admission.
3. **Move the frontier to P:** also correct, but every row loses P − frontier columns of capacity.


**Who sets the frontier when the batch is empty:** the first request admitted into an empty batch sets `frontier = P`. Every request after it needs `P ≤ frontier`. The capacity check uses the frontier the request would get (P if the batch is empty, otherwise the current frontier): `frontier + max_new_tokens + 1 ≤ max_len`.

### Shapes

Shapes in `generate_batch`, for B prompts padded to length `t_max`:

```
ids, mask                     (B, t_max)                  left-padded; mask grows by one column per step
k_past, v_past (per layer)    (B, nh, t_max + max_new, hs)
attn_mask[:, None, None, :]   (B, 1, 1, total_t)          broadcasts over heads and query rows
att                           (B, nh, current_t, total_t) prefill: current_t = t_max; decode: current_t = 1
pos_ids                       (B, total_t) -> (B, t)      keep the last t columns, the tokens being embedded
logits[:, -1, :]              (B, vocab_size)             every row's last real token is in the last column
next_id                       (B, 1)                      appended to ids each step
output                        (B, t_max + max_new_tokens)
```

Shapes in `Engine`:

```
k_buf, v_buf (per layer)      (n_slots, n_kv_head, max_len, hs)   shared by all slots, allocated once
mask                          (n_slots, max_len)                  1 = this slot attends to this column
next_id                       (n_slots, 1)                        the token each slot feeds next; PAD if empty

admit (one slot, prompt length P):
  prompt_ids                  (1, P)
  slot cache                  k_buf[slot:slot+1]  (1, n_kv_head, max_len, hs), counter = frontier - P
  slot mask                   mask[slot:slot+1, :frontier]  (1, frontier)
  logits                      (1, P, vocab_size) -> last position (1, vocab_size)

decode (all slots):
  input                       next_id  (n_slots, 1)
  att                         (n_slots, nh, 1, frontier + 1)
  logits                      (n_slots, 1, vocab_size) -> logits[slot:slot+1] per running slot
```

### Testing

- Each sequence in a batch must produce the same tokens it produces alone (greedy). This catches masking, cache-indexing and position-id bugs.
- `test_batching`: compare `generate_batch` against solo runs, with prompts of different lengths.
- Check `GPT.forward` with the new `attn_mask` argument on its own: a left-padded row must give the same logits as the same prompt without padding.
- `test_scheduler`: submit more requests than slots, with different prompt lengths and token limits, and compare every output against solo runs.

### Benchmark

- `bench/batch.py`: decode throughput (total tokens/sec) at batch sizes {1, 8, 16, 32, 64, 128}, and latency per request at each size
- Expect total throughput to rise almost in proportion to batch size at first. Each decode step reads every weight once no matter how many rows there are, and extra rows only add math, which is cheap until the GPU runs out of compute. Latency per request rises once it does.
- `bench/continuous.py`: give static and continuous batching the same set of requests with very different output lengths and fewer slots than requests. Measure each request's completion time, and the time until the last one finishes.
- Expect continuous to have a lower average completion time, because a short request doesn't wait for the longest one in its batch. The time until the last request finishes should be about the same, because the total number of tokens is the same.

---

## 4. Serving

**Goal:** a small serving layer over the engine. A request arrives, the engine admits it into the batch, it generates with the other requests, it's evicted when done, and its tokens are streamed back to its client as they're generated.

### Implementation

1. FastAPI app with `POST /generate`, streaming tokens with SSE (server-sent events)
2. Requests go into an asyncio queue, and the engine runs in its own loop.
3. Settings: max concurrent requests = `n_slots`, max context = `max_len`

```
 client ──POST /generate──► FastAPI handler ──Request──► inbox (asyncio.Queue)
 client ──POST /generate──► FastAPI handler ──Request──►   │
                                                           ▼
                                              ┌─ engine loop task ──────────────┐
                                              │  drain inbox → engine.submit()  │
                                              │  await to_thread(engine.step()) │◄── GPU thread
                                              │  route events to outboxes       │
                                              └─────────────────────────────────┘
                                                           │ (req_id, token, done)
                              one outbox per request ◄─────┘
 client ◄──SSE token stream── handler awaits its outbox, yields as tokens arrive
```

- `EngineLoop.run` waits on the inbox when there's nothing to do. Otherwise it moves new requests into the engine, runs one `engine.step()` in a separate thread, and puts each `(req_id, token, done)` result into that request's outbox. Running `step()` in a thread lets the event loop keep sending tokens to other clients while the GPU works.
- uvicorn starts and runs the event loop, so the server code has no `asyncio.run`.
- Each time a token arrives, decode the request's whole output so far and send only the new text. A single token can be half of a multi-byte character, which decodes on its own as `�`.

To check it works, start the server and send two requests from two terminals at the same time:

```bash
uvicorn server.app:app --port 8000
```

```bash
curl -N localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt": "Hello"}'
```

```bash
curl -N localhost:8000/generate -H 'Content-Type: application/json' -d '{"prompt": "The meaning of life is"}'
```

`-N` turns off curl's buffering. Both terminals should print tokens at the same time.

### Testing

- `tests/test_server.py`: send 20 requests at once and check that each output matches its solo generation and that no tokens end up in the wrong stream.

### Benchmark

- `bench/load.py` (asyncio): send requests at concurrency {1, 4, 8, 16}. Measure requests/sec, p50/p95/p99 latency, and time to first token.
- Compare the tokens/sec the server delivers against what the same number of busy slots can decode in `bench/batch.py`. The server will deliver less, for three reasons:
  1. A slot can stay empty while the next request waits for the frontier to reach its prompt length.
  2. Each admission runs a prefill for the new request alone, and the other requests don't decode during it.
  3. Each token goes through the async queue, HTTP and SSE before the client gets it.
- Expect throughput to stop rising once every slot is busy. After that, extra clients wait in the queue, so p95 latency rises quickly.

**Don't time mechanisms at the same time:** if you compare mechanisms by running them at the same time on one GPU, the results come out much closer together than they really are. Example: four jobs that take 1s, 1s, 1s and 6s alone share one processor equally. The three 1s jobs finish at about 4s (4x slower than alone), and the 6s job finishes at 9s (1.5x slower than alone). Run them one after another instead.

```
time:      0s     1s     2s     3s     4s     5s     6s     7s     8s     9s
           │      │      │      │      │      │      │      │      │      │
A (1s)     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ done at 4s   (4x)
B (1s)     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ done at 4s   (4x)
C (1s)     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│ done at 4s   (4x)
D (6s)     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒███████████████████████████████████│ done at 9s (1.5x)

▒ = sharing the GPU 4 ways (each job gets ¼)    █ = running alone
ratio alone 6:1  ->  ratio when shared 9:4 ≈ 2.3:1
```

**Static vs continuous on the same prompts:** if every prompt starts at the same time and generates the same number of tokens, static batching finishes first. Continuous does extra work: a separate prefill per request, scheduler bookkeeping every step, and a message per token. Continuous is faster to the first token, though, because each request starts streaming as soon as it's admitted. Continuous batching is faster overall when requests arrive and finish at different times.

### Try the dashboard

The server also serves a page that runs the four mechanisms (naive, cached, static, continuous) on the same prompts and draws each one's work per step. It runs on your own machine: GPT-2 is small enough for a laptop CPU or an Apple-silicon GPU.

```bash
git clone https://github.com/archangelinux/llm-inference.git
```

```bash
cd llm-inference && pip install -e .
```

```bash
uvicorn server.app:app --port 8000
```

Then open http://localhost:8000 and pick GPT-2 or Qwen3-0.6B from the dropdown. Qwen3 needs a few more GB of memory and takes longer to start. To load only GPT-2, start the server with `MODELS=gpt2` in front of the command. Click run twice, because the first run includes compiling GPU kernels.

<video src="media/llm-dashboard.mp4" controls muted loop playsinline width="100%"></video>

<details>
<summary><b>Why isn't cached faster than naive, and why is continuous slower than static?</b></summary>

**Cached vs naive:** the prompts are short, so there's almost nothing for the cache to save, and the two take about the same time. Either one can come out ahead from run to run.

- Each prompt is 5-20 tokens plus 40 generated, so naive's longest step runs about 60 tokens through the model. At that size, a step is mostly the cost of launching ~200 kernels from Python, whether it processes 60 tokens or 1.
- The cached path adds a few launches per step, because every layer writes the new k and v into the cache. At this size that extra cost is about the same as the recomputation it skips.
- With a prompt of a few hundred tokens, naive's work per step grows with the context and it falls far behind, while cached stays flat.

**Continuous vs static:** every prompt starts at the same time and generates the same number of tokens, which is the case static batching handles best.

- Static runs one prefill for all prompts and then decodes them together.
- Continuous runs a separate prefill for each prompt when it's admitted, does scheduler work every step, and sends each token to its client as a separate message. A prompt longer than the frontier also waits a few steps to be admitted.
- Continuous is faster to the first token, because each prompt starts streaming as soon as it's admitted. It's faster overall when requests arrive and finish at different times.
</details>

---

## Checkpoint: the bandwidth floor

Before optimizing further, calculate the fastest a decode step could possibly be. Every decode step reads every weight once, so:

```
minimum time per step = weight bytes / memory bandwidth
GPT-2 fp32 on an A10G: (124M params × 4 bytes) / 600 GB/s ≈ 0.83 ms
```

Then measure ms per token (1000 / tokens per second) and divide by the minimum. If the measured time is several times the minimum, the extra time isn't spent reading weights. Here it's launch overhead. Each layer's forward runs ~15-20 separate GPU operations, each started from Python, about 200 per token, and starting them takes longer than the math.

A **kernel** is a small program that runs on the GPU. When Python runs `out = x + y` on GPU tensors, PyTorch tells the GPU to run its addition kernel on the two arrays. Each PyTorch operation is one kernel launch, so a forward pass of ~200 operations is ~200 kernel launches.

You can also compare against llama.cpp on the same weights (`brew install llama.cpp`, download a GGUF of GPT-2, then `llama-bench -m gpt2.f16.gguf -p 512 -n 50 -r 3`). Expect to lose by more at decode than at prefill. Launch overhead is paid once per forward pass: prefill pays it once for the whole prompt, and decode pays it for every token. llama.cpp uses fused C++/Metal kernels, so it launches far fewer of them.

---

## 5. Precision: fp16, int8, and a Triton kernel

**Goal:** make each decode step read fewer bytes by storing the weights in fewer bits

PyTorch stores weights in fp32 by default (4 bytes each). Training needs that precision because gradient updates are tiny and get added up over millions of steps, so small errors compound. Inference runs one forward pass and picks a token, and the result only changes if a rounding error changes which logit is largest.

```
fp32: 1 sign + 8 exponent + 23 mantissa bits   ~7 significant digits, range ±3.4e38
fp16: 1 sign + 5 exponent + 10 mantissa bits   ~3 significant digits, range ±65,504
```

Decode time ≈ weight bytes / bandwidth + launch overhead. Fewer bytes per weight only reduces the first part.

### Implementation

#### fp16

1. Load the weights with `.to(DEVICE, DTYPE)` where `DTYPE=fp16` (instead of `model.half()`). The KV cache follows if its buffers use the weights' dtype.
2. Replace any large constants in masks. A mask value of `-1e9` becomes `-inf` in fp16 (the largest fp16 value is 65,504), and fully masked rows become NaN after softmax. Use `torch.finfo(dtype).min`.

Before measuring, predict the result: half the bytes halves the bandwidth floor, but the launch overhead stays the same, so the total only drops by the floor's share of the step (Amdahl's law).

Exact-match tests stop working at fp16. GPT-2's logits are around 100 and fp16 keeps about 3 significant digits, so an error of ~0.3 per logit is normal. When the top two tokens' logits are closer than that, greedy can pick the other one, and every token after it is different. Keep exact match for fp32, loosen the logit threshold for fp16, and add the quality tests below.

#### int8

int8 stores whole numbers from -128 to 127. A weight like 0.0234 can't be stored as an int8, but it can be stored as an int8 `q` and a float **scale** `s` (the step size between values) such that `w ≈ q × s`:

```
s = max|w| / 127
q = round(w / s)
w ≈ q × s          error ≤ s / 2
```

Use one scale per output row (**per-channel**), not one per matrix. With one scale per matrix, the single largest weight sets the step size for every weight, and rows of small weights lose most of their precision. `quantize_weight` computes `w.abs().max(dim=1)`.

The scheme is **W8A16**: int8 weights, fp16 activations.

```
                       W8A16
             ┌───────────┴────────────┐
       int8 weights             fp16 activations
             │                        │
  quantized once at load       computed every forward pass;
  half the bytes of fp16       a few KB per layer at b=1,
                               tiny next to the weights
```

- Quantize the weights once at load. `quantize_model` replaces every `nn.Linear` except `lm_head` with a `QuantLinear`. `lm_head` shares its weights with the embedding table, so it stays fp16, and so do the norms.
- **W8A8** (int8 activations too) would barely reduce memory reads at decode, because one token's activations are a few KB per layer compared to megabytes of weights. It would make the math faster, but at b=1 the math is a tiny part of each step (a GPT-2 decode step is ~250M FLOPs, ~8 µs on an A10G). It matters for prefill and large batches.
- Activation values depend on the input and have outliers (the LLM.int8() paper), so int8 activations would need their scale recomputed for every input.

`QuantLinear.forward` first gets a slow version: convert the weights back to fp16 (`q.to(x.dtype) * scale`) and call `F.linear`. It writes a full fp16 copy of the weights to memory and reads it back, so it's slower than plain fp16. Its job is to be the correct answer the fast kernel gets checked against.

#### The fused dequant-matmul kernel

<details>
<summary><b>What are CUDA, Triton and cuBLAS?</b></summary>

- **CUDA** is NVIDIA's platform for programming its GPUs. CUDA kernels are usually written in CUDA C++, where you control what each GPU thread does.
- **Triton** is a Python-based language (from OpenAI) for writing GPU kernels. You write the code for one block of data, and the compiler handles the thread-level details, like how data is loaded from memory and when to use the GPU's matrix-multiply hardware. It compiles to the same kind of GPU machine code as CUDA. Triton runs on NVIDIA and AMD GPUs but not on Apple's, so this stage needs an NVIDIA GPU. I rented one in the cloud.
- **cuBLAS** is NVIDIA's library of prewritten, heavily tuned matrix-multiply kernels. When PyTorch runs a matmul on an NVIDIA GPU, it usually calls cuBLAS.
</details>

Each PyTorch kernel does one operation and writes its result to GPU memory. In the slow int8 version, `q.to(x.dtype) * self.scale` is one kernel that writes a full fp16 weight matrix to memory, and the matmul is a second kernel that reads it back. A custom kernel can do both in one launch, so the fp16 weights are never written to memory. That's what **fused** means.

PyTorch has one kernel per operation, and there are too many combinations to ship a fused kernel for each. `torch.compile` fuses simple chains of elementwise operations, but not an int8-to-fp16 conversion followed by a matmul. Libraries do have W8A16 kernels (bitsandbytes, GPTQ/AWQ, Marlin in vLLM). I wrote my own because the point of this project is to build each part. Decode at b=1 is limited by memory reads, so a simple kernel that reads each int8 weight once can get a large fraction of the maximum bandwidth.

Writing it (`kernels/dequant_matmul.py`):

1. Each program computes one (BLOCK_M × BLOCK_N) tile of the output.
2. Loop over K in chunks of BLOCK_K: load a chunk of int8 weights, convert it to fp16 in registers, and add `tl.dot(a, b)` to an fp32 accumulator.
3. After the loop, multiply by the per-row scales once (the scale is the same for every k in a row, so it can be applied after the sum), then store.
4. Add the bias in PyTorch afterwards.

### Shapes

Quantizing one weight matrix:

```
w        (N, K)    e.g. c_attn: (2304, 768), stored (out_features, in_features)
row max  (N, 1)    max |w| along each row
scale    (N, 1)    fp16, row max / 127
q        (N, K)    int8, round(w / scale)
```

The kernel:

```
x                    (..., K)          e.g. (4, 10, 768)
x2 = x.reshape(-1, K) (M, K)           (40, 768): every token is one row
q                    (N, K) int8       read through swapped strides as (K, N), so no transpose is copied
scale                (N,)              one per output column of the result
out                  (M, N)            (40, 2304)
grid                 (M / BLOCK_M, N / BLOCK_N) programs, rounded up
result               out.reshape(..., N)   (4, 10, 2304)
```

### Testing

- `|q × scale − w|` is at most half a step.
- `QuantLinear` gives nearly the same output as the `Linear` it replaced.
- The quantized model's logits stay close to the fixtures.
- Compare the kernel against the slow version on every layer shape in the model.

#### Quality gates

Once precision is lossy, measure quality directly. `eval/quality.py` runs each precision over a slice of WikiText-2 in 512-token chunks and reports:

- **Perplexity:** how surprised the model is by real text. Lower is better. Compare fp16 and int8 against fp32 on the same chunks. Each chunk starts with no context, so the numbers won't match published ones, but they can be compared to each other.
- **Top-1 agreement:** how often the variant picks the same next token as fp32
- **Max centered logit difference:** the largest logit difference from fp32 after subtracting each position's mean logit

Center the logits before comparing them. Softmax only depends on the differences between logits, so adding the same number to every logit doesn't change the output. Quantization can do exactly that. The logits are `h · (every token embedding)`, and the trained embeddings all share a large common part. Example with 2-number vectors and 3 tokens:

```
cat = (10, 10) + ( 0.3, -0.1) = (10.3,  9.9)
dog = (10, 10) + (-0.2,  0.4) = ( 9.8, 10.4)
pie = (10, 10) + ( 0.1,  0.2) = (10.1, 10.2)
       shared       unique

fp32  h  = (5.0, 5.0):   cat 101.00   dog 101.00   pie 101.50
int8  h' = (5.2, 5.2):   cat 105.04   dog 105.04   pie 105.56
```

h changed by +0.2 in each number. Multiplied by the shared (10, 10), that adds +4 to every score, while the differences between scores change by ~0.06. An uncentered comparison would report an error of 4 for an output that's almost unchanged.

Use perplexity and top-1 agreement as the pass/fail tests, and the centered logit difference for debugging.

### Benchmark

- **Don't time it with a Python loop:** launching a Triton kernel from Python takes ~20 µs, and the kernel itself takes a few µs, so a Python loop measures the launch. If every block size gives the same time, you're measuring the launch.
- **Time it with a CUDA graph:** record 100 launches in a graph and time replaying it (`kernels/sweep_graph.py`). That removes the launch cost.
- **Don't reuse one weight tensor:** a small weight fits in the GPU's L2 cache, so repeated reads come from the cache and look faster than memory allows. Cycle through several copies of the weight (I used 8).
- **Report achieved bandwidth:** bytes read and written / time. Compare it against the GPU's peak, and compare the time against cuBLAS fp16 (the stock kernel a server would use), not fp32.
- **Choose block sizes so every SM has work:** at decode M = 1, so the grid is `N / BLOCK_N` programs. With N = 2304 and BLOCK_N = 64 that's 36 programs on a GPU with 72 SMs, and half the GPU sits idle. BLOCK_N = 32 gives 72.

Expect the kernel to be faster than cuBLAS fp16 per matmul, because it reads half the bytes, and slower for the whole decode step. Every matmul is now a Triton launch, and each one costs more from Python than the GPU work it starts. That's the launch overhead from the bandwidth check again, and stage 7 removes it.

---

## 6. Porting a modern architecture (Qwen3-0.6B)

**Goal:** implement the architecture changes between GPT-2 (2019) and Qwen3 (2025) by hand, keeping everything else the same. `engine/qwen.py` is a copy of `model.py` with the changed parts replaced. The KV cache, `generate`, `generate_batch`, the scheduler, the server and quantization work unchanged.

Qwen3-0.6B-Base: 28 layers, hidden size 1024, 16 q-heads and 8 kv-heads, head_dim 128, vocab 151,936, 32k context. head_dim is 128 even though 1024 / 16 = 64, so q projects to 16 × 128 = 2048, k and v to 8 × 128 = 1024, and `o_proj` maps 2048 → 1024.

| GPT-2 | Qwen3 |
|---|---|
| LayerNorm | RMSNorm |
| GELU MLP | SwiGLU |
| learned position table `wpe` | RoPE |
| 12 q, 12 k/v heads | GQA: 16 q, 8 k/v |
| — | QK-Norm |

### Implementation

#### 1. Weight-name mapping

Rename Qwen's checkpoint keys (`model.layers.N.self_attn.q_proj`, `input_layernorm`, ...) to your module names. No transposes this time, because Qwen uses `nn.Linear`, not Conv1D.

#### 2. LayerNorm → RMSNorm

```python
rms = sqrt(mean(x²) + eps)
out = x / rms * weight
```

RMSNorm divides by the root mean square and scales. Unlike LayerNorm it doesn't subtract the mean and has no bias. Compute it in fp32 and convert back, because squaring fp16 values can overflow.

#### 3. GELU MLP → SwiGLU

GPT-2's MLP is `down(gelu(up(x)))`: make the token's vector wider, put every value through GELU, and make it narrow again. Whether a value survives depends only on that value: GELU keeps positive values and shrinks negative ones toward 0.

Qwen's MLP (SwiGLU) is `down(silu(gate(x)) * up(x))`:

1. **`up_proj(x)`:** make the vector wider (1024 → 3072 values), the same as GPT-2's first layer. These are the values the MLP wants to pass on.
2. **`gate_proj(x)`:** a second, separate matrix that also turns x into 3072 values. Each row of `gate_proj` is a learned question asked of the token's vector, and its answer decides how much of the matching `up` value gets through.
3. **SiLU on the gate values:** SiLU is `x * sigmoid(x)`, almost the same curve as GELU. A large positive answer stays large, and a negative answer becomes close to 0.
4. **Multiply element by element:** each `up` value is multiplied by its gate value. A gate near 0 removes that value, and a large gate keeps it.
5. **`down_proj`:** make the vector narrow again (3072 → 1024).

With 3 values instead of 3072:

```
up(x)          = [ 5.0,  -2.0,  3.0 ]
silu(gate(x))  = [ 0.9,   0.1,  0.0 ]
product        = [ 4.5,  -0.2,  0.0 ]   first kept, second reduced, third removed
```

The third value was 3.0, and GELU would have kept it. In SwiGLU it's removed, because a different part of the network (`gate_proj`) decided this token shouldn't pass it on. The model learns two things separately: what to compute (`up_proj`) and when to let it through (`gate_proj`). Multiplying one learned signal by another like this is called **gating**.

Naming: GLU (gated linear unit) is the multiplication, and the prefix says which function is applied to the gate: sigmoid → GLU, GELU → GEGLU, SiLU (also called Swish) → SwiGLU. In the paper that introduced them (Shazeer 2020), the gated versions all beat non-gated MLPs, and SwiGLU and GEGLU scored about the same. Most models use SwiGLU because PaLM and then Llama did, and because SiLU is cheaper to compute than exact GELU.

Qwen's checkpoint has three MLP matrices, `gate_proj`, `up_proj` and `down_proj`, trained to be used this way. Three matrices have more parameters than two, so Qwen uses a 3x hidden size instead of GPT-2's 4x (`intermediate_size` = 3072 = 3 × 1024).

#### 4. Learned positions → RoPE

1. **Why positions are needed:** attention has no information about order. q·k only compares content, so attention never sees position directly.
   - GPT-2 adds a learned position vector (`wpe`) to each token's embedding. Content and position are then fused into one vector, and q·k gets no direct information about how far apart two tokens are. Any notion of distance has to be reconstructed by the model.
   - GPT-2 stores 1024 position vectors (1024 × 768 numbers). Positions beyond 1023 don't exist.
2. **The idea:** instead of adding anything to the embedding, rotate q and k by an angle proportional to their position, right before scoring. Position p rotates by p·θ: position 0 by 0, position 1 by θ, position 2 by 2θ, and so on.
   - q and k themselves contain no position at all. If q is at position m and k is at position n, after rotating them their dot product depends only on the angle between them, (m − n)θ, which is how far apart they are.
   - Position is applied inside every attention layer, not once at the bottom.
   - Qwen can handle 32k context because it doesn't need a 32k-row table (32768 × 1024 = 33M parameters).
3. **Rotating a 128-number vector:** split each head's 128 numbers into 64 pairs. Each pair is a 2D arrow that rotates at its own speed:
   ```python
   inv_freq[i] = rope_theta ** (-2i / 128)   # theta = 1,000,000; pair 0 turns 1 radian per position, later pairs turn slower
   angle       = p * inv_freq[i]
   ```
   A fast pair tells nearby positions apart precisely, but it wraps around and becomes ambiguous over long distances. The slow pairs cover long distances. Like hands on a clock.
4. **Code:** `inv_freq` is computed once. Each forward computes the angle `p · inv_freq[i]` for every position p and pair i from `pos_ids`. Apply the rotation in attention's forward to q and k, after the projections, head split and QK-Norm, and before `q @ k.T`. v is not rotated. Rotating one pair (a, b) by θ:
   ```
   a' = a·cosθ − b·sinθ
   b' = a·sinθ + b·cosθ
   ```
   To do this to all 64 pairs at once, store pair i as `(x[i], x[i + 64])` and use `rotate_half`:
   ```
   x              = [ a0,  a1, ...,  a63,  b0,  b1, ...,  b63]
   rotate_half(x) = [-b0, -b1, ..., -b63,  a0,  a1, ...,  a63]
   cos            = [cosθ0, ..., cosθ63, cosθ0, ..., cosθ63]   each angle duplicated
   sin            = [sinθ0, ..., sinθ63, sinθ0, ..., sinθ63]

   x * cos + rotate_half(x) * sin
   ```
   Element 0 gives a0' = a0·cosθ0 + (−b0·sinθ0), and element 64 gives b0' = b0·cosθ0 + a0·sinθ0, which is the rotation formula for pair 0.

This matters because a head is one set of weights applied to every position. Trained models have heads that do things like "attend to the previous token". With RoPE, "1 back" is the same angle difference at position 7 as at position 30,000, so the head learns it once. With `wpe`, "1 back" at position 7 uses table rows 7 and 6, and at position 800 it uses rows 800 and 799, so training has to make every pair of rows relate the same way.

- Positions still come from the cumsum of the mask, the same as GPT-2's `pos_ids`. Pass them to attention instead of the embedding layer.
- Build the causal mask on every call: `ones(current_t, total_t).tril(diagonal=total_t - current_t)`. A precomputed `tril` matrix for 32k context would be 32768² floats, 4.3 GB per layer.
- Test the relative-position property directly: rotating q and k by (m, n) and by (m + s, n + s) must give the same score.

#### 5. GQA (grouped-query attention)

Qwen3 has 16 q-heads but only 8 k-heads and 8 v-heads. Each k/v head is shared by 2 q-heads:

```
q-heads:   0  1   2  3   4  5   ...  14 15
            \/     \/     \/          \/
kv-heads:   0      1      2    ...    7
```

The reason is the KV cache size: `layers × 2 × n_kv_head × head_dim × context`. q isn't cached. Halving the k/v heads halves the cache and the cache memory read each step, so twice as many tokens or requests fit, with very little quality loss.

`q @ k.T` needs k to have as many heads as q, so right before scoring, `repeat_interleave(2, dim=1)` copies each k and v head twice (8 heads → 16). The copy is temporary, and the cache stores only the 8 heads. Allocate the engine's cache buffers with `n_kv_head`, and give the GPT-2 config `n_kv_head = 12` so the scheduler works for both.

#### 6. QK-Norm

Every layer already normalizes before attention and before the MLP. QK-Norm adds two more RMSNorms inside attention: one on q and one on k, after the token has been projected to q and k and split into heads, and before RoPE.

The input to attention was already normalized, so why normalize again? `q_proj` and `k_proj` are free matrices that can stretch a unit-size input to any size, so q and k, and their dot products, can get very large. Normalizing them keeps the attention scores in a stable range.

### Shapes

MLP:

```
x                          (B, T, 1024)
gate_proj(x), up_proj(x)   (B, T, 3072) each
silu(gate) * up            (B, T, 3072)
down_proj                  (B, T, 1024)
```

Attention:

```
x                                  (B, t, 1024)
q_proj(x).view.transpose           (B, 16, t, 128)
k_proj(x), v_proj(x) (same)        (B, 8, t, 128)
q_norm(q), k_norm(k)               same shapes, RMSNorm over the last dim (128)
angles = pos_ids * inv_freq        (B, 1, t, 64)      one angle per pair, shared by all heads
cat(angles, angles) -> cos, sin    (B, 1, t, 128)
rope(q), rope(k)                   (B, 16, t, 128), (B, 8, t, 128)
write k, v into cache              k_past, v_past (B, 8, max_len, 128)
k.repeat_interleave(2, dim=1)      (B, 16, total_t, 128)   same for v; temporary
att = q @ k.T / sqrt(128)          (B, 16, t, total_t)
att @ v                            (B, 16, t, 128)
transpose + view                   (B, t, 2048)
o_proj                             (B, t, 1024)
```

### Testing

- Generate fixtures from HF's Qwen3: `python tests/make_fixtures.py Qwen/Qwen3-0.6B-Base`.
- Logits match HF's in fp32, greedy generation is token-identical, batched == solo, and the continuous-batching engine == solo.

### Benchmark

Rerun every benchmark with `MODEL=qwen`.

---

## 7. CUDA-graph decode

**Goal:** remove the per-kernel launch cost from every decode step

You may have noticed that stage 5 was underwhelming. fp16 halved the bytes read per step and decode barely got faster. The int8 kernel was faster than cuBLAS for each matmul and made the whole decode step slower. Both have the same cause, and it's the one the bandwidth check found: most of each decode step is spent launching kernels from Python, not reading weights.

- fp16 and int8 only shrink the time spent reading weights, which was a small part of each step to begin with.
- Every PyTorch operation is a separate kernel launch, and each launch costs roughly 20 µs of Python work before the GPU starts, often longer than the kernel itself runs. A decode step is ~200 launches for GPT-2 and ~1,000 for Qwen3, and the GPU sits idle between them.
- A Triton launch costs more from Python than a cuBLAS launch, so swapping cuBLAS matmuls for the int8 kernel added launch cost even though each kernel ran faster.

Fewer bytes only speeds up decode once the launch cost is gone, and CUDA graphs remove it.

A **CUDA graph** is a recording of GPU work. You record one decode step, and the GPU stores the list of kernels it ran and the memory addresses each one read and wrote. `replay()` runs that list again without Python, so the hundreds of launches per step become one.

```
eager:   CPU  │launch│launch│launch│launch│ ... ×200
         GPU      █      █      █      █          (idle between kernels)

graph:   CPU  │replay│
         GPU   ████████████ ... ×200 back to back
```

This works for decode because every decode step runs the same kernels on tensors of the same shapes. Only the values change: the input token, the mask, the cache contents. Kernels read their inputs from memory when they run, so if the new values are written into the same memory before each replay, the replay computes the next step.

### Implementation

A CUDA graph saves the exact shapes and memory addresses it ran on, and replays them unchanged. The work is making every decode step look identical to the GPU, so one recording works for all of them.

1. **Attend over the whole cache every step:** the eager decode step sliced `mask[:, :frontier+1]` and `k_past[:, :, :total]`, and both grew by one column per step. A recording can't change shapes, so pass the full mask (n_slots, max_len) and have attention read all `max_len` columns of the cache. The mask is already 0 for every column after the frontier, so those columns get no attention weight.
2. **Make the write position a GPU tensor:** `k_past[:, :, past_len:total] = k` uses a Python int, and the recording would save that number, so every replay would write the same column. Create a GPU tensor `frontier_t` once, and write the new k and v with `k_past.index_copy_(2, frontier_t, k)`. That reads the column from `frontier_t` on every replay. `self.frontier`, the Python int, is still used by the scheduler.
3. **Update inputs in place:** before each replay, write into the same tensors (`mask[s, f] = 1`, `next_id[s] = pick`, `frontier_t.fill_(f)`). Never replace one with a new tensor (`mask = torch.cat(...)`), because the recording would keep reading the old one's address. The scheduler's mask, `next_id` and KV buffers already worked this way.
4. **Allocate the cache with `torch.zeros`, not `torch.empty`:** attention now reads every column, including ones never written, and `torch.empty` can leave NaN in them. Masked columns get a softmax weight of 0, but 0 × NaN is NaN.
5. **Put the decode forward in its own function:** `_decode_forward()` runs `logits = model(next_id, kv_past, mask)` and nothing else. Picking a token (`.item()`), checking EOS, eviction and admission are Python decisions and can't be recorded, so they stay outside it.
6. **Record it on the first decode step:** call `_decode_forward()` once normally, so Triton compiles its kernels and cuBLAS initializes outside the recording. Then `torch.cuda.synchronize()`, and record:
   ```python
   self.graph = torch.cuda.CUDAGraph()
   with torch.cuda.graph(self.graph):
       self.static_logits = self._decode_forward()   # recorded, not run
   ```
7. **Replay it on every step after that:** fill the inputs, call `self.graph.replay()`, and read the result from `self.static_logits`, which is the same tensor every time. Each step becomes:
   ```
   Python (evict, admit, set mask column, frontier_t.fill_)  →  replay  →  Python (pick, harvest, detect, frontier += 1)
   ```

Only decode is recorded. Prefill has a different shape for every prompt length, so it runs normally. CUDA graphs are CUDA only. On a Mac, the same fixed-shape `_decode_forward()` runs without recording, which lets the correctness tests check it locally.

### Shapes

The recorded decode step, with every shape fixed:

```
next_id                       (n_slots, 1)                       filled in place each step
mask                          (n_slots, max_len)                 full width, never sliced
frontier_t                    (1,) long, on the GPU              the column to write
k (new token)                 (n_slots, n_kv_head, 1, hs)
k_past.index_copy_(2, frontier_t, k)  writes into (n_slots, n_kv_head, max_len, hs)
att                           (n_slots, nh, 1, max_len)          columns past the frontier are masked
static_logits                 (n_slots, 1, vocab_size)           same tensor every replay
```

### Testing

On an NVIDIA GPU, run the engine tests with the graph on, and check that the graphed engine matches solo generation for both models and with the int8 kernel.

### Benchmark

Time ms per decode step with and without the graph, for each model and precision. Expect each step to be several times faster. Once the launch cost is gone, the variants that read fewer bytes per step become faster, which is when the int8 kernel finally pays off. Compare the result against the bandwidth floor again to see how much overhead is left.

---

## What's next

Part 1 ends with decode still several times slower than the bandwidth floor, and with every slot reserving `max_len` columns of cache whether it uses them or not. Part 2 covers:

- **Speculative decoding (fewer forward passes):** a small model guesses several tokens and the large model checks all of them in one forward pass.
- **Fused kernels (fewer kernels):** RMSNorm, RoPE, SiLU × up and QK-Norm as single Triton kernels.
- **Paged KV cache (less wasted memory):** fixed-size blocks per request instead of `max_len` columns per slot, so more requests fit.
