# llm-inference-by-parts

GPT-2 inference engine built from scratch in PyTorch. The stages of implemenation are as follows, with each stage verified against it's previous and benchmarked:
1) forward pass
2) KV cache
3) static + continuous batching
4) async serving layer (FastAPI)

Future:
5) Quantization + GPU kernel
6) Porting to Qwen (RoPE, GQA, SwiGLU, RMSNorm) 


![decode throughput stage by stage](bench/progression.png)

Each engine optimization, measured (`bench/chart_progression.py`): the KV cache
is worth ~60x at 512-token context on my Macbook Air (naive full-recompute is
pathologically slow under MPS; on an A10G the same gap is 3.4x; the cache's real
product is decode that doesn't slow down with context, see the A10G section);
batching is another ~12x on top (decode is memory-bound, so batch rows ride the same
weight-stream nearly free). Continuous batching and the serving layer improve
different metrics — per-request completion time and delivered throughput, respectively —
and are measured on their own axes in the benchmark sections below.

## To run

```
pip install -e . && pip install fastapi uvicorn
uvicorn server.app:app --port 8000     # then open http://localhost:8000/ in two tabs to gernearte streams in both over one shared batch
```

## Architecture

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

Inside `engine.step()` — ORCA-style iteration-level scheduling (continuous batching):

```
        ┌──────────────────────── one lap ───────────────────────────┐
        │                                                            │
 waiting│  EVICT      ADMIT            one batched     PICK/HARVEST/ │
 queue ─┼► finished ─► newcomers ────► decode forward ─► DETECT      ├─► next lap
        │  rows free   solo-prefill    (all slots,      per slot     │
        │  their slot  into the slot   one column)                   │
        └────────────────────────────────────────────────────────────┘

 shared KV cache: (n_slots, n_head, max_len, head_size) per layer -- slots are
 recycled; stale k/v from evicted tenants stays in place, masked off forever.
 the attention mask (n_slots, max_len) is the single source of truth for what
 exists; position ids derive from it by cumsum.
```

## Correctness chain

Every layer is tested against the layer below it, anchored to HuggingFace at the bottom:

```
HF fixtures ─► forward pass ─► solo generate ─► static batch ─► EOS ─► engine ─► HTTP server
(test_logits)  (max err ~1e-4)  (test_greedy)   (test_batching)        (test_scheduler)
                                                                       (test_server: 20
                                                                        concurrent, exact)
```

Eeach sequence in a batch is checked to produce exactly the tokens it produces alone (greedy). This catches any masking, cache-indexing, and position-id bugs.

## Benchmarks (M-series MacBook Air, MPS)

**Decode throughput vs batch size** (`bench/batch.py`, median of 5, per-size
warmup). Extra batch rows are nearly free until compute saturates: decode is
memory-bound, so all rows share one read of the weights.

| b | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| tok/s | ~40 | ~113 | ~513 | ~908 | ~1460 | ~1998 |

Potential MPS platform quirk (reproducible across reruns) per-row throughput drops from ~40
tok/s at b=1 to ~14 at b=8, then jumps to ~32 at b=16. Likely that MPS uses slow
matmul kernels for mid-size batch shapes and a better path from b=16 up. The
A10G sweep below doesn't have this.

**Continuous vs static batching** (`bench/continuous.py`): staggered workload,
3 slots. Mean completion time is 1.5-1.8x better under continuous scheduling
because short requests stop waiting behind long ones.

![continuous vs static completion times](bench/continuous.png)

Makespan is unchanged — it's a scheduling win, not a speed win. On the A10G the
same experiment is a wash (0.31s vs 0.32s mean): laps are ~15x faster there, so
with a small model and 8 requests there's barely any waiting to reclaim, and the
scheduler still pays its per-lap overhead. The win scales with how long requests
actually wait, which is why it matters for real serving (big models, slow laps,
constant arrivals) and not here.

**Serving under load** (`bench/load.py`, 4 slots):

![load results](bench/load_results.png)

Throughput scales until the slots fill (~c=8), then stalls while p95 latency
blows up — added clients past that point just wait.

Peak delivered throughput is ~48 tok/s against the 4-slot raw decode ceiling of
~88 (dashed line). The gap is utilization: empty slots while queued requests
wait to be admitted, prefill stalls on every admission, and the asyncio/HTTP/SSE
overhead per token.

## Layout

```
engine/model.py        GPT-2 from scratch: forward, KV cache, generate, generate_batch
engine/scheduler.py    Engine: continuous batching (slots, frontier, admit/evict)
server/engine_loop.py  async bridge: inbox -> engine loop task -> per-request outboxes
server/app.py          FastAPI: POST /generate (SSE), browser demo page at /
tests/                 the correctness chain (pytest tests/)
bench/                 throughput, latency, and load benchmarks + results
```

## On NVIDIA A10G via Modal

Same engine and protocol on an A10G (`bench/modal_bench.py` ->
`modal_results.json`):

| b | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| total tok/s | 159 | 1,210 | 2,414 | 4,909 | 8,789 | 16,428 |
| per-row tok/s | 159 | 151 | 151 | 153 | 137 | 128 |

![A10G results](bench/modal.png)

Per-row throughput stays flat and declines gently, which is the expected memory-bound
shape, without the cliff seen with MPS.

Naive vs cached is also cleaner here: naive is actually faster at a 16-token
prompt (201 vs 186 tok/s), decays to 52 tok/s at 512 context, and keeps getting
worse with length. Cached holds ~180 tok/s at any length, showing decode speed that doesn't depend on context length as expected.

**Serving under load on the A10G** (`bench/modal_load.py`, loopback): delivered
throughput plateaus at ~156 tok/s from c=8 (TTFT 55ms at c=1; p95 2.8s at
c=16). That's ~25% of the 4-slot raw ceiling (~634 tok/s), worse than the M1's
~55% — the scheduling overheads (admission waits, prefill stalls, async/HTTP
hops) are fixed costs, so on faster hardware they eat a bigger share of each
second.

**Bandwidth check**: each decode step has to read all 124M x 4bytes ~= 0.50 GB of fp32 weights,
and the A10G moves ~600 GB/s, so a step can't take less than 0.5GB / 600GB/s = 0.83 ms. 
Measured:
5.75 ms/token, 6.9x that floor. The difference is per-op overhead — each layer
runs as many separate PyTorch kernel launches driven from Python. (5.75 ms is
the single-stream `model.generate` path; the batch table's b=1 of 159 tok/s
goes through `generate_batch`, which adds mask/position bookkeeping.)

## vs llama.cpp

Macbook Air with same GPT-2 weights (GGUF from `mradermacher/gpt2-GGUF'), single stream and greedy.

| | this repo (fp32) | llama.cpp f16 | llama.cpp Q8_0 |
|---|---|---|---|
| decode tok/s (b=1) | ~38-50 | 179 | 256 |
| prefill tok/s (512-token prompt) | ~4,000 | 8,825 | 8,162 |

We lose ~4-5x on decode but only ~2x on prefill. 
In ours, Python/dispatch overhead (fixed cost of a forward pass) for prefill is a single forward pass spread over 512 tokens of matmul, whereas decode runs one forward per token.
llama.cpp avoids this overhead with fused C++/Metal kernels (which reduces setup cost / launch overhead) and it's Q8_0 quantization format, which is 1.4x faster than f16 at decode because its weights are half the size. The bottleneck for decode is the memory reads, so less bytes per step means more tokens per second. Prefill is still limited by compute.

This comparison is single-stream only. Both engines can batch; we only
benchmarked ours, so the table above measures per-stream speed, not total
throughput.

Reproduce: `brew install llama.cpp`, download the GGUF, then
`llama-bench -m gpt2.f16.gguf -p 512 -n 50 -r 3`.

## Limitations

Deliberate scope cuts:

- **No custom kernels** — every decode step pays per-op Python dispatch.
- **fp32, no quantization** — 2-4x more memory traffic than fp16/int8, and
  decode is memory-bound. (planned future step)
- **Pad-and-mask batching, not paged attention** — recycled cache slots waste
  columns on masked-off junk, and the shared frontier burns a column per lap.
- **No chunked prefill** — admitting a long prompt stalls all running streams
  for one forward pass.
- **No per-request cancellation** — a disconnected client's request generates to
  budget anyway.

## Scripts

```
pytest tests/            # the correctness chain (slow: real model, real server)
python bench/run.py      # naive vs KV cache          -> results.jsonl
python bench/batch.py    # throughput vs batch size   -> batch_results.jsonl
python bench/continuous.py  # continuous vs static    -> continuous_results.jsonl
python bench/load.py     # serving under load         -> load_results.json
python bench/chart_progression.py && python bench/chart_load.py && python bench/chart_continuous.py && python bench/chart_modal.py
```
