# llm-inference

GPT-2 inference engine built from scratch in PyTorch — forward pass, KV cache,
static + continuous batching, and an async serving layer — each stage verified
against the one before it and benchmarked on its own numbers.

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

The load-bearing test: **each sequence in a batch produces exactly the tokens it
produces alone (greedy)** — catches masking, cache-indexing, and position-id bugs.

## Benchmarks (M-series MacBook Air, MPS)

**Decode throughput vs batch size** (`bench/batch.py`): below saturation, extra
batch rows are nearly free — decode is memory-bound, all rows ride the same weight
stream. Compute saturates around b=32–64; past it, throughput is paid for in latency.

| b | 1 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| tok/s | ~25 | ~143 | ~552 | ~944 | ~1541 | ~1945 |

**Continuous vs static batching** (`bench/continuous.py`): staggered workload,
3 slots — mean completion time 1.5–1.8× better under continuous scheduling, because
short requests stop waiting behind long ones. Makespan unchanged (it's a
scheduling win, not a speed win).

**Serving under load** (`bench/load.py`, 4 slots):

![load results](bench/load_results.png)

Throughput scales until the slots saturate (~c=8), then queueing takes over:
req/s stalls while the p95 tail explodes — TTFT degrades first, the early-warning
metric of an overloaded service.

## Layout

```
engine/model.py        GPT-2 from scratch: forward, KV cache, generate, generate_batch
engine/scheduler.py    Engine: continuous batching (slots, frontier, admit/evict)
server/engine_loop.py  async bridge: inbox -> engine loop task -> per-request outboxes
server/app.py          FastAPI: POST /generate (SSE), browser demo page at /
tests/                 the correctness chain (pytest tests/)
bench/                 throughput, latency, and load benchmarks + results
```

## Run it

```
pip install -e . && pip install fastapi uvicorn httpx
uvicorn server.app:app --port 8000        # then open http://localhost:8000/
curl -N localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt": "The meaning of life is"}'
pytest tests/                             # the whole chain (slow: real model, real server)
```
