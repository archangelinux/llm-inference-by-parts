# Roadmap: the rest of the project

This file supersedes every previous plan fragment. One order, one why per phase,
concrete steps with done-when gates. Read the Purpose section once; execute the
phases top to bottom; ignore anything from older plans that contradicts this.

## Purpose (why any of this impresses anyone)

What exists today: "rebuilt GPT-2 inference from scratch, batched it, served it,
benchmarked it honestly on laptop + cloud." Strong, but the shape of a student
exercise.

The remaining phases each remove one "but it's just—" objection:

| Phase | Removes the objection | The credential |
|---|---|---|
| 1. Precision (fp16 -> int8 -> Triton) | "...but it's fp32, nobody serves fp32" | can make models fast AND prove they stayed smart; wrote a GPU kernel |
| 2. Qwen port | "...but it's GPT-2, a 2019 museum piece" | engine handles modern architectures (RoPE, GQA, SwiGLU, RMSNorm) |
| 3. Writeup | "...but nobody will ever see the work" | can explain systems work clearly — half the job at any company |

Together: "takes a model, makes it fast, proves it still works, on modern
architectures, and explains it." That is the literal job description for
inference/serving roles, and a high-signal artifact for any technical audience.

**Cut lines** (each phase ends at a shippable state): after phase 1 you have a
quantization + kernel story; after phase 2 the "modern engine" claim; phase 3 is
never cut — an unwritten project is an unfinished one. The add-ons (routing,
speculative decoding, paged cache) are OPTIONAL and come only after phase 3.

## Standing rules (learned the hard way in stages 1-5, still binding)

1. **Gates before speed claims.** Every change reruns the correctness chain
   before any benchmark is quoted.
2. **One variable at a time.** New precision on the old model; new model at the
   old precision. Never both in one step.
3. **Measure, don't assume** — and clean protocol always: per-config warmup,
   median of 5, one config at a time.
4. **A number a reader can recompute must check out or be pre-explained.**
5. **Weird-but-reproducible = a finding, not noise.** Investigate, then document
   (see: the MPS kernel cliff, the utilization-drops-on-faster-hardware result).

---

## Phase 0 — Push (today, 10 minutes)

The README is done. `git add -A && git commit && git push`. Everything after
this builds on a published baseline. Done when: the link exists.

## Phase 1 — Precision: fp16, int8, one Triton kernel (~2 weeks)

**The idea in one paragraph:** weights are fp32 (4 bytes) only because that's
PyTorch's default and training wants the precision headroom — inference doesn't.
Decode speed = weight bytes / bandwidth (your own bandwidth-check section), so
halving the bytes ≈ doubles decode. fp16 is free (PyTorch native). int8 stores
`int8 weight x per-channel fp16 scale`; works because trained weights are small,
zero-centered, and redundant. This is industry floor practice (llama.cpp's Q8_0
already beat you with it in your own README table).

**Why int8 needs a custom kernel:** PyTorch has no "fp16 activations x int8
weights" matmul. Dequantizing to fp16 first writes a full-size copy — MORE
memory traffic than fp32, slower. The win only exists if dequantization happens
inside the matmul, in on-chip registers. That's a ~60-line Triton kernel, and
it is the single strongest resume line in this phase.

**Steps, in order (each gated before the next):**

1. `--dtype` knob; `model.half()` after loading. Rerun `test_logits.py`;
   RECORD the new max errors (expect ~100x looser — measure, don't assume the
   tolerance). Bench fp32 vs fp16 on A10G (`modal_bench.py` grows a dtype arg).
   Done when: gates green at measured fp16 tolerance + a 2-point speed table.
2. `engine/quant.py`: `QuantLinear` (int8 weights + per-channel scales, forward
   = dequantize-then-matmul — deliberately slow, correctness first) and
   `quantize_model()` swapping the four Linear types. Done when: greedy
   generation still speaks English.
3. Quality gates for lossy compression (new — exact-match gates can't work
   here): max logit diff, top-1 agreement % across fixture positions, and
   perplexity on a WikiText slice. Produce the fp32/fp16/int8 table.
   Done when: int8 perplexity delta is ~1% or explained.
4. Triton on Modal: official matmul tutorial first (until the tile loop makes
   sense), then the fused dequant-matmul (W8A16; GPTQ-style references abound).
   Done when: kernel output matches step 2's slow path to tolerance.
5. Final bench: fp32 / fp16 / int8-torch / int8-kernel on A10G + % of 600 GB/s
   achieved per variant. Done when: the quality-vs-speed frontier chart exists
   and supports the sentence "my kernel reaches X% of A10G bandwidth; the
   remaining gap is Y." (That sentence is the resume line.)

## Phase 2 — Qwen2.5-0.5B port (~1 week)

**Why:** proves the engine isn't a GPT-2-only trick. The port is the 2019->2024
architectural diff, implemented by hand — four part-swaps, same skeleton
(blocks, residuals, KV cache, scheduler, server all carry over):

- LayerNorm -> **RMSNorm** (~5 lines: drop the mean subtraction)
- GELU MLP -> **SwiGLU** (~10 lines: two up-projections, one gates the other via SiLU)
- learned `wpe` -> **RoPE** (rotate q,k by position-dependent angles; the
  conceptual heavyweight — be able to whiteboard why rotation encodes RELATIVE
  position; connects directly to your positions-from-mask work)
- full multi-head k/v -> **GQA** (few k/v heads shared by many q heads; exists
  purely to shrink the KV cache — you of all people will appreciate why)

**Steps:** (1) new fixtures via the existing `make_fixtures.py` pattern against
HF's Qwen2.5-0.5B; (2) the four modules + config-driven assembly + weight-name
mapping in the loader; (3) the same gate ladder: logits vs HF -> greedy identity
-> batched == solo -> engine -> server; (4) rerun the bench suite on Qwen.
Done when: gates green and the README says what porting required.
Cut-line inside the phase: if GQA vs your cache layout fights you past ~3 days,
ship the port without batched-Qwen and say so.

## Phase 3 — The writeup (~1 week, never cut)

One blog post / site page telling the by-parts story: what each mechanism
bought, measured (charts mostly exist), the bandwidth-floor sanity check per
stage, the llama.cpp gap analysis, limitations. README links it; it links the
repo. Then resume bullets distilled from it. Equal in value to any code week:
it is the difference between work done and work seen. Done when: published URL.

## Add-ons — only after phase 3, pick AT MOST one

Ranked: **A. difficulty-aware routing** (small+large Qwen behind your server;
hypothesis -> baseline table; requires building a small honest eval harness
first — the originality pick, this one is YOURS) > **B. speculative decoding**
(known algorithm, prestige) > **C. paged KV cache** (deepest systems work).
Int8 is no longer an add-on — it's phase 1. If energy is finite, shipping
phases 1-3 well beats phases 1-3 plus a rushed add-on.
