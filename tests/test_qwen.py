"""Our Qwen vs the HF reference fixtures (tests/fixtures/qwen3-0.6b-base/, from
tests/make_fixtures.py Qwen/Qwen3-0.6B-Base). Same ladder as the GPT-2 tests:
rope property (no weights needed) -> logits vs HF -> greedy identity -> batched == solo.
Run with pytest or plain python."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engine.config import DEVICE, DTYPE, QwenConfig
from engine.qwen import Qwen, apply_rope
from engine.scheduler import Engine, Request

TESTS_DIR = Path(__file__).parent
FIXTURES = TESTS_DIR / "fixtures" / "qwen3-0.6b-base"
with open(FIXTURES / "fixture_generations.json") as f:
    GENERATIONS = json.load(f)

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
cfg = QwenConfig()

#same placement logic as test_logits: a few times above correct-code error, far
#below broken-code error. qwen logits are smaller-magnitude than gpt-2's (~±30),
#so measured errors land lower; set from the first green run
TOL = 1e-3 if DTYPE == torch.float32 else 0.5 # fp32 measured 2e-5..1.5e-4


def test_rope_relative():  #rotating q and k by (m, n) vs (m+s, n+s) must give the same score; different gap must not
    inv = cfg.rope_theta ** (-torch.arange(0, cfg.head_dim, 2) / cfg.head_dim)
    def rot(x, p):
        a = torch.tensor([[[[float(p)]]]]) * inv
        e = torch.cat((a, a), -1)
        return apply_rope(x, e.cos(), e.sin())
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)
    same_gap = (rot(q, 3) * rot(k, 7)).sum().item(), (rot(q, 103) * rot(k, 107)).sum().item()
    other_gap = (rot(q, 3) * rot(k, 8)).sum().item()
    assert abs(same_gap[0] - same_gap[1]) < 1e-3, same_gap
    assert abs(same_gap[0] - other_gap) > 1e-3
    print(f"rope: gap-4 scores {same_gap[0]:.4f} == {same_gap[1]:.4f}; gap-5 {other_gap:.4f} differs")


def _model():
    return Qwen.from_pretrained(cfg).to(DEVICE, DTYPE).eval()


def test_logits():  #one forward pass vs HF's logits for each fixture prompt
    model = _model()
    for i, prompt in enumerate(GENERATIONS, 1):
        ref = torch.load(FIXTURES / f"fixture_logits_{i}.pt", map_location="cpu")
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.inference_mode():
            logits, _ = model(ids)
        mine = logits.float().cpu()
        err = (mine - ref).abs().max().item()
        print(f"[{i}] {prompt[:40]!r:42s} max abs err = {err:.2e}")
        assert mine.shape == ref.shape, (mine.shape, ref.shape)
        assert err < TOL


def test_greedy():  #50-token greedy continuation must match HF token for token (fp32); fp16 reports
    model = _model()
    mismatches = 0
    for prompt, expected in GENERATIONS.items():
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out = model.generate(ids, max_new_tokens=50, eos_id=tok.eos_token_id) #stop at eos like hf does
        match = tok.decode(out[0]) == expected
        mismatches += not match
        if DTYPE == torch.float32:
            assert match, prompt
        elif not match:
            print(f"greedy diverged ({DTYPE}): {prompt[:40]!r}")
    print(f"greedy: {len(GENERATIONS) - mismatches}/{len(GENERATIONS)} match")


def test_batched_equals_solo():  #left-padded batch must reproduce each prompt's solo greedy tokens (masks, rope positions, gqa cache)
    model = _model()
    prompts = list(GENERATIONS)[:4]  #4 prompts of different lengths
    all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]
    n_new = 20
    batched = model.generate_batch(all_ids, max_new_tokens=n_new)
    for i, ids in enumerate(all_ids):
        solo = model.generate(ids, max_new_tokens=n_new)[0, -n_new:]
        assert torch.equal(batched[i, -n_new:].cpu(), solo.cpu()), prompts[i]
    print("batched == solo: all match")


def test_engine():  #continuous batching on qwen: every request's output_ids == solo greedy, up to and including eos
    model = _model()
    n_new, n_slots = 30, 3  #fewer slots than prompts -> queueing and slot recycling
    eos = tok.eos_token_id
    engine = Engine(model=model, n_slots=n_slots, max_len=256)
    reqs = {}
    for prompt in GENERATIONS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        r = Request(prompt_ids=ids, max_new_tokens=n_new, eos_id=eos)
        reqs[r.req_id] = (prompt, ids)
        engine.submit(r)
    for r in engine.run():
        prompt, ids = reqs[r.req_id]
        solo = model.generate(ids, max_new_tokens=n_new, eos_id=eos)[0, ids.shape[1]:].tolist()
        assert r.output_ids == solo, prompt
    print(f"engine: all {len(reqs)} requests match solo (n_slots={n_slots})")


if __name__ == "__main__":
    test_rope_relative()
    test_logits()
    test_greedy()
    test_batched_equals_solo()
    test_engine()
