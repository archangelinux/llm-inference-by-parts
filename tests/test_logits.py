"""Our GPT vs the HF reference fixtures from tests/make_fixtures.py:
fixture_logits_<i>.pt = HF's (1, t, vocab) logits for prompt i;
fixture_generations.json = prompt -> HF's greedy 50-token continuation
(keys are the prompts, in fixture order). Run with pytest or plain python."""
import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig, DTYPE
from engine.model import GPT

TESTS_DIR = Path(__file__).parent
with open(TESTS_DIR / "fixture_generations.json") as f: #written by make_fixtures.py
    GENERATIONS = json.load(f)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()

# threshold sits between correct-code error and broken-code error (tens) for each dtype:
# fp32 correct ~1e-4; fp16 correct ~0.3 (rounding on ~100-magnitude logits, measured)
TOL = 1e-3 if DTYPE == torch.float32 else 1.0

def test_logits(): #check if one forward pass produces the same values as HF
    for i, prompt in enumerate(GENERATIONS, 1):
        ref = torch.load(TESTS_DIR / f"fixture_logits_{i}.pt", map_location="cpu")
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE) # to mps (input and model must be on same device)
        with torch.inference_mode():
            logits, kv = model(ids) # no cache passed => kv = [None]*n_layer
            mine = logits.cpu() #back to cpu so it can be subtracted from ref
        if i == 1:
            print(f"shapes: mine {tuple(mine.shape)}  ref {tuple(ref.shape)}  ({mine.dtype}, {DEVICE})")
        err = (mine - ref).abs().max().item() # elementwise difference -> abs val -> biggest single element -> .item() converts the 0-dim tensor to a plain Python float
        print(f"[{i}] {prompt[:40]!r:42s} max abs err = {err:.2e}") #!r prints with quotes/escapes visible, .2e is for sci notation
        assert mine.shape == ref.shape
        assert err < TOL


def test_greedy():
    # exact match is only a valid gate at fp32; fp16 rounding legitimately flips
    # near-tied tokens (5/8 match, measured), so there we report instead of assert
    mismatches = 0
    for prompt, expected in GENERATIONS.items(): #gives both the prompt and HFs expected output string
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE) #tokenize
        out = model.generate(ids, max_new_tokens=50) #greedy by default (do_sample=False)
        match = tok.decode(out[0]) == expected #decode and compare
        mismatches += not match
        if DTYPE == torch.float32:
            assert match
        elif not match:
            print(f"greedy diverged ({DTYPE}): {prompt[:40]!r}")
    print(f"greedy: {len(GENERATIONS) - mismatches}/{len(GENERATIONS)} match")


if __name__ == "__main__":
    test_logits()
    test_greedy()
