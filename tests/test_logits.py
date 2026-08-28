"""Our GPT vs the HF reference fixtures from tests/make_fixtures.py:
fixture_logits_<i>.pt = HF's (1, t, vocab) logits for prompt i;
fixture_generations.json = prompt -> HF's greedy 50-token continuation
(keys are the prompts, in fixture order). Run with pytest or plain python."""
import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig
from engine.model import GPT

TESTS_DIR = Path(__file__).parent
with open(TESTS_DIR / "fixture_generations.json") as f: #writte by make_fixtures.py
    GENERATIONS = json.load(f)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()

def test_logits(): #check if one forward pass produces the same values as HF
    for i, prompt in enumerate(GENERATIONS, 1):
        ref = torch.load(TESTS_DIR / f"fixture_logits_{i}.pt", map_location="cpu")
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE) # to mps (input and model must be on same device)
        with torch.inference_mode():
            mine = model(ids).cpu() #back to cpu so it can be subtracted from ref
        if i == 1:
            print(f"shapes: mine {tuple(mine.shape)}  ref {tuple(ref.shape)}  ({mine.dtype}, {DEVICE})")
        err = (mine - ref).abs().max().item() # elementwise difference -> abs val -> biggest single element -> .item() converts the 0-dim tensor to a plain Python float
        print(f"[{i}] {prompt[:40]!r:42s} max abs err = {err:.2e}") #!r prints with quotes/escapes visible, .2e is for sci notation
        assert mine.shape == ref.shape
        assert err < 1e-3  # mine come in around 1e~4


def test_greedy():
    for prompt, expected in GENERATIONS.items(): #gives both the prompt and HFs expected output string
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE) #tokenize
        out = model.generate(ids, max_new_tokens=50) #greedy by default (do_sample=False)
        assert tok.decode(out[0]) == expected #decode and compare
    print("greedy: all match")


if __name__ == "__main__":
    test_logits()
    test_greedy()
