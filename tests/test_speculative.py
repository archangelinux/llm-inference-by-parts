"""Greedy speculative decoding must return exactly what target.generate does.
Target: Qwen3-0.6B. Draft: the same model cut to its first 8 layers -- a worse model
that disagrees often, so the reject + rollback path runs constantly. The real pair
(0.6B drafting for 1.7B) is too big for local memory; the algorithm doesn't care.
Run with pytest or plain python."""
import copy
import dataclasses
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from engine.config import DEVICE, DTYPE, QwenConfig
from engine.qwen import Qwen
from engine.speculative import spec_generate

FIXTURES = Path(__file__).parent / "fixtures" / "qwen3-0.6b-base"
with open(FIXTURES / "fixture_generations.json") as f:
    PROMPTS = list(json.load(f))[1:4]  # "The meaning of life is", "In 2019, OpenAI...", "def fibonacci(n):"

DRAFT_LAYERS = 8
cfg = QwenConfig()
tok = AutoTokenizer.from_pretrained(cfg.hf_id)

target = Qwen.from_pretrained(cfg).to(DEVICE, DTYPE).eval()
draft = copy.deepcopy(target)  # deepcopy, not a second from_pretrained: no second HF copy in memory
draft.transformer.h = draft.transformer.h[:DRAFT_LAYERS]
draft.config = dataclasses.replace(cfg, n_layer=DRAFT_LAYERS)  # _new_cache sizes the cache from config.n_layer


def test_draft_disagrees():  #otherwise the equality test below never exercises a rejection
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        assert not torch.equal(draft.generate(ids, max_new_tokens=20), target.generate(ids, max_new_tokens=20))


def test_greedy_equals_target():  #token for token, every k, lengths that end mid-round
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        for n in (7, 13, 30):
            expected = target.generate(ids, max_new_tokens=n)
            for k in (1, 2, 4, 8):
                got = spec_generate(target, draft, ids, k=k, max_new_tokens=n)
                assert torch.equal(got, expected), (prompt, n, k, tok.decode(got[0]), tok.decode(expected[0]))
        print(f"{prompt[:40]!r:42s} ok for n in (7, 13, 30), k in (1, 2, 4, 8)")


if __name__ == "__main__":
    test_draft_disagrees()
    test_greedy_equals_target()
