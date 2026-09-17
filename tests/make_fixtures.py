#writes the HF reference fixtures the test files check against:
#tests/fixtures/<name>/fixture_logits_<i>.pt (one forward pass) and
#fixture_generations.json (greedy 50-token continuations)

# usage: python tests/make_fixtures.py                     -> tests/fixtures/gpt2/
#        python tests/make_fixtures.py Qwen/Qwen2.5-0.5B   -> tests/fixtures/qwen2.5-0.5b/
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.config import DEVICE

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
OUT = Path(__file__).parent / "fixtures" / MODEL.split("/")[-1].lower()
OUT.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).eval().to(DEVICE)

prompts = [
    "Hello",
    "The meaning of life is",
    "In 2019, OpenAI released a language model that",
    "def fibonacci(n):",
    "1, 1, 2, 3, 5, 8, 13,",
    "Q: What is the capital of France?\nA: Paris.\nQ: What is the capital of Japan?\nA:",
    "The quick brown fox jumps over the lazy dog. The quick brown fox",
    ("Alice gave the book to Bob because he had asked her politely. "
     "Later that afternoon, Bob returned it to"),
]

generations = {}
with torch.inference_mode():
    for i, p in enumerate(prompts, 1):
        enc = tokenizer(p, return_tensors="pt")
        ids = enc.input_ids.to(DEVICE)
        logits = model(ids).logits
        torch.save(logits.cpu(), OUT / f"fixture_logits_{i}.pt") # test forward pass in isolation
        out = model.generate( #to test against generation loop
            ids,
            attention_mask=enc.attention_mask.to(DEVICE), #no padding
            max_new_tokens=50,
            do_sample=False, #argmax every step
        )
        generations[p] = tokenizer.decode(out[0])
        print(f"[{i}/{len(prompts)}] ok: {p[:30]!r}")

with open(OUT / "fixture_generations.json", "w") as f:
    json.dump(generations, f, indent=2)
print(f"[device: {DEVICE}] fixtures written to {OUT}")
