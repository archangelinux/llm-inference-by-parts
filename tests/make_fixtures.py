import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from engine.config import DEVICE

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

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
        torch.save(logits.cpu(), f"tests/fixture_logits_{i}.pt") # test forward pass in isolation
        out = model.generate( #to test against generation loop
            ids,
            attention_mask=enc.attention_mask.to(DEVICE), #no padding 
            max_new_tokens=50,
            do_sample=False, #argmax every step
        )
        generations[p] = tokenizer.decode(out[0])
        print(f"[{i}/{len(prompts)}] ok: {p[:30]!r}")

with open("tests/fixture_generations.json", "w") as f:
    json.dump(generations, f, indent=2)
print(f"[device: {DEVICE}] fixtures written")