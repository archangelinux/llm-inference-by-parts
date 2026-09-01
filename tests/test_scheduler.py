"""correctness gate for continuous batching: every request served must produce exactly the ids solo greedy generation produces
Engine: submit(Request) -> run()/step();
Postcondition: request.output_ids holds the generated ids (ints), up to and including the first eos token, otherwise up to the budget length.
Unlike generate_batch there is no pad tail; harvesting stops at eos

1) test batch mode (submit all, run) to show slot recycling;
2) test mid-run admit (submit, step, submit more) to show continuous."""
import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig
from engine.model import GPT
from engine.scheduler import Engine, Request

N_NEW = 50
EOS = 13  # "." -- greedy gpt2 never emits the real eos in 50 tokens; borrow a frequent token
N_SLOTS = 3  # fewer slots than prompts -> forces queueing, waiting, slot recycling

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE).eval()

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

_expected_cache = {}  # prompt -> list of ints; solo generate is slow, compute once per session

def expected_for(prompt):
    """baseline solo greedy picks, truncated after the first eos (kept), else full budget."""
    if prompt not in _expected_cache:
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        solo_gen = model.generate(ids, max_new_tokens=N_NEW)[0, -N_NEW:].tolist()
        exp = []
        for t in solo_gen:
            exp.append(t)
            if t == EOS:
                break  # eos included, nothing after -> matches per-step harvest semantics
        _expected_cache[prompt] = exp
    return _expected_cache[prompt]


def make_requests():
    return [Request(prompt_ids=tok(p, return_tensors="pt").input_ids.to(DEVICE),
                    max_new_tokens=N_NEW, eos_id=EOS)
            for p in prompts]

def check(reqs):
    for i, (p, req) in enumerate(zip(prompts, reqs)):
        assert req.done, f"prompt {i} {p[:40]!r} never finished"
        exp = expected_for(p)
        print(f"[{i}] {p[:40]!r:42s} {len(req.output_ids)} tokens")
        if req.output_ids != exp:
            n = min(len(req.output_ids), len(exp))
            diffs = [j for j in range(n) if req.output_ids[j] != exp[j]]
            step = diffs[0] if diffs else n  # first differing step, or a length mismatch
            raise AssertionError(
                f"prompt {i} {p[:40]!r} diverged at generated step {step}\n"
                f"  engine:   {req.output_ids}\n"
                f"  expected: {exp}"
            )


def test_batch_mode():
    engine = Engine(model=model, n_slots=N_SLOTS, max_len=512)
    reqs = make_requests()
    for r in reqs:
        engine.submit(r)
    engine.run()
    check(reqs)
    print("batch mode: all match")

def test_midrun_admission():
    engine = Engine(model=model, n_slots=N_SLOTS, max_len=512)
    reqs = make_requests()
    for r in reqs[:2]:
        engine.submit(r)
    for _ in range(5):  # let the first two get going before anyone else exists
        engine.step()
    for r in reqs[2:]:  # arrive mid-flight -> admitted into live/recycled slots
        engine.submit(r)
    engine.run()
    check(reqs)
    print("mid-run admission: all match")


if __name__ == "__main__":
    test_batch_mode()
    test_midrun_admission()
