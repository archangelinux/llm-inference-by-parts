"""correctness gate for batching: each sequence in a greedy batch must produce exactly the ids it produces in a solo run

Contract this test imposes on model.generate_batch:
  in:  list of (1, t) id tensors, ragged lengths; max_new_tokens
  out: (b, padded_len + max_new_tokens) tensor, LEFT-padded
       -> generated tokens are the last max_new_tokens columns of every row
"""
import torch
from transformers import GPT2Tokenizer
from engine.config import DEVICE, GPTConfig, DTYPE
from engine.model import GPT

N_NEW = 50

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()

# token lengths span 1..27
# "Hello" (1 token) needs 26 pad slots when batched with the longest
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

def solo_baselines():
    output = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        output.append(model.generate(ids, max_new_tokens=N_NEW))  # (1, t + N_NEW)
    return output

def generate_batching():
    all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]
    return model.generate_batch(all_ids, max_new_tokens=N_NEW)  # (b, padded + N_NEW)

def test_batched_matches_solo():
    solo = solo_baselines() #list of tensors
    batched = generate_batching() # one tensor
    for i, p in enumerate(prompts):
        solo_gen = solo[i][0, -N_NEW:] #generated region = last N_NEW ids of the solo row
        batch_gen = batched[i, -N_NEW:]  #same slice since left-padding
        if not torch.equal(solo_gen.cpu(), batch_gen.cpu()):
            step = (solo_gen.cpu() != batch_gen.cpu()).nonzero()[0].item()  # marks the first diverging step
            raise AssertionError(
                f"prompt {i} {p[:40]!r} diverged at generated step {step}: "
                f"got {batch_gen[step].item()}, expected {solo_gen[step].item()}\n"
                f"  batched: {tok.decode(batch_gen)!r}\n"
                f"  solo: {tok.decode(solo_gen)!r}"
            )
    print("batched == solo: all match")

# debug probe: check if the initial (prefill) step was the cause of any output mismatch
#if prefill is fine then bug is in decode
#imposes a second contract, on GPT.forward: model(ids, attn_mask=mask) where mask is (b, t) with 1=real/0=pad; forward applies the mask in attention and derives per-row position ids from it.
def test_prefill_logits_match_solo():
    # padding built by HF's tokenizer instea dof generate_batch() -> independent of the code under test, so a padding bug there can't hide from this probe
    tok.pad_token = tok.eos_token  # gpt2 has no pad token; any id works (eos conventional), the mask hides it
    tok.padding_side = "left"
    enc = tok(prompts, padding=True, return_tensors="pt")
    ids = enc.input_ids.to(DEVICE) # (b, padded_len)
    mask = enc.attention_mask.to(DEVICE) # (b, padded_len), 1=real, 0=pad

    with torch.inference_mode():
        batched_logits, _ = model(ids, attn_mask=mask)  #(b, padded_len, vocab), no cache
    for i, p in enumerate(prompts):
        solo_ids = tok(p, return_tensors="pt").input_ids.to(DEVICE)
        with torch.inference_mode():
            solo_logits, _ = model(solo_ids) # (1, t, vocab)
        # compare next-token scores at each row's LAST REAL position:
        err = (batched_logits[i, -1].cpu() - solo_logits[0, -1].cpu()).abs().max().item()
        print(f"[{i}] {p[:40]!r:42s} prefill max abs err = {err:.2e}")
        assert err < 1e-3, f"prompt {i} {p[:40]!r}: prefill logits diverge (err {err:.2e}) => bug is in mask/positions/padding"
    print("prefill: all match")

#staggered-EOS gate: rows must stop at different steps without disturbing each other.
#greedy gpt2 never emits the real eos (50256) in 50 tokens, so borrow "." (13) as the stop
#token purely to exercise the machinery; two prompts never emit it -> "absent" branch covered.
#expected is computable from the solo baseline: greedy is deterministic, so a correct row
#matches solo's picks up to and including its first eos, then pads to full width.
def test_batched_eos():
    from engine.model import PAD_TOKEN
    EOS = 13  # "."
    solo = solo_baselines()
    all_ids = [tok(p, return_tensors="pt").input_ids.to(DEVICE) for p in prompts]
    batched = model.generate_batch(all_ids, max_new_tokens=N_NEW, eos_id=EOS)
    for i, p in enumerate(prompts):
        solo_gen = solo[i][0, -N_NEW:].cpu()
        hits = (solo_gen == EOS).nonzero()
        if len(hits) == 0:
            expected = solo_gen  # never finishes: all 50 must match, no pads
            finish = "-"
        else:
            j = hits[0].item()  # first eos position; row keeps it, then pads out
            expected = torch.cat((solo_gen[:j + 1],
                                  torch.full((N_NEW - j - 1,), PAD_TOKEN, dtype=solo_gen.dtype)))
            finish = j
        batch_gen = batched[i, -N_NEW:].cpu()
        print(f"[{i}] {p[:40]!r:42s} finishes at step {finish}")
        if not torch.equal(batch_gen, expected):
            step = (batch_gen != expected).nonzero()[0].item()
            raise AssertionError(
                f"prompt {i} {p[:40]!r} diverged at generated step {step}: "
                f"got {batch_gen[step].item()}, expected {expected[step].item()} (eos at {finish})\n"
                f"  batched:  {batch_gen.tolist()}\n"
                f"  expected: {expected.tolist()}"
            )
    print("eos: all match")


if __name__ == "__main__":
    test_batched_matches_solo()
    test_prefill_logits_match_solo()
    test_batched_eos()
