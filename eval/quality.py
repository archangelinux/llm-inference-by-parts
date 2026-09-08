'''quality gates for lossy precision: 
    - top-1 agreement: (argmax) % of positions where the variant picks the same token as fp32
    - max logit diff: vs fp32 over the same positions
    - perplexity (ppl, the effective number of choices the model was hedging between): (on a WikiText-2 slice) how well the model predicts real text

exact-match tests stop working once precision is lossy (one near-tie flips a
token and the whole tail diverges), so these measure quality directly instead.
ppl here is chunked (no sliding window; blind spot on start of each chunk), so it isn't comparable to
published numbers, but is valid here for the purpose of comparing the variants
'''
# usage: python eval/quality.py; writes eval/quality_results.json
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig, sync
from engine.model import GPT
from engine.quant import quantize_model

CHUNK = 512 #tokens per forward pass (511 predictions each)
N_CHUNKS = 32 #~16k tokens total
RESULTS_FILE = Path(__file__).parent / "quality_results.json"

tok = GPT2Tokenizer.from_pretrained("gpt2")

#one long token stream from the wikitext test split, cut into chunks
text = "\n\n".join(t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0, : CHUNK * N_CHUNKS]
chunks = ids.view(N_CHUNKS, CHUNK).to(DEVICE)
print(f"eval corpus: {ids.numel()} tokens in {N_CHUNKS} chunks of {CHUNK}")


@torch.inference_mode()
def run_chunks(model):
    #per chunk: logits for positions :-1 (each predicts the next token in the chunk)
    for i in range(N_CHUNKS):
        logits, _ = model(chunks[i : i + 1])
        yield logits[0, :-1].float(), chunks[i, 1:] #returns as a generator to iterate over


@torch.inference_mode()
def evaluate(model, ref_model):
    nll_sum, n, agree, max_diff = 0.0, 0, 0, 0.0 #negative log likelihood is the "penalty" for ease of summing and averaging before converting back to ppl
    ref_iter = run_chunks(ref_model) if ref_model is not None else None
    for logits, targets in run_chunks(model):
        nll_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        n += targets.numel()
        if ref_iter is not None:
            ref_logits, _ = next(ref_iter)
            agree += (logits.argmax(-1) == ref_logits.argmax(-1)).sum().item()
            #center each position's logits first: softmax only uses the gaps between logits, so a uniform shift of the whole vector (which quantization causes at some positions) can't affect output and shouldn't count as error
            d = (logits - logits.mean(-1, keepdim=True)) - (ref_logits - ref_logits.mean(-1, keepdim=True))
            max_diff = max(max_diff, d.abs().max().item())
    out = {"perplexity": round(torch.exp(torch.tensor(nll_sum / n)).item(), 3)}
    if ref_iter is not None:
        out["top1_agreement"] = round(agree / n, 4)
        out["max_centered_logit_diff"] = round(max_diff, 3)
    return out


def load(dtype):
    return GPT.from_pretrained(GPTConfig()).to(DEVICE, dtype).eval()


if __name__ == "__main__":
    results = {}
    ref = load(torch.float32) #fp32 is the reference: its agreement is 1.0 by definition
    results["fp32"] = evaluate(ref, None)
    print(f"fp32: {results['fp32']}")

    for name, build in [
        ("fp16", lambda: load(torch.float16)),
        ("int8", lambda: quantize_model(load(torch.float16))), #int8 on top of fp16, as served
    ]:
        model = build()
        results[name] = evaluate(model, ref)
        print(f"{name}: {results[name]}")
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"wrote {RESULTS_FILE}")
