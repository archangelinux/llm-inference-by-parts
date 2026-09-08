"""correctness checks: 
1) |q * scale - w| (error from converting to int8 and back) must stay within half a tick, 
2) QuantLinear matches the Linear it was built from, 
3) quantized model logits stay near the fp32 fixtures. 
The kernel will later add: fused kernel output == QuantLinear's slow path"""

import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer

from engine.config import DEVICE, GPTConfig, DTYPE
from engine.model import GPT
from engine.quant import QuantLinear, quantize_model, quantize_weight

TESTS_DIR = Path(__file__).parent
with open(TESTS_DIR / "fixture_generations.json") as f:
    GENERATIONS = json.load(f)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT.from_pretrained(GPTConfig()).to(DEVICE, DTYPE).eval()


def test_roundtrip(): 
    block = model.transformer.h[0]
    for layer in [block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj]:
        w = layer.weight
        q, s = quantize_weight(w)
        err = (q.float() * s.float() - w.float()).abs() #check in fp32: the bound is about the quantization math, not fp16 multiply rounding
        bound = s.float() * 0.6 #rounding error is <= half a tick, plus the w/scale division in fp16 can shift the quotient ~127*2^-11 = 0.06 ticks
        assert (err <= bound).all(), f"{(err - bound).max().item()} over the half-tick bound"
    print("roundtrip: all four layer shapes within half a tick")


def test_layer(): #QuantLinear vs the Linear it replaces, same input
    layer = model.transformer.h[0].attn.c_attn
    qlayer = QuantLinear(layer)
    x = torch.randn(1, 8, model.config.n_embd, device=DEVICE, dtype=layer.weight.dtype)
    with torch.inference_mode():
        err = (qlayer(x) - layer(x)).abs().max().item()
    print(f"layer out err = {err:.3f}")
    assert err < 1.0 # measured ~0.1; real bugs (wrong scale dim, transposed q) come in at tens


def test_model_logits(): #quantized model vs the fp32 HF fixtures
    quantize_model(model) #mutates -- keep this test last
    for i, prompt in enumerate(GENERATIONS, 1):
        ref = torch.load(TESTS_DIR / f"fixture_logits_{i}.pt", map_location="cpu")
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.inference_mode():
            logits, _ = model(ids)
        err = (logits.float().cpu() - ref).abs().max().item()
        print(f"[{i}] {prompt[:40]!r:42s} max abs err = {err:.3f}")
        assert err < 10 # measured 0.65-4.43 on fp16+int8; wrong-code errors are far past 10


if __name__ == "__main__":
    test_roundtrip()
    test_layer()
    test_model_logits()
