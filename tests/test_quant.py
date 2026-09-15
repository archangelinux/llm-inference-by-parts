"""correctness checks: 
1) |q * scale - w| (error from converting to int8 and back) must stay within half a tick, 
2) QuantLinear matches the Linear it was built from, 
3) quantized model logits stay near the fp32 fixtures. 
The kernel will later add: fused kernel output == QuantLinear's slow path"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn

from engine.config import DEVICE, DTYPE, MODEL
from engine.load import load_model
from engine.quant import QuantLinear, quantize_model, quantize_weight

TESTS_DIR = Path(__file__).parent
FIXTURES = TESTS_DIR / "fixtures" / {"gpt2": "gpt2", "qwen": "qwen3-0.6b-base"}[MODEL]
with open(FIXTURES / "fixture_generations.json") as f:
    GENERATIONS = json.load(f)

model, tok = load_model() #MODEL=gpt2|qwen
#every Linear in block 0 (gpt-2: c_attn, c_proj, c_fc, c_proj; qwen3: q/k/v/o_proj, gate/up/down_proj)
BLOCK0_LINEARS = [m for m in model.transformer.h[0].modules() if isinstance(m, nn.Linear)]


def test_roundtrip(): 
    for layer in BLOCK0_LINEARS:
        w = layer.weight
        q, s = quantize_weight(w)
        err = (q.float() * s.float() - w.float()).abs() #check in fp32: the bound is about the quantization math, not fp16 multiply rounding
        bound = s.float() * 0.6 #rounding error is <= half a tick, plus the w/scale division in fp16 can shift the quotient ~127*2^-11 = 0.06 ticks
        assert (err <= bound).all(), f"{(err - bound).max().item()} over the half-tick bound"
    print(f"roundtrip: all {len(BLOCK0_LINEARS)} block-0 layer shapes within half a tick")


def test_layer(): #QuantLinear vs the Linear it replaces, same input
    layer = BLOCK0_LINEARS[0] #the widest projection: c_attn (gpt-2) / q_proj (qwen)
    qlayer = QuantLinear(layer)
    x = torch.randn(1, 8, layer.in_features, device=DEVICE, dtype=layer.weight.dtype)
    with torch.inference_mode():
        err = (qlayer(x) - layer(x)).abs().max().item()
    print(f"layer out err = {err:.3f}")
    assert err < 1.0 # measured ~0.1; real bugs (wrong scale dim, transposed q) come in at tens


def test_kernel(): #fused triton kernel == the slow dequant path (cuda only; triton has no mps build)
    if not torch.cuda.is_available():
        print("kernel: skipped (needs cuda)")
        return
    for layer in BLOCK0_LINEARS:
        #fresh fp16 copy of the layer so this test works whatever DTYPE the model loaded as
        lin = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None, device=DEVICE, dtype=torch.float16) #qwen3 linears have no bias
        with torch.inference_mode():
            lin.weight.copy_(layer.weight)
            if layer.bias is not None:
                lin.bias.copy_(layer.bias)
            qlayer = QuantLinear(lin)
            x = torch.randn(2, 5, layer.in_features, device=DEVICE, dtype=torch.float16)
            fused = qlayer(x) #cuda + fp16 -> kernel path
            slow = F.linear(x, qlayer.q.to(x.dtype) * qlayer.scale, qlayer.bias)
            err = (fused - slow).abs().max().item()
        print(f"kernel vs slow path ({layer.out_features}x{layer.in_features}): max abs err = {err:.4f}")
        assert err < 0.1 # same q both sides, so only summation rounding differs; stride bugs come in at tens


def test_model_logits(): #quantized model vs the fp32 HF fixtures
    quantize_model(model) #mutates -- keep this test last
    for i, prompt in enumerate(GENERATIONS, 1):
        ref = torch.load(FIXTURES / f"fixture_logits_{i}.pt", map_location="cpu")
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        with torch.inference_mode():
            logits, _ = model(ids)
        err = (logits.float().cpu() - ref).abs().max().item()
        print(f"[{i}] {prompt[:40]!r:42s} max abs err = {err:.3f}")
        assert err < {"gpt2": 10, "qwen": 50}[MODEL] # gpt-2 measured 0.65-4.4; qwen 9-26 (raw max diff carries a uniform logit shift, see eval/quality.py); wrong-code errors are far past these


if __name__ == "__main__":
    test_roundtrip()
    test_layer()
    test_kernel()
    test_model_logits()
