#render the phase-1 quant results (modal_quant_results.json) -> quant.png

"""Two panels: per-op time at the decode shape (where the fused kernel wins)
and end-to-end decode tok/s (where launch overhead takes it back).

Usage: python bench/chart_quant.py   (after bench/modal_quant.py)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
r = json.loads((HERE / "modal_quant_results.json").read_text())

GREEN, OLIVE, VIOLET, RED = "#c4473a", "#8b5e3c", "#2e8f95", "#a9d0d3"  #cuBLAS fp16 red, int8 slow path brown, int8 kernel teal; right panel: shades of teal, darker = fewer bytes read
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#8b8a85", "#e4e3dd", "#fdfdfc"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

# panel 1: per-op time, c_attn decode shape (lower is better); kernel is the dark bar
ops = [("cuBLAS fp16", "fp16_linear", GREEN),
       ("int8 dequant\nthen matmul", "int8_slow", OLIVE),
       ("int8 fused\nkernel", "int8_kernel", VIOLET)]
xs = range(len(ops))
for i, (label, key, color) in enumerate(ops):
    v = r["c_attn_op"][key]
    ax1.bar(i, v["us"], width=0.55, color=color)
    ax1.annotate(f'{v["us"]:.1f} us\n{v["achieved_gb_s"]:.0f} GB/s',
                 (i, v["us"]), xytext=(0, 4), textcoords="offset points",
                 ha="center", color=INK, fontsize=8)
ax1.set_xticks(list(xs), [o[0] for o in ops], fontsize=8, color=INK)
ax1.set_ylim(0, r["c_attn_op"]["int8_slow"]["us"] * 1.25)
ax1.set_title("One attention matmul, microseconds (lower is better)", color=INK, fontsize=10, loc="left")

# panel 2: end-to-end decode tok/s; kernel variant is the dark bar
names = ["fp32", "fp16", "int8-torch", "int8-kernel"]
xs2 = range(len(names))
for i, name in enumerate(names):
    v = r["variants"][name]
    ax2.bar(i, v["decode_tok_s"], width=0.55, color={"fp32": "#a9d0d3", "fp16": "#6fb0b5", "int8-torch": "#2e8f95", "int8-kernel": "#1f6469"}[name])
    ax2.annotate(f'{v["decode_tok_s"]:.0f}', (i, v["decode_tok_s"]), xytext=(0, 4),
                 textcoords="offset points", ha="center", color=INK, fontsize=9)
ax2.set_xticks(list(xs2), names, fontsize=8, color=INK)
ax2.set_title("Full decode, tok/s at b=1", color=INK, fontsize=10, loc="left")

fig.suptitle("GPT-2 on the A10G: int8 and the fused Triton kernel", color=INK, fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.885, "Left: one matmul timed by CUDA-graph replay over 8 weight copies. Right: full decode in eager PyTorch, where each launch costs ~20us.",
         color=MUTED, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "quant.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
