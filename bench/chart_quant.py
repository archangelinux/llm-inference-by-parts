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

BLUE, DARK, INK, MUTED, GRID, SURFACE = "#86b6ef", "#1c5cab", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

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
ops = [("cuBLAS fp16", "fp16_linear", BLUE),
       ("int8 dequant\nthen matmul", "int8_slow", BLUE),
       ("int8 fused\nkernel", "int8_kernel", DARK)]
xs = range(len(ops))
for i, (label, key, color) in enumerate(ops):
    v = r["c_attn_op"][key]
    ax1.bar(i, v["us"], width=0.55, color=color)
    ax1.annotate(f'{v["us"]:.1f} us\n{v["achieved_gb_s"]:.0f} GB/s',
                 (i, v["us"]), xytext=(0, 4), textcoords="offset points",
                 ha="center", color=INK, fontsize=8)
ax1.set_xticks(list(xs), [o[0] for o in ops], fontsize=8, color=INK)
ax1.set_ylim(0, r["c_attn_op"]["int8_slow"]["us"] * 1.25)
ax1.set_title("one c_attn op at decode shape (us, lower is better)", color=INK, fontsize=10, loc="left")

# panel 2: end-to-end decode tok/s; kernel variant is the dark bar
names = ["fp32", "fp16", "int8-torch", "int8-kernel"]
xs2 = range(len(names))
for i, name in enumerate(names):
    v = r["variants"][name]
    ax2.bar(i, v["decode_tok_s"], width=0.55, color=DARK if name == "int8-kernel" else BLUE)
    ax2.annotate(f'{v["decode_tok_s"]:.0f}', (i, v["decode_tok_s"]), xytext=(0, 4),
                 textcoords="offset points", ha="center", color=INK, fontsize=9)
ax2.set_xticks(list(xs2), names, fontsize=8, color=INK)
ax2.set_title("end-to-end decode (tok/s, b=1)", color=INK, fontsize=10, loc="left")

fig.suptitle("int8 + fused kernel on the A10G", color=INK, fontsize=12, x=0.02, ha="left")
fig.text(0.02, 0.885, "per-op: cuda-graph replay over 8 weight copies (defeats L2)  •  "
         "end-to-end: eager pytorch, ~20us python dispatch per triton launch",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "quant.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
