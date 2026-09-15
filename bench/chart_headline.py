#the headline figure: decode throughput by engine stage, both models, one GPU -> headline.png

"""Reads modal_results.json (GPT-2) and modal_results.qwen.json (Qwen3), fp32,
A10G. Three stages per model: naive full-recompute at 512 context, KV-cached
decode at 512 context, batched decode (b=128, 16-token prompts). Log scale.

Usage: python bench/chart_headline.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
gpt2 = json.loads((HERE / "modal_results.json").read_text())
qwen = json.loads((HERE / "modal_results.qwen.json").read_text())

LIGHT, DARK, INK, MUTED, GRID, SURFACE = "#86b6ef", "#1c5cab", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

stages = ["naive\n(512 ctx)", "KV cache\n(512 ctx)", "batched\n(b=128)"]
def vals(r):
    return [r["naive"]["512"], r["cached"]["512"], r["batched"]["128"]]
series = [("GPT-2 124M", vals(gpt2), LIGHT), ("Qwen3 0.6B", vals(qwen), DARK)]

fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)

w = 0.36
for j, (label, v, color) in enumerate(series):
    xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(stages))]
    bars = ax.bar(xs, v, width=w, color=color, label=label)
    for b, val in zip(bars, v):
        ax.annotate(f"{val:,.0f}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK, fontsize=8)
ax.set_yscale("log")
ax.set_xticks(range(len(stages)), stages, color=INK, fontsize=9)
ax.set_ylabel("tok/s (log)", color=MUTED, fontsize=9)
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
ax.set_title("Decode throughput by engine stage", color=INK, fontsize=12, loc="left", pad=22)
ax.text(0, 1.04, "A10G, fp32, greedy, median of 5  •  same engine, both architectures",
        transform=ax.transAxes, color=MUTED, fontsize=8)
fig.tight_layout()
out = HERE / "headline.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
