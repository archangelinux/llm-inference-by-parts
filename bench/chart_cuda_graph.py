#render modal_graph_results.json -> cuda_graph.png: ms per decode step, eager vs graphed, b=1

"""Two panels (GPT-2, Qwen3), three variants each, paired bars eager/graphed.

Usage: python bench/chart_graph.py   (after bench/modal_graph.py)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
runs = json.loads((HERE / "modal_graph_results.json").read_text())["runs"]
LIGHT, DARK, INK, MUTED, GRID, SURFACE = "#86b6ef", "#1c5cab", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
VARIANTS = ["fp32", "fp16", "int8-kernel"]

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
for ax, (name, title) in zip(axes, [("gpt2", "GPT-2 124M"), ("qwen", "Qwen3 0.6B")]):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    rows = {r["variant"]: r for r in runs if r["model"] == name and r["n_slots"] == 1}
    w = 0.36
    for j, (key, color, label) in enumerate([("eager_ms_per_step", LIGHT, "eager"), ("graphed_ms_per_step", DARK, "cuda graph")]):
        xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(VARIANTS))]
        vals = [rows[v][key] for v in VARIANTS]
        bars = ax.bar(xs, vals, width=w, color=color, label=label)
        for b, val in zip(bars, vals):
            ax.annotate(f"{val:.1f}", (b.get_x() + b.get_width() / 2, val), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(len(VARIANTS)), VARIANTS, color=INK, fontsize=9)
    ax.set_title(f"{title}: ms per decode step, b=1", color=INK, fontsize=10, loc="left")
    ax.set_ylim(0, max(rows[v]["eager_ms_per_step"] for v in VARIANTS) * 1.2)
axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
fig.suptitle("Decode step: eager launches vs one captured CUDA graph", color=INK, fontsize=12, x=0.02, ha="left")
fig.text(0.02, 0.885, "A10G, 50 tokens, median of 5  •  graphed replays ~200 (GPT-2) / ~1,000 (Qwen3) launches as one",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "cuda_graph.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
