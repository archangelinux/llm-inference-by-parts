#render the A10G results (modal_results.json + modal_load_results.json) -> modal.png

"""Two panels: batch sweep (total tok/s vs b, log-log-ish) and serving load
(delivered tok/s vs concurrency, with the 4-slot raw ceiling).

Usage: python bench/chart_modal.py   (after both modal benches have run)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sweep = json.loads((HERE / "modal_results.json").read_text())["batched"]
load = json.loads((HERE / "modal_load_results.json").read_text())

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

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

# panel 1: batch sweep, total tok/s (log y -- spans 168 to 16.5k)
bs = sorted(int(b) for b in sweep)
totals = [sweep[str(b)] for b in bs]
x1 = range(len(bs))
bars = ax1.bar(list(x1), totals, width=0.6, color=BLUE)
ax1.set_yscale("log")
for bar, v in zip(bars, totals):
    ax1.annotate(f"{v:,.0f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                 textcoords="offset points", ha="center", color=INK, fontsize=8)
ax1.set_xticks(list(x1), [str(b) for b in bs])
ax1.set_xlabel("batch size", color=MUTED, fontsize=9)
ax1.set_title("decode throughput vs batch size (tok/s, log)", color=INK, fontsize=10, loc="left")

# panel 2: serving, delivered tok/s vs concurrency, vs the 4-slot raw ceiling
CEILING = 634  # b=4 median from this sweep: 4 always-busy slots
x2 = range(len(load))
tps = [r["delivered_tok_s"] for r in load]
bars = ax2.bar(list(x2), tps, width=0.55, color=BLUE)
for bar, v in zip(bars, tps):
    ax2.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                 textcoords="offset points", ha="center", color=INK, fontsize=9)
ax2.axhline(CEILING, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
ax2.annotate("raw decode ceiling, 4 slots (~634)", (0, CEILING), xytext=(0, 4),
             textcoords="offset points", color=MUTED, fontsize=8)
ax2.set_ylim(0, CEILING * 1.18)
ax2.set_xticks(list(x2), [str(r["concurrency"]) for r in load])
ax2.set_xlabel("concurrent requests", color=MUTED, fontsize=9)
ax2.set_title("serving: delivered throughput (tok/s)", color=INK, fontsize=10, loc="left")

fig.suptitle("A10G results (Modal)", color=INK, fontsize=12, x=0.02, ha="left")
fig.text(0.02, 0.885, "GPT-2 124M fp32, greedy, median of 5 (sweep) / 3 waves (serving, loopback, 25 tok/request)",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "modal.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
