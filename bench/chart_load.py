#render bench/load_results.json -> load_results.png for the README

"""Two panels, one story: as concurrency rises past the slot count,
latency (esp. the p95 tail) explodes while req/s stalls -- saturation.

Usage: python bench/chart_load.py   (after bench/load.py has written the json)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = json.loads((HERE / "load_results.json").read_text())

x = range(len(rows))
labels = [str(r["concurrency"]) for r in rows]

# reference palette (light): categorical slots in fixed order + chrome ink
VIOLET, GREEN, RED, OLIVE = "#2e8f95", "#8b5e3c", "#c4473a", "#c4473a"  #p95 teal, p50 red, ttft brown, throughput teal
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
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("concurrent requests", color=MUTED, fontsize=9)

# panel 1: latency percentiles + ttft (3 series -> categorical, direct-labeled)
series = [("p95 latency", [r["p95_s"] for r in rows], VIOLET),
          ("p50 latency", [r["p50_s"] for r in rows], RED),
          ("TTFT", [r["ttft_s"] for r in rows], GREEN)]
for name, ys, color in series:
    ax1.plot(list(x), ys, color=color, linewidth=2, marker="o", markersize=5)
    ax1.annotate(name, (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                 color=INK, fontsize=9, va="center")
ax1.set_title("Latency per request, seconds", color=INK, fontsize=10, loc="left")
ax1.set_xlim(-0.3, len(rows) + 1.1)  # room for the direct labels

# panel 2: DELIVERED tokens/s (req/s x 25), vs the 4-slot raw decode ceiling
CEILING = 88  # b=4 median from the batch sweep: what 4 always-busy slots produce
tps = [r["req_per_s"] * 25 for r in rows]
bars = ax2.bar(list(x), tps, width=0.55, color=VIOLET)
for b, v in zip(bars, tps):
    ax2.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                 textcoords="offset points", ha="center", color=INK, fontsize=9)
ax2.axhline(CEILING, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
ax2.annotate("raw decode ceiling, 4 slots (~88)", (0, CEILING), xytext=(0, 4),
             textcoords="offset points", color=MUTED, fontsize=8)
ax2.set_ylim(0, CEILING * 1.18)
ax2.set_title("Delivered throughput, tok/s", color=INK, fontsize=10, loc="left")

fig.suptitle("GPT-2 on the M1: serving under load, 4 slots",
             color=INK, fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.885, "End to end over HTTP and SSE. 25 tokens per request, 3 waves per concurrency level, medians.",
         color=MUTED, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "load_results.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
