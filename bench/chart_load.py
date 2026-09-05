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
BLUE, GREEN, MAGENTA = "#2a78d6", "#008300", "#e87ba4"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

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
series = [("p95 latency", [r["p95_s"] for r in rows], MAGENTA),
          ("p50 latency", [r["p50_s"] for r in rows], BLUE),
          ("TTFT", [r["ttft_s"] for r in rows], GREEN)]
for name, ys, color in series:
    ax1.plot(list(x), ys, color=color, linewidth=2, marker="o", markersize=5)
    ax1.annotate(name, (x[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                 color=INK, fontsize=9, va="center")
ax1.set_title("latency per request (s)", color=INK, fontsize=10, loc="left")
ax1.set_xlim(-0.3, len(rows) + 1.1)  # room for the direct labels

# panel 2: DELIVERED tokens/s (req/s x 25), vs the 4-slot raw decode ceiling
CEILING = 88  # b=4 median from the batch sweep: what 4 always-busy slots produce
tps = [r["req_per_s"] * 25 for r in rows]
bars = ax2.bar(list(x), tps, width=0.55, color=BLUE)
for b, v in zip(bars, tps):
    ax2.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                 textcoords="offset points", ha="center", color=INK, fontsize=9)
ax2.axhline(CEILING, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
ax2.annotate("raw decode ceiling, 4 slots (~88)", (0, CEILING), xytext=(0, 4),
             textcoords="offset points", color=MUTED, fontsize=8)
ax2.set_ylim(0, CEILING * 1.18)
ax2.set_title("delivered throughput (tok/s)", color=INK, fontsize=10, loc="left")

fig.suptitle("Serving load test (M1, 4 slots)",
             color=INK, fontsize=12, x=0.02, ha="left")
fig.text(0.02, 0.885, "HTTP+SSE end to end, M1, 4 slots, 25 tokens/request, 3 waves per level, medians",
         color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / "load_results.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
