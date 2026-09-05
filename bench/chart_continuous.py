#continuous vs static batching, per-request completion time: dumbbell chart

"""Reads continuous_results.jsonl -> continuous.png. One row per request,
a dot pair (static vs continuous) joined by a line: the line length IS the win.
Requests are sorted by static completion time so the queueing story reads
top-to-bottom: early-group requests barely move, late/short ones collapse.

Usage: python bench/chart_continuous.py   (after bench/continuous.py)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
rows = [json.loads(l) for l in (HERE / "continuous_results.jsonl").read_text().splitlines()]
rows.sort(key=lambda r: (r["static_s"], r["continuous_s"]))

# one measure, two states -> one hue, two shades (light=before/static, dark=after/continuous)
LIGHT_BLUE, DARK_BLUE = "#86b6ef", "#1c5cab"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)

ys = range(len(rows))
for y, r in zip(ys, rows):
    ax.plot([r["continuous_s"], r["static_s"]], [y, y], color=GRID, linewidth=2, zorder=1)
ax.scatter([r["static_s"] for r in rows], list(ys), s=55, color=LIGHT_BLUE, zorder=2, label="static groups")
ax.scatter([r["continuous_s"] for r in rows], list(ys), s=55, color=DARK_BLUE, zorder=2, label="continuous")

ax.set_yticks(list(ys), [r["prompt"][:28] for r in rows], fontsize=8, color=INK)
ax.tick_params(axis="x", colors=MUTED, labelsize=9)
ax.set_xlabel("completion time (s) — lower is better", color=MUTED, fontsize=9)
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")

ax.set_title("Per-request completion time: static vs continuous batching",
             color=INK, fontsize=12, loc="left", pad=24)
ax.text(0, 1.05, "8 requests, 3 slots, M1  •  finishes stagger (2–50 tokens)  •  "
        "one line = one request under both schedulers",
        transform=ax.transAxes, color=MUTED, fontsize=8)
fig.tight_layout()
out = HERE / "continuous.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
