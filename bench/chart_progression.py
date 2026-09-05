#throughput progression: naive -> KV cache -> batched -> served. README header figure.

"""
log scale (the range spans 1 -> 552 tok/s)

Each bar is a different stage of the project measured in its own bench; configs
differ by construction (that's the point -- each stage unlocked a new regime):
  naive:    b=1, 512-token context, no cache     (bench/run.py, median)
  cached:   b=1, 512-token context, KV cache     (bench/run.py, median)
  batched:  b=16, prompt 16, static batch        (bench/batch.py, median)
(the served/delivered story lives in the load-test section: different metric,
different config -- it doesn't belong on this axis)

"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent

def median_of(path, **match):
    for line in (HERE / path).read_text().splitlines():
        r = json.loads(line)
        if all(r.get(k) == v for k, v in match.items()):
            return statistics.median(r["tok_per_sec"])
    raise KeyError(match)

naive = median_of("results.jsonl", label="naive", prompt_len=512)
cached = median_of("results.jsonl", label="cached", prompt_len=512)
batched = median_of("batch_results.jsonl", batch_size=16)
stages = ["naive\n(no cache)", "+ KV cache", "+ batching\n(b=16)"]
values = [naive, cached, batched]

# reference palette: sequential blue ramp for the engine progression (one hue,
# more-is-darker = magnitude); served is a different kind of number -> green slot
RAMP = ["#9ec5f4", "#5598e7", "#1c5cab"]
GREEN = "#008300"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
bars = ax.bar(stages, values, width=0.55, color=RAMP)
ax.set_yscale("log")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.set_ylabel("tokens / second (log)", color=MUTED, fontsize=9)

for b, v in zip(bars, values):
    ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                textcoords="offset points", ha="center", color=INK, fontsize=10)

ax.set_title("Decode throughput by optimization stage",
             color=INK, fontsize=12, loc="left", pad=30)
ax.text(0, 1.115, "GPT-2 124M, M1 (MPS), greedy, 50 new tokens, median of 5",
        transform=ax.transAxes, color=MUTED, fontsize=8)
ax.text(0, 1.055, "naive/cached: one sequence, 512-token context  •  batched: 16 short prompts, total tok/s",
        transform=ax.transAxes, color=MUTED, fontsize=8)
fig.tight_layout()
out = HERE / "progression.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
for s, v in zip(stages, values):
    print(f"  {s.replace(chr(10), ' '):28s} {v:8.1f} tok/s")
