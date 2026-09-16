#the headline figure: every stage of the engine, both models, one A10G -> headline.png

"""Three panels from the result files:
  1. decode throughput by stage (naive at 512 ctx, KV cache at 512 ctx, batched b=128)   modal_results*.json
  2. one attention projection per op: cuBLAS fp16 vs the fused int8 Triton kernel        modal_quant_results*.json
  3. ms per decode step at b=1, eager vs CUDA graph, fp16 and int8-kernel                modal_graph_results.json

Usage: python bench/chart_headline.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
load = lambda name: json.loads((HERE / name).read_text())
res = {"gpt2": load("modal_results.json"), "qwen": load("modal_results.qwen.json")}
quant = {"gpt2": load("modal_quant_results.json"), "qwen": load("modal_quant_results.qwen.json")}
graph = load("modal_graph_results.json")["runs"]

LIGHT, DARK, INK, MUTED, GRID, SURFACE = "#ae8a41", "#517a97", "#1a1a1a", "#8b8a85", "#e4e3dd", "#fdfdfc"
TEAL_GREEN, LAVENDER = "#3fb59a", "#9c96c7"  # the portfolio palette: one hue per model
MODELS = [("gpt2", "GPT-2 124M", TEAL_GREEN), ("qwen", "Qwen3 0.6B", LAVENDER)]
W = 0.36

#two rows: stages and the kernel on top, precision + cuda graphs full width below
fig = plt.figure(figsize=(11, 8), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.55, wspace=0.25)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
fig.patch.set_facecolor(SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

def grouped(ax, labels, series, fmt="{:,.0f}"):
    #series: list of (name, values, color); one bar group per label
    for j, (name, vals, color) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * (W + 0.03) for i in range(len(labels))]
        bars = ax.bar(xs, vals, width=W, color=color, label=name)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK, fontsize=7)
    ax.set_xticks(range(len(labels)), labels, color=INK, fontsize=8)

# 1) stages
ax = axes[0]
stages = ["naive\n(512 ctx)", "KV cache\n(512 ctx)", "batched\n(b=128)"]
grouped(ax, stages, [(label, [res[m]["naive"]["512"], res[m]["cached"]["512"], res[m]["batched"]["128"]], c) for m, label, c in MODELS])
ax.set_yscale("log"); ax.set_ylabel("tok/s (log)", color=MUTED, fontsize=8)
ax.set_title("Stages 1-3: decode throughput by mechanism, fp32", color=INK, fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

# 2) the kernel, per op
ax = axes[1]
ops = ["PyTorch (cuBLAS)\nfp16 weights", "int8 weights,\nour Triton kernel"]
grouped(ax, ops, [(label, [quant[m]["c_attn_op"]["fp16_linear"]["us"], quant[m]["c_attn_op"]["int8_kernel"]["us"]], c) for m, label, c in MODELS], fmt="{:.1f}")
ax.set_ylabel("microseconds per matmul (b=1)", color=MUTED, fontsize=8)
ax.set_title("Stage 5: one attention matmul, cuBLAS fp16 vs int8 kernel", color=INK, fontsize=10, loc="left")
ax.set_ylim(0, max(quant[m]["c_attn_op"]["fp16_linear"]["us"] for m, _, _ in MODELS) * 1.3)

# 3) eager vs graph
ax = axes[2]
row = {(r["model"], r["variant"]): r for r in graph if r["n_slots"] == 1}
labels = ["fp16\neager", "fp16\nCUDA graph", "int8-kernel\neager", "int8-kernel\nCUDA graph"]
def vals(m):
    return [row[(m, "fp16")]["eager_ms_per_step"], row[(m, "fp16")]["graphed_ms_per_step"],
            row[(m, "int8-kernel")]["eager_ms_per_step"], row[(m, "int8-kernel")]["graphed_ms_per_step"]]
grouped(ax, labels, [(label, vals(m), c) for m, label, c in MODELS], fmt="{:.1f}")
ax.set_ylabel("ms per decode step (b=1)", color=MUTED, fontsize=8)
ax.set_title("Stages 5-7: time per decode step at b=1, eager vs CUDA graph", color=INK, fontsize=10, loc="left")
ax.tick_params(labelsize=9)

fig.suptitle("GPT-2 and Qwen3 on the same engine, A10G", color=INK, fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.935, "All bars: greedy (argmax) decoding, median of 5 runs. Panel numbers are the stages in the table above.",
         color=MUTED, fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = HERE / "headline.png"
fig.savefig(out, facecolor=SURFACE)
print(f"wrote {out}")
