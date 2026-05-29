"""
Figure 4 — Main Results: Accuracy Across Four Benchmarks.

Grouped bar chart of Accuracy (%) on TSQA, HAR-CoT, Sleep-CoT, ECG-QA-CoT,
comparing the three strongest baselines against ATLAS-TS (1B and 3B).

Values are pulled from Table 1 of paper.tex (Acc columns).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data (Accuracy %, Table 1)
# ---------------------------------------------------------------------------
benchmarks = ["TSQA", "HAR-CoT", "Sleep-CoT", "ECG-QA-CoT"]

models = [
    ("OpenTSLM-Chronos2\nLlama-3.2-1B",   [99.71, 70.61, 78.54, 44.01], "baseline"),
    ("OpenTSLM-Chronos2\nLlama-3.2-3B",   [99.58, 72.20, 78.31, 44.27], "baseline"),
    ("OpenTSLM-SoftPrompt\nLlama-3.2-1B", [97.54, 71.48, 81.08, 35.49], "baseline"),
    ("ATLAS-TS (ours)\nLlama-3.2-1B",     [99.89, 73.06, 88.81, 68.14], "ours_1b"),
    ("ATLAS-TS (ours)\nLlama-3.2-3B",     [99.63, 78.96, 90.56, 71.50], "ours_3b"),
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
})

# Muted greys for baselines; dark teal for ATLAS-TS
COLORS = {
    "baseline_1": "#c9ccd1",   # light grey
    "baseline_2": "#a0a4ab",   # medium grey
    "baseline_3": "#6e7480",   # dark grey
    "ours_1b":    "#2f6f73",   # dark teal
    "ours_3b":    "#1f4e52",   # darker teal
}
HATCH = {
    "ours_1b": "///",
    "ours_3b": "xxx",
}

bar_face = [
    COLORS["baseline_1"],
    COLORS["baseline_2"],
    COLORS["baseline_3"],
    COLORS["ours_1b"],
    COLORS["ours_3b"],
]
bar_hatch = [None, None, None, HATCH["ours_1b"], HATCH["ours_3b"]]
bar_edge  = ["#5a5e66", "#5a5e66", "#3a3d42", "#0d2e30", "#0d2e30"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 5.6))

n_models = len(models)
n_bench  = len(benchmarks)
group_width = 0.82
bar_width = group_width / n_models
x = np.arange(n_bench)

for i, (label, vals, kind) in enumerate(models):
    offset = (i - (n_models - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset,
        vals,
        width=bar_width * 0.95,
        color=bar_face[i],
        edgecolor=bar_edge[i],
        linewidth=0.7,
        hatch=bar_hatch[i],
        label=label,
        zorder=3,
    )
    # Numeric labels on top
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.8,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#1a1a1a",
            fontweight="bold" if kind.startswith("ours") else "normal",
            zorder=4,
        )

# Axes formatting
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel("Accuracy (%)", fontsize=12, labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.tick_params(axis="x", length=0, pad=8)
ax.tick_params(axis="y", labelsize=10, color="#7a7a7a")

# Light horizontal gridlines
ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e3e3e3", zorder=0)
ax.set_axisbelow(True)

# Title
ax.set_title(
    "Main Results — Accuracy Across Four Benchmarks",
    fontsize=14,
    fontweight="bold",
    pad=14,
    color="#1a1a1a",
)

# Legend below
legend_handles = [
    Patch(facecolor=bar_face[i], edgecolor=bar_edge[i],
          hatch=bar_hatch[i], linewidth=0.7,
          label=models[i][0].replace("\n", " "))
    for i in range(n_models)
]
leg = ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    frameon=False,
    fontsize=9.5,
    handlelength=2.2,
    handleheight=1.2,
    columnspacing=1.6,
    labelspacing=0.6,
)

# Subtle annotation: largest gap on ECG-QA-CoT
best_baseline_ecg = max(99.71*0+44.01, 44.27, 35.49)  # = 44.27
ours_best_ecg = 71.50
gap_ecg = ours_best_ecg - best_baseline_ecg  # +27.2 pp

ax.annotate(
    f"+{gap_ecg:.1f} pp",
    xy=(x[3] + (n_models-1)/2 * bar_width, ours_best_ecg + 1.0),
    xytext=(x[3] + (n_models-1)/2 * bar_width + 0.05, 95),
    fontsize=10,
    fontweight="bold",
    color="#1f4e52",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#1f4e52", lw=1.0,
                    connectionstyle="arc3,rad=0.15"),
)

# Tight layout leaving room for legend
plt.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.22)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/wangni/notion-figures/atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_004.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
