"""
Figure 6 — Rationale Quality: LLM-as-a-Judge Scores.

Grouped bar chart of paper Table 2: 0-2 LLM-judge scores on three clinically
anchored criteria (pattern recognition, clinical reasoning, and context
integration) for ECG-QA-CoT and Sleep-CoT, comparing OpenTSLM-Chronos2 vs
ATLAS-TS (both on Llama-3.2-3B backbone). ATLAS-TS wins all six criteria.

Values are pulled verbatim from Table 2 of paper.tex.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Data (Table 2 — 0-2 LLM-judge scores)
# ---------------------------------------------------------------------------
criteria = ["Pattern\nRecognition", "Clinical\nReasoning", "Context\nIntegration"]

# {dataset: {model: [Patt, Reas, Ctx]}}
data = {
    "ECG-QA-CoT": {
        "OpenTSLM-Chronos2": [1.22, 1.17, 1.65],
        "ATLAS-TS (ours)":   [1.67, 1.40, 1.79],
    },
    "Sleep-CoT": {
        "OpenTSLM-Chronos2": [1.52, 1.41, 1.80],
        "ATLAS-TS (ours)":   [1.89, 1.55, 1.91],
    },
}

# ---------------------------------------------------------------------------
# Style (consistent with fig_004 — dark teal accent for ATLAS-TS)
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

COLOR_BASELINE = "#a0a4ab"   # muted grey
EDGE_BASELINE  = "#5a5e66"
COLOR_OURS     = "#1f4e52"   # dark teal (same accent as fig_004)
EDGE_OURS      = "#0d2e30"
HATCH_OURS     = "///"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)

n_crit = len(criteria)
group_width = 0.72
bar_width = group_width / 2          # two models per group
x = np.arange(n_crit)

for ax, (dataset, scores) in zip(axes, data.items()):
    baseline_vals = scores["OpenTSLM-Chronos2"]
    ours_vals     = scores["ATLAS-TS (ours)"]

    # Baseline bars (left)
    b1 = ax.bar(
        x - bar_width / 2,
        baseline_vals,
        width=bar_width * 0.95,
        color=COLOR_BASELINE,
        edgecolor=EDGE_BASELINE,
        linewidth=0.7,
        label="OpenTSLM-Chronos2",
        zorder=3,
    )
    # ATLAS-TS bars (right)
    b2 = ax.bar(
        x + bar_width / 2,
        ours_vals,
        width=bar_width * 0.95,
        color=COLOR_OURS,
        edgecolor=EDGE_OURS,
        linewidth=0.7,
        hatch=HATCH_OURS,
        label="ATLAS-TS (ours)",
        zorder=3,
    )

    # Numeric labels on baseline bars
    for bar, val in zip(b1, baseline_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.025,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#3a3a3a",
            zorder=4,
        )

    # Numeric labels + delta annotation on ATLAS-TS bars
    for bar, val, base in zip(b2, ours_vals, baseline_vals):
        delta = val - base
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.025,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color="#1a1a1a",
            fontweight="bold",
            zorder=4,
        )
        # Delta tag, placed just above the value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.14,
            f"+{delta:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color=COLOR_OURS,
            fontweight="bold",
            zorder=4,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#e6efef",
                edgecolor=COLOR_OURS,
                linewidth=0.5,
            ),
        )

    # Axes formatting per panel
    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.arange(0, 2.01, 0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=11)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10, color="#7a7a7a")

    # Light horizontal gridlines
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e3e3e3", zorder=0)
    ax.set_axisbelow(True)

    # Panel title
    ax.set_title(
        dataset,
        fontsize=13,
        fontweight="bold",
        pad=10,
        color="#1a1a1a",
    )

# Shared y-axis label
axes[0].set_ylabel("LLM-judge score (0–2)", fontsize=12, labelpad=8)

# Figure-level title
fig.suptitle(
    "Rationale Quality — LLM-as-a-Judge Scores",
    fontsize=15,
    fontweight="bold",
    y=0.99,
    color="#1a1a1a",
)

# Subtitle (clarifies the comparison)
fig.text(
    0.5, 0.925,
    "Llama-3.2-3B backbone. Higher is better. ATLAS-TS wins all six criteria.",
    ha="center",
    fontsize=10.5,
    color="#4a4a4a",
    style="italic",
)

# Shared legend at the bottom
legend_handles = [
    Patch(facecolor=COLOR_BASELINE, edgecolor=EDGE_BASELINE,
          linewidth=0.7, label="OpenTSLM-Chronos2 (baseline)"),
    Patch(facecolor=COLOR_OURS, edgecolor=EDGE_OURS,
          hatch=HATCH_OURS, linewidth=0.7, label="ATLAS-TS (ours)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.005),
    ncol=2,
    frameon=False,
    fontsize=11,
    handlelength=2.4,
    handleheight=1.2,
    columnspacing=2.4,
)

plt.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.16, wspace=0.10)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/wangni/notion-figures/atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_006.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
