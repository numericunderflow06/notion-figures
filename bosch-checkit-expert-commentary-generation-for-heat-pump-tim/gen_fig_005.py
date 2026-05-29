"""
fig_005: NLG Metrics by Subset (All / Space Heating / DHW)

Grouped bar chart comparing BLEU, ROUGE-L, BERTScore F1, and STS cosine
across All / Space Heating / DHW subsets for Bosch CheckIt Training Run #1.
"""

import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_005.png"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

metrics = ["BLEU", "ROUGE-L", "BERTScore F1", "STS cosine"]
subsets = ["All (n=76)", "Space Heating (n=38)", "DHW (n=38)"]

values = np.array([
    [30.16, 42.41, 13.75],   # BLEU
    [46.86, 58.47, 35.25],   # ROUGE-L
    [79.80, 83.21, 76.38],   # BERTScore F1
    [86.06, 92.29, 79.84],   # STS cosine
])

bosch_blue = "#1F3A93"
bosch_orange = "#E67E22"
bosch_green = "#27AE60"
colors = [bosch_blue, bosch_orange, bosch_green]

fig, ax = plt.subplots(figsize=(11, 6))

n_groups = len(metrics)
n_bars = len(subsets)
bar_width = 0.25
x = np.arange(n_groups)

for i, (subset, color) in enumerate(zip(subsets, colors)):
    offset = (i - (n_bars - 1) / 2) * bar_width
    bars = ax.bar(
        x + offset,
        values[:, i],
        bar_width,
        label=subset,
        color=color,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, values[:, i]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_xlabel("Metric", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 105)
ax.set_yticks(np.arange(0, 101, 10))
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.set_title(
    "NLG Metrics by Subset — Bosch CheckIt Training Run #1 (76-sample test split)",
    fontsize=13,
    pad=18,
)

leg = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.06),
    ncol=3,
    frameon=False,
    fontsize=11,
)

fig.text(
    0.5, 0.005,
    "OpenTSLM-Flamingo + Llama-3.2-1B  ·  Pattern: Space Heating > All > DHW across all four metrics",
    ha="center", va="bottom", fontsize=9, style="italic", color="#555555",
)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

fig.savefig(OUT_PATH, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PATH}")
