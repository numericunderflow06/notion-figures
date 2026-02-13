#!/usr/bin/env python3
"""
fig_003: Parameter Distribution — Frozen vs Trainable
Horizontal stacked bar chart with breakdown of trainable components.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data from verified facts ---
# Total: 10.5B parameters
# Frozen: 7B (Qwen 2.5-7B LLM body)
# Trainable: ~3.5B split across components
total_params = 10.5  # billions
frozen_params = 7.0
trainable_params = 3.5

# Trainable component breakdown (approximate splits based on architecture)
# Components: CNN Encoder, Perceiver Resampler, Gated Cross-Attention layers, LLM Input Embeddings
# Gated cross-attention at every decoder layer dominates trainable params
trainable_components = {
    "Gated Cross-Attention\nLayers": 2.4,
    "Perceiver\nResampler": 0.6,
    "LLM Input\nEmbeddings": 0.4,
    "CNN\nEncoder": 0.1,
}

# --- Colors ---
frozen_color = "#9CA8B8"       # muted gray
trainable_color = "#3B82F6"    # vibrant blue
component_colors = [
    "#1D4ED8",  # Gated Cross-Attention - darkest blue
    "#3B82F6",  # Perceiver Resampler
    "#7CB3F7",  # LLM Input Embeddings
    "#B8D4FA",  # CNN Encoder - lightest blue
]

# --- Figure setup ---
fig = plt.figure(figsize=(12, 5.0), facecolor="white")

# Use manual axes placement for precise control
ax1 = fig.add_axes([0.06, 0.58, 0.88, 0.25])   # [left, bottom, width, height]
ax2 = fig.add_axes([0.06, 0.10, 0.88, 0.25])

bar_height = 0.5

# ============================================================
# TOP BAR: Frozen vs Trainable
# ============================================================
frozen_pct = frozen_params / total_params * 100
trainable_pct = trainable_params / total_params * 100

# Draw bars
ax1.barh(0, frozen_params, height=bar_height, color=frozen_color,
         edgecolor="white", linewidth=0.8)
ax1.barh(0, trainable_params, height=bar_height, left=frozen_params,
         color=trainable_color, edgecolor="white", linewidth=0.8)

# Labels inside bars
ax1.text(frozen_params / 2, 0,
         f"Frozen  |  7.0B  ({frozen_pct:.0f}%)",
         ha="center", va="center", fontsize=11.5, fontweight="bold",
         color="white")
ax1.text(frozen_params + trainable_params / 2, 0,
         f"Trainable  |  3.5B  ({trainable_pct:.0f}%)",
         ha="center", va="center", fontsize=10.5, fontweight="bold",
         color="white")

# Styling
ax1.set_xlim(0, total_params)
ax1.set_ylim(-0.38, 0.38)
ax1.set_yticks([])
ax1.set_xlabel("Parameters (Billions)", fontsize=10, labelpad=6)
ax1.set_xticks(np.arange(0, 11.5, 1))
ax1.tick_params(axis="x", labelsize=9)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)

# Title
fig.text(0.06, 0.93, "Parameter Distribution: Frozen vs Trainable",
         fontsize=14, fontweight="bold", color="#1E293B")
fig.text(0.06, 0.88,
         "Frozen: Qwen 2.5-7B LLM body   |   "
         "Trainable: CNN Encoder, Perceiver Resampler, "
         "Gated Cross-Attention, LLM Input Embeddings",
         fontsize=9, color="#64748B")

# ============================================================
# BOTTOM BAR: Trainable breakdown by component
# ============================================================
# Section label
fig.text(0.06, 0.42, "Trainable Parameter Breakdown (~3.5B)",
         fontsize=12, fontweight="bold", color="#1E293B")

left = 0
comp_names = list(trainable_components.keys())
comp_values = list(trainable_components.values())

for i, (name, val) in enumerate(zip(comp_names, comp_values)):
    ax2.barh(0, val, height=bar_height, left=left,
             color=component_colors[i], edgecolor="white", linewidth=0.8)

    pct = val / trainable_params * 100
    cx = left + val / 2

    if val >= 0.5:
        # Label inside bar
        ax2.text(cx, 0, f"{name}\n{val}B ({pct:.0f}%)",
                 ha="center", va="center", fontsize=9, fontweight="600",
                 color="white", linespacing=1.2)
    elif val >= 0.3:
        # Label inside bar, smaller font
        ax2.text(cx, 0, f"{name}\n{val}B ({pct:.0f}%)",
                 ha="center", va="center", fontsize=7.5, fontweight="600",
                 color="white", linespacing=1.1)
    else:
        # Place label above for narrow segments
        ax2.annotate(f"{name}\n{val}B ({pct:.0f}%)",
                     xy=(cx, bar_height / 2), xytext=(cx - 0.05, 0.52),
                     ha="center", va="bottom", fontsize=8, fontweight="500",
                     color="#334155", linespacing=1.1,
                     arrowprops=dict(arrowstyle="-", color="#94A3B8",
                                     lw=0.8))
    left += val

# Styling
ax2.set_xlim(0, trainable_params * 1.01)
ax2.set_ylim(-0.38, 0.80)
ax2.set_yticks([])
ax2.set_xlabel("Trainable Parameters (Billions)", fontsize=10, labelpad=6)
ax2.set_xticks(np.arange(0, 4.0, 0.5))
ax2.tick_params(axis="x", labelsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

# ============================================================
# Save
# ============================================================
fig.savefig("/home/wangni/notion-figures/polymarkets/fig_003.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved: /home/wangni/notion-figures/polymarkets/fig_003.png")
