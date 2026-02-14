"""
Figure 007: Experiment Design — Model Variants Comparison
Shows which components are active in each of the three model variants
as a clean grid/matrix with checkmarks and X marks, plus a progressive
building-blocks sidebar.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
variants = ["Baseline", "TPA-Enc", "TPA-Full"]
components = ["ATPE", "Anchors", "Cross-Attn"]

# True = component active
matrix = np.array([
    [False, False, False],  # Baseline
    [True,  False, False],  # TPA-Enc
    [True,  True,  True],   # TPA-Full
])

variant_descriptions = [
    "Learnable PE only",
    "Temporal encoding",
    "Full alignment",
]

param_costs = ["~16K params", "~260K params", "~4M params"]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
BG        = "#FFFFFF"
HEADER_BG = "#2C3E6B"
HEADER_FG = "#FFFFFF"
ROW_EVEN  = "#F0F3F9"
ROW_ODD   = "#FFFFFF"
CHECK_CLR = "#27AE60"
CROSS_CLR = "#E74C3C"
BORDER    = "#D5D8DC"
ACCENT    = "#2C3E6B"

# Progressive block colors
BLOCK_COLORS = ["#3498DB", "#E67E22", "#9B59B6"]  # blue, orange, purple

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 7), facecolor=BG)

# Two areas: left = matrix grid, right = building blocks
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.15,
                      left=0.03, right=0.97, top=0.84, bottom=0.12)

ax_grid = fig.add_subplot(gs[0, 0])
ax_blocks = fig.add_subplot(gs[0, 1])

# Title
fig.text(0.50, 0.94, "Experiment Design: Model Variants Comparison",
         fontsize=17, fontweight="bold", ha="center", va="center",
         color=ACCENT, fontfamily="sans-serif")
fig.text(0.50, 0.895, "Three ablation variants progressively adding TPA components",
         fontsize=11, ha="center", va="center", color="#5D6D7E",
         fontfamily="sans-serif")

# =========================================================================
# LEFT PANEL — Matrix / Grid
# =========================================================================
ax = ax_grid
ax.set_xlim(-0.1, 4.0)
ax.set_ylim(-0.1, 4.3)
ax.axis("off")
ax.set_aspect("equal")

cell_w = 0.95
cell_h = 0.88
x_start = 1.05
y_start = 0.25

# Column headers
for j, comp in enumerate(components):
    cx = x_start + j * cell_w + cell_w / 2
    cy = y_start + 3 * cell_h + cell_h / 2 + 0.15
    rect = FancyBboxPatch((x_start + j * cell_w + 0.03, y_start + 3 * cell_h + 0.10),
                          cell_w - 0.06, cell_h + 0.10,
                          boxstyle="round,pad=0.05", facecolor=HEADER_BG,
                          edgecolor="none")
    ax.add_patch(rect)
    ax.text(cx, cy + 0.08, comp, ha="center", va="center",
            fontsize=11, fontweight="bold", color=HEADER_FG, fontfamily="sans-serif")
    ax.text(cx, cy - 0.22, param_costs[j], ha="center", va="center",
            fontsize=7.5, color="#C0C8D8", fontfamily="sans-serif")

# Row headers + cells
for i, var in enumerate(variants):
    row_y = y_start + (2 - i) * cell_h
    row_color = ROW_EVEN if i % 2 == 0 else ROW_ODD

    # Row background
    row_rect = FancyBboxPatch((-0.05, row_y + 0.02),
                              x_start + 3 * cell_w + 0.10, cell_h - 0.04,
                              boxstyle="round,pad=0.04", facecolor=row_color,
                              edgecolor=BORDER, linewidth=0.5)
    ax.add_patch(row_rect)

    # Variant label
    ax.text(0.48, row_y + cell_h / 2 + 0.08, var, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=ACCENT, fontfamily="sans-serif")
    ax.text(0.48, row_y + cell_h / 2 - 0.16, variant_descriptions[i],
            ha="center", va="center", fontsize=8, color="#7F8C8D",
            fontfamily="sans-serif", style="italic")

    for j in range(3):
        cx = x_start + j * cell_w + cell_w / 2
        cy = row_y + cell_h / 2

        if matrix[i, j]:
            circle = plt.Circle((cx, cy), 0.24, facecolor="#E8F8F0",
                                edgecolor=CHECK_CLR, linewidth=2)
            ax.add_patch(circle)
            ax.text(cx, cy, "\u2713", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=CHECK_CLR,
                    fontfamily="sans-serif")
        else:
            circle = plt.Circle((cx, cy), 0.24, facecolor="#FDEDEC",
                                edgecolor=CROSS_CLR, linewidth=2)
            ax.add_patch(circle)
            ax.text(cx, cy, "\u2717", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=CROSS_CLR,
                    fontfamily="sans-serif")

# =========================================================================
# RIGHT PANEL — Progressive Building Blocks
# =========================================================================
ax2 = ax_blocks
ax2.set_xlim(-0.5, 4.0)
ax2.set_ylim(-1.0, 5.5)
ax2.axis("off")

ax2.text(1.75, 5.2, "Progressive Architecture", ha="center", va="center",
         fontsize=12, fontweight="bold", color=ACCENT, fontfamily="sans-serif")

block_h = 0.50
block_w = 2.6
gap = 0.10
base_x = 0.45

# Three variant groups positioned from top to bottom
# Baseline at top, TPA-Enc in middle, TPA-Full at bottom
group_tops = [4.5, 3.0, 0.7]

for vi, (var, desc) in enumerate(zip(variants, variant_descriptions)):
    top_y = group_tops[vi]
    active = [j for j in range(3) if matrix[vi, j]]
    n_blocks = max(len(active), 1)

    # Variant label above the group
    ax2.text(base_x + block_w / 2, top_y + 0.35, var,
             ha="center", va="center", fontsize=10.5, fontweight="bold",
             color=ACCENT, fontfamily="sans-serif")

    if not active:
        # Baseline — dashed empty box
        rect = FancyBboxPatch((base_x, top_y - block_h), block_w, block_h,
                              boxstyle="round,pad=0.06", facecolor="#F5F5F5",
                              edgecolor="#BDC3C7", linewidth=1.3, linestyle="--")
        ax2.add_patch(rect)
        ax2.text(base_x + block_w / 2, top_y - block_h / 2,
                 "Learnable PE  (no TPA components)", ha="center", va="center",
                 fontsize=8.5, color="#95A5A6", fontfamily="sans-serif",
                 style="italic")
        group_bottom = top_y - block_h
    else:
        for k, j in enumerate(active):
            by = top_y - (k + 1) * (block_h + gap) + gap
            rect = FancyBboxPatch((base_x, by), block_w, block_h,
                                  boxstyle="round,pad=0.06",
                                  facecolor=BLOCK_COLORS[j],
                                  edgecolor="white", linewidth=1.5, alpha=0.90)
            ax2.add_patch(rect)
            ax2.text(base_x + block_w / 2, by + block_h / 2 + 0.04,
                     components[j], ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color="white",
                     fontfamily="sans-serif")
            ax2.text(base_x + block_w / 2, by + block_h / 2 - 0.14,
                     param_costs[j], ha="center", va="center",
                     fontsize=7.5, color="#E8E8E8", fontfamily="sans-serif")
        group_bottom = by

    # Store bottom for arrow drawing
    if vi == 0:
        g0_bottom = group_bottom
    elif vi == 1:
        g1_bottom = group_bottom

# Arrows between groups
arrow_x = base_x + block_w / 2
# Baseline → TPA-Enc
ax2.annotate("", xy=(arrow_x, group_tops[1] + 0.35 + 0.18),
             xytext=(arrow_x, g0_bottom - 0.05),
             arrowprops=dict(arrowstyle="-|>", color="#BDC3C7",
                             lw=1.8, mutation_scale=15))
ax2.text(arrow_x + 0.05, (g0_bottom + group_tops[1] + 0.35) / 2 + 0.05,
         "+ ATPE", ha="left", va="center", fontsize=8, fontweight="bold",
         color="#3498DB", fontfamily="sans-serif")

# TPA-Enc → TPA-Full
ax2.annotate("", xy=(arrow_x, group_tops[2] + 0.35 + 0.18),
             xytext=(arrow_x, g1_bottom - 0.05),
             arrowprops=dict(arrowstyle="-|>", color="#BDC3C7",
                             lw=1.8, mutation_scale=15))
ax2.text(arrow_x + 0.05, (g1_bottom + group_tops[2] + 0.35) / 2 + 0.10,
         "+ Anchors\n+ Cross-Attn", ha="left", va="center", fontsize=8,
         fontweight="bold", color="#8E44AD", fontfamily="sans-serif")

# Bottom note
fig.text(0.50, 0.04,
         "Total new parameters for TPA-Full: ~4.3M  (small vs. encoder ~1.5M, tiny vs. LLM ~1B)",
         ha="center", va="center", fontsize=9, color="#7F8C8D",
         fontfamily="sans-serif", style="italic")

# Legend for block colors at bottom
for j, (comp, clr) in enumerate(zip(components, BLOCK_COLORS)):
    lx = 0.30 + j * 0.18
    fig.patches.append(mpatches.FancyBboxPatch(
        (lx, 0.07), 0.013, 0.018,
        boxstyle="round,pad=0.002", facecolor=clr, edgecolor="none",
        transform=fig.transFigure, alpha=0.88))
    fig.text(lx + 0.019, 0.079, comp, fontsize=8, va="center",
             color="#5D6D7E", fontfamily="sans-serif")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.savefig("/home/wangni/notion-figures/alignment/fig_007.png",
            dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.15)
plt.close()
print("Saved: /home/wangni/notion-figures/alignment/fig_007.png")
