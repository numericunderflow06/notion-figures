#!/usr/bin/env python3
"""
fig_004: Component Comparison — OpenTSLM vs QoQ-Med vs JEPA-Flamingo-DRPO
Three-column visual comparison across five key dimensions with color-coded
strength/weakness indicators.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Data from the verified spec comparison table
# ---------------------------------------------------------------------------
systems = ["OpenTSLM", "QoQ-Med", "JEPA-Flamingo-DRPO"]

dimensions = [
    "TS Encoding",
    "Fusion Efficiency",
    "Training Objective",
    "Domain Balance",
    "Trainable Params",
]

# (label, quality) where quality: "good", "neutral", "weak"
cells = {
    # TS Encoding
    ("TS Encoding", "OpenTSLM"):              ("Conv1D\nfrom scratch",              "weak"),
    ("TS Encoding", "QoQ-Med"):               ("ECG-JEPA\n(pretrained)",             "good"),
    ("TS Encoding", "JEPA-Flamingo-DRPO"):    ("ECG-JEPA\n(pretrained, frozen)",     "best"),

    # Fusion Efficiency
    ("Fusion Efficiency", "OpenTSLM"):            ("Perceiver +\ncross-attn\n(constant memory)", "good"),
    ("Fusion Efficiency", "QoQ-Med"):             ("Token\nconcatenation\n(linear memory)",      "weak"),
    ("Fusion Efficiency", "JEPA-Flamingo-DRPO"):  ("Perceiver +\ncross-attn\n(constant memory)", "best"),

    # Training Objective
    ("Training Objective", "OpenTSLM"):            ("Supervised\ncross-entropy",      "neutral"),
    ("Training Objective", "QoQ-Med"):             ("DRPO\n(domain-aware RL)",        "good"),
    ("Training Objective", "JEPA-Flamingo-DRPO"):  ("Supervised \u2192 DRPO\n(both phases)", "best"),

    # Domain Balance
    ("Domain Balance", "OpenTSLM"):            ("None",                       "weak"),
    ("Domain Balance", "QoQ-Med"):             ("DRPO temperature\nscaling",  "good"),
    ("Domain Balance", "JEPA-Flamingo-DRPO"):  ("DRPO temperature\nscaling",  "best"),

    # Trainable Params
    ("Trainable Params", "OpenTSLM"):            ("Encoder + cross-attn\n+ LoRA",       "neutral"),
    ("Trainable Params", "QoQ-Med"):             ("Full model\nor LoRA",                "weak"),
    ("Trainable Params", "JEPA-Flamingo-DRPO"):  ("Perceiver + cross-attn\nonly (~200M)", "best"),
}

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COL_BEST    = "#1B813E"   # deep green — best / combined strength
COL_GOOD    = "#4A90D9"   # blue — good
COL_NEUTRAL = "#E8A838"   # amber — acceptable / neutral
COL_WEAK    = "#C0392B"   # red — weakness

BG_BEST     = "#E8F5E9"
BG_GOOD     = "#E3F0FC"
BG_NEUTRAL  = "#FFF8E1"
BG_WEAK     = "#FDEDEC"

quality_map = {
    "best":    (COL_BEST,    BG_BEST,    "\u2714"),   # heavy check
    "good":    (COL_GOOD,    BG_GOOD,    "\u2714"),
    "neutral": (COL_NEUTRAL, BG_NEUTRAL, "\u25CB"),   # circle
    "weak":    (COL_WEAK,    BG_WEAK,    "\u2718"),   # cross
}

HEADER_COLORS = {
    "OpenTSLM":            ("#34495E", "#D5DBDB"),
    "QoQ-Med":             ("#34495E", "#D5DBDB"),
    "JEPA-Flamingo-DRPO":  ("#145A32", "#A9DFBF"),
}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
n_rows = len(dimensions)
n_cols = len(systems)

col_width  = 3.6
row_height = 1.25
label_col_width = 2.0
header_height = 1.0
title_height = 0.85

fig_w = label_col_width + n_cols * col_width + 0.4
fig_h = title_height + header_height + n_rows * row_height + 0.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")
fig.patch.set_facecolor("white")

x_start = label_col_width + 0.2          # left edge of data columns
y_top   = fig_h - title_height - 0.15    # top of header row

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(
    fig_w / 2, fig_h - 0.15,
    "Component Comparison: OpenTSLM  vs  QoQ-Med  vs  JEPA-Flamingo-DRPO",
    fontsize=14, fontweight="bold", ha="center", va="top",
    color="#1a1a1a",
)

# Subtitle
ax.text(
    fig_w / 2, fig_h - 0.55,
    "How JEPA-Flamingo-DRPO selects the best component from each predecessor",
    fontsize=10, ha="center", va="top", color="#555555", style="italic",
)

# ---------------------------------------------------------------------------
# Column headers
# ---------------------------------------------------------------------------
for j, sys in enumerate(systems):
    cx = x_start + j * col_width + col_width / 2
    cy = y_top - header_height / 2

    fg, bg = HEADER_COLORS[sys]
    rect = FancyBboxPatch(
        (x_start + j * col_width + 0.08, y_top - header_height + 0.05),
        col_width - 0.16, header_height - 0.10,
        boxstyle="round,pad=0.08", facecolor=bg, edgecolor=fg,
        linewidth=2 if sys == "JEPA-Flamingo-DRPO" else 1.2,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, sys, fontsize=11.5 if sys != "JEPA-Flamingo-DRPO" else 11,
            fontweight="bold", ha="center", va="center", color=fg)

# ---------------------------------------------------------------------------
# Row labels (left column)
# ---------------------------------------------------------------------------
for i, dim in enumerate(dimensions):
    cy = y_top - header_height - i * row_height - row_height / 2
    ax.text(
        label_col_width, cy, dim,
        fontsize=11, fontweight="semibold", ha="right", va="center",
        color="#2C3E50",
    )

# ---------------------------------------------------------------------------
# Data cells
# ---------------------------------------------------------------------------
for i, dim in enumerate(dimensions):
    for j, sys in enumerate(systems):
        label, quality = cells[(dim, sys)]
        fg, bg, icon = quality_map[quality]

        cx = x_start + j * col_width + col_width / 2
        cy = y_top - header_height - i * row_height - row_height / 2

        # Cell background
        pad_x, pad_y = 0.08, 0.06
        rw = col_width - 2 * pad_x
        rh = row_height - 2 * pad_y

        lw = 2.2 if (sys == "JEPA-Flamingo-DRPO") else 1.0
        ec = fg if (sys == "JEPA-Flamingo-DRPO") else "#CCCCCC"

        rect = FancyBboxPatch(
            (x_start + j * col_width + pad_x, cy - rh / 2),
            rw, rh,
            boxstyle="round,pad=0.06",
            facecolor=bg, edgecolor=ec, linewidth=lw,
        )
        ax.add_patch(rect)

        # Icon
        ax.text(
            cx - rw / 2 + 0.35, cy + 0.05,
            icon, fontsize=16, ha="center", va="center", color=fg,
            fontweight="bold",
        )

        # Label text
        ax.text(
            cx + 0.15, cy + 0.02,
            label, fontsize=9.5, ha="center", va="center", color="#1a1a1a",
            linespacing=1.25,
        )

# ---------------------------------------------------------------------------
# Horizontal separator lines
# ---------------------------------------------------------------------------
for i in range(n_rows + 1):
    y = y_top - header_height - i * row_height
    ax.plot(
        [0.15, fig_w - 0.15], [y, y],
        color="#D5D8DC", linewidth=0.6, zorder=0,
    )

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_y = 0.35
legend_items = [
    (COL_BEST,    BG_BEST,    "\u2714  Best (combined strength)"),
    (COL_GOOD,    BG_GOOD,    "\u2714  Strength"),
    (COL_NEUTRAL, BG_NEUTRAL, "\u25CB  Acceptable"),
    (COL_WEAK,    BG_WEAK,    "\u2718  Weakness"),
]

total_legend_w = len(legend_items) * 3.0
lx = (fig_w - total_legend_w) / 2

for k, (fg, bg, txt) in enumerate(legend_items):
    x = lx + k * 3.0
    rect = FancyBboxPatch(
        (x, legend_y - 0.18), 2.7, 0.38,
        boxstyle="round,pad=0.04", facecolor=bg, edgecolor=fg, linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(x + 1.35, legend_y + 0.01, txt, fontsize=8.5,
            ha="center", va="center", color=fg, fontweight="semibold")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout(pad=0.2)
out = "/home/wangni/notion-figures/qoq-med/fig_004.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
