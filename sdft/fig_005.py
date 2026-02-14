"""
fig_005: Curriculum Learning Pipeline with SDFT Stage Mapping
Horizontal pipeline showing OpenTSLM curriculum stages, color-coded SFT (blue) vs SDFT (green).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- Configuration ---
stages = [
    {"stage": "Stage 1", "name": "MCQ",       "method": "SFT"},
    {"stage": "Stage 2", "name": "Captioning", "method": "SFT"},
    {"stage": "Stage 3", "name": "HAR CoT",    "method": "SDFT"},
    {"stage": "Stage 4", "name": "Sleep CoT",  "method": "SDFT"},
    {"stage": "Stage 5", "name": "ECG CoT",    "method": "SDFT"},
    {"stage": "Stage 6+", "name": "Future CoT", "method": "SDFT"},
]

# Colors
SFT_COLOR = "#4A90D9"       # Blue for SFT
SFT_COLOR_LIGHT = "#D6E6F7" # Light blue fill
SDFT_COLOR = "#3AAA6D"      # Green for SDFT
SDFT_COLOR_LIGHT = "#D4F0E0" # Light green fill
ARROW_COLOR = "#666666"
BG_COLOR = "white"
TEXT_COLOR = "#222222"
STAGE_TEXT_COLOR = "#555555"

# Layout
fig_width = 16
fig_height = 4.2
box_w = 1.8
box_h = 1.5
gap = 0.7
y_center = 2.1
x_start = 1.2

fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis("off")

# Title
ax.text(
    fig_width / 2, fig_height - 0.35,
    "OpenTSLM Curriculum Learning Pipeline",
    ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT_COLOR,
    fontfamily="sans-serif",
)

# Draw stages
box_centers = []
for i, s in enumerate(stages):
    x = x_start + i * (box_w + gap)
    cx = x + box_w / 2
    cy = y_center
    box_centers.append((cx, cy))

    is_sft = s["method"] == "SFT"
    edge_color = SFT_COLOR if is_sft else SDFT_COLOR
    fill_color = SFT_COLOR_LIGHT if is_sft else SDFT_COLOR_LIGHT

    # Box
    box = FancyBboxPatch(
        (x, cy - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.12",
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=2.2,
    )
    ax.add_patch(box)

    # Stage label (top of box)
    ax.text(
        cx, cy + box_h / 2 - 0.25,
        s["stage"],
        ha="center", va="center", fontsize=10, color=STAGE_TEXT_COLOR,
        fontweight="medium", fontfamily="sans-serif",
    )

    # Dataset/task name (center)
    ax.text(
        cx, cy - 0.03,
        s["name"],
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color=TEXT_COLOR, fontfamily="sans-serif",
    )

    # Method badge (bottom of box)
    badge_y = cy - box_h / 2 + 0.32
    badge_color = SFT_COLOR if is_sft else SDFT_COLOR
    badge = FancyBboxPatch(
        (cx - 0.5, badge_y - 0.15), 1.0, 0.30,
        boxstyle="round,pad=0.06",
        facecolor=badge_color,
        edgecolor="none",
        alpha=0.92,
    )
    ax.add_patch(badge)
    ax.text(
        cx, badge_y,
        s["method"],
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="white", fontfamily="sans-serif",
    )

# Draw arrows between stages
for i in range(len(box_centers) - 1):
    x1 = box_centers[i][0] + box_w / 2 + 0.04
    x2 = box_centers[i + 1][0] - box_w / 2 - 0.04
    cy = y_center

    # Arrow line
    arrow = FancyArrowPatch(
        (x1, cy), (x2, cy),
        arrowstyle="->,head_width=0.18,head_length=0.12",
        color=ARROW_COLOR,
        linewidth=1.8,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arrow)

    # "checkpoint" label above arrow
    mid_x = (x1 + x2) / 2
    ax.text(
        mid_x, cy + 0.42,
        "ckpt",
        ha="center", va="center", fontsize=8, color="#888888",
        fontstyle="italic", fontfamily="sans-serif",
    )

# Legend
legend_y = 0.42
legend_x_start = fig_width / 2 - 2.5

# SFT legend
sft_patch = FancyBboxPatch(
    (legend_x_start, legend_y - 0.13), 0.4, 0.26,
    boxstyle="round,pad=0.04",
    facecolor=SFT_COLOR, edgecolor="none",
)
ax.add_patch(sft_patch)
ax.text(
    legend_x_start + 0.55, legend_y,
    "SFT (Supervised Fine-Tuning)",
    ha="left", va="center", fontsize=10, color=TEXT_COLOR,
    fontfamily="sans-serif",
)

# SDFT legend
sdft_x = legend_x_start + 3.6
sdft_patch = FancyBboxPatch(
    (sdft_x, legend_y - 0.13), 0.4, 0.26,
    boxstyle="round,pad=0.04",
    facecolor=SDFT_COLOR, edgecolor="none",
)
ax.add_patch(sdft_patch)
ax.text(
    sdft_x + 0.55, legend_y,
    "SDFT (Self-Distillation Fine-Tuning)",
    ha="left", va="center", fontsize=10, color=TEXT_COLOR,
    fontfamily="sans-serif",
)

plt.tight_layout(pad=0.3)
fig.savefig(
    "/home/wangni/notion-figures/sdft/fig_005.png",
    dpi=200,
    facecolor=BG_COLOR,
    bbox_inches="tight",
    pad_inches=0.2,
)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/sdft/fig_005.png")
