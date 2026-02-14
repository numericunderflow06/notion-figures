"""
fig_001: OpenTSLM 10-Stage Curriculum Learning Pipeline
Horizontal dataflow diagram with domain color-coding and Stage 6 highlighted.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- Stage data ---
stages = [
    {"id": "1", "label": "Time-Series\nMCQ (TSQA)", "domain": "general"},
    {"id": "2", "label": "Time-Series\nCaptioning (M4)", "domain": "general"},
    {"id": "3", "label": "HAR\nCoT Reasoning", "domain": "biosignal"},
    {"id": "4", "label": "Sleep Stage\nClassification CoT", "domain": "biosignal"},
    {"id": "5", "label": "ECG QA\nCoT", "domain": "biosignal"},
    {"id": "6", "label": "Financial Reports\nPost-Filing Return\nPrediction", "domain": "financial"},
    {"id": "7", "label": "ZuCo 1.0 EEG\nReading Task\nClassification", "domain": "eeg"},
    {"id": "8", "label": "ZuCo 1.0 EEG\nSentiment\nClassification", "domain": "eeg"},
    {"id": "9a", "label": "ZuCo 1.0\nEye-Tracking\nReading Task", "domain": "eeg"},
    {"id": "9b", "label": "ZuCo 2.0\nEye-Tracking\nReading Task", "domain": "eeg"},
]

# --- Color palette ---
domain_colors = {
    "general":   {"box": "#4A90D9", "bg": "#E8F0FE", "border": "#2B5EA7", "text": "#FFFFFF"},
    "biosignal": {"box": "#3AAA6D", "bg": "#E6F5EC", "border": "#267A4B", "text": "#FFFFFF"},
    "financial": {"box": "#D4A017", "bg": "#FFF8E1", "border": "#9B7510", "text": "#FFFFFF"},
    "eeg":       {"box": "#7B5EA7", "bg": "#F0EAF8", "border": "#553D7A", "text": "#FFFFFF"},
}

domain_labels = {
    "general": "General Time-Series",
    "biosignal": "Biosignal",
    "financial": "Financial",
    "eeg": "EEG / Eye-Tracking",
}

# --- Layout: two rows of 5 ---
n_cols = 5
row_y = [4.0, 1.0]  # y-centers for each row
x_positions = [1.6 + i * 3.2 for i in range(n_cols)]  # x-centers

box_w = 2.6
box_h = 1.6
stage6_w = 2.9
stage6_h = 1.95

fig, ax = plt.subplots(figsize=(18, 9.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# --- Draw domain background bands ---
band_specs = [
    (x_positions[0] - box_w/2 - 0.35, x_positions[1] + box_w/2 + 0.35, row_y[0], "general"),
    (x_positions[2] - box_w/2 - 0.35, x_positions[4] + box_w/2 + 0.35, row_y[0], "biosignal"),
    (x_positions[0] - stage6_w/2 - 0.35, x_positions[0] + stage6_w/2 + 0.35, row_y[1], "financial"),
    (x_positions[1] - box_w/2 - 0.35, x_positions[4] + box_w/2 + 0.35, row_y[1], "eeg"),
]

band_h = 2.8
for x0, x1, yc, domain in band_specs:
    rect = FancyBboxPatch(
        (x0, yc - band_h/2), x1 - x0, band_h,
        boxstyle="round,pad=0.15",
        facecolor=domain_colors[domain]["bg"],
        edgecolor=domain_colors[domain]["border"],
        linewidth=1.0, alpha=0.45, zorder=0
    )
    ax.add_patch(rect)
    # Domain label at top-left of band (outside/above boxes)
    ax.text(
        x0 + 0.25, yc + band_h/2 - 0.12, domain_labels[domain],
        ha="left", va="top", fontsize=9, fontstyle="italic",
        color=domain_colors[domain]["border"], fontweight="semibold", zorder=4
    )

# --- Draw stage boxes ---
box_patches = []
box_centers = []

for idx, stage in enumerate(stages):
    row = 0 if idx < 5 else 1
    col = idx if row == 0 else idx - 5
    cx = x_positions[col]
    cy = row_y[row]

    domain = stage["domain"]
    is_stage6 = (stage["id"] == "6")

    w = stage6_w if is_stage6 else box_w
    h = stage6_h if is_stage6 else box_h
    lw = 3.5 if is_stage6 else 1.5
    fc = domain_colors[domain]["box"]
    ec = domain_colors[domain]["border"]

    # Shadow for stage 6
    if is_stage6:
        shadow = FancyBboxPatch(
            (cx - w/2 + 0.07, cy - h/2 - 0.07), w, h,
            boxstyle="round,pad=0.18",
            facecolor="#00000018", edgecolor="none", zorder=1
        )
        ax.add_patch(shadow)

    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.18",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=2
    )
    ax.add_patch(box)

    # Stage number label
    stage_num_label = f"Stage {stage['id']}"
    ax.text(
        cx, cy + h/2 - 0.28, stage_num_label,
        ha="center", va="top",
        fontsize=10 if not is_stage6 else 11.5,
        fontweight="bold",
        color=domain_colors[domain]["text"],
        zorder=3
    )

    # Stage description label
    desc_y_offset = -0.08 if not is_stage6 else -0.05
    ax.text(
        cx, cy + desc_y_offset, stage["label"],
        ha="center", va="center",
        fontsize=9.5 if not is_stage6 else 10.5,
        fontweight="normal" if not is_stage6 else "semibold",
        color=domain_colors[domain]["text"],
        linespacing=1.15, zorder=3
    )

    # Star marker for stage 6
    if is_stage6:
        ax.text(
            cx, cy - h/2 + 0.22, "★ This Project",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color="#FFFFFF", zorder=3
        )

    box_patches.append(box)
    box_centers.append((cx, cy, w, h, row))

# --- Draw arrows between stages ---
arrow_color = "#555555"
arrow_kw = dict(
    arrowstyle="-|>",
    color=arrow_color,
    linewidth=1.8,
    mutation_scale=16,
    connectionstyle="arc3,rad=0",
    zorder=1
)

# Row 0: horizontal arrows between stages 1-5
for i in range(4):
    cx1, cy1, w1, h1, _ = box_centers[i]
    cx2, cy2, w2, h2, _ = box_centers[i + 1]
    arrow = FancyArrowPatch(
        (cx1 + w1/2 + 0.08, cy1),
        (cx2 - w2/2 - 0.08, cy2),
        **arrow_kw
    )
    ax.add_patch(arrow)

# Curved arrow from stage 5 (end of row 0) down to stage 6 (start of row 1)
cx5, cy5, w5, h5, _ = box_centers[4]
cx6, cy6, w6, h6, _ = box_centers[5]
arrow_turn = FancyArrowPatch(
    (cx5, cy5 - h5/2 - 0.08),
    (cx6, cy6 + h6/2 + 0.08),
    arrowstyle="-|>",
    color=arrow_color,
    linewidth=2.2,
    mutation_scale=18,
    connectionstyle="arc3,rad=-0.5",
    zorder=1
)
ax.add_patch(arrow_turn)

# Label on the turn arrow
mid_x = (cx5 + cx6) / 2 - 1.6
mid_y = (cy5 + cy6) / 2
ax.text(
    mid_x, mid_y, "checkpoint\ntransfer",
    ha="center", va="center", fontsize=8.5,
    color="#555555", fontstyle="italic", zorder=3,
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CCCCCC", alpha=0.9)
)

# Row 1: horizontal arrows between stages 6-9b
for i in range(5, 9):
    cx1, cy1, w1, h1, _ = box_centers[i]
    cx2, cy2, w2, h2, _ = box_centers[i + 1]
    arrow = FancyArrowPatch(
        (cx1 + w1/2 + 0.08, cy1),
        (cx2 - w2/2 - 0.08, cy2),
        **arrow_kw
    )
    ax.add_patch(arrow)

# --- Title ---
ax.text(
    8.4, 7.2, "OpenTSLM 10-Stage Curriculum Learning Pipeline",
    ha="center", va="center",
    fontsize=18, fontweight="bold", color="#222222", zorder=3
)
ax.text(
    8.4, 6.7,
    "Flamingo-style cross-attention on Llama 3.2 3B  ·  Checkpoints transfer sequentially between stages",
    ha="center", va="center",
    fontsize=11, color="#666666", zorder=3
)

# --- Legend (placed below title, compact horizontal layout) ---
legend_items = [
    ("General Time-Series", "general"),
    ("Biosignal", "biosignal"),
    ("Financial", "financial"),
    ("EEG / Eye-Tracking", "eeg"),
]
# Center the legend across the figure
total_legend_width = len(legend_items) * 3.2
legend_start_x = 8.4 - total_legend_width / 2 + 0.5
legend_y = 6.2

for i, (label, domain) in enumerate(legend_items):
    lx = legend_start_x + i * 3.2
    rect = FancyBboxPatch(
        (lx - 0.2, legend_y - 0.15), 0.35, 0.3,
        boxstyle="round,pad=0.04",
        facecolor=domain_colors[domain]["box"],
        edgecolor=domain_colors[domain]["border"],
        linewidth=1.2, zorder=3
    )
    ax.add_patch(rect)
    ax.text(
        lx + 0.3, legend_y, label,
        ha="left", va="center", fontsize=9.5,
        color="#444444", fontweight="medium", zorder=3
    )

# --- Axis settings ---
ax.set_xlim(-0.5, 17.5)
ax.set_ylim(-0.8, 7.8)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0.5)
plt.savefig("/home/wangni/notion-figures/finance/fig_001.png", dpi=200,
            bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("Saved: /home/wangni/notion-figures/finance/fig_001.png")
