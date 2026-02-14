"""
fig_006: OpenTSLM Curriculum with SDFT Stages
Horizontal pipeline/arrow diagram showing all curriculum stages.
SDFT stages highlighted in gold/amber; supervised stages in blue;
evaluation-only stages with dashed borders.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# --- Stage data (from curriculum_learning.py + plan.md SDFT stages) ---
stages = [
    {"id": "1",  "name": "MCQ\n(TSQA)",               "epochs": 30, "type": "supervised"},
    {"id": "2",  "name": "Captioning\n(M4)",            "epochs": 20, "type": "supervised"},
    {"id": "TS", "name": "TSExam\nEval",                "epochs": None, "type": "eval"},
    {"id": "3",  "name": "HAR CoT",                     "epochs": 30, "type": "supervised"},
    {"id": "3s", "name": "HAR\nSDFT",                   "epochs": None, "type": "sdft"},
    {"id": "4",  "name": "Sleep CoT",                   "epochs": 60, "type": "supervised"},
    {"id": "4s", "name": "Sleep\nSDFT",                  "epochs": None, "type": "sdft"},
    {"id": "5",  "name": "ECG QA\nCoT",                 "epochs": 60, "type": "supervised"},
    {"id": "6",  "name": "Financial\nReports",           "epochs": 30, "type": "supervised"},
    {"id": "7",  "name": "EEG\nReading",                "epochs": 30, "type": "supervised"},
    {"id": "8",  "name": "EEG\nSentiment",              "epochs": 30, "type": "supervised"},
    {"id": "9",  "name": "ET Reading\n(ZuCo 1.0)",      "epochs": 30, "type": "supervised"},
    {"id": "9b", "name": "ET Reading\n(ZuCo 2.0)",      "epochs": 30, "type": "supervised"},
]

# --- Color palette ---
colors = {
    "supervised":      "#3B82F6",   # blue-500
    "supervised_edge": "#2563EB",   # blue-600
    "supervised_text": "#FFFFFF",
    "sdft":            "#F59E0B",   # amber-500
    "sdft_edge":       "#D97706",   # amber-600
    "sdft_text":       "#FFFFFF",
    "sdft_glow":       "#FDE68A",   # amber-200
    "eval":            "#9CA3AF",   # gray-400
    "eval_edge":       "#6B7280",   # gray-500
    "eval_text":       "#FFFFFF",
    "arrow":           "#94A3B8",   # slate-400
    "arrow_head":      "#64748B",   # slate-500
    "bg":              "#FFFFFF",
    "label_dark":      "#1E293B",   # slate-800
    "label_mid":       "#475569",   # slate-600
}

# --- Layout parameters ---
box_w = 1.7
box_h = 1.35
gap = 0.5
row_gap = 1.5  # vertical gap between rows

# Two rows: first 7 stages on top, remaining 6 on bottom
split = 7
row1 = stages[:split]
row2 = stages[split:]

total_w_row1 = split * box_w + (split - 1) * gap
total_w_row2 = len(row2) * box_w + (len(row2) - 1) * gap
fig_w = max(total_w_row1, total_w_row2) + 3.0
fig_h = 2 * box_h + row_gap + 3.6

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.patch.set_facecolor(colors["bg"])
ax.set_facecolor(colors["bg"])

# Center each row
x_offset_row1 = (fig_w - total_w_row1) / 2
x_offset_row2 = (fig_w - total_w_row2) / 2
y_row1 = fig_h - 1.7
y_row2 = y_row1 - box_h - row_gap


def draw_stage_box(ax, cx, cy, stage, box_w, box_h):
    """Draw a single stage box at center (cx, cy)."""
    stype = stage["type"]
    fc = colors[stype]
    ec = colors[stype + "_edge"] if stype != "eval" else colors["eval_edge"]
    tc = colors[stype + "_text"]

    x0 = cx - box_w / 2
    y0 = cy - box_h / 2

    ls = "--" if stype == "eval" else "-"
    lw = 2.5 if stype == "sdft" else 1.8

    # Glow effect for SDFT stages
    if stype == "sdft":
        glow = FancyBboxPatch(
            (x0 - 0.06, y0 - 0.06), box_w + 0.12, box_h + 0.12,
            boxstyle="round,pad=0.10",
            facecolor="none", edgecolor=colors["sdft_glow"],
            linewidth=4.0, zorder=2, alpha=0.6,
        )
        ax.add_patch(glow)

    box = FancyBboxPatch(
        (x0, y0), box_w, box_h,
        boxstyle="round,pad=0.08",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, linestyle=ls,
        zorder=3,
    )
    ax.add_patch(box)

    # Stage number label (top of box)
    id_text = stage["id"]
    if id_text == "TS":
        label = "Eval"
    elif id_text == "3s":
        label = "Stage 3\u2032"
    elif id_text == "4s":
        label = "Stage 4\u2032"
    else:
        label = f"Stage {id_text}"

    ax.text(cx, cy + 0.25, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=tc, zorder=4)

    # Stage name (center-lower of box)
    name_text = stage["name"]
    ax.text(cx, cy - 0.15, name_text,
            ha="center", va="center", fontsize=8.5,
            color=tc, zorder=4, linespacing=1.15)

    # Epochs label below box
    if stage["epochs"] is not None:
        ax.text(cx, y0 - 0.15, f"{stage['epochs']} ep",
                ha="center", va="top", fontsize=7.5, color=colors["label_mid"],
                zorder=4)
    elif stype == "eval":
        ax.text(cx, y0 - 0.15, "eval only",
                ha="center", va="top", fontsize=7.5, color=colors["label_mid"],
                style="italic", zorder=4)
    elif stype == "sdft":
        ax.text(cx, y0 - 0.15, "self-distill",
                ha="center", va="top", fontsize=7.5, color="#B45309",
                style="italic", fontweight="bold", zorder=4)


def draw_arrow_h(ax, x1, y, x2):
    """Draw a horizontal connecting arrow."""
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="->,head_width=0.14,head_length=0.1",
        color=colors["arrow_head"], linewidth=1.6,
        zorder=2,
    )
    ax.add_patch(arrow)


# --- Draw Row 1 ---
centers_row1 = []
for i, stage in enumerate(row1):
    cx = x_offset_row1 + i * (box_w + gap) + box_w / 2
    cy = y_row1
    centers_row1.append((cx, cy))
    draw_stage_box(ax, cx, cy, stage, box_w, box_h)

for i in range(len(centers_row1) - 1):
    x1 = centers_row1[i][0] + box_w / 2 + 0.04
    x2 = centers_row1[i + 1][0] - box_w / 2 - 0.04
    draw_arrow_h(ax, x1, y_row1, x2)

# --- Draw Row 2 ---
centers_row2 = []
for i, stage in enumerate(row2):
    cx = x_offset_row2 + i * (box_w + gap) + box_w / 2
    cy = y_row2
    centers_row2.append((cx, cy))
    draw_stage_box(ax, cx, cy, stage, box_w, box_h)

for i in range(len(centers_row2) - 1):
    x1 = centers_row2[i][0] + box_w / 2 + 0.04
    x2 = centers_row2[i + 1][0] - box_w / 2 - 0.04
    draw_arrow_h(ax, x1, y_row2, x2)

# --- Connecting arrow: row 1 last -> row 2 first (right-angle path) ---
last_r1 = centers_row1[-1]
first_r2 = centers_row2[0]

# Path: go right from last box in row1, then down, then left to first box in row2
turn_x = last_r1[0] + box_w / 2 + 0.6  # go right past the last box
mid_y = (y_row1 - box_h / 2 + y_row2 + box_h / 2) / 2  # halfway point vertically
turn_x2 = first_r2[0] - box_w / 2 - 0.6  # approach from left of first row2 box

# Segment 1: right from last row1 box
ax.annotate("", xy=(turn_x, y_row1), xytext=(last_r1[0] + box_w / 2 + 0.04, y_row1),
            arrowprops=dict(arrowstyle="-", color=colors["arrow_head"], lw=1.6))

# Segment 2: down from turn point
ax.annotate("", xy=(turn_x, y_row2), xytext=(turn_x, y_row1),
            arrowprops=dict(arrowstyle="-", color=colors["arrow_head"], lw=1.6))

# Segment 3: left to approach row2
ax.annotate("", xy=(turn_x2, y_row2), xytext=(turn_x, y_row2),
            arrowprops=dict(arrowstyle="-", color=colors["arrow_head"], lw=1.6))

# Segment 4: into first row2 box (with arrowhead)
ax.annotate("", xy=(first_r2[0] - box_w / 2 - 0.04, y_row2),
            xytext=(turn_x2, y_row2),
            arrowprops=dict(arrowstyle="->,head_width=0.14,head_length=0.1",
                            color=colors["arrow_head"], lw=1.6))

# --- Title ---
ax.text(fig_w / 2, fig_h - 0.30, "OpenTSLM Curriculum with SDFT Stages",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color=colors["label_dark"], zorder=5)

# --- Legend ---
legend_y = y_row2 - box_h / 2 - 1.0
legend_items = [
    ("Supervised Training", colors["supervised"], colors["supervised_edge"], "-"),
    ("Self-Distilled Fine-Tuning (SDFT)", colors["sdft"], colors["sdft_edge"], "-"),
    ("Evaluation Only", colors["eval"], colors["eval_edge"], "--"),
]

total_legend_w = sum(len(lab) * 0.095 + 0.8 for lab, _, _, _ in legend_items) + 0.6
legend_x_start = fig_w / 2 - total_legend_w / 2
lx = legend_x_start

for label, fc, ec, ls in legend_items:
    rect = FancyBboxPatch(
        (lx, legend_y - 0.13), 0.36, 0.26,
        boxstyle="round,pad=0.03",
        facecolor=fc, edgecolor=ec,
        linewidth=1.3, linestyle=ls, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(lx + 0.52, legend_y, label,
            ha="left", va="center", fontsize=9.5, color=colors["label_dark"],
            zorder=4)
    lx += len(label) * 0.095 + 0.8

# --- Axis cleanup ---
ax.set_xlim(0, fig_w)
ax.set_ylim(legend_y - 0.6, fig_h)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0.3)
out_path = "/home/wangni/notion-figures/self-distillation/fig_006.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=colors["bg"], edgecolor="none")
plt.close()
print(f"Figure saved to {out_path}")
