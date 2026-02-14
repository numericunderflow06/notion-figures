"""
fig_002: Curriculum Learning Stages
Visual representation of the 10-stage curriculum learning pipeline showing
progression from general time-series understanding to specific eye-tracking tasks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- Data: all curriculum stages from curriculum_learning.py ---
stages = [
    {"num": "1",  "name": "MCQ",                "dataset": "TSQA",               "task": "Time-series MCQ",                    "domain": "general"},
    {"num": "2",  "name": "Captioning",          "dataset": "M4",                 "task": "Time-series captioning",             "domain": "general"},
    {"num": "3",  "name": "HAR CoT",             "dataset": "HAR",                "task": "Activity recognition CoT",           "domain": "motion"},
    {"num": "4",  "name": "Sleep CoT",           "dataset": "SleepEDF",           "task": "Sleep stage classification CoT",     "domain": "sleep"},
    {"num": "5",  "name": "ECG CoT",             "dataset": "ECG QA",             "task": "ECG question answering CoT",         "domain": "cardiac"},
    {"num": "6",  "name": "Financial Reports",   "dataset": "Financial Reports",  "task": "Post-filing return prediction",      "domain": "financial"},
    {"num": "7",  "name": "EEG Reading Task",    "dataset": "ZuCo 1.0 EEG",      "task": "Reading task classification",        "domain": "cognitive"},
    {"num": "8",  "name": "EEG Sentiment",       "dataset": "ZuCo 1.0 EEG",      "task": "Sentiment classification",           "domain": "cognitive"},
    {"num": "9",  "name": "ET Reading Task",     "dataset": "ZuCo 1.0 ET",       "task": "Eye-tracking NR vs TSR",             "domain": "cognitive"},
    {"num": "9b", "name": "ET2 Reading Task",    "dataset": "ZuCo 2.0 ET",       "task": "Eye-tracking NR vs TSR",             "domain": "cognitive"},
]

# Domain colors
domain_colors = {
    "general":   "#4A90D9",  # Blue
    "motion":    "#E8913A",  # Orange
    "sleep":     "#9B59B6",  # Purple
    "cardiac":   "#E74C3C",  # Red
    "financial": "#27AE60",  # Green
    "cognitive": "#2C3E50",  # Dark navy
}

domain_labels = {
    "general":   "General TS",
    "motion":    "Motion/HAR",
    "sleep":     "Sleep",
    "cardiac":   "Cardiac",
    "financial": "Financial",
    "cognitive": "Cognitive/EEG/ET",
}

# Branch points: 4 initialization options for ZuCo experiments
branch_points = {
    "scratch":       {"source_stage": None,  "label": "From Scratch\n(Llama 3.2 3B)", "color": "#95A5A6"},
    "HAR (stage 3)": {"source_stage": "3",   "label": "HAR-flamingo\n(after stage 3)", "color": "#E8913A"},
    "Sleep (stage 4)":{"source_stage": "4",  "label": "Sleep-flamingo\n(after stage 4)", "color": "#9B59B6"},
    "Full (stage 8)": {"source_stage": "8",  "label": "Full curriculum\n(after stage 8)", "color": "#2C3E50"},
}

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(-1, 10)
ax.set_ylim(-3.0, 22)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(5, 21.3, "OpenTSLM Curriculum Learning Pipeline",
        fontsize=18, fontweight="bold", ha="center", va="center",
        color="#1a1a1a")
ax.text(5, 20.6, "10 stages from general time-series understanding to eye-tracking classification",
        fontsize=11, ha="center", va="center", color="#555555", style="italic")

# --- Draw main staircase pipeline ---
box_width = 8.0
box_height = 1.4
x_center = 5.0
y_start = 19.2
y_step = 1.85

for i, stage in enumerate(stages):
    y = y_start - i * y_step
    color = domain_colors[stage["domain"]]

    # Main stage box
    box = FancyBboxPatch(
        (x_center - box_width / 2, y - box_height / 2),
        box_width, box_height,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.92,
    )
    ax.add_patch(box)

    # Stage number badge (circle on left)
    badge_x = x_center - box_width / 2 + 0.55
    badge = plt.Circle((badge_x, y), 0.38, color="white", ec=color, linewidth=2, zorder=5)
    ax.add_patch(badge)
    ax.text(badge_x, y, stage["num"], fontsize=12, fontweight="bold",
            ha="center", va="center", color=color, zorder=6)

    # Stage name
    ax.text(badge_x + 0.8, y + 0.22, stage["name"],
            fontsize=12, fontweight="bold", ha="left", va="center", color="white")

    # Dataset + Task
    ax.text(badge_x + 0.8, y - 0.22,
            f'{stage["dataset"]}  \u2022  {stage["task"]}',
            fontsize=9, ha="left", va="center", color=(1, 1, 1, 0.88))

    # Connecting arrow between stages (except last)
    if i < len(stages) - 1:
        arrow_y_start = y - box_height / 2 - 0.02
        arrow_y_end = y - y_step + box_height / 2 + 0.02
        ax.annotate(
            "", xy=(x_center, arrow_y_end), xytext=(x_center, arrow_y_start),
            arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.8,
                            connectionstyle="arc3,rad=0"),
        )

    # Mark branch points with a diamond marker on the right side
    is_branch = stage["num"] in ["3", "4", "8"]
    if is_branch:
        marker_x = x_center + box_width / 2 + 0.35
        ax.plot(marker_x, y, marker="D", markersize=10, color=color,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)


# --- Draw branch arrows to stages 9 / 9b ---
# The target stages are at indices 8 (stage 9) and 9 (stage 9b)
target_y_9 = y_start - 8 * y_step
target_y_9b = y_start - 9 * y_step

# Right-side panel: branch point legend
panel_x = x_center + box_width / 2 + 1.2
panel_y_start = y_start - 2 * y_step  # near stage 3

ax.text(panel_x + 1.0, panel_y_start + 2.0, "Initialization\nBranch Points",
        fontsize=11, fontweight="bold", ha="center", va="center", color="#333333")

bp_items = [
    ("From Scratch", "Llama 3.2 3B, no pretraining", "#95A5A6", None),
    ("HAR-flamingo", "Checkpoint after stage 3", "#E8913A", "3"),
    ("Sleep-flamingo", "Checkpoint after stage 4", "#9B59B6", "4"),
    ("Full Curriculum", "Checkpoint after stage 8", "#2C3E50", "8"),
]

for j, (label, desc, color, src) in enumerate(bp_items):
    by = panel_y_start + 0.8 - j * 1.3
    # Small colored box
    bp_box = FancyBboxPatch(
        (panel_x - 0.2, by - 0.38), 2.4, 0.76,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.15,
    )
    ax.add_patch(bp_box)

    # Diamond marker
    ax.plot(panel_x + 0.05, by, marker="D" if src else "o", markersize=8,
            color=color, markeredgecolor="white", markeredgewidth=1, zorder=5)

    ax.text(panel_x + 0.4, by + 0.07, label,
            fontsize=9.5, fontweight="bold", ha="left", va="center", color=color)
    ax.text(panel_x + 0.4, by - 0.2, desc,
            fontsize=7.5, ha="left", va="center", color="#777777")

# --- Draw curved branch arrows from branch points to stages 9/9b ---
# These show that stages 9 and 9b can be initialized from different checkpoints

# We draw small arrows from the right-side branch panel down to the ZuCo stages
# with labels showing what each produces

zuco_box_y9 = y_start - 8 * y_step
zuco_box_y9b = y_start - 9 * y_step

# Highlight stages 9 and 9b with a background glow
for idx, zy in [(8, zuco_box_y9), (9, zuco_box_y9b)]:
    glow = FancyBboxPatch(
        (x_center - box_width / 2 - 0.12, zy - box_height / 2 - 0.08),
        box_width + 0.24, box_height + 0.16,
        boxstyle="round,pad=0.15",
        facecolor="none",
        edgecolor="#F39C12",
        linewidth=2.5,
        linestyle="--",
        zorder=3,
    )
    ax.add_patch(glow)

# Label the ZuCo target stages
ax.text(x_center - box_width / 2 - 0.4, (zuco_box_y9 + zuco_box_y9b) / 2,
        "ZuCo\nTargets",
        fontsize=9, fontweight="bold", ha="center", va="center",
        color="#F39C12", rotation=90,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#F39C12", linewidth=1))

# --- Domain legend at the bottom ---
legend_y = -0.8
legend_x_start = 0.5
ax.text(x_center, legend_y - 0.6, "Domain Color Legend", fontsize=10,
        fontweight="bold", ha="center", va="center", color="#333333")

domain_order = ["general", "motion", "sleep", "cardiac", "financial", "cognitive"]
n_domains = len(domain_order)
spacing = box_width / n_domains
start_x = x_center - box_width / 2 + spacing / 2

for k, dom in enumerate(domain_order):
    lx = start_x + k * spacing
    ly = legend_y - 1.2
    circle = plt.Circle((lx, ly), 0.2, color=domain_colors[dom], zorder=5)
    ax.add_patch(circle)
    ax.text(lx, ly - 0.45, domain_labels[dom],
            fontsize=8, ha="center", va="center", color="#444444")

# --- Save ---
plt.tight_layout(pad=1.0)
fig.savefig("/home/wangni/notion-figures/zuco/fig_002.png", dpi=200,
            bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()

print("Saved: /home/wangni/notion-figures/zuco/fig_002.png")
