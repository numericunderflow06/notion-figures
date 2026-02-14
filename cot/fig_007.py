"""
fig_007: Curriculum Integration — SFT + GRPO Stages
Two-row horizontal timeline showing the modified curriculum.
Row 1: Stages 1–5g (original + new GRPO stages)
Row 2: Stages 6–9 (unchanged, shown in grey), left-aligned under row 1
Checkpoint flow arrows connect stages; SFT→GRPO pairs are bracketed by domain.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ── colours ──────────────────────────────────────────────────────────────
SFT_BLUE       = "#2563EB"
SFT_BLUE_LIGHT = "#DBEAFE"
GRPO_ORANGE    = "#D97706"
GRPO_ORANGE_LT = "#FEF3C7"
GREY           = "#9CA3AF"
GREY_LIGHT     = "#F3F4F6"
ARROW_COLOR    = "#6B7280"
TEXT_DARK      = "#111827"
TEXT_MID       = "#4B5563"
TEXT_LIGHT     = "#6B7280"
CKPT_GREEN     = "#059669"
WHITE          = "#FFFFFF"

# ── figure setup ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7.0), dpi=200)
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ── layout constants ─────────────────────────────────────────────────────
box_w    = 1.45
box_h    = 1.05
gap_norm = 0.55   # normal gap between stages
gap_ckpt = 0.80   # wider gap for checkpoint arrows (SFT↔GRPO)
row1_y   = 3.0    # vertical centre of row 1
row2_y   = 0.0    # vertical centre of row 2

# ── row 1 stage definitions ─────────────────────────────────────────────
row1_stages = [
    ("Stage 1",  "MCQ\n(TSQA)",        "sft",   None),
    ("Stage 2",  "Captioning\n(M4)",   "sft",   None),
    ("Stage 3",  "HAR CoT\n(SFT)",     "sft",   None),
    ("Stage 3g", "HAR CoT\n(GRPO)",    "grpo",  "68K"),
    ("Stage 4",  "Sleep CoT\n(SFT)",   "sft",   None),
    ("Stage 4g", "Sleep CoT\n(GRPO)",  "grpo",  "7K"),
    ("Stage 5",  "ECG-QA CoT\n(SFT)", "sft",   None),
    ("Stage 5g", "ECG-QA CoT\n(GRPO)","grpo",  "159K"),
]

def get_gap(i, stages):
    if i >= len(stages) - 1:
        return 0
    k1, k2 = stages[i][2], stages[i+1][2]
    if (k1 == "sft" and k2 == "grpo") or (k1 == "grpo" and k2 == "sft"):
        return gap_ckpt
    return gap_norm

row1_x = []
x = 0.0
for i in range(len(row1_stages)):
    row1_x.append(x)
    x += box_w + get_gap(i, row1_stages)
row1_total_w = row1_x[-1] + box_w

# ── row 2: left-aligned under row 1 ─────────────────────────────────────
row2_stages = [
    ("Stage 6",  "...",  "unchanged", None),
    ("Stage 7",  "...",  "unchanged", None),
    ("Stage 8",  "...",  "unchanged", None),
    ("Stage 9",  "...",  "unchanged", None),
]
row2_x = []
for i in range(len(row2_stages)):
    row2_x.append(i * (box_w + gap_norm))


# ── helper: draw a stage box ────────────────────────────────────────────
def draw_stage(x, y, stage, show_samples_below=True):
    label_top, label_bot, kind, samples = stage
    cx = x + box_w / 2

    if kind == "sft":
        fc, ec, lw = SFT_BLUE_LIGHT, SFT_BLUE, 2.2
    elif kind == "grpo":
        fc, ec, lw = GRPO_ORANGE_LT, GRPO_ORANGE, 2.2
    else:
        fc, ec, lw = GREY_LIGHT, GREY, 1.5

    rect = mpatches.FancyBboxPatch(
        (x, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=2,
    )
    ax.add_patch(rect)

    top_color = SFT_BLUE if kind == "sft" else (GRPO_ORANGE if kind == "grpo" else GREY)
    ax.text(cx, y + 0.24, label_top,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=top_color, zorder=3)

    txt_color = TEXT_DARK if kind != "unchanged" else TEXT_LIGHT
    ax.text(cx, y - 0.15, label_bot,
            ha="center", va="center", fontsize=9,
            color=txt_color, zorder=3, linespacing=1.15)

    if kind == "grpo" and samples and show_samples_below:
        ax.text(cx, y - box_h / 2 - 0.14, f"{samples} samples",
                ha="center", va="top", fontsize=9, fontstyle="italic",
                color=GRPO_ORANGE, fontweight="semibold", zorder=3)

    return cx


# ── draw rows ────────────────────────────────────────────────────────────
row1_cx = [draw_stage(row1_x[i], row1_y, s) for i, s in enumerate(row1_stages)]
row2_cx = [draw_stage(row2_x[i], row2_y, s, False) for i, s in enumerate(row2_stages)]

# ── arrows within row 1 ─────────────────────────────────────────────────
for i in range(len(row1_stages) - 1):
    x_from = row1_cx[i] + box_w / 2 + 0.04
    x_to   = row1_cx[i + 1] - box_w / 2 - 0.04
    k1, k2 = row1_stages[i][2], row1_stages[i+1][2]
    is_ckpt = (k1 == "sft" and k2 == "grpo") or (k1 == "grpo" and k2 == "sft")

    col = CKPT_GREEN if is_ckpt else ARROW_COLOR
    lw  = 2.0 if is_ckpt else 1.4

    arr = FancyArrowPatch(
        (x_from, row1_y), (x_to, row1_y),
        arrowstyle="->,head_width=0.12,head_length=0.10",
        color=col, linewidth=lw, zorder=4,
    )
    ax.add_patch(arr)

    if is_ckpt:
        mid_x = (x_from + x_to) / 2
        ax.text(mid_x, row1_y + 0.20, "ckpt",
                ha="center", va="bottom", fontsize=7.5,
                color=CKPT_GREEN, fontweight="bold", zorder=5)

# ── transition arrow: Stage 5g → Stage 6 (row jump) ─────────────────────
# Downward arrow from bottom of Stage 5g to top of Stage 6
# Since Stage 5g is far right and Stage 6 is far left, use an L-shaped path
x_5g = row1_cx[-1]
x_6  = row2_cx[0]

# Path: bottom of 5g → down to row2 level → left to Stage 6
drop_x = x_5g
drop_start_y = row1_y - box_h / 2 - 0.14
# The domain brackets and sample labels are at about row1_y - box_h/2 - 0.8
# Go below the brackets
travel_y = row1_y - box_h / 2 - 1.2

# horizontal travel left
arrive_x = x_6
arrive_y = row2_y + box_h / 2 + 0.14

# Draw with dashed line to distinguish from checkpoint flow
dash_color = ARROW_COLOR
dash_lw = 1.3

# Vertical down from 5g
ax.plot([drop_x, drop_x], [drop_start_y, travel_y],
        color=dash_color, lw=dash_lw, ls=(0, (5, 3)), zorder=3,
        solid_capstyle='round')
# Horizontal left
ax.plot([drop_x, arrive_x], [travel_y, travel_y],
        color=dash_color, lw=dash_lw, ls=(0, (5, 3)), zorder=3,
        solid_capstyle='round')
# Vertical up to Stage 6 (with arrowhead)
arr_down = FancyArrowPatch(
    (arrive_x, travel_y), (arrive_x, arrive_y),
    arrowstyle="->,head_width=0.10,head_length=0.08",
    color=dash_color, linewidth=dash_lw, linestyle='dashed', zorder=4,
)
ax.add_patch(arr_down)

# Label on the transition
mid_travel_x = (drop_x + arrive_x) / 2
ax.text(mid_travel_x, travel_y - 0.15, "continues...",
        ha="center", va="top", fontsize=8, fontstyle="italic",
        color=TEXT_LIGHT, zorder=3)

# ── arrows within row 2 ─────────────────────────────────────────────────
for i in range(len(row2_stages) - 1):
    x_from = row2_cx[i] + box_w / 2 + 0.04
    x_to   = row2_cx[i + 1] - box_w / 2 - 0.04
    arr = FancyArrowPatch(
        (x_from, row2_y), (x_to, row2_y),
        arrowstyle="->,head_width=0.10,head_length=0.08",
        color=ARROW_COLOR, linewidth=1.2, zorder=4,
    )
    ax.add_patch(arr)

# ── domain brackets under row 1 SFT→GRPO pairs ─────────────────────────
pairs = [
    (2, 3, "HAR"),
    (4, 5, "Sleep"),
    (6, 7, "ECG-QA"),
]
brace_y = row1_y - box_h / 2 - 0.48

for left_i, right_i, domain in pairs:
    x_left  = row1_cx[left_i] - box_w / 2 + 0.10
    x_right = row1_cx[right_i] + box_w / 2 - 0.10
    mid_x   = (row1_cx[left_i] + row1_cx[right_i]) / 2

    tick_h = 0.08
    ax.plot([x_left, x_right], [brace_y, brace_y],
            color=TEXT_MID, linewidth=1.0, zorder=3)
    ax.plot([x_left, x_left], [brace_y, brace_y + tick_h],
            color=TEXT_MID, linewidth=1.0, zorder=3)
    ax.plot([x_right, x_right], [brace_y, brace_y + tick_h],
            color=TEXT_MID, linewidth=1.0, zorder=3)

    ax.text(mid_x, brace_y - 0.08, f"Domain: {domain}",
            ha="center", va="top", fontsize=8.5,
            color=TEXT_MID, fontweight="bold", zorder=3)

# ── "Unchanged" label for row 2 ─────────────────────────────────────────
row2_mid_x = (row2_cx[0] + row2_cx[-1]) / 2
ax.text(row2_mid_x, row2_y + box_h / 2 + 0.28,
        "Remaining stages (unchanged)",
        ha="center", va="bottom", fontsize=9.5, fontstyle="italic",
        color=TEXT_LIGHT, zorder=3)

# ── legend ───────────────────────────────────────────────────────────────
legend_y_pos = row1_y + box_h / 2 + 0.80

legend_items = [
    (SFT_BLUE_LIGHT, SFT_BLUE,    "SFT Stage"),
    (GRPO_ORANGE_LT, GRPO_ORANGE, "GRPO Stage (new)"),
    (GREY_LIGHT,     GREY,        "Unchanged"),
]

item_widths = [2.20, 2.60, 2.20]
ckpt_legend_w = 3.0
total_legend_w = sum(item_widths) + ckpt_legend_w
legend_start_x = row1_total_w / 2 - total_legend_w / 2

cur_x = legend_start_x
for j, (fc, ec, label) in enumerate(legend_items):
    rect = mpatches.FancyBboxPatch(
        (cur_x, legend_y_pos - 0.12), 0.28, 0.24,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(cur_x + 0.42, legend_y_pos, label,
            ha="left", va="center", fontsize=9, color=TEXT_DARK, zorder=5)
    cur_x += item_widths[j]

# Checkpoint arrow in legend
arr = FancyArrowPatch(
    (cur_x + 0.05, legend_y_pos), (cur_x + 0.60, legend_y_pos),
    arrowstyle="->,head_width=0.10,head_length=0.10",
    color=CKPT_GREEN, linewidth=2.0, zorder=5,
)
ax.add_patch(arr)
ax.text(cur_x + 0.75, legend_y_pos, "Checkpoint handoff",
        ha="left", va="center", fontsize=9, color=TEXT_DARK, zorder=5)

# ── title ────────────────────────────────────────────────────────────────
title_y = legend_y_pos + 0.62
ax.text(row1_total_w / 2, title_y,
        "Modified Curriculum: SFT + GRPO Stages",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=TEXT_DARK, zorder=5)

subtitle_y = title_y - 0.36
ax.text(row1_total_w / 2, subtitle_y,
        "GRPO follow-up stage inserted after each CoT SFT stage (3\u20135); "
        "later stages remain unchanged",
        ha="center", va="center", fontsize=10, color=TEXT_MID, zorder=5)

# ── axis cleanup ─────────────────────────────────────────────────────────
ax.set_xlim(-0.8, row1_total_w + 1.2)
ax.set_ylim(row2_y - box_h / 2 - 0.8, title_y + 0.5)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0.3)
plt.savefig("/home/wangni/notion-figures/cot/fig_007.png",
            dpi=200, bbox_inches="tight", facecolor=WHITE)
plt.close()

print("Saved: /home/wangni/notion-figures/cot/fig_007.png")
