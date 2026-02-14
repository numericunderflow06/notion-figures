#!/usr/bin/env python3
"""
fig_007: Reward Computation Pipeline
Detailed dataflow: K responses per input → deduplication → batch to judge →
receive 3-dimensional scores → apply weighted formula → apply format bonus/penalty →
cache results → group normalization → advantages for GRPO.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
BG          = "#FFFFFF"
HEADER_BG   = "#1B2A4A"
HEADER_FG   = "#FFFFFF"

INPUT_CLR   = "#E8EEF6"
DEDUP_CLR   = "#FFF3CD"
JUDGE_CLR   = "#D4EDDA"
SCORE_CLR_C = "#4A90D9"   # blue  – correctness
SCORE_CLR_R = "#E67E22"   # orange – reasoning quality
SCORE_CLR_K = "#8E44AD"   # purple – consistency
FORMULA_CLR = "#F0E6F6"
FORMAT_CLR  = "#FDEBD0"
CACHE_CLR   = "#D5F5E3"
NORM_CLR    = "#D6EAF8"
ADV_CLR     = "#1B2A4A"

ARROW_CLR   = "#555555"
BORDER_CLR  = "#34495E"

fig, ax = plt.subplots(figsize=(16, 19), facecolor=BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 19)
ax.set_aspect("equal")
ax.axis("off")

# ── helpers ─────────────────────────────────────────────────────────────────
def draw_box(x, y, w, h, color, label, fontsize=11, fontweight="normal",
             border_color=BORDER_CLR, lw=1.5, text_color="#222222",
             zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor=border_color,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=zorder + 1,
            linespacing=1.4)
    return box

def draw_arrow(x1, y1, x2, y2, color=ARROW_CLR, lw=1.8,
               connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", color=color,
                            linewidth=lw, mutation_scale=16,
                            connectionstyle=connectionstyle,
                            zorder=1)
    ax.add_patch(arrow)

def draw_data_shape(x, y, w, h, label, color="#EAF2F8", fontsize=9,
                    border_color="#85C1E9"):
    skew = 0.25
    pts = np.array([
        [x + skew, y + h],
        [x + w + skew, y + h],
        [x + w - skew, y],
        [x - skew, y],
    ])
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor=border_color,
                   linewidth=1.2, zorder=2)
    ax.add_patch(poly)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color="#333333", zorder=3, style="italic")

def step_number(x, y, h, num):
    circle = plt.Circle((x, y + h / 2), 0.28,
                         facecolor=HEADER_BG, edgecolor="white",
                         linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y + h / 2, str(num), ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6)

# ── title banner ────────────────────────────────────────────────────────────
title_box = FancyBboxPatch((1.5, 17.6), 13, 0.9,
                           boxstyle="round,pad=0.15",
                           facecolor=HEADER_BG, edgecolor=HEADER_BG,
                           linewidth=0, zorder=5)
ax.add_patch(title_box)
ax.text(8, 18.05, "Reward Computation Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=HEADER_FG, zorder=6)

# ── layout constants ────────────────────────────────────────────────────────
CX = 8.0
BW = 5.0
BH = 0.72
GAP = 0.38
NUM_X = 3.2   # step number x position

current_y = 17.0

# ── STEP 1: Generate K Responses ────────────────────────────────────────────
y1 = current_y - BH
current_y = y1 - GAP

draw_box(CX - BW/2, y1, BW, BH, INPUT_CLR,
         "Generate K Responses\nper Input Prompt $x_i$",
         fontsize=11, fontweight="bold")
step_number(NUM_X, y1, BH, 1)
draw_data_shape(CX + BW/2 + 0.7, y1 + 0.05, 3.0, 0.62,
                "K candidate CoT\nresponses per group",
                color="#E8EEF6", border_color="#7FB3D8")

# ── STEP 2: Deduplicate ────────────────────────────────────────────────────
y2 = current_y - BH
current_y = y2 - GAP

draw_box(CX - BW/2, y2, BW, BH, DEDUP_CLR,
         "Deduplicate Responses",
         fontsize=11, fontweight="bold")
step_number(NUM_X, y2, BH, 2)
draw_data_shape(CX + BW/2 + 0.7, y2 + 0.05, 3.0, 0.62,
                "Unique responses\n(duplicates removed)",
                color="#FFF9E6", border_color="#F0C75E")

# ── STEP 3: Batch to Judge ──────────────────────────────────────────────────
y3 = current_y - BH
current_y = y3 - GAP

draw_box(CX - BW/2, y3, BW, BH, JUDGE_CLR,
         "Batch to LLM Judge\n(GPT-4o / local 70B)",
         fontsize=11, fontweight="bold")
step_number(NUM_X, y3, BH, 3)

# ── STEP 4: 3-Dimensional Scores ───────────────────────────────────────────
BH_SCORE = 0.65
score_w = 3.0
score_gap_x = 0.3

y4 = current_y - BH_SCORE
current_y = y4 - GAP

sx1 = CX - score_w * 1.5 - score_gap_x
sx2 = CX - score_w / 2
sx3 = CX + score_w / 2 + score_gap_x

draw_box(sx1, y4, score_w, BH_SCORE, SCORE_CLR_C,
         "Correctness\n(0\u20135)",
         fontsize=11, fontweight="bold", text_color="#FFFFFF",
         border_color="#2E6DA4")
draw_box(sx2, y4, score_w, BH_SCORE, SCORE_CLR_R,
         "Reasoning\nQuality (0\u20135)",
         fontsize=11, fontweight="bold", text_color="#FFFFFF",
         border_color="#C0651A")
draw_box(sx3, y4, score_w, BH_SCORE, SCORE_CLR_K,
         "Consistency\n(0\u20135)",
         fontsize=11, fontweight="bold", text_color="#FFFFFF",
         border_color="#6C3483")
step_number(NUM_X, y4, BH_SCORE, 4)

# Weight labels below score boxes
ax.text(sx1 + score_w/2, y4 - 0.18, "weight = 0.5",
        ha="center", va="center", fontsize=9.5, color=SCORE_CLR_C,
        fontweight="bold", zorder=4)
ax.text(sx2 + score_w/2, y4 - 0.18, "weight = 0.3",
        ha="center", va="center", fontsize=9.5, color=SCORE_CLR_R,
        fontweight="bold", zorder=4)
ax.text(sx3 + score_w/2, y4 - 0.18, "weight = 0.2",
        ha="center", va="center", fontsize=9.5, color=SCORE_CLR_K,
        fontweight="bold", zorder=4)

# ── STEP 5: Weighted Formula ───────────────────────────────────────────────
current_y -= 0.15   # extra space for weight labels
BH_FORMULA = 1.0
y5 = current_y - BH_FORMULA
current_y = y5 - GAP

fw = BW + 4.0   # wider formula box to fit text
formula_box = FancyBboxPatch((CX - fw/2, y5), fw, BH_FORMULA,
                             boxstyle="round,pad=0.18",
                             facecolor=FORMULA_CLR, edgecolor="#7D3C98",
                             linewidth=2.5, zorder=3)
ax.add_patch(formula_box)
step_number(NUM_X, y5, BH_FORMULA, 5)

ax.text(CX, y5 + BH_FORMULA * 0.70, "Weighted Reward Formula",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#4A235A", zorder=4)

# Color-coded formula rendered piece by piece
parts = [
    ("r  =  ", "#333333"),
    ("0.5\u00d7correctness", SCORE_CLR_C),
    ("  +  ", "#333333"),
    ("0.3\u00d7reasoning", SCORE_CLR_R),
    ("  +  ", "#333333"),
    ("0.2\u00d7consistency", SCORE_CLR_K),
]
x_cursor = CX - 3.7
for text, color in parts:
    ax.text(x_cursor, y5 + BH_FORMULA * 0.30, text,
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            fontfamily="monospace", color=color, zorder=4)
    x_cursor += len(text) * 0.165

# ── STEP 6: Format Bonus/Penalty ───────────────────────────────────────────
BH_FORMAT = 0.80
y6 = current_y - BH_FORMAT
current_y = y6 - GAP

draw_box(CX - BW/2 - 0.3, y6, BW + 0.6, BH_FORMAT, FORMAT_CLR,
         "Apply Format Bonus / Penalty",
         fontsize=11, fontweight="bold", border_color="#D4850F")
step_number(NUM_X, y6, BH_FORMAT, 6)

ax.text(CX + BW/2 + 0.9, y6 + BH_FORMAT/2 + 0.15,
        '+0.5  if "<reasoning> Answer: <label>"',
        ha="left", va="center", fontsize=9, color="#27AE60",
        fontweight="bold", fontfamily="monospace", zorder=4)
ax.text(CX + BW/2 + 0.9, y6 + BH_FORMAT/2 - 0.15,
        '\u22121.0  if format violated',
        ha="left", va="center", fontsize=9, color="#E74C3C",
        fontweight="bold", fontfamily="monospace", zorder=4)

# ── STEP 7: Cache Results ──────────────────────────────────────────────────
y7 = current_y - BH
current_y = y7 - GAP

draw_box(CX - BW/2, y7, BW, BH, CACHE_CLR,
         "Cache Results\n(dedup key \u2192 score)",
         fontsize=11, fontweight="bold", border_color="#28B463")
step_number(NUM_X, y7, BH, 7)

# Cache annotation – right side to avoid left-edge clipping
draw_data_shape(CX + BW/2 + 0.7, y7 + 0.05, 3.0, 0.62,
                "Cached scores avoid\nredundant judge calls",
                color="#D5F5E3", border_color="#7DCEA0")

# ── STEP 8: Group Normalization ─────────────────────────────────────────────
BH_NORM = 0.80
y8 = current_y - BH_NORM
current_y = y8 - GAP

draw_box(CX - BW/2 - 0.3, y8, BW + 0.6, BH_NORM, NORM_CLR,
         "Group Normalization\n(mean-subtract, std-divide within group)",
         fontsize=10.5, fontweight="bold", border_color="#2980B9")
step_number(NUM_X, y8, BH_NORM, 8)

# ── STEP 9: Advantages for GRPO ────────────────────────────────────────────
y9 = current_y - BH

draw_box(CX - BW/2, y9, BW, BH, ADV_CLR,
         "Advantages for GRPO",
         fontsize=12, fontweight="bold", text_color="#FFFFFF",
         border_color="#0E1F3A")
step_number(NUM_X, y9, BH, 9)

# ── Arrows ──────────────────────────────────────────────────────────────────
# Straight vertical arrows
draw_arrow(CX, y1, CX, y2 + BH, lw=2.0)
draw_arrow(CX, y2, CX, y3 + BH, lw=2.0)

# Judge → three score boxes (fan out)
draw_arrow(CX, y3, sx1 + score_w/2, y4 + BH_SCORE,
           lw=1.8, connectionstyle="arc3,rad=0.15")
draw_arrow(CX, y3, sx2 + score_w/2, y4 + BH_SCORE,
           lw=1.8, connectionstyle="arc3,rad=0.0")
draw_arrow(CX, y3, sx3 + score_w/2, y4 + BH_SCORE,
           lw=1.8, connectionstyle="arc3,rad=-0.15")

# Three score boxes → formula (fan in, color-coded)
draw_arrow(sx1 + score_w/2, y4 - 0.28, CX - 1.5, y5 + BH_FORMULA,
           lw=1.8, color=SCORE_CLR_C, connectionstyle="arc3,rad=0.12")
draw_arrow(sx2 + score_w/2, y4 - 0.28, CX, y5 + BH_FORMULA,
           lw=1.8, color=SCORE_CLR_R, connectionstyle="arc3,rad=0.0")
draw_arrow(sx3 + score_w/2, y4 - 0.28, CX + 1.5, y5 + BH_FORMULA,
           lw=1.8, color=SCORE_CLR_K, connectionstyle="arc3,rad=-0.12")

# Remaining vertical arrows
draw_arrow(CX, y5, CX, y6 + BH_FORMAT, lw=2.0)
draw_arrow(CX, y6, CX, y7 + BH, lw=2.0)
draw_arrow(CX, y7, CX, y8 + BH_NORM, lw=2.0)
draw_arrow(CX, y8, CX, y9 + BH, lw=2.0)

# ── Section brackets on the left ────────────────────────────────────────────
BRK_X = 2.0

# SCORE bracket (steps 3–7)
score_top = y3 + BH
score_bot = y7
ax.plot([BRK_X, BRK_X], [score_bot + 0.05, score_top - 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.plot([BRK_X, BRK_X + 0.2], [score_bot + 0.05, score_bot + 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.plot([BRK_X, BRK_X + 0.2], [score_top - 0.05, score_top - 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.text(BRK_X - 0.5, (score_top + score_bot) / 2,
        "SCORE\n(\u00a74.6 step 2)",
        ha="center", va="center", fontsize=8.5, color="#666666",
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.95))

# NORMALIZE bracket (steps 8–9)
norm_top = y8 + BH_NORM
norm_bot = y9
ax.plot([BRK_X, BRK_X], [norm_bot + 0.05, norm_top - 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.plot([BRK_X, BRK_X + 0.2], [norm_bot + 0.05, norm_bot + 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.plot([BRK_X, BRK_X + 0.2], [norm_top - 0.05, norm_top - 0.05],
        color="#888888", lw=1.3, zorder=1)
ax.text(BRK_X - 0.5, (norm_top + norm_bot) / 2,
        "NORMALIZE\n(\u00a74.6 step 3)",
        ha="center", va="center", fontsize=8.5, color="#666666",
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#CCCCCC", alpha=0.95))

# ── Legend ───────────────────────────────────────────────────────────────────
legend_y = y9 - 0.9
legend_x = 3.5
ax.text(legend_x, legend_y, "Score Dimensions:", fontsize=10.5,
        fontweight="bold", color="#333333", va="center")
for i, (label, clr) in enumerate([
    ("Correctness (w=0.5)", SCORE_CLR_C),
    ("Reasoning Quality (w=0.3)", SCORE_CLR_R),
    ("Consistency (w=0.2)", SCORE_CLR_K),
]):
    bx = legend_x + 3.5 * i
    rect = FancyBboxPatch((bx, legend_y - 0.42), 0.30, 0.22,
                          boxstyle="round,pad=0.02",
                          facecolor=clr, edgecolor="none", zorder=3)
    ax.add_patch(rect)
    ax.text(bx + 0.45, legend_y - 0.31, label,
            ha="left", va="center", fontsize=9.5, color="#333333", zorder=4)

plt.savefig("/home/wangni/notion-figures/llm-judge/fig_007.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
plt.close()
print("Saved: /home/wangni/notion-figures/llm-judge/fig_007.png")
