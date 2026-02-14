"""
fig_005: SDFT End-to-End Pipeline for OpenTSLM
Horizontal data flow diagram showing the complete SDFT pipeline.

Color coding:
  - Blue:   Student path
  - Orange: Teacher path (EMA)
  - Green:  Reward / scoring path
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
BLUE        = "#3B82F6"   # student boxes
BLUE_LIGHT  = "#DBEAFE"   # student fill
BLUE_DARK   = "#1E40AF"   # student text

ORANGE      = "#F59E0B"   # teacher boxes
ORANGE_LIGHT= "#FEF3C7"   # teacher fill
ORANGE_DARK = "#92400E"   # teacher text

GREEN       = "#10B981"   # reward boxes
GREEN_LIGHT = "#D1FAE5"   # reward fill
GREEN_DARK  = "#065F46"   # reward text

GRAY        = "#6B7280"
GRAY_LIGHT  = "#F3F4F6"
DARK        = "#1F2937"
PURPLE      = "#7C3AED"
PURPLE_LIGHT= "#EDE9FE"

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 11))
ax.set_xlim(-0.5, 21.5)
ax.set_ylim(-1.5, 10.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper functions ────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, facecolor, edgecolor, textcolor=DARK,
             fontsize=11, fontweight="bold", lw=2.0, style="round,pad=0.15",
             subtext=None, subtextsize=8.5):
    """Draw a rounded box with centred text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=style,
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x + w / 2, y + h / 2 + 0.22, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.28, subtext,
                ha="center", va="center", fontsize=subtextsize,
                color=textcolor, style="italic", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=4)
    return box

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.0, style="-|>",
          connectionstyle="arc3,rad=0.0", linestyle="-", zorder=2):
    """Draw an arrow between two points."""
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle=style, color=color,
                         lw=lw, connectionstyle=connectionstyle,
                         linestyle=linestyle, zorder=zorder,
                         mutation_scale=18)
    ax.add_patch(ar)
    return ar

def annotation(ax, x, y, text, color=GRAY, fontsize=8, ha="center",
               bbox_color="white"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=bbox_color,
                      edgecolor=color, alpha=0.85, lw=0.8))

# ── Title ───────────────────────────────────────────────────────────
ax.text(10.5, 10.0, "SDFT End-to-End Pipeline for OpenTSLM",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color=DARK, zorder=5)
ax.text(10.5, 9.4, "Self-Distilled Fine-Tuning: from time-series input to gradient update",
        ha="center", va="center", fontsize=11, color=GRAY, zorder=5)

# ════════════════════════════════════════════════════════════════════
# ROW POSITIONS  (top to bottom)
#   Student row  : y_s = 6.0  (centre of boxes)
#   Reward row   : y_r = 3.2
#   Teacher row  : y_t = 0.8
# ════════════════════════════════════════════════════════════════════

BH = 1.3   # box height
BW = 2.8   # box width (default)

# ────────────────────────────────────────────────────────────────────
# STUDENT PATH (top row, blue)
# ────────────────────────────────────────────────────────────────────
y_s = 6.2

# Box 1: Input
draw_box(ax, 0.0, y_s, 2.4, BH,
         "Time-Series\nInput + Prompt", BLUE_LIGHT, BLUE, BLUE_DARK,
         fontsize=10.5)

# Box 2: OpenTSLM generate()
draw_box(ax, 3.6, y_s, 3.0, BH,
         "OpenTSLM\ngenerate()", BLUE_LIGHT, BLUE, BLUE_DARK,
         fontsize=11)
annotation(ax, 5.1, y_s - 0.35, "temp > 0, sampling", BLUE, 8)

# Box 3: N Rollouts
draw_box(ax, 7.8, y_s, 2.6, BH,
         "N Rollouts", BLUE_LIGHT, BLUE, BLUE_DARK,
         fontsize=12, subtext="CoT + answer")
annotation(ax, 9.1, y_s - 0.35, "N = 8", BLUE, 9, bbox_color=BLUE_LIGHT)

# Box 4: Gradient Update (student)
draw_box(ax, 16.0, y_s, 3.0, BH,
         "Gradient Update\n(Student)", BLUE_LIGHT, BLUE, BLUE_DARK,
         fontsize=11)

# ────────────────────────────────────────────────────────────────────
# REWARD / SCORING PATH (middle row, green)
# ────────────────────────────────────────────────────────────────────
y_r = 3.6

# Box 5: Binary Reward Scoring
draw_box(ax, 7.8, y_r, 2.6, BH,
         "Binary Reward\nScoring", GREEN_LIGHT, GREEN, GREEN_DARK,
         fontsize=11)

# 3-tier match detail box
draw_box(ax, 4.0, y_r - 0.1, 3.0, 1.5,
         "3-Tier Match", GREEN_LIGHT, GREEN, GREEN_DARK,
         fontsize=10, subtext="exact | prefix | MCQ-3char",
         subtextsize=8)
annotation(ax, 5.5, y_r - 0.55, "reward ∈ {0, 1}", GREEN, 8,
           bbox_color=GREEN_LIGHT)

# Box 6: Successful CoT Selection
draw_box(ax, 11.4, y_r, 2.8, BH,
         "Successful CoT\nSelection", GREEN_LIGHT, GREEN, GREEN_DARK,
         fontsize=10.5)

# ────────────────────────────────────────────────────────────────────
# TEACHER PATH (bottom row, orange)
# ────────────────────────────────────────────────────────────────────
y_t = 0.8

# Box 7: Self-Teacher (EMA Model)
draw_box(ax, 7.8, y_t, 3.2, BH,
         "Self-Teacher", ORANGE_LIGHT, ORANGE, ORANGE_DARK,
         fontsize=12, subtext="EMA model + augmented prompt")
annotation(ax, 9.4, y_t - 0.38, "EMA rate = 0.05", ORANGE, 9,
           bbox_color=ORANGE_LIGHT)

# Box 8: KL Distillation Loss
draw_box(ax, 12.5, y_t, 2.8, BH,
         "KL Distillation\nLoss", ORANGE_LIGHT, ORANGE, ORANGE_DARK,
         fontsize=11)
annotation(ax, 13.9, y_t - 0.38, "α = 0.5", ORANGE, 9,
           bbox_color=ORANGE_LIGHT)

# Box 9: EMA Teacher Update
draw_box(ax, 16.0, y_t, 3.0, BH,
         "EMA Teacher\nUpdate", ORANGE_LIGHT, ORANGE, ORANGE_DARK,
         fontsize=11)
annotation(ax, 17.5, y_t - 0.38, "θ_T ← (1−τ)θ_T + τθ_S", ORANGE, 8,
           bbox_color=ORANGE_LIGHT)

# ════════════════════════════════════════════════════════════════════
# ARROWS
# ════════════════════════════════════════════════════════════════════

# --- Student row (left to right) ---
# Input → generate()
arrow(ax, 2.4, y_s + BH/2, 3.6, y_s + BH/2, BLUE, 2.5)
# generate() → N Rollouts
arrow(ax, 6.6, y_s + BH/2, 7.8, y_s + BH/2, BLUE, 2.5)

# --- N Rollouts → Binary Reward (down) ---
arrow(ax, 9.1, y_r + BH + 0.7, 9.1, y_r + BH, GREEN, 2.5)

# --- 3-tier detail → Binary Reward ---
arrow(ax, 7.0, y_r + BH/2 - 0.1 + 0.75, 7.8, y_r + BH/2, GREEN, 2.0)

# --- Binary Reward → Successful CoT Selection ---
arrow(ax, 10.4, y_r + BH/2, 11.4, y_r + BH/2, GREEN, 2.5)

# --- Successful CoT → Self-Teacher (down-left curve) ---
arrow(ax, 12.0, y_r, 11.0, y_t + BH,
      GREEN, 2.0, connectionstyle="arc3,rad=0.25")
# small annotation on this arrow
annotation(ax, 11.0, y_r - 0.7, "demonstration", GREEN, 8)

# --- Self-Teacher → KL Loss ---
arrow(ax, 11.0, y_t + BH/2, 12.5, y_t + BH/2, ORANGE, 2.5)

# --- KL Loss → Gradient Update (up-right) ---
arrow(ax, 15.3, y_t + BH/2, 16.5, y_s,
      PURPLE, 2.5, connectionstyle="arc3,rad=-0.2")
annotation(ax, 15.0, 3.8, "combined\nloss", PURPLE, 8.5,
           bbox_color=PURPLE_LIGHT)

# --- Gradient Update → EMA Teacher Update (down) ---
arrow(ax, 17.5, y_s, 17.5, y_t + BH,
      ORANGE, 2.0, style="-|>", connectionstyle="arc3,rad=0.0",
      linestyle="--")
annotation(ax, 18.5, 3.8, "after each\nstep", ORANGE, 8)

# --- N Rollouts direct to Gradient Update (student logprobs) ---
# This represents the student logprobs used in KL divergence
arrow(ax, 10.4, y_s + BH/2, 16.0, y_s + BH/2, BLUE, 2.0,
      linestyle="--")
annotation(ax, 13.0, y_s + BH + 0.1, "student log-probs\n(response tokens)", BLUE, 8,
           bbox_color=BLUE_LIGHT)

# --- Feedback loop: Gradient Update back to Input (curved top) ---
# Draw as segmented path: up, left, down using line segments
loop_top = y_s + BH + 0.9
# Up from Gradient Update
arrow(ax, 17.5, y_s + BH, 17.5, loop_top,
      BLUE_DARK, 1.8, linestyle="--", style="-")
# Left across the top
arrow(ax, 17.5, loop_top, 1.2, loop_top,
      BLUE_DARK, 1.8, linestyle="--", style="-")
# Down into Input
arrow(ax, 1.2, loop_top, 1.2, y_s + BH,
      BLUE_DARK, 1.8, linestyle="--", style="-|>")
annotation(ax, 9.5, loop_top + 0.35, "next training iteration", BLUE_DARK, 9,
           bbox_color=BLUE_LIGHT)

# ════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════
leg_y = -0.8
leg_x = 1.5
spacing = 5.5

# Student
draw_box(ax, leg_x, leg_y, 1.2, 0.55, "Student", BLUE_LIGHT, BLUE, BLUE_DARK,
         fontsize=9, lw=1.5, style="round,pad=0.08")
ax.text(leg_x + 1.4, leg_y + 0.28, "Student path", fontsize=9, color=BLUE_DARK,
        va="center")

# Reward
draw_box(ax, leg_x + spacing, leg_y, 1.2, 0.55, "Reward", GREEN_LIGHT, GREEN, GREEN_DARK,
         fontsize=9, lw=1.5, style="round,pad=0.08")
ax.text(leg_x + spacing + 1.4, leg_y + 0.28, "Reward / scoring path", fontsize=9,
        color=GREEN_DARK, va="center")

# Teacher
draw_box(ax, leg_x + 2*spacing, leg_y, 1.2, 0.55, "Teacher", ORANGE_LIGHT, ORANGE, ORANGE_DARK,
         fontsize=9, lw=1.5, style="round,pad=0.08")
ax.text(leg_x + 2*spacing + 1.4, leg_y + 0.28, "Teacher path (EMA)", fontsize=9,
        color=ORANGE_DARK, va="center")

# ════════════════════════════════════════════════════════════════════
# TARGET TASKS annotation (bottom-right)
# ════════════════════════════════════════════════════════════════════
task_text = ("Target tasks:\n"
             "  Stage 3 — HAR CoT  (8 activity classes)\n"
             "  Stage 4 — Sleep CoT (6 sleep stages)")
ax.text(20.5, -0.5, task_text, fontsize=8.5, color=GRAY,
        va="center", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GRAY_LIGHT,
                  edgecolor=GRAY, lw=0.8))

# ────────────────────────────────────────────────────────────────────
# Save
# ────────────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/self-distillation/fig_005.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {out_path}")
