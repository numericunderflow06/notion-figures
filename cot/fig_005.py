"""
fig_005: GRPO Training Pipeline — Data Flow Diagram
Visualizes one GRPO training step per Section 4.2, 4.3, 4.5 of the plan.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
BLUE        = "#3B82F6"   # policy-model operations
BLUE_LIGHT  = "#DBEAFE"
GREEN       = "#16A34A"   # reward computation
GREEN_LIGHT = "#DCFCE7"
ORANGE      = "#EA8C00"   # advantage calculation
ORANGE_LIGHT= "#FEF3C7"
RED         = "#EF4444"   # reference model / KL
RED_LIGHT   = "#FEE2E2"
GRAY        = "#6B7280"
DARK        = "#1E293B"
WHITE       = "#FFFFFF"
INPUT_BG    = "#F1F5F9"

# ── figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(19, 9.6), dpi=200)
ax.set_xlim(0, 19)
ax.set_ylim(0, 9.6)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ── helper: rounded box ────────────────────────────────────────────
def draw_box(ax, x, y, w, h, facecolor, edgecolor, label_lines,
             sublabel=None, fontsize=11, fontweight="bold",
             sublabel_fontsize=9, linewidth=2.0):
    """Draw a rounded rectangle with label and optional sublabel."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.18",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.20, label_lines,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=DARK, zorder=3)
        ax.text(cx, cy - 0.28, sublabel,
                ha="center", va="center", fontsize=sublabel_fontsize,
                color=GRAY, zorder=3, style="italic")
    else:
        ax.text(cx, cy, label_lines,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=DARK, zorder=3)
    return (x, y, w, h)  # return geometry for arrow anchoring

# ── helper: arrow ──────────────────────────────────────────────────
def draw_arrow(ax, start, end, color=GRAY, lw=2.0, style="-|>",
               conn="arc3,rad=0", ls="-", zorder=1, ms=18):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color,
        linewidth=lw, connectionstyle=conn,
        mutation_scale=ms, zorder=zorder, linestyle=ls)
    ax.add_patch(arrow)

# ── helper: circled step number ────────────────────────────────────
def step_num(ax, x, y, num, color):
    c = plt.Circle((x, y), 0.24, facecolor=color, edgecolor="white",
                    linewidth=1.5, zorder=5)
    ax.add_patch(c)
    ax.text(x, y, str(num), ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white", zorder=6)

# ── helper: arrow label on white bg ────────────────────────────────
def arrow_label(ax, x, y, text, color=GRAY, fontsize=9, rotation=0):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=4, rotation=rotation,
            bbox=dict(boxstyle="round,pad=0.12", facecolor=WHITE,
                      edgecolor="none", alpha=0.92))

# ====================================================================
# LAYOUT — five main stages left-to-right
# ====================================================================
MAIN_Y = 5.6          # vertical centre of main pipeline
BH     = 1.4          # box height
GAP    = 0.55         # horizontal gap between boxes

# Widths
W_INP  = 2.2
W_POL  = 2.5
W_SAMP = 2.8
W_REW  = 2.6
W_ADV  = 2.8
W_UPD  = 2.5

# X positions (left edge of each box)
x1 = 0.5
x2 = x1 + W_INP + GAP
x3 = x2 + W_POL + GAP
x4 = x3 + W_SAMP + GAP
x5 = x4 + W_REW + GAP
x6 = x5 + W_ADV + GAP

by = MAIN_Y - BH / 2  # box bottom-y

# ── (1) Time-series input ──────────────────────────────────────────
b1 = draw_box(ax, x1, by, W_INP, BH,
              INPUT_BG, GRAY,
              "Time-Series\nInput", sublabel="prompt  x")
step_num(ax, x1 + 0.28, by + BH + 0.35, 1, BLUE)

# ── (2a) Policy model ─────────────────────────────────────────────
b2 = draw_box(ax, x2, by, W_POL, BH,
              BLUE_LIGHT, BLUE,
              "Policy Model", sublabel="$\\pi_\\theta$",
              fontsize=12)

# ── (2b) Sampling ─────────────────────────────────────────────────
b3 = draw_box(ax, x3, by, W_SAMP, BH,
              BLUE_LIGHT, BLUE,
              "Sample G = 8\nCoT Completions", sublabel="T = 0.7",
              fontsize=10.5)
step_num(ax, x3 + 0.28, by + BH + 0.35, 2, BLUE)

# ── (3) Reward function ───────────────────────────────────────────
b4 = draw_box(ax, x4, by, W_REW, BH,
              GREEN_LIGHT, GREEN,
              "Reward\nFunction", sublabel="$r = r_c + r_f$")
step_num(ax, x4 + 0.28, by + BH + 0.35, 3, GREEN)
# sub-annotation
ax.text(x4 + W_REW / 2, by - 0.30,
        "correctness + format", ha="center", va="center",
        fontsize=8.5, color=GREEN, zorder=3)

# ── (4) Advantages ────────────────────────────────────────────────
b5 = draw_box(ax, x5, by, W_ADV, BH,
              ORANGE_LIGHT, ORANGE,
              "Group-Normalized\nAdvantages",
              sublabel="$\\hat{A}_i = (r_i - \\mu)\\,/\\,(\\sigma + \\delta)$",
              fontsize=10.5, sublabel_fontsize=8.5)
step_num(ax, x5 + 0.28, by + BH + 0.35, 4, ORANGE)

# ── (5) Policy gradient update ────────────────────────────────────
b6 = draw_box(ax, x6, by, W_UPD, BH,
              BLUE_LIGHT, BLUE,
              "Policy Gradient\nUpdate",
              sublabel="$\\epsilon = 0.2$,  $\\beta = 0.04$",
              fontsize=10.5, sublabel_fontsize=8.5)
step_num(ax, x6 + 0.28, by + BH + 0.35, 5, BLUE)

# ── Reference model (below pipeline, centered) ────────────────────
REF_W, REF_H = 3.0, 1.2
ref_x = (x4 + W_REW / 2 + x6 + W_UPD / 2) / 2 - REF_W / 2
ref_y = 1.5
bref = draw_box(ax, ref_x, ref_y, REF_W, REF_H,
                RED_LIGHT, RED,
                "Reference Model",
                sublabel="$\\pi_{\\mathrm{ref}}$  (SFT policy)",
                fontsize=11, sublabel_fontsize=9)

# ====================================================================
# MAIN-FLOW ARROWS (left → right)
# ====================================================================
arr_y = MAIN_Y  # arrows at vertical centre

# input → policy
draw_arrow(ax, (x1 + W_INP, arr_y), (x2, arr_y), color=GRAY, lw=2.2)
# policy → sampling
draw_arrow(ax, (x2 + W_POL, arr_y), (x3, arr_y), color=BLUE, lw=2.2)
# sampling → reward
draw_arrow(ax, (x3 + W_SAMP, arr_y), (x4, arr_y), color=BLUE, lw=2.2)
# reward → advantages
draw_arrow(ax, (x4 + W_REW, arr_y), (x5, arr_y), color=GREEN, lw=2.2)
# advantages → update
draw_arrow(ax, (x5 + W_ADV, arr_y), (x6, arr_y), color=ORANGE, lw=2.2)

# ====================================================================
# FEEDBACK ARROW: update → policy (curved, on top)
# ====================================================================
# Use annotation with a manual curved path going ABOVE the boxes
draw_arrow(ax,
           (x6 + W_UPD / 2, by + BH),          # top of update box
           (x2 + W_POL / 2, by + BH),           # top of policy box
           color=BLUE, lw=2.2, style="-|>",
           conn="arc3,rad=0.35", zorder=3)
# label for feedback arrow — positioned above the arc
mid_fb_x = (x6 + W_UPD / 2 + x2 + W_POL / 2) / 2
ax.text(mid_fb_x, 8.55, "update  $\\theta$",
        ha="center", va="center", fontsize=10, color=BLUE,
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.15", facecolor=WHITE,
                  edgecolor=BLUE, alpha=0.9, linewidth=0.8))

# ====================================================================
# REFERENCE MODEL ARROWS (dashed red)
# ====================================================================

# ref → update  (KL penalty)
draw_arrow(ax,
           (ref_x + REF_W * 0.75, ref_y + REF_H),
           (x6 + W_UPD / 2, by),
           color=RED, lw=1.8, style="-|>",
           conn="arc3,rad=-0.12", ls="--", ms=16)
# label
kl_lx = (ref_x + REF_W * 0.75 + x6 + W_UPD / 2) / 2 + 0.5
kl_ly = (ref_y + REF_H + by) / 2
ax.text(kl_lx, kl_ly, "KL penalty",
        ha="center", va="center", fontsize=9, color=RED,
        fontweight="bold", zorder=4, rotation=38,
        bbox=dict(boxstyle="round,pad=0.1", facecolor=WHITE,
                  edgecolor="none", alpha=0.9))

# ref → sampling area (ratio π_ref(y|x))
draw_arrow(ax,
           (ref_x + REF_W * 0.25, ref_y + REF_H),
           (x3 + W_SAMP / 2, by),
           color=RED, lw=1.5, style="-|>",
           conn="arc3,rad=0.15", ls="--", ms=14)
ratio_lx = (ref_x + REF_W * 0.25 + x3 + W_SAMP / 2) / 2 - 0.3
ratio_ly = (ref_y + REF_H + by) / 2
ax.text(ratio_lx, ratio_ly, "$\\pi_{\\mathrm{ref}}(y_i \\mid x)$",
        ha="center", va="center", fontsize=9, color=RED,
        zorder=4, rotation=22,
        bbox=dict(boxstyle="round,pad=0.1", facecolor=WHITE,
                  edgecolor="none", alpha=0.9))

# ====================================================================
# DATA-FLOW LABELS on main arrows
# ====================================================================
label_dy = 0.38  # offset above arrow

arrow_label(ax, (x1 + W_INP + x2) / 2, arr_y + label_dy,
            "$x$", GRAY, 10)
arrow_label(ax, (x2 + W_POL + x3) / 2, arr_y + label_dy,
            "$x$", BLUE, 10)
arrow_label(ax, (x3 + W_SAMP + x4) / 2, arr_y + label_dy,
            "$\\{y_1, \\ldots, y_8\\}$", BLUE, 9.5)
arrow_label(ax, (x4 + W_REW + x5) / 2, arr_y + label_dy,
            "$\\{r_1, \\ldots, r_8\\}$", GREEN, 9.5)
arrow_label(ax, (x5 + W_ADV + x6) / 2, arr_y + label_dy,
            "$\\{\\hat{A}_1, \\ldots, \\hat{A}_8\\}$", ORANGE, 9.5)

# ====================================================================
# LEGEND (bottom-left)
# ====================================================================
legend_items = [
    (BLUE_LIGHT, BLUE,   "Policy model operations"),
    (GREEN_LIGHT, GREEN,  "Reward computation"),
    (ORANGE_LIGHT, ORANGE,"Advantage calculation"),
    (RED_LIGHT, RED,      "Reference model / KL"),
]
lx_start, ly = 0.6, 0.35
for i, (fc, ec, label) in enumerate(legend_items):
    lx = lx_start + i * 4.3
    box = FancyBboxPatch((lx, ly), 0.4, 0.3,
                         boxstyle="round,pad=0.05",
                         facecolor=fc, edgecolor=ec,
                         linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(lx + 0.55, ly + 0.15, label,
            ha="left", va="center", fontsize=9, color=DARK, zorder=3)

# ====================================================================
# TITLE
# ====================================================================
ax.text(19 / 2, 9.2, "GRPO Training Pipeline — One Training Step",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=DARK, zorder=4)

# ====================================================================
# SAVE
# ====================================================================
plt.tight_layout(pad=0.3)
fig.savefig("/home/wangni/notion-figures/cot/fig_005.png",
            dpi=200, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/cot/fig_005.png")
