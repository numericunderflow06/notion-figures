#!/usr/bin/env python3
"""
fig_005 — End-to-End RL Data Flow
Detailed data flow diagram from input to policy update:
  time-series + text prompt → OpenTSLM generates K responses → responses batched
  to LLM judge → judge returns scores → rewards computed and normalized →
  GRPO loss → gradient update.
Shows the reference policy branch for KL computation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Color palette ─────────────────────────────────────────────────────────────
COL_INPUT      = "#6366F1"   # indigo — input data
COL_INPUT_EDGE = "#4338CA"
COL_POLICY     = "#3B82F6"   # blue — policy / model
COL_POLICY_EDGE= "#1E40AF"
COL_GEN        = "#8B5CF6"   # violet — generation fan-out
COL_GEN_EDGE   = "#6D28D9"
COL_JUDGE      = "#F97316"   # orange — LLM judge
COL_JUDGE_EDGE = "#C2410C"
COL_REWARD     = "#EF4444"   # red — reward / score
COL_REWARD_EDGE= "#B91C1C"
COL_LOSS       = "#10B981"   # emerald — loss / gradient
COL_LOSS_EDGE  = "#047857"
COL_REF        = "#94A3B8"   # slate — reference policy
COL_REF_EDGE   = "#64748B"
COL_ARROW      = "#475569"
COL_REF_ARROW  = "#94A3B8"
COL_TEXT_DARK   = "#1E293B"
COL_TEXT_MED    = "#475569"
COL_TEXT_WHITE  = "#FFFFFF"
COL_FANOUT_BG   = "#F5F3FF"  # light violet background for fan-out region
COL_FANOUT_EDGE = "#C4B5FD"

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 10.5), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Helper: draw a rounded box with text ─────────────────────────────────────
def draw_box(ax, cx, cy, w, h, label, sublabel=None, fc="#3B82F6", ec="#1E40AF",
             tc="#FFFFFF", fontsize=11, sublabel_fontsize=8.5, zorder=5,
             icon=None, icon_fontsize=16, boxstyle="round,pad=0.08"):
    """Draw a rounded rectangle centered at (cx, cy)."""
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=boxstyle,
        facecolor=fc, edgecolor=ec, linewidth=2.0,
        zorder=zorder,
    )
    ax.add_patch(box)

    text_y = cy + (0.12 if sublabel else 0) + (0.0 if icon is None else -0.05)
    if icon is not None:
        ax.text(cx, cy + 0.22, icon, ha="center", va="center",
                fontsize=icon_fontsize, color=tc, zorder=zorder+1)
        text_y = cy - 0.02

    ax.text(cx, text_y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc, zorder=zorder+1)

    if sublabel:
        ax.text(cx, text_y - 0.32, sublabel, ha="center", va="center",
                fontsize=sublabel_fontsize, color=tc, alpha=0.88,
                zorder=zorder+1)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color=COL_ARROW, lw=2.0, style="-|>",
               connectionstyle=None, linestyle="-", zorder=3, mutation_scale=16):
    """Draw an arrow from (x0,y0) to (x1,y1)."""
    props = dict(
        arrowstyle=style,
        color=color,
        lw=lw,
        mutation_scale=mutation_scale,
        linestyle=linestyle,
    )
    if connectionstyle:
        props["connectionstyle"] = connectionstyle
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=props, zorder=zorder)


def draw_label_on_arrow(ax, x, y, text, fontsize=8.5, color=COL_TEXT_MED, zorder=6,
                        bg="white"):
    """Place a label with white background along an arrow path."""
    t = ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                color=color, fontstyle="italic", zorder=zorder,
                bbox=dict(facecolor=bg, edgecolor="none", pad=1.5, alpha=0.92))
    return t


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — left-to-right flow
# ══════════════════════════════════════════════════════════════════════════════

# Vertical centers
Y_MAIN = 5.0       # main policy flow
Y_REF  = 1.6       # reference policy branch

# Horizontal positions (center x of each box)
X_INPUT_TS   = 1.3
X_INPUT_TXT  = 1.3
X_POLICY     = 4.8
X_GEN_FAN    = 8.2   # center of the K=4 fan-out region
X_JUDGE      = 12.0
X_SCORES     = 15.5
X_NORM       = 19.0
X_LOSS       = 22.0
X_GRAD       = 25.0

BOX_W = 2.2
BOX_H = 1.0
SMALL_W = 1.7
SMALL_H = 0.7

# ── 1. Input boxes (time-series + text prompt) ──────────────────────────────
INPUT_Y_TS  = Y_MAIN + 0.75
INPUT_Y_TXT = Y_MAIN - 0.75

draw_box(ax, X_INPUT_TS, INPUT_Y_TS, BOX_W, SMALL_H,
         "Time-Series", sublabel=None,
         fc=COL_INPUT, ec=COL_INPUT_EDGE, tc=COL_TEXT_WHITE,
         fontsize=10.5, icon="/\\/\\", icon_fontsize=10)

draw_box(ax, X_INPUT_TXT, INPUT_Y_TXT, BOX_W, SMALL_H,
         "Text Prompt", sublabel=None,
         fc=COL_INPUT, ec=COL_INPUT_EDGE, tc=COL_TEXT_WHITE,
         fontsize=10.5, icon="Aa", icon_fontsize=11)

# Brace/merge indicator
merge_x = X_INPUT_TS + BOX_W/2 + 0.45
ax.plot([merge_x, merge_x], [INPUT_Y_TXT, INPUT_Y_TS], color=COL_ARROW,
        lw=1.5, zorder=2)
ax.plot([merge_x, merge_x + 0.3], [Y_MAIN, Y_MAIN], color=COL_ARROW,
        lw=1.5, zorder=2)
# small circle at merge
ax.plot(merge_x, Y_MAIN, 'o', color=COL_ARROW, markersize=5, zorder=4)
# connecting lines from input boxes to merge point
ax.plot([X_INPUT_TS + BOX_W/2, merge_x], [INPUT_Y_TS, INPUT_Y_TS],
        color=COL_ARROW, lw=1.5, zorder=2)
ax.plot([X_INPUT_TXT + BOX_W/2, merge_x], [INPUT_Y_TXT, INPUT_Y_TXT],
        color=COL_ARROW, lw=1.5, zorder=2)

# Label x_i
draw_label_on_arrow(ax, merge_x + 0.55, Y_MAIN + 0.28, "$x_i$", fontsize=11,
                    color=COL_TEXT_DARK)

# Arrow from merge to policy
draw_arrow(ax, merge_x + 0.3, Y_MAIN, X_POLICY - BOX_W/2, Y_MAIN)

# ── 2. OpenTSLM Policy (pi_theta) ────────────────────────────────────────────
draw_box(ax, X_POLICY, Y_MAIN, BOX_W, BOX_H * 1.2,
         "OpenTSLM", sublabel="Policy  $\\pi_\\theta$",
         fc=COL_POLICY, ec=COL_POLICY_EDGE, tc=COL_TEXT_WHITE,
         fontsize=13, sublabel_fontsize=10.5)

# ── 3. K=4 Generation Fan-out ────────────────────────────────────────────────
# Draw a light background region for fan-out
fan_pad = 0.25
gen_box_w = 1.9
gen_box_h = 0.58
gen_y_positions = [Y_MAIN + 1.2, Y_MAIN + 0.4, Y_MAIN - 0.4, Y_MAIN - 1.2]

fan_region_x0 = X_GEN_FAN - gen_box_w/2 - fan_pad
fan_region_x1 = X_GEN_FAN + gen_box_w/2 + fan_pad
fan_region_y0 = gen_y_positions[-1] - gen_box_h/2 - fan_pad
fan_region_y1 = gen_y_positions[0] + gen_box_h/2 + fan_pad
fan_bg = FancyBboxPatch(
    (fan_region_x0, fan_region_y0),
    fan_region_x1 - fan_region_x0,
    fan_region_y1 - fan_region_y0,
    boxstyle="round,pad=0.15",
    facecolor=COL_FANOUT_BG, edgecolor=COL_FANOUT_EDGE,
    linewidth=1.5, linestyle="--", zorder=1,
)
ax.add_patch(fan_bg)
ax.text((fan_region_x0 + fan_region_x1)/2, fan_region_y1 + 0.15,
        "K = 4 responses", ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", color=COL_GEN_EDGE, zorder=6)

# Four response boxes
for k, gy in enumerate(gen_y_positions):
    draw_box(ax, X_GEN_FAN, gy, gen_box_w, gen_box_h,
             f"Response {k+1}", sublabel=None,
             fc=COL_GEN, ec=COL_GEN_EDGE, tc=COL_TEXT_WHITE,
             fontsize=9.5, zorder=5)
    # Arrow from policy to each response (fan-out)
    draw_arrow(ax, X_POLICY + BOX_W/2, Y_MAIN,
               X_GEN_FAN - gen_box_w/2, gy,
               color=COL_GEN_EDGE, lw=1.5, style="-|>")
    # Arrow from each response to judge (fan-in)
    draw_arrow(ax, X_GEN_FAN + gen_box_w/2, gy,
               X_JUDGE - BOX_W/2, Y_MAIN,
               color=COL_JUDGE_EDGE, lw=1.5, style="-|>")

# Label fan-out arrow
draw_label_on_arrow(ax, (X_POLICY + BOX_W/2 + X_GEN_FAN - gen_box_w/2)/2,
                    Y_MAIN + 1.7, "sample K=4", fontsize=11, color=COL_GEN_EDGE)

# ── 4. LLM Judge ────────────────────────────────────────────────────────────
draw_box(ax, X_JUDGE, Y_MAIN, BOX_W + 0.2, BOX_H * 1.2,
         "LLM Judge", sublabel="GPT-4o / 70B",
         fc=COL_JUDGE, ec=COL_JUDGE_EDGE, tc=COL_TEXT_WHITE,
         fontsize=13, sublabel_fontsize=9.5)

# Arrow from judge to scores
draw_arrow(ax, X_JUDGE + (BOX_W+0.2)/2, Y_MAIN, X_SCORES - SMALL_W/2 - 0.6, Y_MAIN)

# ── 5. Scores breakdown ─────────────────────────────────────────────────────
# Show the three scoring dimensions
score_y_offsets = [0.7, 0.0, -0.7]
score_labels = ["Correctness", "Reasoning", "Consistency"]
score_weights = ["w = 0.5", "w = 0.3", "w = 0.2"]
score_colors = ["#FEE2E2", "#FEF3C7", "#E0E7FF"]
score_edges  = ["#FCA5A5", "#FCD34D", "#A5B4FC"]
score_text_c = ["#991B1B", "#92400E", "#3730A3"]

# Background for score group
score_bg = FancyBboxPatch(
    (X_SCORES - 1.1, Y_MAIN - 1.3),
    2.2, 2.6,
    boxstyle="round,pad=0.12",
    facecolor="#FFF7ED", edgecolor="#FDBA74",
    linewidth=1.2, linestyle="--", zorder=1,
)
ax.add_patch(score_bg)
ax.text(X_SCORES, Y_MAIN + 1.55, "Scores (0-5)", ha="center", va="bottom",
        fontsize=9.5, fontweight="bold", color=COL_JUDGE_EDGE, zorder=6)

for j, (sy, slbl, sw, sc, se, stc) in enumerate(
        zip(score_y_offsets, score_labels, score_weights, score_colors, score_edges,
            score_text_c)):
    draw_box(ax, X_SCORES, Y_MAIN + sy, SMALL_W + 0.2, 0.5,
             slbl, sublabel=None,
             fc=sc, ec=se, tc=stc,
             fontsize=9, zorder=5)
    # Weight annotation to the right
    ax.text(X_SCORES + SMALL_W/2 + 0.3, Y_MAIN + sy, sw,
            ha="left", va="center", fontsize=8.5, color=COL_TEXT_MED,
            fontstyle="italic", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))

# ── 6. Reward computation & normalization ────────────────────────────────────
draw_box(ax, X_NORM, Y_MAIN, BOX_W, BOX_H * 1.2,
         "Normalize", sublabel="$\\hat{r} = (r - \\mu_G) / \\sigma_G$",
         fc=COL_REWARD, ec=COL_REWARD_EDGE, tc=COL_TEXT_WHITE,
         fontsize=12, sublabel_fontsize=11)

# Arrow from scores to normalize
arrow_scores_start = X_SCORES + SMALL_W/2 + 1.05
draw_arrow(ax, arrow_scores_start, Y_MAIN, X_NORM - BOX_W/2, Y_MAIN)
draw_label_on_arrow(ax, (arrow_scores_start + X_NORM - BOX_W/2)/2,
                    Y_MAIN + 0.3, "$r = \\Sigma w_i \\cdot s_i$", fontsize=11,
                    color=COL_REWARD_EDGE)

# ── 7. GRPO Loss ─────────────────────────────────────────────────────────────
draw_box(ax, X_LOSS, Y_MAIN, BOX_W, BOX_H * 1.2,
         "GRPO Loss", sublabel="$\\mathrm{clip}(\\rho) \\cdot \\hat{A} - \\beta \\cdot KL$",
         fc=COL_LOSS, ec=COL_LOSS_EDGE, tc=COL_TEXT_WHITE,
         fontsize=13, sublabel_fontsize=9.5)

# Arrow from normalize to loss
draw_arrow(ax, X_NORM + BOX_W/2, Y_MAIN, X_LOSS - BOX_W/2, Y_MAIN)
draw_label_on_arrow(ax, (X_NORM + BOX_W/2 + X_LOSS - BOX_W/2)/2,
                    Y_MAIN + 0.3, "advantages $\\hat{A}$", fontsize=11,
                    color=COL_LOSS_EDGE)

# ── 8. Gradient Update ───────────────────────────────────────────────────────
draw_box(ax, X_GRAD, Y_MAIN, BOX_W - 0.2, BOX_H * 1.2,
         "Update $\\theta$", sublabel="$\\nabla$ GRPO loss",
         fc=COL_LOSS, ec=COL_LOSS_EDGE, tc=COL_TEXT_WHITE,
         fontsize=13, sublabel_fontsize=9.5)

# Arrow from loss to gradient
draw_arrow(ax, X_LOSS + BOX_W/2, Y_MAIN, X_GRAD - (BOX_W-0.2)/2, Y_MAIN)

# ── 9. Feedback loop (gradient back to policy) ──────────────────────────────
# Segmented path going BELOW the main flow: Update → down → left → up → Policy
feedback_y = Y_MAIN - 4.2  # how far below to route
grad_bottom_x = X_GRAD
policy_bottom_x = X_POLICY

# Vertical line down from Update
ax.plot([grad_bottom_x, grad_bottom_x],
        [Y_MAIN - BOX_H*1.2/2, feedback_y + 0.25],
        color=COL_LOSS_EDGE, lw=2.2, linestyle="--", zorder=2)
# Rounded corner: small arc or just a line from Update down to horizontal
ax.plot([grad_bottom_x, policy_bottom_x],
        [feedback_y + 0.25, feedback_y + 0.25],
        color=COL_LOSS_EDGE, lw=2.2, linestyle="--", zorder=2)
# Vertical line up to Policy (with arrowhead)
ax.annotate(
    "",
    xy=(policy_bottom_x, Y_MAIN - BOX_H*1.2/2),
    xytext=(policy_bottom_x, feedback_y + 0.25),
    arrowprops=dict(
        arrowstyle="-|>",
        color=COL_LOSS_EDGE,
        lw=2.2,
        mutation_scale=18,
        linestyle="--",
    ),
    zorder=2,
)
draw_label_on_arrow(ax, (X_POLICY + X_GRAD)/2, feedback_y + 0.25,
                    "update encoder, projector, LoRA weights",
                    fontsize=9, color=COL_LOSS_EDGE)

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE POLICY BRANCH (dashed path for KL)
# ══════════════════════════════════════════════════════════════════════════════

# Reference policy box
draw_box(ax, X_POLICY, Y_REF, BOX_W, BOX_H,
         "Ref. Policy", sublabel="$\\pi_{ref}$  (frozen SFT)",
         fc=COL_REF, ec=COL_REF_EDGE, tc=COL_TEXT_WHITE,
         fontsize=11, sublabel_fontsize=9)

# Dashed arrow from input merge to reference policy
draw_arrow(ax, merge_x + 0.3, Y_MAIN - 0.15, X_POLICY - BOX_W/2, Y_REF + 0.15,
           color=COL_REF_ARROW, lw=1.8, style="-|>", linestyle="--",
           connectionstyle="arc3,rad=0.25")

# ── Log-prob computation boxes ────────────────────────────────────────────────
log_prob_x = (X_POLICY + X_LOSS) / 2 + 0.5
log_prob_main_y = Y_REF + (Y_MAIN - Y_REF) * 0.48
LOG_BOX_W = 2.5
LOG_BOX_H = 0.65

# log pi_theta(y|x) — from main policy
draw_box(ax, log_prob_x, log_prob_main_y, LOG_BOX_W, LOG_BOX_H,
         "$\\log \\pi_\\theta(y|x)$", sublabel=None,
         fc="#DBEAFE", ec=COL_POLICY_EDGE, tc=COL_TEXT_DARK,
         fontsize=10.5)

# log pi_ref(y|x) — from reference policy
draw_box(ax, log_prob_x, Y_REF, LOG_BOX_W, LOG_BOX_H,
         "$\\log \\pi_{ref}(y|x)$", sublabel=None,
         fc="#E2E8F0", ec=COL_REF_EDGE, tc=COL_TEXT_DARK,
         fontsize=10.5)

# Arrow from main policy down to log pi_theta
draw_arrow(ax, X_POLICY + BOX_W/2, Y_MAIN - 0.25,
           log_prob_x - LOG_BOX_W/2, log_prob_main_y + 0.1,
           color=COL_POLICY_EDGE, lw=1.5, style="-|>", linestyle="--",
           connectionstyle="arc3,rad=0.12")

# Arrow from ref policy to log pi_ref
draw_arrow(ax, X_POLICY + BOX_W/2, Y_REF,
           log_prob_x - LOG_BOX_W/2, Y_REF,
           color=COL_REF_ARROW, lw=1.8, style="-|>", linestyle="--")

# Arrows from both log probs to GRPO Loss for KL computation
draw_arrow(ax, log_prob_x + LOG_BOX_W/2, Y_REF,
           X_LOSS - BOX_W/2, Y_MAIN - 0.35,
           color=COL_REF_ARROW, lw=1.8, style="-|>", linestyle="--",
           connectionstyle="arc3,rad=-0.12")

draw_arrow(ax, log_prob_x + LOG_BOX_W/2, log_prob_main_y,
           X_LOSS - BOX_W/2, Y_MAIN - 0.15,
           color=COL_POLICY_EDGE, lw=1.5, style="-|>", linestyle="--",
           connectionstyle="arc3,rad=-0.08")

# KL label between the two log prob arrows
kl_label_x = (log_prob_x + LOG_BOX_W/2 + X_LOSS - BOX_W/2) / 2
kl_label_y = (Y_REF + log_prob_main_y) / 2
draw_label_on_arrow(ax, kl_label_x, kl_label_y,
                    "KL divergence\n($\\beta = 0.04$)",
                    fontsize=9.5, color=COL_REF_EDGE)

# ── "frozen" indicator for reference policy ───────────────────────────────────
# Use text instead of emoji to avoid font issues
ax.text(X_POLICY - BOX_W/2 - 0.35, Y_REF, "FROZEN",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=COL_REF_EDGE, rotation=90, zorder=6,
        bbox=dict(facecolor="#F1F5F9", edgecolor=COL_REF_EDGE,
                  pad=3, boxstyle="round,pad=0.2", linewidth=1.2))

# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATIONS & LABELS
# ══════════════════════════════════════════════════════════════════════════════

# Epsilon annotation near GRPO loss
ax.text(X_LOSS, Y_MAIN + BOX_H*1.2/2 + 0.18,
        "$\\epsilon = 0.2$  (clip range)",
        ha="center", va="bottom", fontsize=11, color=COL_LOSS_EDGE,
        fontstyle="italic", zorder=6)

# Per-token annotation near GRPO loss
ax.text(X_LOSS, Y_MAIN - BOX_H*1.2/2 - 0.15,
        "per-token log-probs",
        ha="center", va="top", fontsize=11, color=COL_TEXT_MED,
        fontstyle="italic", zorder=6)

# ── Title ─────────────────────────────────────────────────────────────────────
title_x = (X_INPUT_TS + X_GRAD) / 2
title_y = Y_MAIN + 3.0
ax.text(
    title_x, title_y,
    "End-to-End RL Data Flow  --  GRPO with LLM-as-a-Judge",
    ha="center", va="bottom",
    fontsize=16, fontweight="bold", color=COL_TEXT_DARK,
    zorder=10,
)
ax.text(
    title_x, title_y - 0.3,
    "From time-series input through policy generation, judging, reward normalization, to gradient update",
    ha="center", va="bottom",
    fontsize=10.5, color=COL_TEXT_MED,
    zorder=10,
)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=COL_INPUT, edgecolor=COL_INPUT_EDGE, linewidth=1.5,
                   label="Input data"),
    mpatches.Patch(facecolor=COL_POLICY, edgecolor=COL_POLICY_EDGE, linewidth=1.5,
                   label="Policy  $\\pi_\\theta$"),
    mpatches.Patch(facecolor=COL_GEN, edgecolor=COL_GEN_EDGE, linewidth=1.5,
                   label="Generated responses (K=4)"),
    mpatches.Patch(facecolor=COL_JUDGE, edgecolor=COL_JUDGE_EDGE, linewidth=1.5,
                   label="LLM Judge"),
    mpatches.Patch(facecolor=COL_REWARD, edgecolor=COL_REWARD_EDGE, linewidth=1.5,
                   label="Reward & normalization"),
    mpatches.Patch(facecolor=COL_LOSS, edgecolor=COL_LOSS_EDGE, linewidth=1.5,
                   label="GRPO loss & update"),
    mpatches.Patch(facecolor=COL_REF, edgecolor=COL_REF_EDGE, linewidth=1.5,
                   label="Reference policy  $\\pi_{ref}$  (frozen)"),
]
legend = ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=4,
    fontsize=9.5,
    frameon=True,
    fancybox=True,
    shadow=False,
    edgecolor="#D1D5DB",
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=1.2,
)

# ── Dashed-line legend note ───────────────────────────────────────────────────
ax.plot([2.0, 3.2], [-0.25, -0.25], color=COL_REF_ARROW, lw=1.8, linestyle="--",
        zorder=6)
ax.text(3.4, -0.25, "= reference / KL path (dashed)", ha="left", va="center",
        fontsize=9, color=COL_TEXT_MED, zorder=6)

# ── Axis cleanup ──────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, X_GRAD + BOX_W/2 + 1.0)
ax.set_ylim(-2.0, Y_MAIN + 3.6)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/llm-judge/fig_005.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
