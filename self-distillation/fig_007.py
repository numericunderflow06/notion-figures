"""
fig_007: Self-Distillation Loss Computation
Dataflow diagram showing the mathematical pipeline from logits to final scalar loss.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
C_DATA      = "#E8F0FE"   # light blue – data boxes
C_DATA_BD   = "#4285F4"   # blue border – data boxes
C_OP        = "#FFF3E0"   # light orange – operation boxes
C_OP_BD     = "#F4880B"   # orange border – operation boxes
C_FORMULA   = "#455A64"   # dark slate – formula text
C_PARAM     = "#D32F2F"   # red – parameter annotations
C_ARROW     = "#546E7A"   # muted blue-grey arrows
C_BG        = "white"
C_TITLE     = "#263238"

# ── layout constants ────────────────────────────────────────────────
FIG_W, FIG_H = 10, 18
BOX_W = 5.2
BOX_H_DATA = 0.72
BOX_H_OP   = 0.82
COL_X = FIG_W / 2          # centre column
PARAM_X = COL_X + BOX_W / 2 + 0.35   # parameter annotation x

# vertical positions (top → bottom)
Y_POSITIONS = [
    16.2,   # 0  Student Logits  /  Teacher Logits
    14.6,   # 1  Top-K Approximation
    13.0,   # 2  Tail Bucket
    11.3,   # 3  Alpha-Interpolated JSD Mixture
    9.5,    # 4  KL Divergence Terms
    8.0,    # 5  Sum across vocab
    6.4,    # 6  Importance Sampling Clipping
    4.8,    # 7  Token-Mean Aggregation
    3.3,    # 8  Final Loss Scalar
]


def draw_box(ax, cx, cy, w, h, text, style="data", formula=None,
             fontsize=11, formula_size=8.5):
    """Draw a data box (sharp corners) or operation box (rounded)."""
    x0 = cx - w / 2
    y0 = cy - h / 2
    if style == "op":
        box = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.12",
            facecolor=C_OP, edgecolor=C_OP_BD, linewidth=1.6,
            zorder=3,
        )
    else:  # data
        box = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="square,pad=0.04",
            facecolor=C_DATA, edgecolor=C_DATA_BD, linewidth=1.6,
            zorder=3,
        )
    ax.add_patch(box)

    if formula:
        # two-line: label on top, formula below
        ax.text(cx, cy + 0.14, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_TITLE, zorder=4)
        ax.text(cx, cy - 0.16, formula, ha="center", va="center",
                fontsize=formula_size, fontstyle="italic", color=C_FORMULA,
                zorder=4, family="monospace")
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_TITLE, zorder=4)
    return cy - h / 2   # return bottom y


def draw_arrow(ax, y_from, y_to, x=None):
    """Vertical downward arrow between boxes."""
    if x is None:
        x = COL_X
    ax.annotate(
        "", xy=(x, y_to), xytext=(x, y_from),
        arrowprops=dict(
            arrowstyle="-|>", color=C_ARROW, lw=1.6,
            shrinkA=2, shrinkB=2,
        ),
        zorder=2,
    )


def draw_param(ax, cy, text, offset_x=0):
    """Annotation label to the right of a box."""
    ax.text(PARAM_X + offset_x, cy, text, ha="left", va="center",
            fontsize=9, color=C_PARAM, fontweight="bold", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="#FFEBEE", ec=C_PARAM,
                      lw=0.8, alpha=0.9))


# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(2.0, 17.5)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
ax.text(COL_X, 17.15, "Self-Distillation Loss Computation",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=C_TITLE)
ax.text(COL_X, 16.82, "compute_self_distillation_loss()  ·  core_algos.py : 1085-1188",
        ha="center", va="center", fontsize=9, color="#78909C",
        fontstyle="italic")

# ── 0 · input logits (two boxes side by side) ──────────────────────
y0 = Y_POSITIONS[0]
gap = 0.3
sw = (BOX_W - gap) / 2  # sub-box width
sx_left  = COL_X - gap/2 - sw/2
sx_right = COL_X + gap/2 + sw/2

draw_box(ax, sx_left,  y0, sw, BOX_H_DATA, "Student\nLogits", "data",
         fontsize=10)
draw_box(ax, sx_right, y0, sw, BOX_H_DATA, "Teacher\nLogits", "data",
         fontsize=10)

# small note
ax.text(COL_X, y0 - BOX_H_DATA/2 - 0.12,
        "shape: (bs, seq_len, vocab_size)", ha="center", va="top",
        fontsize=7.5, color="#90A4AE", fontstyle="italic")

# arrows from both to next box
y_bot_input = y0 - BOX_H_DATA / 2 - 0.12
y_top_topk  = Y_POSITIONS[1] + BOX_H_OP / 2
draw_arrow(ax, y_bot_input - 0.06, y_top_topk, x=sx_left)
draw_arrow(ax, y_bot_input - 0.06, y_top_topk, x=sx_right)

# ── 1 · Top-K Approximation ────────────────────────────────────────
y1 = Y_POSITIONS[1]
draw_box(ax, COL_X, y1, BOX_W, BOX_H_OP, "Top-K Selection", "op",
         formula="topk_logps = topk_logits − logsumexp(full_logits)")
draw_param(ax, y1, "k = 100")

y_bot = y1 - BOX_H_OP / 2
y_top = Y_POSITIONS[2] + BOX_H_OP / 2
draw_arrow(ax, y_bot, y_top)

# ── 2 · Tail Bucket ────────────────────────────────────────────────
y2 = Y_POSITIONS[2]
draw_box(ax, COL_X, y2, BOX_W, BOX_H_OP, "Tail Bucket Append", "op",
         formula="P(tail) = 1 − Σ P_i   →  dim: k+1 = 101")
draw_param(ax, y2, "add_tail = True")

# shape note
ax.text(COL_X, y2 - BOX_H_OP/2 - 0.12,
        "shape: (bs, seq_len, 101)", ha="center", va="top",
        fontsize=7.5, color="#90A4AE", fontstyle="italic")

y_bot = y2 - BOX_H_OP / 2 - 0.12
y_top = Y_POSITIONS[3] + BOX_H_OP / 2
draw_arrow(ax, y_bot - 0.04, y_top)

# ── 3 · JSD Mixture ────────────────────────────────────────────────
y3 = Y_POSITIONS[3]
draw_box(ax, COL_X, y3, BOX_W, BOX_H_OP + 0.18,
         "α-Interpolated JSD Mixture", "op",
         formula="M = (1−α)·S + α·T   (in log-space via logsumexp)")
draw_param(ax, y3, "α = 0.5")

y_bot = y3 - (BOX_H_OP + 0.18) / 2
y_top = Y_POSITIONS[4] + (BOX_H_OP + 0.18) / 2
draw_arrow(ax, y_bot, y_top)

# ── 4 · KL Divergence Terms ────────────────────────────────────────
y4 = Y_POSITIONS[4]
draw_box(ax, COL_X, y4, BOX_W, BOX_H_OP + 0.18,
         "KL Divergence Terms", "op",
         formula="JSD = (1−α)·KL(T‖M) + α·KL(S‖M)")

ax.text(COL_X, y4 - (BOX_H_OP + 0.18)/2 - 0.12,
        "Generalized Jensen-Shannon Divergence", ha="center", va="top",
        fontsize=7.5, color="#90A4AE", fontstyle="italic")

y_bot = y4 - (BOX_H_OP + 0.18) / 2 - 0.12
y_top = Y_POSITIONS[5] + BOX_H_OP / 2
draw_arrow(ax, y_bot - 0.04, y_top)

# ── 5 · Sum across vocab ───────────────────────────────────────────
y5 = Y_POSITIONS[5]
draw_box(ax, COL_X, y5, BOX_W, BOX_H_OP, "Sum Across Vocab Dim", "op",
         formula="per_token_loss = kl_loss.sum(dim=−1)")

ax.text(COL_X, y5 - BOX_H_OP/2 - 0.1,
        "shape: (bs, seq_len)", ha="center", va="top",
        fontsize=7.5, color="#90A4AE", fontstyle="italic")

y_bot = y5 - BOX_H_OP / 2 - 0.1
y_top = Y_POSITIONS[6] + BOX_H_OP / 2
draw_arrow(ax, y_bot - 0.04, y_top)

# ── 6 · Importance Sampling Clipping ───────────────────────────────
y6 = Y_POSITIONS[6]
draw_box(ax, COL_X, y6, BOX_W, BOX_H_OP + 0.1,
         "Importance Sampling Clipping", "op",
         formula="ratio = clamp(exp(log π_θ − log π_old), max=c)")
draw_param(ax, y6, "is_clip = 2.0")

# extra formula detail below
ax.text(COL_X, y6 - (BOX_H_OP + 0.1)/2 - 0.12,
        "per_token_loss *= ratio", ha="center", va="top",
        fontsize=8, color=C_FORMULA, fontstyle="italic", family="monospace")

y_bot = y6 - (BOX_H_OP + 0.1) / 2 - 0.22
y_top = Y_POSITIONS[7] + BOX_H_OP / 2
draw_arrow(ax, y_bot, y_top)

# ── 7 · Token-Mean Aggregation ─────────────────────────────────────
y7 = Y_POSITIONS[7]
draw_box(ax, COL_X, y7, BOX_W, BOX_H_OP, "Token-Mean Aggregation", "op",
         formula="loss = Σ(loss · mask) / Σ(mask)")
draw_param(ax, y7, "mode = token-mean")

y_bot = y7 - BOX_H_OP / 2
y_top = Y_POSITIONS[8] + BOX_H_DATA / 2
draw_arrow(ax, y_bot, y_top)

# ── 8 · Final Loss Scalar ──────────────────────────────────────────
y8 = Y_POSITIONS[8]
draw_box(ax, COL_X, y8, BOX_W * 0.65, BOX_H_DATA + 0.08,
         "Final Loss Scalar", "data", fontsize=12)

ax.text(COL_X, y8 - (BOX_H_DATA + 0.08)/2 - 0.15,
        "scalar → backward()", ha="center", va="top",
        fontsize=8, color="#90A4AE", fontstyle="italic")

# ── legend ──────────────────────────────────────────────────────────
legend_y = 2.6
legend_x = 1.0
# data box
leg_data = FancyBboxPatch((legend_x, legend_y - 0.12), 0.5, 0.24,
                           boxstyle="square,pad=0.02",
                           facecolor=C_DATA, edgecolor=C_DATA_BD, lw=1.2)
ax.add_patch(leg_data)
ax.text(legend_x + 0.65, legend_y, "Data (logits / distributions)",
        va="center", fontsize=8, color=C_TITLE)

# op box
leg_op = FancyBboxPatch((legend_x + 5.0, legend_y - 0.12), 0.5, 0.24,
                          boxstyle="round,pad=0.06",
                          facecolor=C_OP, edgecolor=C_OP_BD, lw=1.2)
ax.add_patch(leg_op)
ax.text(legend_x + 5.65, legend_y, "Operation",
        va="center", fontsize=8, color=C_TITLE)

# ── save ────────────────────────────────────────────────────────────
out = "/home/wangni/notion-figures/self-distillation/fig_007.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BG,
            pad_inches=0.3)
plt.close(fig)
print(f"Saved → {out}")
