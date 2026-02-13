"""
fig_006: Gated Cross-Attention Integration in LLM
Shows how FlamingoLayer wraps each LLM decoder layer with gated cross-attention,
illustrating the integration of visual (time series) features into the language model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Color palette ───────────────────────────────────────────────────────────
C_FROZEN_BG    = "#D6E4F0"     # light steel blue for frozen components
C_FROZEN_EDGE  = "#4A7FB5"     # medium blue border
C_TRAIN_BG     = "#FFE0B2"     # warm peach for trainable components
C_TRAIN_EDGE   = "#E65100"     # deep orange border
C_GATE_BG      = "#C8E6C9"     # light green for gating
C_GATE_EDGE    = "#2E7D32"     # dark green border
C_PERCEIVER_BG = "#E1BEE7"     # light purple for perceiver
C_PERCEIVER_EDGE = "#6A1B9A"   # deep purple border
C_ARROW        = "#37474F"     # dark grey arrows
C_TEXT         = "#212121"      # near-black text
C_REPEAT_BG    = "#F5F5F5"     # very light grey for repeat indicators
C_WHITE        = "#FFFFFF"
C_RESIDUAL     = "#78909C"     # blue-grey for residual connections

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ─── Helper functions ────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label, fc, ec, fontsize=11, fontweight="normal",
             alpha=1.0, linestyle="-", linewidth=1.5, sublabel=None, sublabel_fs=9):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fc, edgecolor=ec,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha,
        zorder=2
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=C_TEXT, zorder=3)
        ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                ha="center", va="center", fontsize=sublabel_fs,
                fontstyle="italic", color="#616161", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, color=C_TEXT, zorder=3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, style="-|>",
               connectionstyle="arc3,rad=0", zorder=4):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        lw=lw, connectionstyle=connectionstyle,
        mutation_scale=15, zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

def draw_add_circle(ax, cx, cy, r=0.3):
    """Draw a circled + symbol for residual addition."""
    circle = plt.Circle((cx, cy), r, fc=C_WHITE, ec=C_ARROW, lw=1.8, zorder=5)
    ax.add_patch(circle)
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=16,
            fontweight="bold", color=C_ARROW, zorder=6)
    return circle


# ═══════════════════════════════════════════════════════════════════════════════
# Title
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(7, 17.5, "Gated Cross-Attention Integration in LLM",
        ha="center", va="center", fontsize=16, fontweight="bold", color=C_TEXT)
ax.text(7, 17.1, "FlamingoLayer wraps each decoder layer with trainable cross-attention",
        ha="center", va="center", fontsize=11, color="#616161")

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT COORDINATES
# ═══════════════════════════════════════════════════════════════════════════════

# Main column center
cx = 7.0
box_w = 4.2
box_h = 0.75

# Left perceiver column
pcx = 1.8

# ─── "Repeat for all N layers" top indicator ─────────────────────────────────
repeat_top_y = 16.5
ax.text(cx, repeat_top_y, "⋮  Layer N  ⋮",
        ha="center", va="center", fontsize=12, color="#9E9E9E",
        fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# LARGE DASHED BOUNDING BOX for FlamingoLayer
# ═══════════════════════════════════════════════════════════════════════════════
flamingo_box_x = 0.5
flamingo_box_y = 3.7
flamingo_box_w = 13.0
flamingo_box_h = 12.4

flamingo_rect = FancyBboxPatch(
    (flamingo_box_x, flamingo_box_y), flamingo_box_w, flamingo_box_h,
    boxstyle="round,pad=0.3",
    facecolor="#FAFAFA", edgecolor="#455A64",
    linewidth=2.5, linestyle=(0, (6, 3)),
    alpha=0.5, zorder=1
)
ax.add_patch(flamingo_rect)

# FlamingoLayer label
ax.text(flamingo_box_x + 0.5, flamingo_box_y + flamingo_box_h - 0.15,
        "FlamingoLayer (wraps Decoder Layer i)",
        ha="left", va="top", fontsize=13, fontweight="bold",
        color="#37474F", zorder=3,
        bbox=dict(boxstyle="round,pad=0.2", fc="#FAFAFA", ec="none", alpha=0.9))

# ═══════════════════════════════════════════════════════════════════════════════
# PERCEIVER OUTPUT (left side)
# ═══════════════════════════════════════════════════════════════════════════════
perc_y = 10.0
perc_w = 3.2
perc_h = 1.8

draw_box(ax, pcx - perc_w/2, perc_y, perc_w, perc_h,
         "Perceiver\nResampler Output", C_PERCEIVER_BG, C_PERCEIVER_EDGE,
         fontsize=11, fontweight="bold")
ax.text(pcx, perc_y - 0.25, "(fixed-length visual tokens)",
        ha="center", va="top", fontsize=9, color="#6A1B9A", fontstyle="italic")

# Small label: TRAINABLE
ax.text(pcx, perc_y + perc_h + 0.15, "TRAINABLE",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
        color=C_TRAIN_EDGE,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_TRAIN_BG, ec=C_TRAIN_EDGE, lw=1))

# Arrow from perceiver to cross-attention
cross_attn_y = 10.7
draw_arrow(ax, pcx + perc_w/2 + 0.05, perc_y + perc_h/2,
           cx - box_w/2 - 0.05, cross_attn_y + box_h/2,
           color=C_PERCEIVER_EDGE, lw=2.0)
ax.text((pcx + perc_w/2 + cx - box_w/2) / 2,
        (perc_y + perc_h/2 + cross_attn_y + box_h/2) / 2 + 0.35,
        "K, V", ha="center", va="center", fontsize=10,
        fontweight="bold", color=C_PERCEIVER_EDGE,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=C_PERCEIVER_EDGE, lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (bottom to top)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 1. Input from previous layer ───────────────────────────────────────────
input_y = 2.6
ax.text(cx, input_y, "Input from Decoder Layer (i−1)",
        ha="center", va="center", fontsize=11, color="#616161",
        fontweight="bold")
draw_arrow(ax, cx, input_y + 0.25, cx, input_y + 0.8, color=C_ARROW, lw=2)

# ─── 2. Branching point (split into residual + cross-attention path) ────────
branch_y = 3.9
# Small dot at branch
ax.plot(cx, branch_y, 'o', color=C_ARROW, markersize=6, zorder=5)

# ─── 3. Self-Attention (frozen) ─────────────────────────────────────────────
sa_y = 4.4
draw_box(ax, cx - box_w/2, sa_y, box_w, box_h,
         "Multi-Head Self-Attention", C_FROZEN_BG, C_FROZEN_EDGE,
         fontsize=11, fontweight="bold")
# FROZEN label
ax.text(cx + box_w/2 + 0.3, sa_y + box_h/2, "FROZEN",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_FROZEN_EDGE,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_FROZEN_BG, ec=C_FROZEN_EDGE, lw=1))

draw_arrow(ax, cx, branch_y, cx, sa_y, color=C_ARROW, lw=2)

# ─── 4. Add & Norm after Self-Attention ──────────────────────────────────────
addnorm1_y = 5.55
draw_box(ax, cx - 1.5, addnorm1_y, 3.0, 0.55,
         "Add & LayerNorm", C_FROZEN_BG, C_FROZEN_EDGE,
         fontsize=10)
draw_arrow(ax, cx, sa_y + box_h, cx, addnorm1_y, color=C_ARROW, lw=2)

# ─── 5. FFN (frozen) ────────────────────────────────────────────────────────
ffn_y = 6.55
draw_box(ax, cx - box_w/2, ffn_y, box_w, box_h,
         "Feed-Forward Network (FFN)", C_FROZEN_BG, C_FROZEN_EDGE,
         fontsize=11, fontweight="bold")
ax.text(cx + box_w/2 + 0.3, ffn_y + box_h/2, "FROZEN",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_FROZEN_EDGE,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_FROZEN_BG, ec=C_FROZEN_EDGE, lw=1))
draw_arrow(ax, cx, addnorm1_y + 0.55, cx, ffn_y, color=C_ARROW, lw=2)

# ─── 6. Add & Norm after FFN ────────────────────────────────────────────────
addnorm2_y = 7.7
draw_box(ax, cx - 1.5, addnorm2_y, 3.0, 0.55,
         "Add & LayerNorm", C_FROZEN_BG, C_FROZEN_EDGE,
         fontsize=10)
draw_arrow(ax, cx, ffn_y + box_h, cx, addnorm2_y, color=C_ARROW, lw=2)

# ─── Original decoder output label ──────────────────────────────────────────
decoder_out_y = 8.55
ax.text(cx, decoder_out_y, "Decoder Layer Output (text features)",
        ha="center", va="center", fontsize=10, color="#455A64",
        fontstyle="italic")
draw_arrow(ax, cx, addnorm2_y + 0.55, cx, decoder_out_y - 0.2, color=C_ARROW, lw=2)

# ═══════════════════════════════════════════════════════════════════════════════
# GATED CROSS-ATTENTION PATH (the new trainable components)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── 7. Branch point to gated cross-attention ────────────────────────────────
# The decoder output feeds as Query into cross-attention
draw_arrow(ax, cx, decoder_out_y + 0.2, cx, cross_attn_y - 0.05, color=C_ARROW, lw=2)

# ─── Q label on upward arrow ─────────────────────────────────────────────────
ax.text(cx + 0.35, (decoder_out_y + cross_attn_y) / 2 + 0.25, "Q",
        ha="left", va="center", fontsize=10, fontweight="bold", color=C_ARROW,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=C_ARROW, lw=0.8))

# ─── 8. Gated Cross-Attention block ─────────────────────────────────────────
draw_box(ax, cx - box_w/2, cross_attn_y, box_w, box_h,
         "Cross-Attention", C_TRAIN_BG, C_TRAIN_EDGE,
         fontsize=11, fontweight="bold",
         sublabel="query=text, key/value=vision", sublabel_fs=9)
ax.text(cx + box_w/2 + 0.3, cross_attn_y + box_h/2, "TRAINABLE",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_TRAIN_EDGE,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_TRAIN_BG, ec=C_TRAIN_EDGE, lw=1))

# ─── 9. Tanh Gate ───────────────────────────────────────────────────────────
gate_y = 12.0
gate_w = 3.6
gate_h = 0.85
draw_box(ax, cx - gate_w/2, gate_y, gate_w, gate_h,
         "tanh Gate (α)", C_GATE_BG, C_GATE_EDGE,
         fontsize=12, fontweight="bold")
ax.text(cx + gate_w/2 + 0.3, gate_y + gate_h/2, "TRAINABLE",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color=C_GATE_EDGE,
        bbox=dict(boxstyle="round,pad=0.15", fc=C_GATE_BG, ec=C_GATE_EDGE, lw=1))
ax.text(cx, gate_y - 0.25, "α = tanh(learned_param)",
        ha="center", va="top", fontsize=9, color=C_GATE_EDGE, fontstyle="italic")

draw_arrow(ax, cx, cross_attn_y + box_h, cx, gate_y, color=C_ARROW, lw=2)

# Multiply symbol between gate output
mult_y = 13.2
# Circle with × inside (like the + circle but for multiply)
mult_circle = plt.Circle((cx, mult_y), 0.25, fc=C_GATE_BG, ec=C_GATE_EDGE, lw=1.8, zorder=5)
ax.add_patch(mult_circle)
ax.text(cx, mult_y, "×", ha="center", va="center", fontsize=14,
        fontweight="bold", color=C_GATE_EDGE, zorder=6)
draw_arrow(ax, cx, gate_y + gate_h, cx, mult_y - 0.25, color=C_ARROW, lw=2)

# ─── 10. Residual Addition (Add circle) ─────────────────────────────────────
add_y = 13.9
draw_add_circle(ax, cx, add_y)

draw_arrow(ax, cx, mult_y + 0.25, cx, add_y - 0.3, color=C_ARROW, lw=2)

# Residual skip connection from decoder output to add circle
# Draw a curved residual path on the right side
res_x = cx + box_w/2 + 2.0
# Vertical line up from decoder output level
ax.annotate("", xy=(res_x, add_y), xytext=(res_x, decoder_out_y),
            arrowprops=dict(arrowstyle="-", color=C_RESIDUAL, lw=2,
                           linestyle="--"))
# Horizontal line from decoder output to residual path
ax.annotate("", xy=(res_x, decoder_out_y), xytext=(cx + 0.3, decoder_out_y),
            arrowprops=dict(arrowstyle="-", color=C_RESIDUAL, lw=2,
                           linestyle="--"))
# Horizontal line from residual path to add circle
draw_arrow(ax, res_x, add_y, cx + 0.3, add_y, color=C_RESIDUAL, lw=2,
           style="-|>")

# Label on residual
ax.text(res_x + 0.3, (decoder_out_y + add_y) / 2, "Residual\nConnection",
        ha="left", va="center", fontsize=9, color=C_RESIDUAL,
        fontweight="bold", rotation=0)

# ─── 11. Output ──────────────────────────────────────────────────────────────
output_y = 14.7
ax.text(cx, output_y, "Enhanced Output → Decoder Layer (i+1)",
        ha="center", va="center", fontsize=11, color="#455A64",
        fontweight="bold")
draw_arrow(ax, cx, add_y + 0.3, cx, output_y - 0.2, color=C_ARROW, lw=2)

# Arrow to repeat indicator
draw_arrow(ax, cx, output_y + 0.2, cx, repeat_top_y - 0.3, color=C_ARROW, lw=2)

# ─── "Repeat" bottom indicator ──────────────────────────────────────────────
ax.text(cx, 2.0, "⋮  Layer 1  ⋮",
        ha="center", va="center", fontsize=12, color="#9E9E9E",
        fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_x = 0.6
legend_y = 1.2
legend_spacing = 0.45
legend_box_w = 0.5
legend_box_h = 0.3

legend_items = [
    (C_FROZEN_BG, C_FROZEN_EDGE, "Frozen LLM Components"),
    (C_TRAIN_BG, C_TRAIN_EDGE, "Trainable Cross-Attention"),
    (C_GATE_BG, C_GATE_EDGE, "Trainable Gating Mechanism"),
    (C_PERCEIVER_BG, C_PERCEIVER_EDGE, "Perceiver Resampler (Trainable)"),
]

for i, (fc, ec, label) in enumerate(legend_items):
    ly = legend_y - i * legend_spacing
    box = FancyBboxPatch(
        (legend_x, ly - legend_box_h/2), legend_box_w, legend_box_h,
        boxstyle="round,pad=0.05",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2
    )
    ax.add_patch(box)
    ax.text(legend_x + legend_box_w + 0.2, ly, label,
            ha="left", va="center", fontsize=10, color=C_TEXT)

# Dashed line legend item
ly = legend_y - len(legend_items) * legend_spacing
ax.plot([legend_x, legend_x + legend_box_w], [ly, ly],
        linestyle=(0, (6, 3)), color="#455A64", lw=2)
ax.text(legend_x + legend_box_w + 0.2, ly,
        "FlamingoLayer Boundary (per decoder layer)",
        ha="left", va="center", fontsize=10, color=C_TEXT)

# ═══════════════════════════════════════════════════════════════════════════════
# Annotation: cross_attn_every_n_layers
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(flamingo_box_x + flamingo_box_w - 0.3, flamingo_box_y + 0.3,
        "cross_attn_every_n_layers = 1\n(applied at every decoder layer)",
        ha="right", va="bottom", fontsize=9, color="#455A64",
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#B0BEC5", lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
plt.savefig("/home/wangni/notion-figures/genomics/fig_006.png",
            dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print("Figure saved: /home/wangni/notion-figures/genomics/fig_006.png")
