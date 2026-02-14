"""
fig_001: OpenTSLM-NoMask Architecture Overview
Horizontal flow diagram showing the full pipeline with color-coded components.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
ORANGE      = "#E8853D"   # trainable components
ORANGE_LIGHT = "#FFF0E0"  # trainable fill
BLUE        = "#4A90D9"   # frozen components
BLUE_LIGHT  = "#E0EDFA"   # frozen fill
GREY        = "#F5F5F5"   # input/output background
GREY_BORDER = "#999999"
DARK        = "#2C3E50"   # text
HIGHLIGHT   = "#E74C3C"   # annotation highlight (red)
HIGHLIGHT_BG = "#FFF5F5"
ARROW_CLR   = "#555555"
WHITE       = "#FFFFFF"

# ── figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7.5), dpi=200)
ax.set_xlim(0, 18)
ax.set_ylim(0, 7.5)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)

# ── helper: rounded box ────────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, sublabel, facecolor, edgecolor, lw=1.8,
             fontsize=12, subfontsize=9, text_color=DARK):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=2
    )
    ax.add_patch(box)
    # main label
    ax.text(x + w / 2, y + h / 2 + 0.18, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=3)
    # sub-label (specs)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.28, sublabel,
                ha="center", va="center", fontsize=subfontsize,
                color=text_color, alpha=0.82, zorder=3,
                fontstyle="italic")
    return (x + w, y + h / 2)  # right-center anchor for arrows


def draw_arrow(ax, x_start, y_start, x_end, y_end, color=ARROW_CLR):
    ax.annotate(
        "", xy=(x_end, y_end), xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.0,
            mutation_scale=18,
            shrinkA=2, shrinkB=2,
        ),
        zorder=4,
    )

# ── title ───────────────────────────────────────────────────────────
ax.text(9, 7.15, "OpenTSLM-NoMask Architecture Overview",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=DARK)

# ── legend ──────────────────────────────────────────────────────────
legend_y = 6.65
legend_items = [
    (ORANGE_LIGHT, ORANGE, "Trainable (~250M params)"),
    (BLUE_LIGHT, BLUE, "Frozen (~500M params)"),
]
legend_x = 5.0
for fc, ec, txt in legend_items:
    rect = FancyBboxPatch((legend_x, legend_y - 0.15), 0.4, 0.3,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(legend_x + 0.55, legend_y, txt, ha="left", va="center",
            fontsize=10, color=DARK)
    legend_x += 4.2

# ── row-y positions ─────────────────────────────────────────────────
main_y = 3.0      # vertical center for main boxes
box_h = 1.8       # box height
gap = 0.45        # horizontal gap between boxes

# ── 1. Multi-Channel Sensor Data (input) ───────────────────────────
bx, by, bw = 0.4, main_y, 2.2
draw_box(ax, bx, by, bw, box_h,
         "Multi-Channel\nSensor Data", "NGAFID sensors",
         GREY, GREY_BORDER, fontsize=11, subfontsize=9)
p1_right = (bx + bw, by + box_h / 2)

# ── 2. CNN Tokenizer (trainable) ───────────────────────────────────
bx2 = p1_right[0] + gap
bw2 = 2.6
draw_box(ax, bx2, by, bw2, box_h,
         "CNN Tokenizer", "patch=4, dim=128",
         ORANGE_LIGHT, ORANGE, fontsize=12, subfontsize=10)
p2_right = (bx2 + bw2, by + box_h / 2)
draw_arrow(ax, p1_right[0] + 0.05, p1_right[1], bx2 - 0.05, p1_right[1])

# ── 3. Perceiver Resampler (trainable) ─────────────────────────────
bx3 = p2_right[0] + gap
bw3 = 2.8
draw_box(ax, bx3, by, bw3, box_h,
         "Perceiver\nResampler", "64 latents, depth 2, 8 heads",
         ORANGE_LIGHT, ORANGE, fontsize=12, subfontsize=9.5)
p3_right = (bx3 + bw3, by + box_h / 2)
draw_arrow(ax, p2_right[0] + 0.05, p2_right[1], bx3 - 0.05, p2_right[1])

# ── 4. Gated Cross-Attention + Qwen2.5 (combined block) ────────────
bx4 = p3_right[0] + gap
bw4 = 4.2

# Outer container (trainable cross-attn border)
outer_box = FancyBboxPatch(
    (bx4, by - 0.15), bw4, box_h + 0.3,
    boxstyle="round,pad=0.15",
    facecolor=WHITE, edgecolor=ORANGE, linewidth=2.2, linestyle="--",
    zorder=1
)
ax.add_patch(outer_box)

# Sub-box: Gated Cross-Attention (trainable, top-left area)
ca_w, ca_h = 1.75, 1.35
ca_x, ca_y = bx4 + 0.2, by + box_h - ca_h - 0.05
ca_box = FancyBboxPatch(
    (ca_x, ca_y), ca_w, ca_h,
    boxstyle="round,pad=0.1",
    facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.5, zorder=2
)
ax.add_patch(ca_box)
ax.text(ca_x + ca_w / 2, ca_y + ca_h / 2 + 0.15,
        "Gated Cross-\nAttention", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=DARK, zorder=3)
ax.text(ca_x + ca_w / 2, ca_y + ca_h / 2 - 0.38,
        "24 layers", ha="center", va="center",
        fontsize=9, color=DARK, alpha=0.8, fontstyle="italic", zorder=3)

# Sub-box: Qwen2.5-0.5B (frozen, right area)
qw_w, qw_h = 1.75, 1.35
qw_x, qw_y = ca_x + ca_w + 0.25, ca_y
qw_box = FancyBboxPatch(
    (qw_x, qw_y), qw_w, qw_h,
    boxstyle="round,pad=0.1",
    facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.5, zorder=2
)
ax.add_patch(qw_box)
ax.text(qw_x + qw_w / 2, qw_y + qw_h / 2 + 0.15,
        "Qwen2.5-0.5B", ha="center", va="center",
        fontsize=10.5, fontweight="bold", color=DARK, zorder=3)
ax.text(qw_x + qw_w / 2, qw_y + qw_h / 2 - 0.38,
        "24 layers, hidden 896", ha="center", va="center",
        fontsize=8.5, color=DARK, alpha=0.8, fontstyle="italic", zorder=3)

# Label: "Interleaved" between the two sub-boxes
mid_x = (ca_x + ca_w + qw_x) / 2
ax.annotate("", xy=(qw_x - 0.02, ca_y + ca_h / 2 + 0.12),
            xytext=(ca_x + ca_w + 0.02, ca_y + ca_h / 2 + 0.12),
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.2))
ax.text(mid_x, ca_y + ca_h / 2 - 0.12, "interleaved",
        ha="center", va="center", fontsize=8, color=DARK, zorder=3)

# Container label at bottom
ax.text(bx4 + bw4 / 2, by - 0.0,
        "Gated Cross-Attention Layers (trainable) + LLM Backbone (frozen)",
        ha="center", va="center", fontsize=8, color=DARK, alpha=0.7, zorder=3)

p4_right = (bx4 + bw4, by + box_h / 2)
draw_arrow(ax, p3_right[0] + 0.05, p3_right[1], bx4 - 0.05, p3_right[1])

# ── 5. Text Generation (output) ────────────────────────────────────
bx5 = p4_right[0] + gap
bw5 = 2.0
draw_box(ax, bx5, by, bw5, box_h,
         "Text\nGeneration", "Diagnosis output",
         GREY, GREY_BORDER, fontsize=12, subfontsize=9.5)
draw_arrow(ax, p4_right[0] + 0.05, p4_right[1], bx5 - 0.05, p4_right[1])

# ── annotation callout: eq → ge change ─────────────────────────────
ann_x, ann_y = bx4 + 0.5, 0.25
ann_w, ann_h = 3.2, 1.6

ann_box = FancyBboxPatch(
    (ann_x, ann_y), ann_w, ann_h,
    boxstyle="round,pad=0.18",
    facecolor=HIGHLIGHT_BG, edgecolor=HIGHLIGHT, linewidth=2.2,
    zorder=5
)
ax.add_patch(ann_box)

ax.text(ann_x + ann_w / 2, ann_y + ann_h - 0.25,
        "NoMask Modification", ha="center", va="center",
        fontsize=11, fontweight="bold", color=HIGHLIGHT, zorder=6)

ax.text(ann_x + ann_w / 2, ann_y + ann_h - 0.65,
        "MaskedCrossAttention mask_op:",
        ha="center", va="center", fontsize=9.5, color=DARK, zorder=6)

# eq → ge with strikethrough on eq
ax.text(ann_x + ann_w / 2 - 0.55, ann_y + ann_h - 1.0,
        "torch.eq", ha="center", va="center", fontsize=11,
        color="#999999", zorder=6, fontstyle="italic",
        fontfamily="monospace")
# strikethrough line
ax.plot([ann_x + ann_w / 2 - 1.05, ann_x + ann_w / 2 - 0.05],
        [ann_y + ann_h - 1.0, ann_y + ann_h - 1.0],
        color="#999999", lw=1.5, zorder=7)
# arrow
ax.text(ann_x + ann_w / 2 + 0.05, ann_y + ann_h - 1.0,
        "→", ha="center", va="center", fontsize=13,
        color=DARK, zorder=6)
# ge
ax.text(ann_x + ann_w / 2 + 0.6, ann_y + ann_h - 1.0,
        "torch.ge", ha="center", va="center", fontsize=11,
        color=HIGHLIGHT, fontweight="bold", zorder=6,
        fontfamily="monospace")

ax.text(ann_x + ann_w / 2, ann_y + 0.22,
        "Enables cross-channel attention",
        ha="center", va="center", fontsize=9, color=DARK,
        alpha=0.85, zorder=6)

# Arrow from annotation to the cross-attention sub-box
ann_top_x = ann_x + ann_w / 2
ann_top_y = ann_y + ann_h
target_y = ca_y
target_x = ca_x + ca_w / 2

ax.annotate(
    "", xy=(target_x, target_y - 0.05),
    xytext=(ann_top_x, ann_top_y + 0.02),
    arrowprops=dict(
        arrowstyle="-|>",
        color=HIGHLIGHT,
        lw=1.8,
        connectionstyle="arc3,rad=-0.15",
        mutation_scale=15,
    ),
    zorder=5,
)

# ── data-shape annotations (above the arrows) ──────────────────────
shape_y = main_y + box_h + 0.15
shapes = [
    ((0.4 + 2.2 / 2), "C channels ×\nL samples"),
    ((bx2 + bw2 / 2), "C × (L/4) ×\n128-d patches"),
    ((bx3 + bw3 / 2), "C × 64 ×\n128-d latents"),
]
for sx, txt in shapes:
    ax.text(sx, shape_y + 0.15, txt, ha="center", va="bottom",
            fontsize=8, color=DARK, alpha=0.6, zorder=3,
            linespacing=1.1)

# ── parameter labels below boxes ────────────────────────────────────
param_y = main_y - 0.35
params = [
    (bx2 + bw2 / 2, "trainable"),
    (bx3 + bw3 / 2, "trainable"),
    (bx4 + bw4 * 0.27, "trainable"),
    (bx4 + bw4 * 0.72, "frozen\n~500M params"),
]
for px, txt in params:
    ax.text(px, param_y, txt, ha="center", va="top",
            fontsize=8, color=DARK, alpha=0.55, zorder=3)

# total trainable label
ax.text(bx3, param_y - 0.45,
        "Total trainable: ~250M params   |   Total frozen: ~500M params",
        ha="left", va="top", fontsize=9, color=DARK, alpha=0.6,
        zorder=3, fontweight="bold")

plt.tight_layout(pad=0.5)
fig.savefig("/home/wangni/notion-figures/nomask/fig_001.png",
            dpi=200, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/nomask/fig_001.png")
