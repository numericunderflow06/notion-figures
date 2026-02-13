"""
OpenTSLM-Flamingo Architecture for Genomics
End-to-end architecture diagram showing the full pipeline from DNA input to prediction output.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
TRAINABLE_BLUE   = "#3B82F6"   # vivid blue for trainable components
TRAINABLE_LIGHT  = "#DBEAFE"   # light blue fill for trainable
FROZEN_GRAY      = "#6B7280"   # gray for frozen components
FROZEN_LIGHT     = "#E5E7EB"   # light gray fill for frozen
INPUT_GREEN      = "#10B981"   # green for input data
INPUT_LIGHT      = "#D1FAE5"
OUTPUT_PURPLE    = "#8B5CF6"   # purple for output
OUTPUT_LIGHT     = "#EDE9FE"
ENCODING_TEAL    = "#0D9488"   # teal for encoding step
ENCODING_LIGHT   = "#CCFBF1"
ARROW_COLOR      = "#374151"
TEXT_COLOR        = "#111827"
SHAPE_COLOR      = "#DC2626"   # red for tensor shapes
BG_COLOR         = "#FFFFFF"
SUBTITLE_COLOR   = "#6B7280"

fig, ax = plt.subplots(figsize=(24, 8), dpi=200)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-0.3, 23.3)
ax.set_ylim(-1.8, 6.2)
ax.axis("off")

# ── Title ───────────────────────────────────────────────────────────────────
mid_x = 11.5
ax.text(mid_x, 5.7, "OpenTSLM-Flamingo Architecture for Genomics",
        fontsize=17, fontweight="bold", ha="center", va="center",
        color=TEXT_COLOR, family="sans-serif")
ax.text(mid_x, 5.28, "Flamingo-style multimodal architecture adapting DNA sequences as time series for LLM reasoning",
        fontsize=10, ha="center", va="center", color=SUBTITLE_COLOR, style="italic")

# ── Helper to draw a rounded box ───────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, sublabel, fill_color, edge_color, fontsize=11,
             sublabel_fontsize=8.5, bold=True, radius=0.15):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.08,rounding_size={radius}",
                         facecolor=fill_color, edgecolor=edge_color,
                         linewidth=1.8, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2 + 0.13, label,
            fontsize=fontsize, fontweight=weight, ha="center", va="center",
            color=TEXT_COLOR, zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                fontsize=sublabel_fontsize, ha="center", va="center",
                color=SUBTITLE_COLOR, zorder=3, style="italic")
    return x + w  # return right edge

def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8,
                                connectionstyle="arc3,rad=0.0"),
                zorder=1)

def draw_shape_label(ax, x, y, text, fontsize=8.5):
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center",
            color=SHAPE_COLOR, fontweight="bold", family="monospace",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FEF2F2",
                      edgecolor="#FECACA", linewidth=0.8, alpha=0.95),
            zorder=4)

# ── Main pipeline layout ───────────────────────────────────────────────────
Y = 2.5      # vertical centre of boxes
H = 1.3      # box height
gap = 0.35   # gap between boxes (arrow space)

# Pre-calculate positions to ensure everything fits
# Total width budget: ~23 units
# Boxes: 2.2 + 2.7 + 2.6 + 2.7 + 2.6 + 2.5 + 2.2 = 17.5
# Gaps:  6 * 0.35 = 2.1   + 6 * 0.15 = 0.9  → 3.0
# Total: ~20.5 — fits nicely

# ═══════════════════════════════════════════════════════════════════════════
# BOX 1: DNA Sequence Input
# ═══════════════════════════════════════════════════════════════════════════
x = 0.2
W1 = 2.2
draw_box(ax, x, Y, W1, H, "DNA Sequence", "ACGTACGT...", INPUT_LIGHT, INPUT_GREEN)
draw_shape_label(ax, x + W1/2, Y - 0.38, "[B, seq_len]")
x1_right = x + W1

# ═══════════════════════════════════════════════════════════════════════════
# BOX 2: 5-Channel Encoding
# ═══════════════════════════════════════════════════════════════════════════
x2 = x1_right + gap + 0.15
W2 = 2.7
draw_box(ax, x2, Y, W2, H, "5-Channel Encoding", "Biophysical properties", ENCODING_LIGHT, ENCODING_TEAL)
draw_shape_label(ax, x2 + W2/2, Y - 0.38, "[B, 5, 4096]")
draw_arrow(ax, x1_right + 0.05, Y + H/2, x2 - 0.05, Y + H/2)

# Draw the 5 channel names below
channels = [
    "Ch0: Base Identity",
    "Ch1: Purine/Pyrimidine",
    "Ch2: Amino/Keto",
    "Ch3: H-Bonds",
    "Ch4: GC Content",
]
for i, ch in enumerate(channels):
    ax.text(x2 + W2/2, Y - 0.7 - i * 0.24, ch,
            fontsize=7.0, ha="center", va="center", color=ENCODING_TEAL,
            family="monospace")

x2_right = x2 + W2

# ═══════════════════════════════════════════════════════════════════════════
# BOX 3: CNN Tokenizer (Trainable)
# ═══════════════════════════════════════════════════════════════════════════
x3 = x2_right + gap + 0.15
W3 = 2.6
draw_box(ax, x3, Y, W3, H, "CNNTokenizer", "Conv1D + PosEmbed + LN", TRAINABLE_LIGHT, TRAINABLE_BLUE)
draw_shape_label(ax, x3 + W3/2, Y - 0.38, "[B, 1024, 128]")
draw_arrow(ax, x2_right + 0.05, Y + H/2, x3 - 0.05, Y + H/2)

# Annotation above
ax.text(x3 + W3/2, Y + H + 0.2, "patch_size=4, embed_dim=128",
        fontsize=7.5, ha="center", va="center", color=TRAINABLE_BLUE,
        family="monospace", style="italic")

x3_right = x3 + W3

# ═══════════════════════════════════════════════════════════════════════════
# BOX 4: Perceiver Resampler (Trainable)
# ═══════════════════════════════════════════════════════════════════════════
x4 = x3_right + gap + 0.15
W4 = 2.7
draw_box(ax, x4, Y, W4, H, "Perceiver Resampler", "Cross-attn compression", TRAINABLE_LIGHT, TRAINABLE_BLUE)
draw_shape_label(ax, x4 + W4/2, Y - 0.38, "[B, 64, dim]")
draw_arrow(ax, x3_right + 0.05, Y + H/2, x4 - 0.05, Y + H/2)

# Annotation above
ax.text(x4 + W4/2, Y + H + 0.2, "5×1024 patches → fixed latents",
        fontsize=7.5, ha="center", va="center", color=TRAINABLE_BLUE,
        family="monospace", style="italic")

x4_right = x4 + W4

# ═══════════════════════════════════════════════════════════════════════════
# BOX 5: Gated Cross-Attention (Trainable)
# ═══════════════════════════════════════════════════════════════════════════
x5 = x4_right + gap + 0.15
W5 = 2.6
draw_box(ax, x5, Y, W5, H, "Gated Cross-Attn", "tanh gate, every layer", TRAINABLE_LIGHT, TRAINABLE_BLUE)
draw_arrow(ax, x4_right + 0.05, Y + H/2, x5 - 0.05, Y + H/2)

# Annotation above
ax.text(x5 + W5/2, Y + H + 0.2, "Q: LLM hidden  K,V: Perceiver",
        fontsize=7.5, ha="center", va="center", color=TRAINABLE_BLUE,
        family="monospace", style="italic")

x5_right = x5 + W5

# ═══════════════════════════════════════════════════════════════════════════
# BOX 6: Frozen Qwen LLM (Frozen)
# ═══════════════════════════════════════════════════════════════════════════
x6 = x5_right + gap + 0.15
W6 = 2.5
draw_box(ax, x6, Y, W6, H, "Qwen2.5-0.5B", "Frozen decoder LLM", FROZEN_LIGHT, FROZEN_GRAY)
draw_arrow(ax, x5_right + 0.05, Y + H/2, x6 - 0.05, Y + H/2)

# Show the integration bracket: cross-attn is inserted INTO the LLM
brace_y_top = Y + H + 0.42
brace_y_bot = Y + H + 0.34
ax.plot([x5, x5, x6 + W6, x6 + W6], [brace_y_bot, brace_y_top, brace_y_top, brace_y_bot],
        color=SUBTITLE_COLOR, lw=1.2, zorder=1)
ax.text((x5 + x6 + W6) / 2, brace_y_top + 0.2,
        "Integrated: cross-attn inserted into each LLM layer",
        fontsize=7.5, ha="center", va="center", color=SUBTITLE_COLOR,
        family="sans-serif", style="italic")

x6_right = x6 + W6

# ═══════════════════════════════════════════════════════════════════════════
# BOX 7: Prediction Output
# ═══════════════════════════════════════════════════════════════════════════
x7 = x6_right + gap + 0.15
W7 = 2.2
draw_box(ax, x7, Y, W7, H, "Prediction", "Classification answer", OUTPUT_LIGHT, OUTPUT_PURPLE)
draw_arrow(ax, x6_right + 0.05, Y + H/2, x7 - 0.05, Y + H/2)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND (centered)
# ═══════════════════════════════════════════════════════════════════════════
legend_y = 0.15
legend_x = 3.5
legend_gap = 5.5

# Trainable
legend_box1 = FancyBboxPatch((legend_x, legend_y), 0.5, 0.35,
                              boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor=TRAINABLE_LIGHT, edgecolor=TRAINABLE_BLUE,
                              linewidth=1.5, zorder=2)
ax.add_patch(legend_box1)
ax.text(legend_x + 0.65, legend_y + 0.175, "Trainable",
        fontsize=10, ha="left", va="center", color=TRAINABLE_BLUE, fontweight="bold")

# Frozen
legend_box2 = FancyBboxPatch((legend_x + legend_gap, legend_y), 0.5, 0.35,
                              boxstyle="round,pad=0.05,rounding_size=0.08",
                              facecolor=FROZEN_LIGHT, edgecolor=FROZEN_GRAY,
                              linewidth=1.5, zorder=2)
ax.add_patch(legend_box2)
ax.text(legend_x + legend_gap + 0.65, legend_y + 0.175, "Frozen",
        fontsize=10, ha="left", va="center", color=FROZEN_GRAY, fontweight="bold")

# Tensor shapes
draw_shape_label(ax, legend_x + 2 * legend_gap + 0.3, legend_y + 0.175,
                 "[B, ...]", fontsize=9)
ax.text(legend_x + 2 * legend_gap + 1.15, legend_y + 0.175, "Tensor shape",
        fontsize=10, ha="left", va="center", color=SHAPE_COLOR, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# Training info annotation at bottom
# ═══════════════════════════════════════════════════════════════════════════
info_text = (
    "Training: batch=4  |  lr_encoder=2e-4  |  lr_projector=1e-4  |  "
    "patch_size=4  |  embed_dim=128  |  early_stop=5 epochs"
)
ax.text(mid_x, -1.35, info_text,
        fontsize=8, ha="center", va="center", color=SUBTITLE_COLOR,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9FAFB",
                  edgecolor="#E5E7EB", linewidth=0.8))

plt.tight_layout(pad=0.5)
plt.savefig("/home/wangni/notion-figures/genomics/fig_001.png",
            dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print("Figure saved to /home/wangni/notion-figures/genomics/fig_001.png")
