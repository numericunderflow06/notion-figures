#!/usr/bin/env python3
"""
Fig 003: Perceiver Resampler Detail
Shows how variable-length CNN output patches are compressed into 64 fixed
latent tokens via 6 cross-attention layers.

Architecture specs (Section 7.1, Section 12):
- 6 Perceiver layers
- 64 learnable latent tokens
- 8 attention heads
- 64-dim per head (total dim = 512)
- Input: variable N patches from CNN encoder (dim 128)
- Output: 64 fixed tokens for gated cross-attention
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────
C_BG        = '#FFFFFF'
C_INPUT     = '#4A90D9'   # Blue for input patches
C_LATENT    = '#E07B39'   # Orange for latent tokens
C_LAYER     = '#F5F0E8'   # Warm cream for layer backgrounds
C_LAYER_BD  = '#B0A894'   # Layer border
C_CROSS     = '#D94F4F'   # Red for cross-attention arrows
C_SELF      = '#5BA85B'   # Green for self-attention
C_FFN       = '#7B68AE'   # Purple for FFN
C_TEXT      = '#2C2C2C'
C_DIM_TEXT  = '#555555'
C_ARROW     = '#888888'
C_LIGHT_BG  = '#F9F7F3'

fig, ax = plt.subplots(1, 1, figsize=(11, 15), facecolor=C_BG)
ax.set_xlim(0, 11)
ax.set_ylim(0, 15)
ax.set_aspect('equal')
ax.axis('off')

# ── Helper functions ───────────────────────────────────────────────────
def draw_rounded_box(ax, xy, w, h, color, edgecolor='none', lw=1.0,
                     alpha=1.0, radius=0.15, zorder=2):
    box = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_token_row(ax, x_start, y_center, n_tokens, token_w, token_h,
                   color, alpha=1.0, label=None, zorder=3):
    """Draw a row of small rectangles representing tokens."""
    xs = []
    for i in range(n_tokens):
        x = x_start + i * (token_w + 0.02)
        rect = FancyBboxPatch((x, y_center - token_h/2), token_w, token_h,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='white',
                              linewidth=0.5, alpha=alpha, zorder=zorder)
        ax.add_patch(rect)
        xs.append(x + token_w/2)
    return xs


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
ax.text(5.5, 14.55, 'Perceiver Resampler', fontsize=18, fontweight='bold',
        ha='center', va='center', color=C_TEXT)
ax.text(5.5, 14.15, 'Compresses variable-length CNN patches → 64 fixed latent tokens',
        fontsize=11, ha='center', va='center', color=C_DIM_TEXT)

# ══════════════════════════════════════════════════════════════════════
# INPUT SECTION (bottom)
# ══════════════════════════════════════════════════════════════════════
input_y = 0.8

# Input label
ax.text(5.5, 0.35, 'CNN Encoder Output  (5 channels × variable patches)',
        fontsize=10, ha='center', va='center', color=C_DIM_TEXT, style='italic')

# Draw variable-length input patches
input_box_x = 1.0
input_box_w = 9.0
input_box_h = 1.2
draw_rounded_box(ax, (input_box_x, input_y), input_box_w, input_box_h,
                 color='#E8F0FA', edgecolor=C_INPUT, lw=1.5, radius=0.12)

# Show N variable-length patches as small blocks
n_show = 28
pw = 0.22
for i in range(n_show):
    x = 1.25 + i * (pw + 0.06)
    alpha = 1.0 if i < 25 else 0.4  # fade last few to show variability
    rect = FancyBboxPatch((x, input_y + 0.25), pw, 0.7,
                          boxstyle="round,pad=0.02",
                          facecolor=C_INPUT,
                          edgecolor='white', linewidth=0.4,
                          alpha=alpha, zorder=3)
    ax.add_patch(rect)

# Ellipsis for variable length
ax.text(9.3, input_y + 0.6, '···', fontsize=16, ha='center', va='center',
        color=C_INPUT, fontweight='bold', zorder=4)

# Dimension annotations on input
ax.text(1.0, input_y + 1.35, 'N patches', fontsize=9.5,
        ha='left', va='bottom', color=C_INPUT, fontweight='bold')
ax.text(10.0, input_y + 1.35, 'dim = 128', fontsize=9.5,
        ha='right', va='bottom', color=C_INPUT, fontweight='bold')
ax.text(5.5, input_y + 1.35, '(N = L / patch_size, variable per sequence)',
        fontsize=8.5, ha='center', va='bottom', color=C_DIM_TEXT)

# ══════════════════════════════════════════════════════════════════════
# LEARNABLE LATENT TOKENS (initialized, shown on right side)
# ══════════════════════════════════════════════════════════════════════
latent_init_y = 2.65
ax.text(9.4, latent_init_y + 0.55, 'Learnable\nLatent Queries',
        fontsize=8.5, ha='center', va='center', color=C_LATENT,
        fontweight='bold', style='italic')
ax.text(9.4, latent_init_y + 0.05, '64 tokens, dim 512',
        fontsize=8, ha='center', va='center', color=C_DIM_TEXT)

# Arrow from latent init into first layer
ax.annotate('', xy=(8.2, 3.25), xytext=(8.8, latent_init_y + 0.25),
            arrowprops=dict(arrowstyle='->', color=C_LATENT, lw=1.5))

# ══════════════════════════════════════════════════════════════════════
# PERCEIVER LAYERS (6 layers stacked vertically)
# ══════════════════════════════════════════════════════════════════════
n_layers = 6
layer_h = 1.25
layer_gap = 0.22
layer_x = 1.4
layer_w = 7.2
first_layer_y = 2.9

for i in range(n_layers):
    y = first_layer_y + i * (layer_h + layer_gap)
    layer_num = i + 1

    # Layer background
    draw_rounded_box(ax, (layer_x, y), layer_w, layer_h,
                     color=C_LAYER, edgecolor=C_LAYER_BD, lw=1.2, radius=0.1)

    # Layer label (left side)
    ax.text(layer_x + 0.25, y + layer_h/2, f'Layer {layer_num}',
            fontsize=9, fontweight='bold', ha='left', va='center',
            color=C_TEXT)

    # ── Cross-Attention block ──
    ca_x = layer_x + 1.4
    ca_w = 2.1
    ca_h = 0.75
    ca_y = y + (layer_h - ca_h) / 2
    draw_rounded_box(ax, (ca_x, ca_y), ca_w, ca_h,
                     color='#FDEAEA', edgecolor=C_CROSS, lw=1.0, radius=0.08)
    ax.text(ca_x + ca_w/2, ca_y + ca_h/2 + 0.1, 'Cross-Attention',
            fontsize=8, ha='center', va='center', color=C_CROSS,
            fontweight='bold')
    ax.text(ca_x + ca_w/2, ca_y + ca_h/2 - 0.15, 'Q: latents  K,V: inputs',
            fontsize=7, ha='center', va='center', color='#AA4444')

    # ── Self-Attention block ──
    sa_x = ca_x + ca_w + 0.25
    sa_w = 1.6
    sa_h = 0.75
    sa_y = y + (layer_h - sa_h) / 2
    draw_rounded_box(ax, (sa_x, sa_y), sa_w, sa_h,
                     color='#E8F5E8', edgecolor=C_SELF, lw=1.0, radius=0.08)
    ax.text(sa_x + sa_w/2, sa_y + sa_h/2 + 0.1, 'Self-Attention',
            fontsize=8, ha='center', va='center', color=C_SELF,
            fontweight='bold')
    ax.text(sa_x + sa_w/2, sa_y + sa_h/2 - 0.15, '8 heads × 64 dim',
            fontsize=7, ha='center', va='center', color='#3A7A3A')

    # ── FFN block ──
    ff_x = sa_x + sa_w + 0.25
    ff_w = 1.1
    ff_h = 0.75
    ff_y = y + (layer_h - ff_h) / 2
    draw_rounded_box(ax, (ff_x, ff_y), ff_w, ff_h,
                     color='#EEEBF5', edgecolor=C_FFN, lw=1.0, radius=0.08)
    ax.text(ff_x + ff_w/2, ff_y + ff_h/2 + 0.1, 'FFN',
            fontsize=8, ha='center', va='center', color=C_FFN,
            fontweight='bold')
    ax.text(ff_x + ff_w/2, ff_y + ff_h/2 - 0.15, '+ LayerNorm',
            fontsize=7, ha='center', va='center', color='#6A5A8A')

    # ── Internal arrows within layer ──
    # Cross-Attn → Self-Attn
    ax.annotate('', xy=(sa_x, sa_y + sa_h/2),
                xytext=(ca_x + ca_w, ca_y + ca_h/2),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.0))
    # Self-Attn → FFN
    ax.annotate('', xy=(ff_x, ff_y + ff_h/2),
                xytext=(sa_x + sa_w, sa_y + sa_h/2),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.0))

    # ── Arrow between layers (except top) ──
    if i < n_layers - 1:
        next_y = first_layer_y + (i + 1) * (layer_h + layer_gap)
        ax.annotate('', xy=(5.0, next_y), xytext=(5.0, y + layer_h),
                    arrowprops=dict(arrowstyle='->', color=C_LATENT,
                                   lw=1.5, linestyle='-'))

# ══════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION FEED ARROWS (from input to each layer)
# ══════════════════════════════════════════════════════════════════════
# Draw a vertical line on the left representing input broadcast
input_top_y = input_y + input_box_h
feed_x = 0.85

# Vertical line from input box up alongside layers
ax.plot([feed_x, feed_x], [input_top_y + 0.1, first_layer_y + n_layers * (layer_h + layer_gap) - layer_gap + 0.1],
        color=C_INPUT, lw=2.0, linestyle='-', alpha=0.7, zorder=1)

# Small label
ax.text(0.35, first_layer_y + n_layers * (layer_h + layer_gap) / 2,
        'Input\nbroadcast\n(K, V)',
        fontsize=7.5, ha='center', va='center', color=C_INPUT,
        fontweight='bold', rotation=90)

# Horizontal arrows from vertical line into each layer's cross-attention
for i in range(n_layers):
    y = first_layer_y + i * (layer_h + layer_gap) + layer_h / 2
    ax.annotate('', xy=(layer_x, y), xytext=(feed_x, y),
                arrowprops=dict(arrowstyle='->', color=C_INPUT, lw=1.2))

# Arrow from input box up to the vertical line
ax.annotate('', xy=(feed_x, input_top_y + 0.1),
            xytext=(1.5, input_top_y),
            arrowprops=dict(arrowstyle='->', color=C_INPUT, lw=1.5))

# ══════════════════════════════════════════════════════════════════════
# OUTPUT SECTION (top)
# ══════════════════════════════════════════════════════════════════════
output_y = first_layer_y + n_layers * (layer_h + layer_gap) + 0.15
output_box_x = 2.0
output_box_w = 7.0
output_box_h = 0.9

# Arrow from last layer to output
top_layer_y = first_layer_y + (n_layers - 1) * (layer_h + layer_gap) + layer_h
ax.annotate('', xy=(5.5, output_y), xytext=(5.5, top_layer_y),
            arrowprops=dict(arrowstyle='->', color=C_LATENT, lw=1.8))

# Output box
draw_rounded_box(ax, (output_box_x, output_y), output_box_w, output_box_h,
                 color='#FFF3E8', edgecolor=C_LATENT, lw=1.5, radius=0.12)

# Draw 64 output tokens as small blocks
n_out = 32  # show 32, represent 64 with label
ow = 0.16
for i in range(n_out):
    x = 2.25 + i * (ow + 0.03)
    rect = FancyBboxPatch((x, output_y + 0.18), ow, 0.5,
                          boxstyle="round,pad=0.02",
                          facecolor=C_LATENT,
                          edgecolor='white', linewidth=0.4,
                          alpha=0.9, zorder=3)
    ax.add_patch(rect)

# Output dimension annotations
ax.text(output_box_x + output_box_w / 2, output_y + output_box_h + 0.15,
        '64 fixed latent tokens  ×  dim 512  (8 heads × 64 dim/head)',
        fontsize=10, ha='center', va='bottom', color=C_LATENT, fontweight='bold')

# Arrow to next stage
ax.text(5.5, output_y + output_box_h + 0.6,
        '↓  To Gated Cross-Attention in frozen LLM',
        fontsize=10, ha='center', va='bottom', color=C_DIM_TEXT,
        fontweight='bold')

# ══════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════
legend_x = 0.3
legend_y = 13.45

# Background for legend
draw_rounded_box(ax, (legend_x, legend_y), 7.6, 0.55,
                 color='#FAFAFA', edgecolor='#CCCCCC', lw=0.8, radius=0.08)

items = [
    (C_CROSS, 'Cross-Attention'),
    (C_SELF, 'Self-Attention'),
    (C_FFN, 'FFN + LayerNorm'),
    (C_INPUT, 'Input (K,V)'),
    (C_LATENT, 'Latent (Q)'),
]
for j, (color, label) in enumerate(items):
    lx = legend_x + 0.2 + j * 1.5
    rect = FancyBboxPatch((lx, legend_y + 0.15), 0.2, 0.25,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='none',
                          linewidth=0, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.28, legend_y + 0.275, label, fontsize=7.5,
            ha='left', va='center', color=C_TEXT)

# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/genomics/fig_003.png',
            dpi=200, bbox_inches='tight', facecolor=C_BG, edgecolor='none')
plt.close()
print("✓ Saved fig_003.png")
