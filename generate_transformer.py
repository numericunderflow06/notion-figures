"""Generate a Transformer architecture diagram (Vaswani et al., 2017)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 22)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
C_EMBED   = '#4FC3F7'  # light blue
C_POS     = '#81D4FA'  # lighter blue
C_ATTN    = '#FF8A65'  # orange
C_MATTN   = '#EF5350'  # red-orange
C_CATTN   = '#FFA726'  # amber
C_FFN     = '#66BB6A'  # green
C_NORM    = '#CE93D8'  # purple
C_LINEAR  = '#90A4AE'  # grey-blue
C_SOFTMAX = '#78909C'  # darker grey-blue
C_ENC_BG  = '#E3F2FD'  # encoder background
C_DEC_BG  = '#FFF3E0'  # decoder background
C_TEXT    = '#212121'

def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='#424242', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=C_TEXT, fontweight=weight, wrap=True)

def draw_arrow(x1, y1, x2, y2, color='#424242', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_residual(x_start, y_start, x_end, y_end, side='right', box_w=2.8):
    """Draw a residual (skip) connection arc on the given side."""
    if side == 'right':
        offset = box_w / 2 + 0.5
        mid_x = x_start + offset
    else:
        offset = box_w / 2 + 0.5
        mid_x = x_start - offset
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=1.2,
                                connectionstyle=f'arc3,rad={"0.4" if side == "right" else "-0.4"}',
                                linestyle='dashed'))

# ── Layout constants ──
ENC_X = 2.6    # encoder center x
DEC_X = 9.0    # decoder center x
BOX_W = 2.8
BOX_H = 0.8
SMALL_H = 0.6

# ══════════════════════════════════════
# ENCODER SIDE
# ══════════════════════════════════════

# Encoder background
enc_bg = FancyBboxPatch((0.8, 5.2), 4.8, 10.2, boxstyle="round,pad=0.3",
                         facecolor=C_ENC_BG, edgecolor='#90CAF9', linewidth=2, linestyle='--')
ax.add_patch(enc_bg)
ax.text(3.2, 15.15, 'ENCODER  (x N)', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#1565C0')

# Input
draw_box(ENC_X - BOX_W/2, 1.2, BOX_W, SMALL_H, 'Inputs', '#E0E0E0', fontsize=10)
# Input Embedding
draw_box(ENC_X - BOX_W/2, 2.4, BOX_W, BOX_H, 'Input\nEmbedding', C_EMBED, fontsize=10)
# Positional Encoding
draw_box(ENC_X - BOX_W/2, 3.8, BOX_W, BOX_H, '+ Positional\nEncoding', C_POS, fontsize=10)

# Arrows: Input → Embed → PosEnc
draw_arrow(ENC_X, 1.8, ENC_X, 2.4)
draw_arrow(ENC_X, 3.2, ENC_X, 3.8)

# ── Encoder layers (inside background) ──
# Multi-Head Self-Attention
y_mha = 5.8
draw_box(ENC_X - BOX_W/2, y_mha, BOX_W, BOX_H, 'Multi-Head\nSelf-Attention', C_ATTN, fontsize=10, bold=True)
draw_arrow(ENC_X, 4.6, ENC_X, y_mha)

# Add & Norm 1
y_an1 = 7.2
draw_box(ENC_X - BOX_W/2, y_an1, BOX_W, BOX_H, 'Add & Layer Norm', C_NORM, fontsize=10)
draw_arrow(ENC_X, y_mha + BOX_H, ENC_X, y_an1)
draw_residual(ENC_X, y_mha, ENC_X, y_an1 + BOX_H/2, side='right')

# Feed-Forward Network
y_ffn = 8.6
draw_box(ENC_X - BOX_W/2, y_ffn, BOX_W, BOX_H, 'Feed-Forward\nNetwork', C_FFN, fontsize=10, bold=True)
draw_arrow(ENC_X, y_an1 + BOX_H, ENC_X, y_ffn)

# Add & Norm 2
y_an2 = 10.0
draw_box(ENC_X - BOX_W/2, y_an2, BOX_W, BOX_H, 'Add & Layer Norm', C_NORM, fontsize=10)
draw_arrow(ENC_X, y_ffn + BOX_H, ENC_X, y_an2)
draw_residual(ENC_X, y_ffn, ENC_X, y_an2 + BOX_H/2, side='right')

# Encoder output label
y_enc_out = 11.3
draw_box(ENC_X - BOX_W/2, y_enc_out, BOX_W, SMALL_H, 'Encoder Output', '#BBDEFB', fontsize=10, bold=True)
draw_arrow(ENC_X, y_an2 + BOX_H, ENC_X, y_enc_out)

# ══════════════════════════════════════
# DECODER SIDE
# ══════════════════════════════════════

# Decoder background
dec_bg = FancyBboxPatch((7.0, 5.2), 4.8, 13.0, boxstyle="round,pad=0.3",
                         facecolor=C_DEC_BG, edgecolor='#FFCC80', linewidth=2, linestyle='--')
ax.add_patch(dec_bg)
ax.text(9.4, 17.95, 'DECODER  (x N)', ha='center', va='center',
        fontsize=13, fontweight='bold', color='#E65100')

# Output (shifted right)
draw_box(DEC_X - BOX_W/2, 1.2, BOX_W, SMALL_H, 'Outputs (shifted right)', '#E0E0E0', fontsize=9)
# Output Embedding
draw_box(DEC_X - BOX_W/2, 2.4, BOX_W, BOX_H, 'Output\nEmbedding', C_EMBED, fontsize=10)
# Positional Encoding
draw_box(DEC_X - BOX_W/2, 3.8, BOX_W, BOX_H, '+ Positional\nEncoding', C_POS, fontsize=10)

draw_arrow(DEC_X, 1.8, DEC_X, 2.4)
draw_arrow(DEC_X, 3.2, DEC_X, 3.8)

# ── Decoder layers ──
# Masked Multi-Head Self-Attention
y_mmha = 5.8
draw_box(DEC_X - BOX_W/2, y_mmha, BOX_W, BOX_H, 'Masked Multi-Head\nSelf-Attention', C_MATTN, fontsize=9, bold=True)
draw_arrow(DEC_X, 4.6, DEC_X, y_mmha)

# Add & Norm 1
y_dan1 = 7.2
draw_box(DEC_X - BOX_W/2, y_dan1, BOX_W, BOX_H, 'Add & Layer Norm', C_NORM, fontsize=10)
draw_arrow(DEC_X, y_mmha + BOX_H, DEC_X, y_dan1)
draw_residual(DEC_X, y_mmha, DEC_X, y_dan1 + BOX_H/2, side='left')

# Multi-Head Cross-Attention
y_cattn = 8.6
draw_box(DEC_X - BOX_W/2, y_cattn, BOX_W, BOX_H, 'Multi-Head\nCross-Attention', C_CATTN, fontsize=10, bold=True)
draw_arrow(DEC_X, y_dan1 + BOX_H, DEC_X, y_cattn)

# Arrow from encoder output → cross-attention (K, V)
draw_arrow(ENC_X + BOX_W/2 + 0.1, y_enc_out + SMALL_H/2, DEC_X - BOX_W/2 - 0.1, y_cattn + BOX_H/2,
           color='#1565C0', lw=2.0)
ax.text(5.8, 10.3, 'K, V', fontsize=10, fontweight='bold', color='#1565C0',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#1565C0', alpha=0.9))

# Add & Norm 2
y_dan2 = 10.0
draw_box(DEC_X - BOX_W/2, y_dan2, BOX_W, BOX_H, 'Add & Layer Norm', C_NORM, fontsize=10)
draw_arrow(DEC_X, y_cattn + BOX_H, DEC_X, y_dan2)
draw_residual(DEC_X, y_cattn, DEC_X, y_dan2 + BOX_H/2, side='left')

# Feed-Forward Network
y_dffn = 11.4
draw_box(DEC_X - BOX_W/2, y_dffn, BOX_W, BOX_H, 'Feed-Forward\nNetwork', C_FFN, fontsize=10, bold=True)
draw_arrow(DEC_X, y_dan2 + BOX_H, DEC_X, y_dffn)

# Add & Norm 3
y_dan3 = 12.8
draw_box(DEC_X - BOX_W/2, y_dan3, BOX_W, BOX_H, 'Add & Layer Norm', C_NORM, fontsize=10)
draw_arrow(DEC_X, y_dffn + BOX_H, DEC_X, y_dan3)
draw_residual(DEC_X, y_dffn, DEC_X, y_dan3 + BOX_H/2, side='left')

# ══════════════════════════════════════
# OUTPUT HEAD
# ══════════════════════════════════════

y_lin = 14.5
draw_box(DEC_X - BOX_W/2, y_lin, BOX_W, BOX_H, 'Linear', C_LINEAR, fontsize=11, bold=True)
draw_arrow(DEC_X, y_dan3 + BOX_H, DEC_X, y_lin)

y_sm = 15.8
draw_box(DEC_X - BOX_W/2, y_sm, BOX_W, BOX_H, 'Softmax', C_SOFTMAX, fontsize=11, bold=True)
draw_arrow(DEC_X, y_lin + BOX_H, DEC_X, y_sm)

y_out = 17.1
draw_box(DEC_X - BOX_W/2, y_out, BOX_W, SMALL_H, 'Output Probabilities', '#E0E0E0', fontsize=10, bold=True)
draw_arrow(DEC_X, y_sm + BOX_H, DEC_X, y_out)

# ══════════════════════════════════════
# LEGEND
# ══════════════════════════════════════

ax.text(7.0, 21.2, 'Transformer Architecture', ha='center', va='center',
        fontsize=20, fontweight='bold', color=C_TEXT)
ax.text(7.0, 20.6, 'Vaswani et al., "Attention Is All You Need" (2017)',
        ha='center', va='center', fontsize=11, color='#616161', style='italic')

# Legend boxes
legend_y = 19.6
legend_items = [
    (C_ATTN,  'Attention'),
    (C_FFN,   'Feed-Forward'),
    (C_NORM,  'Norm'),
    (C_EMBED, 'Embedding'),
]
for i, (color, label) in enumerate(legend_items):
    lx = 2.5 + i * 2.8
    box = FancyBboxPatch((lx, legend_y), 0.5, 0.35, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='#424242', linewidth=1)
    ax.add_patch(box)
    ax.text(lx + 0.65, legend_y + 0.18, label, fontsize=9, va='center', color=C_TEXT)

# Dashed line legend
ax.plot([2.5, 3.0], [legend_y - 0.4, legend_y - 0.4], '--', color='#9E9E9E', lw=1.5)
ax.text(3.15, legend_y - 0.4, 'Residual connection', fontsize=9, va='center', color=C_TEXT)

plt.tight_layout()
plt.savefig('/home/wangni/notion-figures/transformer/transformer_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved transformer_architecture.png')
