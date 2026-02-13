import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(figsize=(10, 14), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
C_INPUT    = '#E3F2FD'  # light blue - input
C_CNN      = '#1565C0'  # blue - CNN Tokenizer
C_PERCEIVER= '#2E7D32'  # green - Perceiver Resampler
C_XATTN    = '#E65100'  # orange - Gated Cross-Attention
C_LLM      = '#616161'  # gray - Frozen LLM
C_OUTPUT   = '#4A148C'  # purple - output
C_TEXT     = 'white'
C_DIM      = '#37474F'  # dark gray for dimension labels
C_ARROW    = '#455A64'

def draw_block(ax, cx, cy, w, h, color, label, fontsize=12, text_color='white',
               sublabel=None, sublabel_size=9):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.08", linewidth=1.2,
                         edgecolor='white', facecolor=color, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + 0.12, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
        ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                fontsize=sublabel_size, color=text_color, zorder=4, style='italic')
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)

def draw_arrow(ax, x, y_start, y_end, color=C_ARROW):
    ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                connectionstyle='arc3,rad=0'))

def draw_dim_label(ax, cx, cy, text):
    ax.text(cx, cy, text, ha='center', va='center', fontsize=9.5,
            fontfamily='monospace', color=C_DIM, zorder=4,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FAFAFA',
                      edgecolor='#BDBDBD', linewidth=0.8))

# ── Title ──
ax.text(5, 13.5, 'OpenTSLM-Polymarkets Model Architecture',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color='#212121')
ax.text(5, 13.1, 'Flamingo-style Time Series Language Model',
        ha='center', va='center', fontsize=11, color='#757575')

# ── 1. Input: Raw Time Series ──
y_input = 12.2
draw_block(ax, 5, y_input, 5.2, 0.7, '#1E88E5', 'Raw Time Series Input',
           fontsize=13, sublabel='Market probability history (720 points)')
draw_dim_label(ax, 8.5, y_input, '[B, 720]')

# Arrow
draw_arrow(ax, 5, y_input - 0.35, y_input - 0.85)

# ── 2. CNN Tokenizer ──
y_cnn = 10.8
box_w, box_h = 5.8, 1.0
box = FancyBboxPatch((5 - box_w/2, y_cnn - box_h/2), box_w, box_h,
                     boxstyle="round,pad=0.1", linewidth=2,
                     edgecolor='#0D47A1', facecolor=C_CNN, zorder=3)
ax.add_patch(box)
ax.text(5, y_cnn + 0.2, 'CNNTokenizer', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white', zorder=4)
ax.text(5, y_cnn - 0.15, 'Conv1D  |  kernel_size = 4  |  stride = 4',
        ha='center', va='center', fontsize=10, color='#BBDEFB', zorder=4,
        fontfamily='monospace')

# Dim label after CNN
draw_arrow(ax, 5, y_cnn - 0.5, y_cnn - 1.0)
y_dim1 = y_cnn - 1.25
draw_dim_label(ax, 5, y_dim1, '180 patch embeddings  [B, 180, 128]')
ax.text(8.2, y_dim1, 'PATCH_SIZE=4\nEMBED_DIM=128',
        ha='center', va='center', fontsize=8, color='#78909C',
        fontfamily='monospace', linespacing=1.4)

# Arrow
draw_arrow(ax, 5, y_dim1 - 0.3, y_dim1 - 0.8)

# ── 3. Perceiver Resampler ──
y_per = 8.2
box_w, box_h = 5.8, 1.0
box = FancyBboxPatch((5 - box_w/2, y_per - box_h/2), box_w, box_h,
                     boxstyle="round,pad=0.1", linewidth=2,
                     edgecolor='#1B5E20', facecolor=C_PERCEIVER, zorder=3)
ax.add_patch(box)
ax.text(5, y_per + 0.2, 'Perceiver Resampler', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white', zorder=4)
ax.text(5, y_per - 0.15, 'Compress 180 patches \u2192 64 fixed-length tokens',
        ha='center', va='center', fontsize=10, color='#C8E6C9', zorder=4)

# Dim label after Perceiver
draw_arrow(ax, 5, y_per - 0.5, y_per - 1.0)
y_dim2 = y_per - 1.25
draw_dim_label(ax, 5, y_dim2, '64 latent tokens  [B, 64, 4096]')

# Arrow
draw_arrow(ax, 5, y_dim2 - 0.3, y_dim2 - 0.8)

# ── 4. Interleaved Decoder Block ──
y_block = 5.4
block_w, block_h = 7.6, 2.6
# Outer container
outer = FancyBboxPatch((5 - block_w/2, y_block - block_h/2), block_w, block_h,
                       boxstyle="round,pad=0.12", linewidth=1.5,
                       edgecolor='#BDBDBD', facecolor='#FAFAFA', zorder=2)
ax.add_patch(outer)
ax.text(5, y_block + block_h/2 - 0.22, 'Decoder Block (repeated at every layer)',
        ha='center', va='center', fontsize=10, fontweight='bold',
        color='#424242', zorder=4)

# Gated Cross-Attention sub-block
y_xattn = y_block + 0.35
xattn_w, xattn_h = 5.2, 0.8
box_xa = FancyBboxPatch((5 - xattn_w/2, y_xattn - xattn_h/2), xattn_w, xattn_h,
                        boxstyle="round,pad=0.08", linewidth=1.5,
                        edgecolor='#BF360C', facecolor=C_XATTN, zorder=3)
ax.add_patch(box_xa)
ax.text(5, y_xattn + 0.1, 'Gated Cross-Attention', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white', zorder=4)
ax.text(5, y_xattn - 0.2, 'Gates initialized near zero for stability',
        ha='center', va='center', fontsize=8.5, color='#FFCCBC', zorder=4)

# "Keys/Values from time series" annotation on left
ax.annotate('Keys & Values\nfrom time series\ntokens', xy=(5 - xattn_w/2, y_xattn),
            xytext=(0.6, y_xattn + 0.3), fontsize=8, color=C_XATTN,
            fontweight='bold', ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=C_XATTN, lw=1.3,
                            connectionstyle='arc3,rad=0.2'))

# Small arrow between sub-blocks
draw_arrow(ax, 5, y_xattn - 0.4, y_xattn - 0.72)

# Frozen Qwen Self-Attention sub-block
y_llm = y_block - 0.55
llm_w, llm_h = 5.2, 0.8
box_llm = FancyBboxPatch((5 - llm_w/2, y_llm - llm_h/2), llm_w, llm_h,
                         boxstyle="round,pad=0.08", linewidth=1.5,
                         edgecolor='#424242', facecolor=C_LLM, zorder=3)
ax.add_patch(box_llm)
ax.text(5, y_llm + 0.1, 'Frozen Qwen 2.5-7B Self-Attention', ha='center',
        va='center', fontsize=12, fontweight='bold', color='white', zorder=4)
ax.text(5, y_llm - 0.2, '7B parameters (frozen)', ha='center', va='center',
        fontsize=8.5, color='#E0E0E0', zorder=4)

# Frozen indicator
ax.text(8.6, y_llm, 'FROZEN', ha='center', va='center', fontsize=8,
        fontweight='bold', color=C_LLM, zorder=4,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#EEEEEE',
                  edgecolor='#9E9E9E', linewidth=1))

# Arrow out of decoder block
draw_arrow(ax, 5, y_block - block_h/2, y_block - block_h/2 - 0.6)

# ── 5. Output ──
y_out = 3.2
box_w_out, box_h_out = 5.2, 0.7
box_out = FancyBboxPatch((5 - box_w_out/2, y_out - box_h_out/2), box_w_out, box_h_out,
                         boxstyle="round,pad=0.08", linewidth=2,
                         edgecolor='#311B92', facecolor=C_OUTPUT, zorder=3)
ax.add_patch(box_out)
ax.text(5, y_out, 'Autoregressive Answer Token Generation',
        ha='center', va='center', fontsize=12, fontweight='bold',
        color='white', zorder=4)

# ── Legend ──
y_legend = 1.8
ax.text(5, y_legend + 0.7, 'Trainable Components', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#424242')

legend_items = [
    (C_CNN,       'CNN Tokenizer'),
    (C_PERCEIVER, 'Perceiver Resampler'),
    (C_XATTN,     'Gated Cross-Attention'),
    (C_LLM,       'Frozen Qwen 2.5-7B (not trained)'),
]

total_w = 8
start_x = 5 - total_w / 2
item_w = total_w / len(legend_items)
for i, (color, label) in enumerate(legend_items):
    cx = start_x + i * item_w + item_w / 2
    rect = FancyBboxPatch((cx - 0.35, y_legend - 0.12), 0.7, 0.24,
                          boxstyle="round,pad=0.04", facecolor=color,
                          edgecolor='white', linewidth=0.8, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, y_legend - 0.35, label, ha='center', va='top',
            fontsize=7.5, color='#424242')

# Trainable params note
ax.text(5, 1.0, '~3.5B trainable parameters  |  Qwen 2.5-7B frozen  |  14\u00d7 more token-efficient than text-based encoding',
        ha='center', va='center', fontsize=8, color='#9E9E9E', style='italic')

# ── Special tokens note ──
ax.text(5, 0.55, 'Special tokens:  <image> activates cross-attention  |  <|endofchunk|> ends time series section',
        ha='center', va='center', fontsize=7.5, color='#BDBDBD',
        fontfamily='monospace')

plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/polymarkets/fig_001.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Figure saved to /home/wangni/notion-figures/polymarkets/fig_001.png")
