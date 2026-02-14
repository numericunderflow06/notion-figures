"""
fig_005: Multivariate Time Series Processing Pipeline
Shows how 12-lead ECG flows through shared patch encoder → C×N embeddings → Perceiver pooling (K=64).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------- colour palette ----------
LEAD_COLORS = [
    '#E63946',  '#F4845F',  '#F7B267',  '#A8DADC',
    '#457B9D',  '#1D3557',  '#2A9D8F',  '#264653',
    '#E9C46A',  '#F4A261',  '#E76F51',  '#6A0572',
]
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

BG_WHITE = '#FFFFFF'
BOX_ENCODER = '#D6EAF8'
BOX_PERCEIVER = '#FAD7A0'
BOX_MLP = '#D5F5E3'
BOX_LLM = '#F5EEF8'
BORDER_DARK = '#2C3E50'
ARROW_COLOR = '#555555'
DIM_COLOR = '#777777'

fig, ax = plt.subplots(figsize=(18, 12.5))
ax.set_xlim(0, 18)
ax.set_ylim(1.5, 15.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_WHITE)

# ── Helpers ──────────────────────────────────────────
def rounded_box(x, y, w, h, color, ec=BORDER_DARK, lw=1.2, zorder=3):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                         facecolor=color, edgecolor=ec,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)

def arrow_line(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0'),
                zorder=5)

def bracket(x, y_top, y_bot, text, side='left'):
    sign = -1 if side == 'left' else 1
    bx = x + sign * 0.18
    mid = (y_top + y_bot) / 2
    ax.plot([bx, bx], [y_top, y_bot], color=DIM_COLOR, lw=1.0, zorder=4)
    ax.plot([x, bx], [y_top, y_top], color=DIM_COLOR, lw=1.0, zorder=4)
    ax.plot([x, bx], [y_bot, y_bot], color=DIM_COLOR, lw=1.0, zorder=4)
    offset = sign * 0.45
    ax.text(bx + offset, mid, text, ha='center', va='center',
            fontsize=9.5, color=DIM_COLOR, fontstyle='italic', zorder=6,
            rotation=90 if side == 'left' else 270)


# ═══════════════════════════════════════════════════════
# TITLE (plenty of room at top)
# ═══════════════════════════════════════════════════════
ax.text(9, 15.0, 'Multivariate Time Series Processing Pipeline',
        ha='center', va='center', fontsize=17, fontweight='bold',
        color='#1B2631', zorder=10)
ax.text(9, 14.55, '12-lead ECG example  ·  Shared Encoder  ·  Shared Perceiver Pooling (K = 64)',
        ha='center', va='center', fontsize=11, color='#555555', zorder=10)

# ═══════════════════════════════════════════════════════
# COLUMN HEADERS (below subtitle, above channel boxes)
# ═══════════════════════════════════════════════════════
col1_cx = 2.0        # center of input column
col2_cx = 6.4        # center of encoder column

header_y = 13.85
subheader_y = 13.55

ax.text(col1_cx, header_y, 'Input: 12-Lead ECG',
        ha='center', va='center', fontsize=11.5, fontweight='bold',
        color='#1B2631', zorder=6)
ax.text(col1_cx, subheader_y, 'L = 5000 samples (10 s × 500 Hz)',
        ha='center', va='center', fontsize=8.5, color=DIM_COLOR,
        fontstyle='italic', zorder=6)

ax.text(col2_cx, header_y, 'Shared Patch Encoder',
        ha='center', va='center', fontsize=11.5, fontweight='bold',
        color='#2980B9', zorder=6)
ax.text(col2_cx, subheader_y, 'Conv1D (k=p, s=p) + 4-layer Transformer',
        ha='center', va='center', fontsize=8.5, color=DIM_COLOR,
        fontstyle='italic', zorder=6)

# ═══════════════════════════════════════════════════════
# 12 INPUT CHANNELS
# ═══════════════════════════════════════════════════════
chan_x = 0.8
chan_w = 2.4
chan_h = 0.60
chan_gap = 0.15
y_top = 12.15

chan_ys = [y_top - i * (chan_h + chan_gap) for i in range(12)]

np.random.seed(42)
for i, (cy, color, name) in enumerate(zip(chan_ys, LEAD_COLORS, LEAD_NAMES)):
    box = FancyBboxPatch((chan_x, cy), chan_w, chan_h,
                         boxstyle='round,pad=0.08',
                         facecolor=color, edgecolor=BORDER_DARK,
                         linewidth=0.8, alpha=0.22, zorder=2)
    ax.add_patch(box)
    ax.text(chan_x + 0.32, cy + chan_h / 2, name, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=color, zorder=4)
    # mini waveform
    t = np.linspace(0, 2 * np.pi, 60)
    wave = 0.14 * np.sin(t * (2 + 0.3 * i)) + 0.05 * np.random.randn(60)
    wx = np.linspace(chan_x + 0.60, chan_x + chan_w - 0.10, 60)
    wy = cy + chan_h / 2 + wave
    ax.plot(wx, wy, color=color, lw=1.0, zorder=3)

# Left bracket
bracket(chan_x - 0.08, chan_ys[0] + chan_h, chan_ys[-1],
        'C = 12\nchannels', side='left')

# ═══════════════════════════════════════════════════════
# SHARED PATCH ENCODER (12 instances, shared weights)
# ═══════════════════════════════════════════════════════
enc_x = 5.0
enc_w = 2.8
enc_h = 0.60
enc_ys = chan_ys

for i, (ey, color) in enumerate(zip(enc_ys, LEAD_COLORS)):
    box = FancyBboxPatch((enc_x, ey), enc_w, enc_h,
                         boxstyle='round,pad=0.08',
                         facecolor=BOX_ENCODER, edgecolor=color,
                         linewidth=1.0, alpha=0.75, zorder=2)
    ax.add_patch(box)
    ax.text(enc_x + enc_w / 2, ey + enc_h / 2, 'N patches → d_enc',
            ha='center', va='center', fontsize=7.5, color='#2C3E50', zorder=4)
    # Arrow: channel → encoder
    arrow_line(chan_x + chan_w + 0.05, chan_ys[i] + chan_h / 2,
               enc_x - 0.05, ey + chan_h / 2)

# Dashed bounding box (shared weights)
pad = 0.20
enc_bbox = FancyBboxPatch(
    (enc_x - pad, enc_ys[-1] - pad),
    enc_w + 2 * pad,
    (enc_ys[0] + enc_h) - enc_ys[-1] + 2 * pad,
    boxstyle='round,pad=0.12',
    facecolor='none', edgecolor='#2980B9',
    linewidth=2.0, linestyle='--', zorder=1)
ax.add_patch(enc_bbox)

# Shared-weights label to the right of the dashed box (moved to avoid overlapping title)
ax.text(enc_x + enc_w + pad + 1.2, enc_ys[0] + enc_h + pad - 0.05,
        'shared weights', ha='left', va='center',
        fontsize=7.5, color='#2980B9', fontstyle='italic', zorder=6,
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                  edgecolor='#2980B9', lw=0.8, alpha=0.9))

# Right bracket for "C × N embeddings"
bracket(enc_x + enc_w + 0.08, enc_ys[0] + enc_h, enc_ys[-1],
        'C × N\nembeddings', side='right')

# ═══════════════════════════════════════════════════════
# PERCEIVER POOLING (converging arrows + box)
# ═══════════════════════════════════════════════════════
pool_x = 10.5
pool_w = 3.6
pool_h = 3.2
pool_y_center = (enc_ys[0] + enc_h + enc_ys[-1]) / 2
pool_y = pool_y_center - pool_h / 2

# Converging color-coded arrows
for i, ey in enumerate(enc_ys):
    src_x = enc_x + enc_w + 0.08
    src_y = ey + enc_h / 2
    dst_x = pool_x - 0.05
    frac = i / 11.0
    dst_y = pool_y + pool_h - 0.30 - frac * (pool_h - 0.60)
    ax.annotate('', xy=(dst_x, dst_y), xytext=(src_x, src_y),
                arrowprops=dict(arrowstyle='->', color=LEAD_COLORS[i],
                                lw=1.0, alpha=0.60,
                                connectionstyle='arc3,rad=0'),
                zorder=4)

# Perceiver Pooling box
rounded_box(pool_x, pool_y, pool_w, pool_h,
            BOX_PERCEIVER, ec='#D4790E', lw=2.0)

ax.text(pool_x + pool_w / 2, pool_y + pool_h - 0.42,
        'Perceiver Pooling', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#7D3C00', zorder=5)
ax.text(pool_x + pool_w / 2, pool_y + pool_h - 0.82,
        '(Learned Token Compression)', ha='center', va='center',
        fontsize=10, color='#7D3C00', zorder=5)

details = [
    'K = 64 learnable query vectors',
    'Cross-attention:  Q attends to C×N KV',
    'Input:  C×N = 12×100 patch embeddings',
    'Output:  64 tokens,  dim = d_enc',
]
for j, line in enumerate(details):
    ax.text(pool_x + pool_w / 2, pool_y + pool_h - 1.35 - j * 0.40,
            line, ha='center', va='center',
            fontsize=9, color='#5D4037', zorder=5)

# ═══════════════════════════════════════════════════════
# MLP PROJECTOR
# ═══════════════════════════════════════════════════════
mlp_x = 10.5
mlp_w = 3.6
mlp_h = 1.15
mlp_y = pool_y - 1.8

rounded_box(mlp_x, mlp_y, mlp_w, mlp_h,
            BOX_MLP, ec='#1E8449', lw=1.8)

ax.text(mlp_x + mlp_w / 2, mlp_y + mlp_h - 0.28,
        'MLP Projector', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#145A32', zorder=5)
ax.text(mlp_x + mlp_w / 2, mlp_y + mlp_h - 0.62,
        '2-layer MLP + GELU (LLaVA 1.5)', ha='center', va='center',
        fontsize=10, color='#145A32', zorder=5)
ax.text(mlp_x + mlp_w / 2, mlp_y + mlp_h - 0.93,
        'Output: K = 64 tokens, dim = d_LLM', ha='center', va='center',
        fontsize=9, color='#145A32', fontstyle='italic', zorder=5)

# Arrow: pool → MLP
arrow_line(pool_x + pool_w / 2, pool_y - 0.02,
           mlp_x + mlp_w / 2, mlp_y + mlp_h + 0.02)
ax.text(pool_x + pool_w + 0.15, (pool_y + mlp_y + mlp_h) / 2,
        '64 × d_enc', ha='left', va='center',
        fontsize=9.5, color=DIM_COLOR, fontstyle='italic', zorder=6)

# ═══════════════════════════════════════════════════════
# LLaMA (TOKEN CONCAT + LLM)
# ═══════════════════════════════════════════════════════
llm_x = 10.5
llm_w = 3.6
llm_h = 1.5
llm_y = mlp_y - 2.15

rounded_box(llm_x, llm_y, llm_w, llm_h,
            BOX_LLM, ec='#6C3483', lw=1.8)

ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.28,
        'LLaMA 3.2 (Unmodified)', ha='center', va='center',
        fontsize=14, fontweight='bold', color='#4A235A', zorder=5)
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.62,
        'Concatenate TS tokens + text tokens', ha='center', va='center',
        fontsize=10, color='#4A235A', zorder=5)
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.93,
        'Standard causal self-attention', ha='center', va='center',
        fontsize=10, color='#4A235A', zorder=5)
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 1.23,
        'Autoregressive text output', ha='center', va='center',
        fontsize=9, color='#4A235A', fontstyle='italic', zorder=5)

# Arrow: MLP → LLM
arrow_line(mlp_x + mlp_w / 2, mlp_y - 0.02,
           llm_x + llm_w / 2, llm_y + llm_h + 0.02)
ax.text(mlp_x + mlp_w + 0.15, (mlp_y + llm_y + llm_h) / 2,
        '64 × d_LLM', ha='left', va='center',
        fontsize=9.5, color=DIM_COLOR, fontstyle='italic', zorder=6)

# Text tokens box and arrow (from left)
# Dashed box around "Text tokens" for visual clarity and padding
text_box_x = llm_x - 5.5
text_box_w = 2.8
text_box_h = 0.9
text_box_y = llm_y + llm_h / 2 - text_box_h / 2
text_box = FancyBboxPatch(
    (text_box_x, text_box_y), text_box_w, text_box_h,
    boxstyle='round,pad=0.12',
    facecolor='#F5EEF8', edgecolor='#6C3483',
    linewidth=1.2, linestyle='--', zorder=2)
ax.add_patch(text_box)
ax.text(text_box_x + text_box_w / 2, text_box_y + text_box_h / 2 + 0.12,
        'Text tokens', ha='center', va='center',
        fontsize=10, color='#6C3483', fontweight='bold', zorder=6)
ax.text(text_box_x + text_box_w / 2, text_box_y + text_box_h / 2 - 0.20,
        '(instruction + metadata)', ha='center', va='center',
        fontsize=8.5, color='#6C3483', zorder=6)
# Arrow from text box to LLM
ax.annotate('', xy=(llm_x - 0.02, llm_y + llm_h / 2),
            xytext=(text_box_x + text_box_w + 0.02, text_box_y + text_box_h / 2),
            arrowprops=dict(arrowstyle='->', color='#6C3483',
                            lw=1.5, linestyle='--', connectionstyle='arc3,rad=0'),
            zorder=5)

# ═══════════════════════════════════════════════════════
# DIMENSION FLOW SIDEBAR (far right)
# ═══════════════════════════════════════════════════════
sx = 16.2
summary_items = [
    ('Input',          'C × L\n12 × 5000'),
    ('After Patching', 'C × N × d_enc\n12 × 100 × d_enc'),
    ('After Pooling',  'K × d_enc\n64 × d_enc'),
    ('After MLP',      'K × d_LLM\n64 × d_LLM'),
]

sy_start = 11.5
step = 1.7
for k, (stage, dims) in enumerate(summary_items):
    sy = sy_start - k * step
    ax.text(sx, sy, stage, ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#2C3E50', zorder=6)
    ax.text(sx, sy - 0.40, dims, ha='center', va='center',
            fontsize=8.5, color=DIM_COLOR, fontstyle='italic', zorder=6,
            linespacing=1.3)
    if k < len(summary_items) - 1:
        ax.annotate('', xy=(sx, sy - 0.95),
                    xytext=(sx, sy - 0.75),
                    arrowprops=dict(arrowstyle='->', color=DIM_COLOR, lw=1.0),
                    zorder=5)

ax.text(sx, sy_start + 0.65,
        'Dimension Flow', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.30', facecolor='#EBF5FB',
                  edgecolor='#2980B9', lw=1.2),
        zorder=6)

# ═══════════════════════════════════════════════════════
# KEY INSIGHT CALLOUT
# ═══════════════════════════════════════════════════════
ax.text(sx, sy_start - 4 * step + 0.25,
        'Key: shared weights across\nall C channels (encoder\nand pooling queries)',
        ha='center', va='center', fontsize=9,
        color='#2C3E50', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.40', facecolor='#FEF9E7',
                  edgecolor='#D4AC0D', lw=1.2),
        zorder=6)

# ═══════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('/home/wangni/notion-figures/llava/fig_005.png',
            dpi=200, bbox_inches='tight', facecolor=BG_WHITE)
plt.close()
print('Saved: /home/wangni/notion-figures/llava/fig_005.png')
