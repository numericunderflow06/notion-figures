"""
LLaVA-TSM Architecture Overview (fig_003)
End-to-end architecture diagram showing four components:
1. Time Series Patch Encoder (Conv1D + Transformer, d=512)
2. Perceiver Pooling (K=64 queries, 2 cross-attention layers)
3. MLP Projector (2-layer + GELU)
4. Token Concatenation with LLaMA 3.2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────
C_ENCODER   = '#3B82F6'   # blue
C_ENCODER_L = '#DBEAFE'   # light blue
C_PERCEIVER = '#F97316'   # orange
C_PERCEIVER_L = '#FED7AA' # light orange
C_MLP       = '#22C55E'   # green
C_MLP_L     = '#BBF7D0'   # light green
C_LLM       = '#8B5CF6'   # purple
C_LLM_L     = '#DDD6FE'   # light purple
C_INPUT     = '#64748B'   # slate
C_INPUT_L   = '#E2E8F0'   # light slate
C_TEXT      = '#1E293B'   # dark slate
C_ARROW     = '#475569'   # slate-600
C_SHAPE     = '#94A3B8'   # shape annotation color
C_HIGHLIGHT = '#EF4444'   # red for highlights
C_CONCAT_BG = '#F8FAFC'   # very light background for concat region

fig, ax = plt.subplots(1, 1, figsize=(18, 7.5))
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-1.0, 7.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helper functions ───────────────────────────────────────────
def draw_box(ax, x, y, w, h, facecolor, edgecolor, label, fontsize=11,
             fontweight='bold', text_color='white', alpha=1.0, linewidth=1.5,
             sublabel=None, sublabel_fs=9):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                fontsize=sublabel_fs, fontweight='normal', color=text_color,
                zorder=4, fontstyle='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.8, style='-|>',
               connectionstyle='arc3,rad=0', zorder=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=16,
                            connectionstyle=connectionstyle,
                            zorder=zorder)
    ax.add_patch(arrow)
    return arrow

def draw_tensor_label(ax, x, y, text, color=C_SHAPE, fontsize=8.5):
    """Draw a tensor shape annotation."""
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight='normal', fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor=color, linewidth=0.8, alpha=0.9),
            zorder=5)

def draw_component_label(ax, x, y, num, title, color, fontsize=11):
    """Draw a numbered component title above a block."""
    circle_r = 0.22
    circle = plt.Circle((x - 0.4, y), circle_r, color=color, zorder=5)
    ax.add_patch(circle)
    ax.text(x - 0.4, y, str(num), ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=6)
    ax.text(x + 0.0, y, title, ha='left', va='center',
            fontsize=fontsize, fontweight='bold', color=color, zorder=5)


# ══════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════
ax.text(9.0, 6.65, 'LLaVA-TSM Architecture Overview',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=C_TEXT, zorder=10)

# ══════════════════════════════════════════════════════════════
#  (0) RAW INPUT — time series waveform
# ══════════════════════════════════════════════════════════════
# Draw a small waveform icon
inp_x, inp_y = 0.0, 2.8
draw_box(ax, inp_x, inp_y, 1.6, 1.2, C_INPUT_L, C_INPUT,
         'Raw Time\nSeries', fontsize=10, fontweight='bold',
         text_color=C_TEXT, linewidth=1.2)

# Waveform sketch inside
t = np.linspace(0, 4*np.pi, 80)
sig = 0.3 * np.sin(t) + 0.1 * np.sin(3*t)
wx = np.linspace(inp_x + 0.15, inp_x + 1.45, len(t))
wy = sig + inp_y + 0.15
ax.plot(wx, wy, color=C_INPUT, linewidth=0.8, alpha=0.4, zorder=4)

draw_tensor_label(ax, 0.8, inp_y - 0.35, '(L,) or (C, L)', color=C_INPUT)

# ══════════════════════════════════════════════════════════════
#  (1) TIME SERIES PATCH ENCODER — blue
# ══════════════════════════════════════════════════════════════
enc_x, enc_y = 2.6, 1.6
enc_w, enc_h = 2.8, 3.6

# Outer container
enc_box = FancyBboxPatch((enc_x, enc_y), enc_w, enc_h,
                          boxstyle="round,pad=0.15",
                          facecolor=C_ENCODER_L, edgecolor=C_ENCODER,
                          linewidth=2.0, alpha=0.85, zorder=2)
ax.add_patch(enc_box)

# Component label
draw_component_label(ax, enc_x + 0.55, enc_y + enc_h + 0.30, 1,
                     'Time Series Patch Encoder', C_ENCODER, fontsize=11)

# Sub-blocks inside encoder
sb_w = 2.2
sb_x = enc_x + (enc_w - sb_w) / 2

# Normalize
draw_box(ax, sb_x, enc_y + 0.2, sb_w, 0.55, 'white', C_ENCODER,
         'Normalize to [-1, 1]', fontsize=8.5, fontweight='normal',
         text_color=C_TEXT, linewidth=1.0)

# Patch + Conv1D
draw_box(ax, sb_x, enc_y + 0.95, sb_w, 0.55, C_ENCODER, C_ENCODER,
         'Conv1D (k=stride=p)', fontsize=8.5, fontweight='bold',
         text_color='white', linewidth=1.0)

# Positional encoding
draw_box(ax, sb_x, enc_y + 1.7, sb_w, 0.55, 'white', C_ENCODER,
         '+ Positional Encoding', fontsize=8.5, fontweight='normal',
         text_color=C_TEXT, linewidth=1.0)

# Transformer encoder
draw_box(ax, sb_x, enc_y + 2.45, sb_w, 0.75, C_ENCODER, C_ENCODER,
         'Transformer Encoder', fontsize=9.5, fontweight='bold',
         text_color='white', linewidth=1.0,
         sublabel='4 layers, 8 heads', sublabel_fs=8)

# Internal arrows
for base_y in [enc_y + 0.75, enc_y + 1.5, enc_y + 2.25]:
    draw_arrow(ax, enc_x + enc_w/2, base_y, enc_x + enc_w/2, base_y + 0.2,
               color=C_ENCODER, lw=1.2, style='-|>')

# d_enc annotation
draw_tensor_label(ax, enc_x + enc_w/2, enc_y - 0.35,
                  'N patches \u00d7 d_enc=512', color=C_ENCODER)

# Arrow from input to encoder
draw_arrow(ax, inp_x + 1.6, inp_y + 0.6, enc_x, enc_y + 2.0,
           color=C_ARROW, lw=2.0)


# ══════════════════════════════════════════════════════════════
#  (2) PERCEIVER POOLING — orange
# ══════════════════════════════════════════════════════════════
per_x, per_y = 6.3, 1.6
per_w, per_h = 2.6, 3.6

per_box = FancyBboxPatch((per_x, per_y), per_w, per_h,
                          boxstyle="round,pad=0.15",
                          facecolor=C_PERCEIVER_L, edgecolor=C_PERCEIVER,
                          linewidth=2.0, alpha=0.85, zorder=2)
ax.add_patch(per_box)

draw_component_label(ax, per_x + 0.55, per_y + per_h + 0.30, 2,
                     'Perceiver Pooling', C_PERCEIVER, fontsize=11)

sb_w2 = 2.0
sb_x2 = per_x + (per_w - sb_w2) / 2

# Learned queries
draw_box(ax, sb_x2, per_y + 2.5, sb_w2, 0.65, 'white', C_PERCEIVER,
         'K=64 Learned\nQuery Tokens', fontsize=8.5, fontweight='normal',
         text_color=C_TEXT, linewidth=1.0)

# Cross-attention layers
draw_box(ax, sb_x2, per_y + 1.2, sb_w2, 1.05, C_PERCEIVER, C_PERCEIVER,
         'Cross-Attention', fontsize=9.5, fontweight='bold',
         text_color='white', linewidth=1.0,
         sublabel='2 layers', sublabel_fs=8.5)

# Feed-forward
draw_box(ax, sb_x2, per_y + 0.2, sb_w2, 0.7, 'white', C_PERCEIVER,
         'Feed-Forward', fontsize=8.5, fontweight='normal',
         text_color=C_TEXT, linewidth=1.0)

# Internal arrows
draw_arrow(ax, per_x + per_w/2, per_y + 2.5, per_x + per_w/2, per_y + 2.3,
           color=C_PERCEIVER, lw=1.2, style='-|>')
draw_arrow(ax, per_x + per_w/2, per_y + 1.2, per_x + per_w/2, per_y + 1.0,
           color=C_PERCEIVER, lw=1.2, style='-|>')

# Tensor shape annotation
draw_tensor_label(ax, per_x + per_w/2, per_y - 0.35,
                  'K=64 \u00d7 d_enc=512', color=C_PERCEIVER)

# Arrow from encoder to perceiver
draw_arrow(ax, enc_x + enc_w, enc_y + enc_h/2,
           per_x, per_y + enc_h/2,
           color=C_ARROW, lw=2.0)

# Label on arrow: "N patches" being compressed
ax.text((enc_x + enc_w + per_x) / 2, enc_y + enc_h/2 + 0.3,
        'N \u2192 64', ha='center', va='bottom', fontsize=8.5,
        color=C_PERCEIVER, fontweight='bold', zorder=5)


# ══════════════════════════════════════════════════════════════
#  (3) MLP PROJECTOR — green
# ══════════════════════════════════════════════════════════════
mlp_x, mlp_y = 9.8, 2.2
mlp_w, mlp_h = 2.2, 2.4

mlp_box = FancyBboxPatch((mlp_x, mlp_y), mlp_w, mlp_h,
                          boxstyle="round,pad=0.15",
                          facecolor=C_MLP_L, edgecolor=C_MLP,
                          linewidth=2.0, alpha=0.85, zorder=2)
ax.add_patch(mlp_box)

draw_component_label(ax, mlp_x + 0.25, mlp_y + mlp_h + 0.30, 3,
                     'MLP Projector', C_MLP, fontsize=11)

sb_w3 = 1.7
sb_x3 = mlp_x + (mlp_w - sb_w3) / 2

# Linear 1
draw_box(ax, sb_x3, mlp_y + 1.55, sb_w3, 0.5, C_MLP, C_MLP,
         'Linear', fontsize=9, fontweight='bold',
         text_color='white', linewidth=1.0)

# GELU
draw_box(ax, sb_x3, mlp_y + 0.85, sb_w3, 0.5, 'white', C_MLP,
         'GELU', fontsize=9, fontweight='bold',
         text_color=C_MLP, linewidth=1.0)

# Linear 2
draw_box(ax, sb_x3, mlp_y + 0.15, sb_w3, 0.5, C_MLP, C_MLP,
         'Linear', fontsize=9, fontweight='bold',
         text_color='white', linewidth=1.0)

# Internal arrows
draw_arrow(ax, mlp_x + mlp_w/2, mlp_y + 1.55, mlp_x + mlp_w/2, mlp_y + 1.4,
           color=C_MLP, lw=1.2, style='-|>')
draw_arrow(ax, mlp_x + mlp_w/2, mlp_y + 0.85, mlp_x + mlp_w/2, mlp_y + 0.7,
           color=C_MLP, lw=1.2, style='-|>')

# Tensor shape annotation
draw_tensor_label(ax, mlp_x + mlp_w/2, mlp_y - 0.35,
                  '64 \u00d7 d_LLM', color=C_MLP)

# d_LLM values note
ax.text(mlp_x + mlp_w/2, mlp_y - 0.72,
        'd_LLM = 2048 (1B) / 3072 (3B)', ha='center', va='center',
        fontsize=7.5, color=C_MLP, fontstyle='italic', zorder=5)

# Arrow from perceiver to MLP
draw_arrow(ax, per_x + per_w, per_y + per_h/2,
           mlp_x, mlp_y + mlp_h/2,
           color=C_ARROW, lw=2.0)


# ══════════════════════════════════════════════════════════════
#  TOKEN CONCATENATION REGION
# ══════════════════════════════════════════════════════════════
cat_x, cat_y = 12.7, 1.4
cat_w, cat_h = 1.2, 4.0

# Concatenation symbol / merge zone
cat_box = FancyBboxPatch((cat_x, cat_y), cat_w, cat_h,
                          boxstyle="round,pad=0.12",
                          facecolor=C_CONCAT_BG, edgecolor=C_SHAPE,
                          linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)
ax.add_patch(cat_box)

ax.text(cat_x + cat_w/2, cat_y + cat_h + 0.15, 'Concat',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color=C_SHAPE, zorder=5)

# Time series tokens (colored boxes)
for i in range(4):
    ty = cat_y + 2.3 + i * 0.38
    tb = FancyBboxPatch((cat_x + 0.15, ty), 0.9, 0.3,
                         boxstyle="round,pad=0.05",
                         facecolor=C_MLP_L, edgecolor=C_MLP,
                         linewidth=0.8, zorder=3)
    ax.add_patch(tb)
ax.text(cat_x + cat_w/2, cat_y + 3.95, 'TS', ha='center', va='center',
        fontsize=7, color=C_MLP, fontweight='bold', zorder=5)

# Ellipsis
ax.text(cat_x + cat_w/2, cat_y + 2.15, '\u22ee', ha='center', va='center',
        fontsize=12, color=C_SHAPE, zorder=5)

# Text tokens (colored boxes)
for i in range(3):
    ty = cat_y + 0.2 + i * 0.38
    tb = FancyBboxPatch((cat_x + 0.15, ty), 0.9, 0.3,
                         boxstyle="round,pad=0.05",
                         facecolor=C_LLM_L, edgecolor=C_LLM,
                         linewidth=0.8, zorder=3)
    ax.add_patch(tb)
ax.text(cat_x + cat_w/2, cat_y + 0.65, 'Text', ha='center', va='center',
        fontsize=7, color=C_LLM, fontweight='bold', zorder=5)

# Text input label
ax.text(cat_x + cat_w/2, cat_y - 0.35,
        'H_q (text tokens)', ha='center', va='center',
        fontsize=8.5, color=C_LLM, fontweight='normal', fontstyle='italic',
        zorder=5)

# Arrow from MLP to concat (TS tokens)
draw_arrow(ax, mlp_x + mlp_w, mlp_y + mlp_h/2,
           cat_x, cat_y + 3.2,
           color=C_ARROW, lw=2.0)

# Text input arrow (from below)
draw_arrow(ax, cat_x + cat_w/2, cat_y - 0.1,
           cat_x + cat_w/2, cat_y,
           color=C_LLM, lw=1.5, style='-|>')

# Text input box
draw_box(ax, cat_x - 0.2, cat_y - 1.0, cat_w + 0.4, 0.6, C_LLM_L, C_LLM,
         'Text Prompt', fontsize=8.5, fontweight='bold',
         text_color=C_LLM, linewidth=1.0)

# ══════════════════════════════════════════════════════════════
#  (4) LLaMA 3.2 (LLM) — purple
# ══════════════════════════════════════════════════════════════
llm_x, llm_y = 14.6, 1.6
llm_w, llm_h = 2.8, 3.6

llm_box = FancyBboxPatch((llm_x, llm_y), llm_w, llm_h,
                          boxstyle="round,pad=0.15",
                          facecolor=C_LLM_L, edgecolor=C_LLM,
                          linewidth=2.0, alpha=0.85, zorder=2)
ax.add_patch(llm_box)

draw_component_label(ax, llm_x + 0.35, llm_y + llm_h + 0.30, 4,
                     'LLaMA 3.2 (Unmodified)', C_LLM, fontsize=11)

sb_w4 = 2.2
sb_x4 = llm_x + (llm_w - sb_w4) / 2

# Self-attention layers
draw_box(ax, sb_x4, llm_y + 2.3, sb_w4, 0.85, C_LLM, C_LLM,
         'Self-Attention', fontsize=9.5, fontweight='bold',
         text_color='white', linewidth=1.0,
         sublabel='Standard layers', sublabel_fs=8)

# Feed-forward
draw_box(ax, sb_x4, llm_y + 1.2, sb_w4, 0.85, 'white', C_LLM,
         'Feed-Forward', fontsize=9.5, fontweight='bold',
         text_color=C_LLM, linewidth=1.0,
         sublabel='+ RMSNorm', sublabel_fs=8)

# Output head
draw_box(ax, sb_x4, llm_y + 0.2, sb_w4, 0.7, C_LLM, C_LLM,
         'LM Head', fontsize=9.5, fontweight='bold',
         text_color='white', linewidth=1.0)

# Internal arrows
draw_arrow(ax, llm_x + llm_w/2, llm_y + 2.3, llm_x + llm_w/2, llm_y + 2.1,
           color=C_LLM, lw=1.2, style='-|>')
draw_arrow(ax, llm_x + llm_w/2, llm_y + 1.2, llm_x + llm_w/2, llm_y + 1.0,
           color=C_LLM, lw=1.2, style='-|>')

# Arrow from concat to LLM
draw_arrow(ax, cat_x + cat_w, cat_y + cat_h/2,
           llm_x, llm_y + llm_h/2,
           color=C_ARROW, lw=2.0)

# Output arrow
draw_arrow(ax, llm_x + llm_w, llm_y + llm_h/2,
           llm_x + llm_w + 0.7, llm_y + llm_h/2,
           color=C_LLM, lw=2.0)

# Output label
ax.text(llm_x + llm_w + 0.85, llm_y + llm_h/2, 'Language\nResponse',
        ha='left', va='center', fontsize=10, fontweight='bold',
        color=C_LLM, zorder=5)


# ══════════════════════════════════════════════════════════════
#  KEY INSIGHT CALLOUT
# ══════════════════════════════════════════════════════════════
# Highlight box for the key insight
insight_x, insight_y = 6.0, -0.55
insight_w, insight_h = 8.7, 0.55

insight_box = FancyBboxPatch((insight_x, insight_y), insight_w, insight_h,
                              boxstyle="round,pad=0.1",
                              facecolor='#FEF2F2', edgecolor=C_HIGHLIGHT,
                              linewidth=1.5, alpha=0.9, zorder=5)
ax.add_patch(insight_box)

ax.text(insight_x + insight_w/2, insight_y + insight_h/2,
        'Key Insight: Perceiver pooling is front-end only \u2014 the LLM is completely unmodified (no cross-attention, no gating)',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color=C_HIGHLIGHT, zorder=6)


# ══════════════════════════════════════════════════════════════
#  BRACKET / BRACE ANNOTATIONS
# ══════════════════════════════════════════════════════════════
# Front-end bracket (components 1-3)
brace_y = 5.85
ax.annotate('', xy=(enc_x, brace_y), xytext=(mlp_x + mlp_w, brace_y),
            arrowprops=dict(arrowstyle='-', color=C_SHAPE, lw=1.2))
# Vertical ticks
for bx in [enc_x, mlp_x + mlp_w]:
    ax.plot([bx, bx], [brace_y - 0.1, brace_y + 0.1], color=C_SHAPE, lw=1.2, zorder=5)
ax.text((enc_x + mlp_x + mlp_w) / 2, brace_y + 0.25,
        'Front-end (trainable)', ha='center', va='bottom',
        fontsize=9, color=C_SHAPE, fontweight='bold', fontstyle='italic', zorder=5)

# LLM bracket
for bx in [llm_x, llm_x + llm_w]:
    ax.plot([bx, bx], [brace_y - 0.1, brace_y + 0.1], color=C_LLM, lw=1.2, zorder=5)
ax.annotate('', xy=(llm_x, brace_y), xytext=(llm_x + llm_w, brace_y),
            arrowprops=dict(arrowstyle='-', color=C_LLM, lw=1.2))
ax.text(llm_x + llm_w/2, brace_y + 0.25,
        'LLM (fine-tuned, unmodified arch.)', ha='center', va='bottom',
        fontsize=9, color=C_LLM, fontweight='bold', fontstyle='italic', zorder=5)


# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
fig.savefig('/home/wangni/notion-figures/llava/fig_003.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Figure saved to /home/wangni/notion-figures/llava/fig_003.png")
