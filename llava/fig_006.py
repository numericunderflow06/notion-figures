"""
fig_006: Architecture Comparison — OpenTSLM-SoftPrompt vs OpenTSLM-Flamingo vs LLaVA-TSM
Side-by-side schematic comparing the three architectures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────
C_TS_INPUT  = '#4A90D9'   # Time series input
C_ENCODER   = '#5DADE2'   # Patch encoder
C_PERCEIVER = '#AF7AC5'   # Perceiver / compression
C_PROJECTOR = '#F5B041'   # Projection layer (linear or MLP)
C_CONCAT    = '#58D68D'   # Concatenation zone
C_LLM       = '#EC7063'   # LLM backbone
C_TEXT      = '#85929E'   # Text tokens
C_OUTPUT    = '#AAB7B8'   # Output
C_XATTN     = '#E74C3C'   # Cross-attention (Flamingo-specific)
C_ANNOT_BG  = '#FEF9E7'   # Annotation background
C_BORDER    = '#2C3E50'   # Border / arrow colour

# ── Figure setup ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 15), facecolor='white', dpi=200)

fig.suptitle('Architecture Comparison: OpenTSLM-SoftPrompt  vs  OpenTSLM-Flamingo  vs  LLaVA-TSM',
             fontsize=17, fontweight='bold', y=0.97, color=C_BORDER)

gs = fig.add_gridspec(1, 3, left=0.03, right=0.97, top=0.91, bottom=0.07,
                      wspace=0.12)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

for ax in axes:
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 16.5)
    ax.set_aspect('equal')
    ax.axis('off')


# ── Helper functions ──────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label, color, fontsize=10, fontweight='normal',
             edgecolor=C_BORDER, linewidth=1.2, alpha=0.92, text_color='black',
             style='round,pad=0.15'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha,
                         zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color=text_color,
            zorder=3, linespacing=1.3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_BORDER, lw=1.5, style='-|>',
               connectionstyle='arc3,rad=0.0'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=14,
                            connectionstyle=connectionstyle,
                            zorder=4)
    ax.add_patch(arrow)

def draw_annotation(ax, x, y, text, fontsize=9, color='#B03A2E',
                    bgcolor=C_ANNOT_BG, boxstyle='round,pad=0.25'):
    ax.annotate(text, (x, y),
                fontsize=fontsize, fontweight='bold', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle=boxstyle, fc=bgcolor, ec=color,
                          alpha=0.95, lw=1.2),
                zorder=5)


# ═══════════════════════════════════════════════════════════════════════
# COLUMN 1 — OpenTSLM-SoftPrompt
# ═══════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title('OpenTSLM-SoftPrompt', fontsize=14, fontweight='bold',
             pad=12, color=C_BORDER)

# Time Series Input
draw_box(ax, 1.5, 14.2, 7, 1.0, 'Time Series Input\n(L samples, C channels)',
         C_TS_INPUT, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, 14.2, 5, 13.5)

# Patch Encoder
draw_box(ax, 1.5, 12.2, 7, 1.2, 'Patch Encoder\n(Conv1D + Transformer)',
         C_ENCODER, fontsize=10)
draw_arrow(ax, 5, 12.2, 5, 11.5)

# No compression — grey dashed box
draw_box(ax, 1.5, 10.0, 7, 1.2, 'All C×N Patch Tokens\n(no compression)',
         '#D5DBDB', fontsize=10, edgecolor='#7F8C8D', linewidth=1.8,
         style='round,pad=0.15')
draw_arrow(ax, 5, 10.0, 5, 9.3)

# Linear Projection
draw_box(ax, 1.5, 8.0, 7, 1.2, 'Linear Projection\n(d_enc  →  d_LLM)',
         C_PROJECTOR, fontsize=10)
draw_arrow(ax, 5, 8.0, 5, 7.3)

# Concatenation
draw_box(ax, 0.5, 5.9, 3.5, 1.2, 'Text\nTokens', C_TEXT, fontsize=9.5)
draw_box(ax, 4.2, 5.9, 5.5, 1.2, 'C×N Time Series\nTokens', C_CONCAT, fontsize=9.5)
ax.plot([0.5, 9.7], [5.75, 5.75], color=C_BORDER, lw=1.0, zorder=1)
ax.text(5, 5.4, 'Concatenated Sequence (variable length)', ha='center',
        va='center', fontsize=8.5, fontstyle='italic', color=C_BORDER)
draw_arrow(ax, 5, 5.35, 5, 4.85)

# Frozen LLM + LoRA
draw_box(ax, 1.0, 3.0, 8, 1.8, 'Frozen LLM  +  LoRA\n(LLaMA 3.2)\nSelf-attention over all tokens',
         C_LLM, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, 3.0, 5, 2.3)

# Output
draw_box(ax, 2.5, 1.3, 5, 0.9, 'Text Output', C_OUTPUT, fontsize=10)

# Annotation
draw_annotation(ax, 5, 0.35, 'No compression\nTokens grow with series length',
                fontsize=8.5, color='#C0392B')

# Memory footprint
ax.text(5, -0.55, '~110 GB VRAM\n(LLaMA-3B, ECG-QA)',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#922B21',
        bbox=dict(boxstyle='round,pad=0.3', fc='#FDEDEC', ec='#922B21', lw=1.2))


# ═══════════════════════════════════════════════════════════════════════
# COLUMN 2 — OpenTSLM-Flamingo
# ═══════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title('OpenTSLM-Flamingo', fontsize=14, fontweight='bold',
             pad=12, color=C_BORDER)

# Time Series Input
draw_box(ax, 1.5, 14.2, 7, 1.0, 'Time Series Input\n(L samples, C channels)',
         C_TS_INPUT, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, 14.2, 5, 13.5)

# Patch Encoder
draw_box(ax, 1.5, 12.2, 7, 1.2, 'Patch Encoder\n(Conv1D + Transformer)',
         C_ENCODER, fontsize=10)
draw_arrow(ax, 5, 12.2, 5, 11.5)

# Perceiver Resampler
draw_box(ax, 1.5, 10.0, 7, 1.2, 'Perceiver Resampler\n(K=64 learned queries)',
         C_PERCEIVER, fontsize=10, fontweight='bold')

# K=64 Latent Tokens
draw_arrow(ax, 5, 10.0, 5, 9.3)
draw_box(ax, 1.5, 8.0, 7, 1.2, 'K=64 Latent Tokens\n(compressed repr.)',
         '#D7BDE2', fontsize=10)

# --- Flamingo-specific: two parallel paths into the LLM ---
# Text tokens enter the LLM's self-attention normally
draw_box(ax, 0.3, 5.9, 3.5, 1.2, 'Text\nTokens', C_TEXT, fontsize=9.5)

# Arrow: text tokens → self-attention in the LLM
draw_arrow(ax, 2.05, 5.9, 2.05, 5.25, color=C_BORDER, lw=1.5)

# Arrow: K latent tokens → gated cross-attention (curved, from right side)
draw_arrow(ax, 7.5, 8.0, 8.0, 5.25, color=C_PERCEIVER, lw=2.2,
           style='-|>', connectionstyle='arc3,rad=-0.15')

# Label on curved arrow
ax.text(9.1, 6.6, 'K latent\ntokens', fontsize=7.5, color='#7D3C98',
        ha='center', va='center', fontstyle='italic', fontweight='bold')

# Frozen LLM box (larger, showing internal structure)
draw_box(ax, 0.3, 2.8, 9.4, 2.4, '', C_LLM, fontsize=10, alpha=0.25)

# Internal: Self-Attention (frozen)
draw_box(ax, 0.8, 4.15, 3.6, 0.85, 'Self-Attn\n(frozen)', '#F5B7B1', fontsize=9)

# Internal: Gated Cross-Attention (new)
draw_box(ax, 5.0, 4.15, 4.2, 0.85, 'Gated Cross-\nAttn (new)', C_XATTN, fontsize=9,
         text_color='white', fontweight='bold')

# Connecting arrow between self-attn and cross-attn
draw_arrow(ax, 4.4, 4.57, 5.0, 4.57, color='#7B241C', lw=1.2,
           style='-|>')

# Label inside LLM region
ax.text(5, 3.1, 'Frozen LLM (LLaMA 3.2) + LoRA adapters',
        ha='center', va='center', fontsize=9, fontweight='bold', color=C_BORDER)

# Arrow to output
draw_arrow(ax, 5, 2.8, 5, 2.15)

# Output
draw_box(ax, 2.5, 1.15, 5, 0.9, 'Text Output', C_OUTPUT, fontsize=10)

# Annotation
draw_annotation(ax, 5, 0.2, 'Cross-attention inside LLM\nModifies LLM architecture',
                fontsize=8.5, color='#6C3483')

# Memory footprint
ax.text(5, -0.7, '~40 GB VRAM\n(LLaMA-3B, ECG-QA)',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#6C3483',
        bbox=dict(boxstyle='round,pad=0.3', fc='#F4ECF7', ec='#6C3483', lw=1.2))


# ═══════════════════════════════════════════════════════════════════════
# COLUMN 3 — LLaVA-TSM (Proposed)
# ═══════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_title('LLaVA-TSM  (Proposed)', fontsize=14, fontweight='bold',
             pad=12, color='#1A5276')

# Subtle highlight
highlight = mpatches.FancyBboxPatch((-0.3, -1.3), 11.1, 17.2,
                                     boxstyle='round,pad=0.3',
                                     facecolor='#EBF5FB',
                                     edgecolor='#2E86C1',
                                     linewidth=2.5,
                                     zorder=0, alpha=0.35)
ax.add_patch(highlight)

# Time Series Input
draw_box(ax, 1.5, 14.2, 7, 1.0, 'Time Series Input\n(L samples, C channels)',
         C_TS_INPUT, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, 14.2, 5, 13.5)

# Patch Encoder (pre-trained, frozen)
draw_box(ax, 1.5, 12.2, 7, 1.2, 'Patch Encoder (frozen)\n(Conv1D + Transformer)\nPre-trained via masked patches (Stage 0)',
         C_ENCODER, fontsize=8.5)
draw_arrow(ax, 5, 12.2, 5, 11.5)

# Perceiver Pooling — front-end only
draw_box(ax, 1.5, 10.0, 7, 1.2, 'Perceiver Pooling\n(K=64 learned queries)\nFront-end compression only',
         C_PERCEIVER, fontsize=9, fontweight='bold')
draw_arrow(ax, 5, 10.0, 5, 9.3)

# MLP Projector
draw_box(ax, 1.5, 8.0, 7, 1.2, 'MLP Projector (2-layer + GELU)\n(d_enc  →  d_LLM)\nFollowing LLaVA 1.5',
         C_PROJECTOR, fontsize=9, fontweight='bold')
draw_arrow(ax, 5, 8.0, 5, 7.3)

# Concatenation
draw_box(ax, 0.5, 5.9, 3.5, 1.2, 'Text\nTokens', C_TEXT, fontsize=9.5)
draw_box(ax, 4.2, 5.9, 5.5, 1.2, 'K=64 TS\nTokens', C_CONCAT, fontsize=9.5)
ax.plot([0.5, 9.7], [5.75, 5.75], color=C_BORDER, lw=1.0, zorder=1)
ax.text(5, 5.4, 'Concatenated Sequence (fixed length)', ha='center',
        va='center', fontsize=8.5, fontstyle='italic', color=C_BORDER)
draw_arrow(ax, 5, 5.35, 5, 4.85)

# Unmodified LLM — full fine-tuning
draw_box(ax, 1.0, 3.0, 8, 1.8, 'Unmodified LLM (full fine-tuning)\n(LLaMA 3.2)\nStandard self-attention only',
         C_LLM, fontsize=10, fontweight='bold')
draw_arrow(ax, 5, 3.0, 5, 2.3)

# Output
draw_box(ax, 2.5, 1.3, 5, 0.9, 'Text Output', C_OUTPUT, fontsize=10)

# Annotation
draw_annotation(ax, 5, 0.35, 'Front-end compression only\nNo LLM architecture changes',
                fontsize=8.5, color='#1A5276')

# Memory footprint
ax.text(5, -0.55, '~30 GB (LLaMA-1B)\n~60 GB (LLaMA-3B, DeepSpeed)',
        ha='center', va='center', fontsize=9.5, fontweight='bold',
        color='#1A5276',
        bbox=dict(boxstyle='round,pad=0.3', fc='#D6EAF8', ec='#1A5276', lw=1.2))


# ═══════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(facecolor=C_TS_INPUT, edgecolor=C_BORDER, label='Time Series Input'),
    mpatches.Patch(facecolor=C_ENCODER, edgecolor=C_BORDER, label='Patch Encoder'),
    mpatches.Patch(facecolor=C_PERCEIVER, edgecolor=C_BORDER, label='Perceiver (compression)'),
    mpatches.Patch(facecolor=C_PROJECTOR, edgecolor=C_BORDER, label='Projection Layer'),
    mpatches.Patch(facecolor=C_CONCAT, edgecolor=C_BORDER, label='TS Tokens (concatenated)'),
    mpatches.Patch(facecolor=C_TEXT, edgecolor=C_BORDER, label='Text Tokens'),
    mpatches.Patch(facecolor=C_LLM, edgecolor=C_BORDER, label='LLM Backbone'),
    mpatches.Patch(facecolor=C_XATTN, edgecolor=C_BORDER, label='Gated Cross-Attn (Flamingo only)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=10, frameon=True, fancybox=True,
           edgecolor=C_BORDER, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005))


# ═══════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════
outpath = '/home/wangni/notion-figures/llava/fig_006.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white',
            pad_inches=0.3)
plt.close(fig)
print(f"Figure saved to {outpath}")
