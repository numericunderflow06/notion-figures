"""
TPA End-to-End Architecture Diagram (fig_001)
Full architecture diagram showing the complete TPA pipeline in the context of OpenTSLM data flow.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
BLUE_FILL   = '#D6EAF8'   # existing OpenTSLM components
BLUE_EDGE   = '#2980B9'
ORANGE_FILL = '#FEF0DB'   # TPA additions
ORANGE_EDGE = '#E67E22'
GREEN_FILL  = '#D5F5E3'   # data / I/O
GREEN_EDGE  = '#27AE60'
DARK_TEXT    = '#2C3E50'
ARROW_COLOR = '#5D6D7E'
ANNOT_COLOR = '#7F8C8D'

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 26))
ax.set_xlim(0, 20)
ax.set_ylim(0, 26)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Helper functions ────────────────────────────────────────────────
def rbox(x, y, w, h, label, fill, edge, fontsize=11, fontweight='bold',
         sublabel=None, sublabel_fs=8.5, zorder=3):
    """Rounded rectangle with centred label and optional sublabel."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                         facecolor=fill, edgecolor=edge,
                         linewidth=2.0, zorder=zorder)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h/2 + 0.2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=DARK_TEXT,
                zorder=zorder+1)
        ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                fontsize=sublabel_fs, color=ANNOT_COLOR, style='italic',
                zorder=zorder+1)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=DARK_TEXT,
                zorder=zorder+1)
    return (x, y, w, h)

def arr(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.8, head=True,
        connstyle='arc3,rad=0', ls='-', zorder=2):
    """Arrow or plain line."""
    sty = '->' if head else '-'
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=sty,
                        color=color, linewidth=lw,
                        connectionstyle=connstyle, zorder=zorder,
                        linestyle=ls, mutation_scale=15)
    ax.add_patch(a)

def note(x, y, text, fs=8, color=ANNOT_COLOR, ha='center', va='center',
         fw='normal', box=False):
    props = dict(fontsize=fs, color=color, ha=ha, va=va, fontweight=fw, zorder=10)
    if box:
        props['bbox'] = dict(boxstyle='round,pad=0.25', fc='white',
                             ec=color, lw=0.8, alpha=0.95)
    ax.text(x, y, text, **props)

# =====================================================================
#  LAYOUT COORDINATES — top to bottom
# =====================================================================
#  Columns: LEFT path (time series) centred at x≈5, RIGHT path (text) centred at x≈15
CL = 5.0   # left column centre
CR = 14.5  # right column centre

# ── Title ───────────────────────────────────────────────────────────
ax.text(10, 25.4, 'TPA End-to-End Architecture',
        ha='center', va='center', fontsize=20, fontweight='bold', color=DARK_TEXT)
ax.text(10, 24.95, 'Temporal Positional Alignment for OpenTSLM',
        ha='center', va='center', fontsize=13, color=ANNOT_COLOR)

# =====================================================================
#  ROW 0 — Input data  y=23.5
# =====================================================================
Y0 = 23.5
rbox(1.5, Y0, 3.5, 1.0, 'Dataset', GREEN_FILL, GREEN_EDGE,
     sublabel='(time series + text)')
arr(5.0, Y0+0.5, 6.5, Y0+0.5)
rbox(6.5, Y0, 3.5, 1.0, 'DataLoader', GREEN_FILL, GREEN_EDGE,
     sublabel='pad_and_apply_batch')
arr(10.0, Y0+0.5, 11.5, Y0+0.5)
rbox(11.5, Y0, 4.5, 1.0, 'Training Sample', GREEN_FILL, GREEN_EDGE,
     sublabel='ts_values, text, timestamps')

# =====================================================================
#  Split from Training Sample → two paths
# =====================================================================
Y_split = 22.6
# Vertical stub down from Training Sample
arr(13.0, Y0, 13.0, Y_split, head=False)
arr(14.5, Y0, 14.5, Y_split, head=False)

# Left branch (ts_values + timestamps)
arr(13.0, Y_split, CL, Y_split, head=False)
arr(CL, Y_split, CL, Y_split - 0.5)
note(9.0, Y_split + 0.2, 'ts_values + timestamps', fs=9.5, color=DARK_TEXT, fw='bold')

# Right branch (text)
arr(14.5, Y_split, CR, Y_split, head=False)
arr(CR, Y_split, CR, Y_split - 0.5)
note(14.8, Y_split + 0.2, 'text', fs=9.5, color=DARK_TEXT, fw='bold')

# =====================================================================
#  LEFT COLUMN — Time series processing
# =====================================================================

# --- TransformerCNN Encoder ---
Y_enc = 20.5
rbox(2.5, Y_enc, 5.0, 1.2, 'TransformerCNN Encoder', BLUE_FILL, BLUE_EDGE,
     fontsize=11, sublabel='(frozen / fine-tuned)')

# --- ATPE (Component 1) inside encoder ---
Y_atpe = 18.2
rbox(1.8, Y_atpe, 6.4, 1.5, 'ATPE  (Component 1)', ORANGE_FILL, ORANGE_EDGE,
     fontsize=12, sublabel='sinusoidal(timestamps) → time_proj (128×128)')

arr(CL, Y_enc, CL, Y_atpe + 1.5)
note(CL + 1.0, Y_enc - 0.3, 'replaces learnable PE', fs=8.5,
     color=ORANGE_EDGE, fw='bold')

# ATPE annotations
note(CL, Y_atpe - 0.35,
     'Timestamps normalized to [0, 1] per sample\n'
     'Shared across all series → automatic cross-series alignment',
     fs=7.5)
note(0.7, Y_atpe + 0.75, '~16K\nparams', fs=8, color=ORANGE_EDGE, fw='bold', box=True)

# --- Projector ---
Y_proj = 15.8
rbox(2.5, Y_proj, 5.0, 1.0, 'Projector', BLUE_FILL, BLUE_EDGE,
     sublabel='ts_dim → LLM hidden_dim')
arr(CL, Y_atpe, CL, Y_proj + 1.0)
note(CL + 0.8, Y_proj + 1.3, 'encoded ts patches', fs=8.5, color=DARK_TEXT)

# =====================================================================
#  RIGHT COLUMN — Text processing
# =====================================================================

# --- LLM Tokenizer ---
Y_tok = 20.5
rbox(12.0, Y_tok, 5.0, 1.0, 'LLM Tokenizer', BLUE_FILL, BLUE_EDGE, fontsize=11)

# --- Temporal Reference Extraction ---
Y_tref = 18.5
rbox(12.0, Y_tref, 5.0, 1.2, 'Temporal Reference\nExtraction', ORANGE_FILL, ORANGE_EDGE,
     fontsize=10.5, sublabel='rule-based')

# Arrow from tokenizer down, splits into two: Token Embeddings and Temporal Ref
arr(CR, Y_tok, CR, Y_tref + 1.2)

note(CR, Y_tref - 0.45,
     '"first period" → [0, 0.5]\n'
     '"second period" → [0.5, 1.0]\n'
     '"over time" → 0.5\n'
     '"final probability" → 1.0',
     fs=7)

# --- Token Embeddings ---
Y_temb = 16.0
rbox(12.0, Y_temb, 5.0, 1.0, 'Token Embeddings', BLUE_FILL, BLUE_EDGE,
     sublabel='from frozen LLM embedding layer')
arr(CR, Y_tref, CR, Y_temb + 1.0)

# =====================================================================
#  Temporal Anchor Injection (Component 2) — y≈13.5
# =====================================================================
Y_anch = 13.2
rbox(10.5, Y_anch, 7.0, 1.5, 'Temporal Anchor Injection  (Component 2)',
     ORANGE_FILL, ORANGE_EDGE, fontsize=11,
     sublabel='anchor_proj  (Linear + GELU + LayerNorm)')

# Text embeddings → anchor injection
arr(CR, Y_temb, 14.0, Y_anch + 1.5)
note(CR + 0.8, Y_temb - 0.3, 'text\nembeddings', fs=8.5, color=DARK_TEXT, ha='left')

# Temporal refs → anchor injection (dashed, right side)
arr(17.0, Y_tref, 17.0, Y_anch + 1.5,
    color=ORANGE_EDGE, lw=1.4, ls='--')
note(17.9, Y_tref - 0.8, 'temporal\nreferences', fs=8, color=ORANGE_EDGE, fw='bold')

# Shared ATPE → anchor injection (curved arrow from left)
arr(8.2, Y_atpe + 0.75, 10.5, Y_anch + 1.1,
    color=ORANGE_EDGE, lw=1.5, connstyle='arc3,rad=-0.2')
note(9.1, Y_atpe - 0.0, 'shared ATPE\nembeddings', fs=8.5,
     color=ORANGE_EDGE, fw='bold')

# Annotations — shifted down to avoid overlap with keys/values label
note(14.0, Y_anch - 0.5,
     'text_embeds[pos] += anchor_embeds × gate.tanh()\n'
     'gate init = 0    no-op when no temporal refs',
     fs=7)
note(18.5, Y_anch + 0.75, '~260K\nparams', fs=8, color=ORANGE_EDGE, fw='bold', box=True)

# =====================================================================
#  Temporally-Biased Cross-Attention (Component 3) — y≈10.5
# =====================================================================
Y_xattn = 10.2
rbox(4.5, Y_xattn, 11.0, 1.7,
     'Temporally-Biased Cross-Attention  (Component 3)',
     ORANGE_FILL, ORANGE_EDGE, fontsize=12,
     sublabel='4 attention heads, LayerNorm on queries')

# ts patches (queries) → cross-attention
arr(CL, Y_proj, CL, 12.0, head=False)
arr(CL, 12.0, 7.0, 12.0, head=False)
arr(7.0, 12.0, 7.0, Y_xattn + 1.7)
note(4.2, 12.25, 'ts patches\n(queries)', fs=8.5, color=DARK_TEXT, fw='bold', ha='left')

# text embeddings (keys/values) → cross-attention
arr(14.0, Y_anch, 14.0, 12.3, head=False)
arr(14.0, 12.3, 13.0, 12.3, head=False)
arr(13.0, 12.3, 13.0, Y_xattn + 1.7)
note(13.3, 11.85, 'text embeddings\n(keys / values)', fs=8.5, color=DARK_TEXT, fw='bold', ha='left')

# Annotations
note(10.0, Y_xattn - 0.45,
     'bias(i, j) = −α · |t_ts − t_text|    (α learnable, init = 1.0)\n'
     'output = ts_embeds + gate.tanh() × cross_attn(norm(ts_embeds), text_embeds)',
     fs=7.5)
note(16.3, Y_xattn + 0.85, '~4M\nparams', fs=8, color=ORANGE_EDGE, fw='bold', box=True)

# =====================================================================
#  TPA dashed enclosure
# =====================================================================
# TPA dashed box: from just below cross-attention to just above encoder
tpa_top = Y_atpe + 1.5 + 0.3   # above ATPE
tpa_bot = Y_xattn - 0.7        # below cross-attention annotations
tpa_box = FancyBboxPatch((1.0, tpa_bot), 18.5,
                          tpa_top - tpa_bot,
                          boxstyle="round,pad=0.35",
                          facecolor='none', edgecolor=ORANGE_EDGE,
                          linewidth=2.2, linestyle=(0, (6, 4)),
                          alpha=0.45, zorder=1)
ax.add_patch(tpa_box)
note(2.0, tpa_bot + 0.15, 'TPA Module  (~4.3M total new params)',
     fs=10, color=ORANGE_EDGE, fw='bold', ha='left')

# =====================================================================
#  ROW — Concatenation  y≈7.8
# =====================================================================
Y_cat = 7.0
rbox(5.5, Y_cat, 9.0, 1.2, 'Concatenation', BLUE_FILL, BLUE_EDGE,
     fontsize=12,
     sublabel='[aligned ts embeds  ∥  anchor-injected text embeds]')

# Two outputs from cross-attention → concat
arr(8.0, Y_xattn, 8.0, Y_cat + 1.2)
note(6.3, 8.5, 'aligned\nts embeds', fs=8.5, color=DARK_TEXT)

arr(12.0, Y_xattn, 12.0, Y_cat + 1.2)
note(13.2, 8.5, 'injected\ntext embeds', fs=8.5, color=DARK_TEXT)

# =====================================================================
#  ROW — Frozen LLM  y≈5.5
# =====================================================================
Y_llm = 4.8
rbox(4.5, Y_llm, 11.0, 1.6, 'Frozen LLM', BLUE_FILL, BLUE_EDGE,
     fontsize=15, sublabel='~1B parameters (unchanged)')
arr(10.0, Y_cat, 10.0, Y_llm + 1.6)

# =====================================================================
#  ROW — Output  y≈3.5
# =====================================================================
Y_out = 3.0
rbox(5.5, Y_out, 9.0, 1.2, 'Text Generation Output', GREEN_FILL, GREEN_EDGE,
     fontsize=12, sublabel='time series description / analysis')
arr(10.0, Y_llm, 10.0, Y_out + 1.2)

# =====================================================================
#  LEGEND  y≈1.5
# =====================================================================
Y_leg = 0.8
leg_bg = FancyBboxPatch((2.5, Y_leg - 0.15), 15.5, 1.5,
                         boxstyle="round,pad=0.15",
                         facecolor='#FAFAFA', edgecolor='#BDC3C7',
                         linewidth=1.0, zorder=2)
ax.add_patch(leg_bg)
note(10.25, Y_leg + 1.15, 'Legend', fs=10, color=DARK_TEXT, fw='bold')

rbox(3.3, Y_leg + 0.15, 3.0, 0.6, 'Existing OpenTSLM',
     BLUE_FILL, BLUE_EDGE, fontsize=9, fontweight='normal')
rbox(7.0, Y_leg + 0.15, 3.0, 0.6, 'TPA Addition',
     ORANGE_FILL, ORANGE_EDGE, fontsize=9, fontweight='normal')
rbox(10.7, Y_leg + 0.15, 3.0, 0.6, 'Data / I/O',
     GREEN_FILL, GREEN_EDGE, fontsize=9, fontweight='normal')
note(15.8, Y_leg + 0.45, 'Total TPA: ~4.3M params',
     fs=9.5, color=ORANGE_EDGE, fw='bold')

# ── Save ────────────────────────────────────────────────────────────
fig.savefig('/home/wangni/notion-figures/alignment/fig_001.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.3)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/alignment/fig_001.png")
