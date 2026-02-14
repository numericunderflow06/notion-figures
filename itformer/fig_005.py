"""
fig_005: OpenTSLM-ITA End-to-End Pipeline
Horizontal flow diagram showing the complete data pipeline with component provenance.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette for provenance ──
# Blue: ITFormer-derived components
CLR_ITFORMER      = '#3B82F6'   # blue-500
CLR_ITFORMER_LIGHT = '#DBEAFE'  # blue-100
CLR_ITFORMER_MED   = '#93C5FD'  # blue-300

# Orange: OpenTSLM-Flamingo-derived components
CLR_FLAMINGO       = '#F97316'  # orange-500
CLR_FLAMINGO_LIGHT = '#FFF7ED'  # orange-50
CLR_FLAMINGO_MED   = '#FDBA74'  # orange-300

# Green: Novel hybrid integration points
CLR_HYBRID         = '#10B981'  # emerald-500
CLR_HYBRID_LIGHT   = '#D1FAE5'  # emerald-100
CLR_HYBRID_MED     = '#6EE7B7'  # emerald-300

CLR_TEXT           = '#1E293B'  # slate-800
CLR_SUBTEXT        = '#475569'  # slate-600
CLR_ARROW          = '#64748B'  # slate-500
CLR_DATA_SHAPE     = '#7C3AED'  # violet-600

fig, ax = plt.subplots(figsize=(22, 9.5), dpi=200)
ax.set_xlim(0, 22)
ax.set_ylim(0, 9.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Helper functions ──

def draw_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.8, radius=0.15, zorder=2):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=CLR_ARROW, lw=1.8, style='->', zorder=3):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=16,
        color=color,
        linewidth=lw,
        zorder=zorder,
        connectionstyle="arc3,rad=0"
    )
    ax.add_patch(arrow)

def text(ax, x, y, s, fontsize=10, fontweight='normal', color=CLR_TEXT, ha='center', va='center', **kw):
    return ax.text(x, y, s, fontsize=fontsize, fontweight=fontweight, color=color,
                   ha=ha, va=va, zorder=5, **kw)

def math_text(ax, x, y, s, fontsize=9, color=CLR_DATA_SHAPE, ha='center', va='center'):
    return ax.text(x, y, s, fontsize=fontsize, color=color, ha=ha, va=va, zorder=5,
                   fontstyle='italic')

# ── Title ──
text(ax, 11, 9.05, 'OpenTSLM-ITA: End-to-End Pipeline',
     fontsize=16, fontweight='bold', color=CLR_TEXT)

# ── Layout constants ──
# Main pipeline boxes y-center
Y_MID = 4.8
BOX_H = 3.2
BOX_TOP = Y_MID + BOX_H / 2
BOX_BOT = Y_MID - BOX_H / 2

# ============================================================
# BLOCK 0: Raw Input
# ============================================================
b0_x, b0_w = 0.3, 2.4
draw_box(ax, b0_x, BOX_BOT + 0.4, b0_w, BOX_H - 0.8, '#F8FAFC', '#94A3B8', linewidth=1.5)
text(ax, b0_x + b0_w/2, Y_MID + 0.5, 'Raw Multivariate', fontsize=10.5, fontweight='bold')
text(ax, b0_x + b0_w/2, Y_MID, 'Time Series', fontsize=10.5, fontweight='bold')
# Draw a small waveform icon
wave_x = np.linspace(b0_x + 0.4, b0_x + b0_w - 0.4, 60)
for i, offset in enumerate([0.4, 0.0, -0.4]):
    wave_y = Y_MID - 0.9 + offset + 0.18 * np.sin(2 * np.pi * (3 + i) * (wave_x - b0_x) / b0_w)
    colors_wave = ['#3B82F6', '#F97316', '#10B981']
    ax.plot(wave_x, wave_y, color=colors_wave[i], linewidth=1.2, alpha=0.7, zorder=4)

math_text(ax, b0_x + b0_w/2, BOX_BOT + 0.15,
          r'$\mathbf{X} \in \mathbb{R}^{V \times L}$', fontsize=9.5)

# ============================================================
# BLOCK 1: Patch Encoder + TPE (Hybrid: Conv1D from Flamingo, TPE from ITFormer)
# ============================================================
b1_x, b1_w = 3.5, 3.6
draw_box(ax, b1_x, BOX_BOT, b1_w, BOX_H, CLR_HYBRID_LIGHT, CLR_HYBRID, linewidth=2.0)

text(ax, b1_x + b1_w/2, BOX_TOP - 0.32, 'Patch Encoder + Hierarchical TPE',
     fontsize=10.5, fontweight='bold', color=CLR_HYBRID)

# Sub-box: Conv1D (Flamingo-derived)
sb1a_x, sb1a_y, sb1a_w, sb1a_h = b1_x + 0.2, Y_MID + 0.15, b1_w - 0.4, 0.95
draw_box(ax, sb1a_x, sb1a_y, sb1a_w, sb1a_h, CLR_FLAMINGO_LIGHT, CLR_FLAMINGO, linewidth=1.3, radius=0.1)
text(ax, sb1a_x + sb1a_w/2, sb1a_y + sb1a_h/2 + 0.15, 'Conv1D Patch Encoder', fontsize=9.5, fontweight='bold', color=CLR_FLAMINGO)
text(ax, sb1a_x + sb1a_w/2, sb1a_y + sb1a_h/2 - 0.2, '(kernel = stride = p)', fontsize=8.5, color=CLR_SUBTEXT)

# Sub-box: TPE (ITFormer-derived)
sb1b_x, sb1b_y, sb1b_w, sb1b_h = b1_x + 0.2, BOX_BOT + 0.35, b1_w - 0.4, 1.55
draw_box(ax, sb1b_x, sb1b_y, sb1b_w, sb1b_h, CLR_ITFORMER_LIGHT, CLR_ITFORMER, linewidth=1.3, radius=0.1)
text(ax, sb1b_x + sb1b_w/2, sb1b_y + sb1b_h - 0.25, 'Three-Level TPE', fontsize=9.5, fontweight='bold', color=CLR_ITFORMER)
text(ax, sb1b_x + sb1b_w/2, sb1b_y + sb1b_h/2 - 0.0,
     r'$P_{\mathrm{time}}(n)$: Sinusoidal', fontsize=8.5, color=CLR_SUBTEXT)
text(ax, sb1b_x + sb1b_w/2, sb1b_y + sb1b_h/2 - 0.35,
     r'$P_{\mathrm{chan}}(v)$: Learnable', fontsize=8.5, color=CLR_SUBTEXT)
text(ax, sb1b_x + sb1b_w/2, sb1b_y + sb1b_h/2 - 0.70,
     r'$P_{\mathrm{seg}}(s)$: RoPE', fontsize=8.5, color=CLR_SUBTEXT)

# ============================================================
# BLOCK 2: Instruction-Guided Resampler (IGR) — ITFormer-derived, core contribution
# ============================================================
b2_x, b2_w = 7.9, 4.6
draw_box(ax, b2_x, BOX_BOT, b2_w, BOX_H, CLR_ITFORMER_LIGHT, CLR_ITFORMER, linewidth=2.0)

text(ax, b2_x + b2_w/2, BOX_TOP - 0.32, 'Instruction-Guided Resampler (IGR)',
     fontsize=10.5, fontweight='bold', color=CLR_ITFORMER)

# Sub-box: LIT
sb2a_w, sb2a_h = 1.2, 0.75
sb2a_x, sb2a_y = b2_x + 0.2, Y_MID + 0.3
draw_box(ax, sb2a_x, sb2a_y, sb2a_w, sb2a_h, '#EFF6FF', CLR_ITFORMER_MED, linewidth=1.2, radius=0.08)
text(ax, sb2a_x + sb2a_w/2, sb2a_y + sb2a_h/2 + 0.12, 'LIT', fontsize=9, fontweight='bold', color=CLR_ITFORMER)
text(ax, sb2a_x + sb2a_w/2, sb2a_y + sb2a_h/2 - 0.16, 'M=25', fontsize=8, color=CLR_SUBTEXT)

# Sub-box: Query Conditioning
sb2b_w, sb2b_h = 2.8, 0.75
sb2b_x, sb2b_y = b2_x + 1.6, Y_MID + 0.3
draw_box(ax, sb2b_x, sb2b_y, sb2b_w, sb2b_h, '#EFF6FF', CLR_ITFORMER_MED, linewidth=1.2, radius=0.08)
text(ax, sb2b_x + sb2b_w/2, sb2b_y + sb2b_h/2 + 0.12, 'Query Conditioning', fontsize=9, fontweight='bold', color=CLR_ITFORMER)
text(ax, sb2b_x + sb2b_w/2, sb2b_y + sb2b_h/2 - 0.16, r'Self-Attn $[I; Q_{\mathrm{text}}]$', fontsize=8, color=CLR_SUBTEXT)

# Arrow from LIT to Query Conditioning
draw_arrow(ax, sb2a_x + sb2a_w, sb2a_y + sb2a_h/2, sb2b_x, sb2b_y + sb2b_h/2, color=CLR_ITFORMER_MED, lw=1.3)

# Sub-box: Two-Stage ITA
sb2c_w, sb2c_h = b2_w - 0.4, 1.35
sb2c_x, sb2c_y = b2_x + 0.2, BOX_BOT + 0.35
draw_box(ax, sb2c_x, sb2c_y, sb2c_w, sb2c_h, '#EFF6FF', CLR_ITFORMER_MED, linewidth=1.2, radius=0.08)
text(ax, sb2c_x + sb2c_w/2, sb2c_y + sb2c_h - 0.22, 'Two-Stage ITA', fontsize=9.5, fontweight='bold', color=CLR_ITFORMER)
text(ax, sb2c_x + sb2c_w/2, sb2c_y + sb2c_h/2 - 0.0,
     'Stage A: Channel Instruct Fusing', fontsize=8.5, color=CLR_SUBTEXT)
text(ax, sb2c_x + sb2c_w/2, sb2c_y + sb2c_h/2 - 0.35,
     'Stage B: Time Instruct Compressing', fontsize=8.5, color=CLR_SUBTEXT)

# Arrow from Query Conditioning down to ITA
qc_mid_x = sb2b_x + sb2b_w / 2
draw_arrow(ax, qc_mid_x, sb2b_y, qc_mid_x, sb2c_y + sb2c_h, color=CLR_ITFORMER_MED, lw=1.3)
# Label: conditioned I'
text(ax, qc_mid_x + 0.65, (sb2b_y + sb2c_y + sb2c_h)/2, r"$I'$", fontsize=9, color=CLR_ITFORMER)

# ============================================================
# BLOCK 2.5: Latent Vectors (intermediate output)
# ============================================================
b25_x, b25_w = 13.2, 1.8
b25_h = 1.6
b25_y = Y_MID - b25_h / 2
draw_box(ax, b25_x, b25_y, b25_w, b25_h, CLR_HYBRID_LIGHT, CLR_HYBRID, linewidth=1.8, radius=0.12)
text(ax, b25_x + b25_w/2, b25_y + b25_h/2 + 0.2, 'M = 25', fontsize=10.5, fontweight='bold', color=CLR_HYBRID)
text(ax, b25_x + b25_w/2, b25_y + b25_h/2 - 0.2, 'Latent Vectors', fontsize=10, color=CLR_HYBRID)

# ============================================================
# BLOCK 3: Gated Cross-Attention in Frozen LLM (Flamingo-derived)
# ============================================================
b3_x, b3_w = 15.7, 3.6
draw_box(ax, b3_x, BOX_BOT, b3_w, BOX_H, CLR_FLAMINGO_LIGHT, CLR_FLAMINGO, linewidth=2.0)

text(ax, b3_x + b3_w/2, BOX_TOP - 0.32, 'Frozen LLM (Llama-3.2)',
     fontsize=10.5, fontweight='bold', color=CLR_FLAMINGO)

# Sub-box: Gated Cross-Attention
sb3a_w, sb3a_h = b3_w - 0.4, 1.1
sb3a_x, sb3a_y = b3_x + 0.2, Y_MID + 0.15
draw_box(ax, sb3a_x, sb3a_y, sb3a_w, sb3a_h, '#FFF7ED', CLR_FLAMINGO_MED, linewidth=1.2, radius=0.08)
text(ax, sb3a_x + sb3a_w/2, sb3a_y + sb3a_h/2 + 0.15, 'Gated Cross-Attention', fontsize=9.5, fontweight='bold', color=CLR_FLAMINGO)
text(ax, sb3a_x + sb3a_w/2, sb3a_y + sb3a_h/2 - 0.18, 'every K = 4 blocks', fontsize=8.5, color=CLR_SUBTEXT)

# Sub-box: LLM Transformer Blocks
sb3b_w, sb3b_h = b3_w - 0.4, 1.1
sb3b_x, sb3b_y = b3_x + 0.2, BOX_BOT + 0.5
draw_box(ax, sb3b_x, sb3b_y, sb3b_w, sb3b_h, '#FFF7ED', CLR_FLAMINGO_MED, linewidth=1.2, radius=0.08)
text(ax, sb3b_x + sb3b_w/2, sb3b_y + sb3b_h/2 + 0.15, 'LLM Transformer Blocks', fontsize=9.5, fontweight='bold', color=CLR_FLAMINGO)
text(ax, sb3b_x + sb3b_w/2, sb3b_y + sb3b_h/2 - 0.18, '(frozen weights)', fontsize=8.5, color=CLR_SUBTEXT)

# Bidirectional arrow between GCA and LLM blocks
gca_mid = sb3a_x + sb3a_w/2
draw_arrow(ax, gca_mid - 0.3, sb3a_y, gca_mid - 0.3, sb3b_y + sb3b_h, color=CLR_FLAMINGO_MED, lw=1.2)
draw_arrow(ax, gca_mid + 0.3, sb3b_y + sb3b_h, gca_mid + 0.3, sb3a_y, color=CLR_FLAMINGO_MED, lw=1.2)

# ============================================================
# BLOCK 4: Chain-of-Thought Output
# ============================================================
b4_x, b4_w = 20.1, 1.6
b4_h = 2.0
b4_y = Y_MID - b4_h / 2
draw_box(ax, b4_x, b4_y, b4_w, b4_h, '#F0FDF4', '#22C55E', linewidth=1.8, radius=0.12)
text(ax, b4_x + b4_w/2, b4_y + b4_h/2 + 0.35, 'Chain-of-', fontsize=10.5, fontweight='bold', color='#15803D')
text(ax, b4_x + b4_w/2, b4_y + b4_h/2 - 0.0, 'Thought', fontsize=10.5, fontweight='bold', color='#15803D')
text(ax, b4_x + b4_w/2, b4_y + b4_h/2 - 0.35, 'Output', fontsize=10.5, fontweight='bold', color='#15803D')

# ============================================================
# Text Instruction Input (top, feeding into IGR)
# ============================================================
ti_x, ti_w = 9.5, 2.2
ti_h = 0.9
ti_y = BOX_TOP + 0.65
draw_box(ax, ti_x, ti_y, ti_w, ti_h, '#F5F3FF', '#8B5CF6', linewidth=1.5, radius=0.1)
text(ax, ti_x + ti_w/2, ti_y + ti_h/2 + 0.1, 'Text Instruction', fontsize=9.5, fontweight='bold', color='#7C3AED')
text(ax, ti_x + ti_w/2, ti_y + ti_h/2 - 0.2, '(Query)', fontsize=8.5, color='#7C3AED')

# Arrow from instruction down to IGR (Query Conditioning)
draw_arrow(ax, ti_x + ti_w/2, ti_y, ti_x + ti_w/2, BOX_TOP, color='#8B5CF6', lw=1.8)
text(ax, ti_x + ti_w/2 + 0.55, (ti_y + BOX_TOP)/2, r'$Q_{\mathrm{text}}$', fontsize=9, color='#7C3AED')

# ============================================================
# ARROWS between main blocks
# ============================================================

# Arrow 0 → 1: Raw → Patch Encoder
arr_y = Y_MID
draw_arrow(ax, b0_x + b0_w, arr_y, b1_x, arr_y, lw=2.0)
math_text(ax, (b0_x + b0_w + b1_x)/2, arr_y + 0.3,
          r'$\mathbb{R}^{V \times L}$', fontsize=8.5)

# Arrow 1 → 2: Patch Encoder → IGR
draw_arrow(ax, b1_x + b1_w, arr_y, b2_x, arr_y, lw=2.0)
math_text(ax, (b1_x + b1_w + b2_x)/2, arr_y + 0.5,
          r'$H \in \mathbb{R}^{(V \!\cdot\! N) \times d}$', fontsize=8.5)

# Arrow 2 → 2.5: IGR → Latent Vectors
draw_arrow(ax, b2_x + b2_w, arr_y, b25_x, arr_y, lw=2.0)
math_text(ax, (b2_x + b2_w + b25_x)/2, arr_y + 0.35,
          r'$\mathbb{R}^{M \times d}$', fontsize=8.5)

# Arrow 2.5 → 3: Latent Vectors → Frozen LLM
draw_arrow(ax, b25_x + b25_w, arr_y, b3_x, arr_y, lw=2.0)
math_text(ax, (b25_x + b25_w + b3_x)/2, arr_y + 0.35,
          r'$\mathbb{R}^{M \times d_{\mathrm{llm}}}$', fontsize=8.5)

# Arrow 3 → 4: LLM → CoT
draw_arrow(ax, b3_x + b3_w, arr_y, b4_x, arr_y, lw=2.0)

# ============================================================
# LEGEND (bottom)
# ============================================================
legend_y = 0.7
legend_items = [
    (CLR_ITFORMER_LIGHT, CLR_ITFORMER,  'ITFormer-Derived (TPE, IGR)'),
    (CLR_FLAMINGO_LIGHT, CLR_FLAMINGO,  'OpenTSLM-Flamingo-Derived (Gated Cross-Attn, CoT)'),
    (CLR_HYBRID_LIGHT,   CLR_HYBRID,    'Hybrid Integration (novel combination)'),
]
legend_x_start = 4.0
for i, (fc, ec, label) in enumerate(legend_items):
    lx = legend_x_start + i * 6.2
    draw_box(ax, lx, legend_y - 0.22, 0.5, 0.44, fc, ec, linewidth=1.5, radius=0.06)
    text(ax, lx + 0.75, legend_y, label, fontsize=9, ha='left', color=CLR_TEXT, fontweight='medium')

# ── Linear projection annotation between latent vectors and LLM ──
proj_x = (b25_x + b25_w + b3_x) / 2
text(ax, proj_x, arr_y - 0.35, 'Linear Proj.', fontsize=8.5, color=CLR_SUBTEXT, ha='center')

# ── Save ──
plt.tight_layout(pad=0.3)
plt.savefig('/home/wangni/notion-figures/itformer/fig_005.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved fig_005.png")
