#!/usr/bin/env python3
"""
Fig 004: Temporal Anchor Injection Mechanism
Shows how temporal references in text are detected, resolved to normalized
timestamps, encoded via ATPE, projected to LLM hidden dimension, and
additively injected into text embeddings with tanh gating.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ─── Color Palette ───────────────────────────────────────────────────
BG          = '#FFFFFF'
TEXT_TOKEN  = '#E8EDF2'
HIGHLIGHT   = '#FFF3CD'
HIGHLIGHT_B = '#D4A017'
ATPE_COLOR  = '#D6EAF8'
ATPE_BORDER = '#2E86C1'
PROJ_COLOR  = '#D5F5E3'
PROJ_BORDER = '#27AE60'
GATE_COLOR  = '#FADBD8'
GATE_BORDER = '#E74C3C'
INJECT_COLOR= '#E67E22'
ARROW_COLOR = '#2C3E50'
FORMULA_BG  = '#F8F9FA'
OUT_TOKEN   = '#E8F8F5'
OUT_BORDER  = '#1ABC9C'
RULE_COLOR  = '#EBE4F7'
RULE_BORDER = '#8E44AD'

fig, ax = plt.subplots(figsize=(18, 10.5), facecolor=BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 10.5)
ax.set_aspect('equal')
ax.axis('off')

# ─── Helper Functions ────────────────────────────────────────────────
def rbox(x, y, w, h, text, fc, ec, fontsize=10, fontweight='normal',
         lw=1.5, style='round,pad=0.1', zorder=3, alpha=1.0,
         linestyle='-', fontstyle='normal', text_color='#1a1a1a'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fc,
                         edgecolor=ec, linewidth=lw, zorder=zorder,
                         alpha=alpha, linestyle=linestyle)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color,
            zorder=zorder+1, fontstyle=fontstyle)
    return box

def darrow(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.5,
           style='->', connectionstyle='arc3,rad=0',
           shrinkA=3, shrinkB=3, zorder=5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                        lw=lw, zorder=zorder, connectionstyle=connectionstyle,
                        shrinkA=shrinkA, shrinkB=shrinkB, mutation_scale=15)
    ax.add_patch(a)

def cnum(x, y, num, color='#2C3E50'):
    c = plt.Circle((x, y), 0.22, facecolor=color, edgecolor='white',
                   linewidth=1.5, zorder=8)
    ax.add_patch(c)
    ax.text(x, y, str(num), ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=9)

# ─── Title ───────────────────────────────────────────────────────────
ax.text(9, 10.15, 'Temporal Anchor Injection Mechanism',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color='#1a1a1a')
ax.text(9, 9.75, 'Component 2 of Temporal Positional Alignment (TPA)',
        ha='center', va='center', fontsize=11, fontstyle='italic',
        color='#666666')

# ═══════════════════════════════════════════════════════════════════════
# ROW 1 (y~8.5): Input text tokens
# ═══════════════════════════════════════════════════════════════════════
tokens = ['The', 'trend', 'in', 'the', 'first period', 'shows', 'growth']
tw, th, gap = 1.6, 0.58, 0.14
total_w = len(tokens) * tw + (len(tokens)-1) * gap
sx = (18 - total_w) / 2
ty = 8.7

ax.text(sx - 0.15, ty + th/2, 'Input Text\nTokens',
        ha='right', va='center', fontsize=9.5, color='#555',
        fontweight='bold')

tpos = []
for i, tok in enumerate(tokens):
    x = sx + i * (tw + gap)
    if tok == 'first period':
        rbox(x, ty, tw, th, tok, HIGHLIGHT, HIGHLIGHT_B, fontsize=10.5,
             fontweight='bold', lw=2.5)
    else:
        rbox(x, ty, tw, th, tok, TEXT_TOKEN, '#B0BEC5', fontsize=10.5, lw=1.2)
    tpos.append(x + tw/2)

hl_cx = tpos[4]  # center of "first period"

# ═══════════════════════════════════════════════════════════════════════
# TWO PATHS diverge from "first period" token:
#   LEFT PATH: temporal signal (steps 1-5, the anchor injection path)
#   RIGHT PATH: original text embedding (straight down to ⊕)
# ═══════════════════════════════════════════════════════════════════════

# --- PATH LABELS ---
# Left path label
ax.text(hl_cx - 2.0, 8.0, 'Temporal signal path',
        ha='center', va='center', fontsize=8.5, color=RULE_BORDER,
        fontstyle='italic', fontweight='bold')

# Right path label
ax.text(hl_cx + 3.5, 8.0, 'Original embedding path',
        ha='center', va='center', fontsize=8.5, color='#7F8C8D',
        fontstyle='italic', fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════
# LEFT PATH: Temporal anchor injection
# ═══════════════════════════════════════════════════════════════════════

# Step 1: Detection
cnum(hl_cx - 0.6, 8.05, 1, RULE_BORDER)
ax.annotate('temporal reference\ndetected',
            xy=(hl_cx - 0.3, ty), xytext=(hl_cx - 2.0, 7.55),
            ha='center', va='top', fontsize=8.5, color=HIGHLIGHT_B,
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=HIGHLIGHT_B, lw=1.5))

# Step 2: Rule-Based Extraction
rule_x, rule_y = 2.5, 6.6
rule_w, rule_h = 3.2, 0.65
rbox(rule_x, rule_y, rule_w, rule_h, 'Rule-Based Extraction',
     RULE_COLOR, RULE_BORDER, fontsize=10.5, fontweight='bold', lw=1.8)

darrow(hl_cx - 2.0, 7.5, rule_x + rule_w/2, rule_y + rule_h,
       color=RULE_BORDER, lw=1.5)
cnum((rule_x + rule_w + 6.8)/2, rule_y + rule_h + 0.22, 2, '#D4AC0D')

# Step 3: Normalized Timestamp
ts_x, ts_y = 6.8, 6.6
ts_w, ts_h = 2.4, 0.65
rbox(ts_x, ts_y, ts_w, ts_h, 't ∈ [0, 0.5]',
     '#FFF9E6', '#D4AC0D', fontsize=12, fontweight='bold', lw=1.8)

darrow(rule_x + rule_w, rule_y + rule_h/2,
       ts_x, ts_y + ts_h/2, color=ARROW_COLOR, lw=1.5)

# Label on arrow
ax.text((rule_x + rule_w + ts_x)/2, rule_y + rule_h/2 - 0.22,
        '"first period"', fontsize=8, ha='center', va='top',
        color='#888', fontstyle='italic')

# Step 4: ATPE Module
atpe_x, atpe_y = 6.4, 5.1
atpe_w, atpe_h = 3.2, 0.95
rbox(atpe_x, atpe_y, atpe_w, atpe_h, 'ATPE\n(Shared Module)',
     ATPE_COLOR, ATPE_BORDER, fontsize=10.5, fontweight='bold', lw=2.2)

# Notes around ATPE — placed to the right to avoid left-side annotations
ax.text(atpe_x + atpe_w + 0.55, atpe_y + 0.2,
        'sin/cos multi-scale\n+ time_proj',
        ha='left', va='center', fontsize=7.5, color=ATPE_BORDER,
        fontstyle='italic')

ax.text(atpe_x + atpe_w + 0.55, atpe_y + atpe_h - 0.1,
        'shared with encoder\n(Component 1)',
        ha='left', va='top', fontsize=7.5, color=ATPE_BORDER,
        fontstyle='italic', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.08', facecolor='#EBF5FB',
                  edgecolor=ATPE_BORDER, lw=0.6, linestyle='--'))

# Arrow: timestamp → ATPE
darrow(ts_x + ts_w/2, ts_y,
       atpe_x + atpe_w/2, atpe_y + atpe_h,
       color=ATPE_BORDER, lw=1.8)
cnum(atpe_x + atpe_w + 0.3, atpe_y + atpe_h - 0.2, 3, ATPE_BORDER)

# Step 5: anchor_proj
proj_x, proj_y = 6.4, 3.5
proj_w, proj_h = 3.2, 0.75
rbox(proj_x, proj_y, proj_w, proj_h, 'anchor_proj',
     PROJ_COLOR, PROJ_BORDER, fontsize=12, fontweight='bold', lw=2.2)

# Note under anchor_proj
ax.text(proj_x + proj_w/2, proj_y - 0.12,
        'Linear → GELU → LayerNorm',
        ha='center', va='top', fontsize=8, color=PROJ_BORDER,
        fontstyle='italic')

# Arrow: ATPE → anchor_proj
darrow(atpe_x + atpe_w/2, atpe_y,
       proj_x + proj_w/2, proj_y + proj_h,
       color=ARROW_COLOR, lw=1.8)
cnum(proj_x + proj_w + 0.3, proj_y + proj_h - 0.15, 4, PROJ_BORDER)

# Label: dimension transform
ax.text(proj_x + proj_w + 0.15, proj_y + proj_h + 0.15,
        'd_model → d_LLM', fontsize=8, ha='left', va='center',
        color='#555', fontstyle='italic')

# Step 6: tanh Gate (× multiply)
gate_x, gate_y = 6.4, 2.0
gate_w, gate_h = 3.2, 0.7
rbox(gate_x, gate_y, gate_w, gate_h, 'tanh Gate  (×)',
     GATE_COLOR, GATE_BORDER, fontsize=11, fontweight='bold', lw=2.2)

# Note under gate
ax.text(gate_x + gate_w/2, gate_y - 0.12,
        'gate init = 0 (no-op at training start)',
        ha='center', va='top', fontsize=8.5, color=GATE_BORDER,
        fontstyle='italic')

# Arrow: anchor_proj → gate
darrow(proj_x + proj_w/2, proj_y,
       gate_x + gate_w/2, gate_y + gate_h,
       color=ARROW_COLOR, lw=1.8)
cnum(gate_x + gate_w + 0.3, gate_y + gate_h - 0.1, 5, GATE_BORDER)

# ═══════════════════════════════════════════════════════════════════════
# RIGHT PATH: text_embeds straight down
# ═══════════════════════════════════════════════════════════════════════
emb_cx = hl_cx + 3.5
emb_y = 7.2
emb_w, emb_h = 2.6, 0.55
rbox(emb_cx - emb_w/2, emb_y, emb_w, emb_h, 'text_embeds[pos]',
     TEXT_TOKEN, '#7F8C8D', fontsize=10, fontweight='bold', lw=1.5)

ax.text(emb_cx, emb_y + emb_h + 0.1,
        'frozen LLM embedding of "first period"',
        ha='center', va='bottom', fontsize=8, color='#777',
        fontstyle='italic')

# Arrow from highlighted token to text_embeds
darrow(hl_cx + tw/2, ty + th/2,
       emb_cx - emb_w/2, emb_y + emb_h/2,
       color='#95A5A6', lw=1.3, connectionstyle='arc3,rad=-0.2')

# ═══════════════════════════════════════════════════════════════════════
# ⊕ Additive injection (where both paths converge)
# ═══════════════════════════════════════════════════════════════════════
add_cx = emb_cx
add_cy = 2.35
circle_add = plt.Circle((add_cx, add_cy), 0.38, facecolor='#FEF9E7',
                         edgecolor=INJECT_COLOR, linewidth=3.0, zorder=6)
ax.add_patch(circle_add)
ax.text(add_cx, add_cy, '⊕', ha='center', va='center', fontsize=22,
        fontweight='bold', color=INJECT_COLOR, zorder=7)
cnum(add_cx + 0.55, add_cy + 0.3, 6, INJECT_COLOR)

# Arrow: text_embeds → ⊕ (vertical down, right side)
darrow(emb_cx, emb_y, emb_cx, add_cy + 0.42,
       color='#7F8C8D', lw=2.0)

# Arrow: gate → ⊕ (horizontal right)
darrow(gate_x + gate_w, gate_y + gate_h/2,
       add_cx - 0.42, add_cy,
       color=INJECT_COLOR, lw=2.2)

# Label on the horizontal arrow
ax.text((gate_x + gate_w + add_cx - 0.42)/2, gate_y + gate_h/2 + 0.2,
        'gated temporal signal',
        ha='center', va='bottom', fontsize=8, color=INJECT_COLOR,
        fontstyle='italic', fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: Modified Embedding
# ═══════════════════════════════════════════════════════════════════════
out_w, out_h = 2.6, 0.6
out_x = add_cx - out_w/2
out_y = 0.9
rbox(out_x, out_y, out_w, out_h, 'Modified Embedding',
     OUT_TOKEN, OUT_BORDER, fontsize=11, fontweight='bold', lw=2.2)

# Arrow: ⊕ → output
darrow(add_cx, add_cy - 0.42,
       add_cx, out_y + out_h, color=OUT_BORDER, lw=2.2)

# ═══════════════════════════════════════════════════════════════════════
# LEFT SIDE ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════

# Template mapping rules box
mrx, mry = 0.3, 6.4
mrw, mrh = 2.0, 1.25
box_r = FancyBboxPatch((mrx, mry), mrw, mrh, boxstyle='round,pad=0.1',
                        facecolor=RULE_COLOR, edgecolor=RULE_BORDER,
                        linewidth=1.0, zorder=3, alpha=0.5, linestyle='--')
ax.add_patch(box_r)
ax.text(mrx + 0.12, mry + mrh - 0.1,
        'Template Mappings:', fontsize=8.5, fontweight='bold',
        color=RULE_BORDER, va='top', zorder=4)
for i, m in enumerate(['"first period" → [0, 0.5]',
                        '"second period" → [0.5, 1.0]',
                        '"over time" → 0.5',
                        '"final probability" → 1.0']):
    ax.text(mrx + 0.15, mry + mrh - 0.42 - i * 0.22, m,
            fontsize=7.5, color='#555', va='center', zorder=4,
            family='monospace')

# Concrete Example box
ex_x, ex_y = 0.3, 3.4
ex_w, ex_h = 5.3, 2.6
box_ex = FancyBboxPatch((ex_x, ex_y), ex_w, ex_h,
                         boxstyle='round,pad=0.12', facecolor='#FDFEFE',
                         edgecolor='#D5D8DC', linewidth=1.2, zorder=3)
ax.add_patch(box_ex)
ax.text(ex_x + 0.2, ex_y + ex_h - 0.15,
        'Concrete Example Flow:', fontsize=10, fontweight='bold',
        color='#333', va='top', zorder=4)

steps = [
    ('1.', '"first period"  →  rule-based extraction'),
    ('2.', 't ∈ [0, 0.5]  →  ATPE encoding (sinusoidal)'),
    ('3.', 'ATPE output  →  anchor_proj (d_model → d_LLM)'),
    ('4.', 'anchor_embed  ×  gate.tanh()  →  gated signal'),
    ('5.', 'text_embeds[pos]  +=  gated signal'),
    ('', ''),
    ('', '→  Modified Embedding'),
]
for i, (num, desc) in enumerate(steps):
    yy = ex_y + ex_h - 0.58 - i * 0.3
    if num:
        ax.text(ex_x + 0.25, yy, num, fontsize=9, color=ARROW_COLOR,
                va='center', zorder=4, fontweight='bold')
    ax.text(ex_x + 0.58, yy, desc, fontsize=9, color='#444',
            va='center', zorder=4, family='monospace')

# Formula box
fx, fy = 0.3, 1.5
fw, fh = 5.3, 1.6
box_f = FancyBboxPatch((fx, fy), fw, fh, boxstyle='round,pad=0.12',
                        facecolor=FORMULA_BG, edgecolor='#AEB6BF',
                        linewidth=1.5, zorder=3, linestyle='--')
ax.add_patch(box_f)
ax.text(fx + 0.2, fy + fh - 0.18,
        'Gating Formula:', fontsize=10, fontweight='bold',
        color='#333', va='top', zorder=4)
ax.text(fx + fw/2, fy + fh/2 - 0.15,
        'text_embeds[pos] +=\n    anchor_embeds[i] * gate.tanh()',
        fontsize=11, color='#1a1a1a', va='center', ha='center',
        zorder=4, family='monospace', fontweight='bold')

# Parameter count
ax.text(fx + fw/2, fy - 0.12,
        '~260K params total (anchor_proj + gate)',
        ha='center', va='top', fontsize=9, color='#888',
        fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#f8f8f8',
                  edgecolor='#ddd', lw=0.8))

# Graceful no-op note
ax.text(0.3, 0.7,
        'When no temporal references exist in text, no anchors are injected (graceful no-op).',
        fontsize=8.5, color='#999', fontstyle='italic', va='top')

# ═══════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.3)
plt.savefig('/home/wangni/notion-figures/alignment/fig_004.png',
            dpi=200, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print('Saved: /home/wangni/notion-figures/alignment/fig_004.png')
