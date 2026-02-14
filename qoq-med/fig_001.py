"""
JEPA-Flamingo-DRPO Architecture Diagram (fig_001)
End-to-end: Raw ECG → ECG-JEPA → Linear Proj → Perceiver Resampler
            → Gated Cross-Attention in LLM → Text Output
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
FROZEN_BG   = "#DAEAF6"   # light blue fill  (frozen)
FROZEN_BD   = "#2471A3"   # blue border       (frozen)
TRAIN_BG    = "#FDE8D0"   # light orange fill (trainable)
TRAIN_BD    = "#D35400"   # orange border     (trainable)
SIGNAL_CLR  = "#2C3E50"   # dark slate for signal
ARROW_CLR   = "#444444"   # arrow colour
TEXT_CLR    = "#1B1B1B"   # main text
DIM_CLR     = "#6C7A89"   # dimension annotations
LLM_BG      = "#EAF7EA"   # LLM outer bg
LLM_BD      = "#1E8449"   # LLM outer border
BLK_BG      = "#C8E6C9"   # LLM inner block bg
BLK_BD      = "#2E7D32"   # LLM inner block border
TOKEN_BG    = "#F3E5F5"   # token stream bg
TOKEN_BD    = "#7B1FA2"   # token stream border
GATE_BG     = "#FFFDE7"   # gate callout bg
GATE_BD     = "#F9A825"   # gate callout border
OUTPUT_BG   = "#E0F2F1"   # output bg
OUTPUT_BD   = "#00897B"   # output border
NEUTRAL_BG  = "#F5F5F5"
NEUTRAL_BD  = "#90A4AE"
WHITE       = "#FFFFFF"

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(28, 13))
ax.set_xlim(-1.5, 27)
ax.set_ylim(-1.0, 12.2)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)

# ── Helpers ─────────────────────────────────────────────────────────
def rbox(x, y, w, h, bg, bd, lw=2.0, pad=0.15, ls='-', zorder=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad}",
                       facecolor=bg, edgecolor=bd, linewidth=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    return p

def txt(x, y, s, fs=11, fw='normal', fc=TEXT_CLR, ha='center', va='center',
        zorder=5, style='normal', **kw):
    return ax.text(x, y, s, fontsize=fs, fontweight=fw, color=fc,
                   ha=ha, va=va, zorder=zorder, fontstyle=style, **kw)

def arr(x1, y1, x2, y2, color=ARROW_CLR, lw=2.2, style='->', cs=None):
    props = dict(arrowstyle=style, color=color, lw=lw)
    if cs:
        props['connectionstyle'] = cs
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=2)

def dim(x, y, s, fs=8.5):
    txt(x, y, s, fs=fs, fc=DIM_CLR, style='italic',
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))

def badge(x, y, s, bg, bd, fs=8):
    txt(x, y, s, fs=fs, fw='bold', fc=bd,
        bbox=dict(boxstyle='round,pad=0.18', fc=bg, ec=bd, lw=1.2))

def snow(x, y, size=13):
    txt(x, y, '\u2744', fs=size, fc=FROZEN_BD, fontfamily='DejaVu Sans', zorder=6)

# ── Layout constants ────────────────────────────────────────────────
Y_TOP   = 6.2     # main pipeline y-centre
Y_TXT   = 1.7     # text-input row y-centre
BH      = 1.4     # standard box height
BH_SM   = 1.0

# ====================================================================
# TITLE
# ====================================================================
txt(12.8, 11.7, "JEPA-Flamingo-DRPO Architecture", fs=20, fw='bold')
txt(12.8, 11.15, "End-to-end data flow:  Raw ECG  \u2192  Multimodal Fusion  \u2192  Clinical Text Output",
    fs=12, fc=DIM_CLR)

# ====================================================================
# 1  RAW ECG SIGNAL
# ====================================================================
rx, ry, rw, rh = -0.8, Y_TOP - BH/2, 2.2, BH
rbox(rx, ry, rw, rh, NEUTRAL_BG, NEUTRAL_BD)
# Tiny ECG waveform inside
t = np.linspace(0, 4*np.pi, 200)
ecg = (0.25*np.sin(t)
       + np.where((t > 4.5) & (t < 5.0), 0.6, 0)
       + np.where((t > 5.0) & (t < 5.4), -0.2, 0)
       + np.where((t > 10.5) & (t < 11.0), 0.6, 0)
       + np.where((t > 11.0) & (t < 11.4), -0.2, 0))
ax.plot(np.linspace(rx+0.25, rx+rw-0.25, len(ecg)),
        ecg + Y_TOP + 0.22, color=SIGNAL_CLR, lw=1.3, zorder=4, clip_on=True)
txt(rx+rw/2, Y_TOP - 0.15, "Raw ECG Signal", fs=11, fw='bold')
txt(rx+rw/2, Y_TOP - 0.48, "12-lead", fs=9, fc=DIM_CLR, style='italic')
dim(rx+rw/2, ry - 0.35, "T samples \u00d7 12 leads")

# ====================================================================
# 2  ECG-JEPA ENCODER (frozen)
# ====================================================================
jx, jy, jw, jh = 2.8, Y_TOP - BH/2, 2.8, BH
rbox(jx, jy, jw, jh, FROZEN_BG, FROZEN_BD)
txt(jx+jw/2, Y_TOP + 0.18, "ECG-JEPA", fs=13, fw='bold', fc=FROZEN_BD)
txt(jx+jw/2, Y_TOP - 0.22, "Encoder", fs=10, fc=DIM_CLR, style='italic')
snow(jx+jw-0.25, jy+jh-0.2)
badge(jx+jw/2, jy - 0.32, "FROZEN", FROZEN_BG, FROZEN_BD)
dim(jx+jw/2, jy - 0.72, "Output: N_windows \u00d7 d_jepa")

arr(rx+rw+0.05, Y_TOP, jx-0.05, Y_TOP)

# ====================================================================
# 3  LINEAR PROJECTION (trainable)
# ====================================================================
lx, ly, lw_b, lh = 6.8, Y_TOP - BH_SM/2, 2.0, BH_SM
rbox(lx, ly, lw_b, lh, TRAIN_BG, TRAIN_BD)
txt(lx+lw_b/2, Y_TOP + 0.12, "Linear", fs=11, fw='bold', fc=TRAIN_BD)
txt(lx+lw_b/2, Y_TOP - 0.2, "Projection", fs=9, fc=DIM_CLR, style='italic')
badge(lx+lw_b/2, ly - 0.32, "TRAINABLE", TRAIN_BG, TRAIN_BD)
dim(lx+lw_b/2, ly - 0.72, "d_jepa \u2192 d_model")

arr(jx+jw+0.05, Y_TOP, lx-0.05, Y_TOP)

# ====================================================================
# 4  PERCEIVER RESAMPLER (trainable)
# ====================================================================
px, py, pw, ph = 10.0, Y_TOP - BH/2, 3.0, BH
rbox(px, py, pw, ph, TRAIN_BG, TRAIN_BD)
txt(px+pw/2, Y_TOP + 0.18, "Perceiver", fs=13, fw='bold', fc=TRAIN_BD)
txt(px+pw/2, Y_TOP - 0.22, "Resampler", fs=10, fc=DIM_CLR, style='italic')
badge(px+pw/2, py - 0.32, "TRAINABLE", TRAIN_BG, TRAIN_BD)

# Config callout above perceiver
cfg_x, cfg_y = px + pw/2, py + ph + 0.55
txt(cfg_x, cfg_y, "2 cross-attn layers\n8 heads \u2022 64 latent outputs",
    fs=9, fc=TRAIN_BD,
    bbox=dict(boxstyle='round,pad=0.22', fc=TRAIN_BG, ec=TRAIN_BD, lw=1, alpha=0.75))
dim(px+pw/2, py - 0.72, "Output: 64 \u00d7 d_model")

arr(lx+lw_b+0.05, Y_TOP, px-0.05, Y_TOP)

# ====================================================================
# 5  LLM BACKBONE — outer container
# ====================================================================
llm_x  = 14.3
llm_y  = Y_TOP - 2.6
llm_w  = 9.0
llm_h  = 4.5
rbox(llm_x, llm_y, llm_w, llm_h, LLM_BG, LLM_BD, lw=2.8, pad=0.3, zorder=1)
txt(llm_x + llm_w/2, llm_y + llm_h - 0.28,
    "LLaMA-3.2-3B   (Frozen)", fs=14, fw='bold', fc=LLM_BD)
snow(llm_x + llm_w - 0.55, llm_y + llm_h - 0.28, size=14)

# --- Transformer blocks (top row inside LLM) ---
blk_w, blk_h = 1.6, 0.85
blk_xs = [14.9, 17.0, 19.1, 21.2]
blk_y  = Y_TOP + 0.15

for i, bx in enumerate(blk_xs):
    rbox(bx, blk_y, blk_w, blk_h, BLK_BG, BLK_BD, lw=1.5, zorder=4)
    txt(bx + blk_w/2, blk_y + blk_h/2,
        f"Blocks\n{2*i+1}\u2013{2*i+2}", fs=9, fw='bold', fc=BLK_BD)
    if i < len(blk_xs) - 1:
        arr(bx + blk_w, blk_y + blk_h/2, blk_xs[i+1], blk_y + blk_h/2,
            color=BLK_BD, lw=1.5)

# --- Gated cross-attention boxes (bottom row inside LLM) ---
gca_h = 0.8
gca_y = blk_y - 1.6
gca_xs = [15.15, 17.25, 19.35]

for i, gx in enumerate(gca_xs):
    rbox(gx, gca_y, 1.1, gca_h, TRAIN_BG, TRAIN_BD, lw=1.5, zorder=4)
    txt(gx + 0.55, gca_y + gca_h/2, "Gated\nX-Attn", fs=8, fw='bold', fc=TRAIN_BD)
    # vertical arrow GCA → block above
    arr(gx + 0.55, gca_y + gca_h, blk_xs[i] + blk_w/2, blk_y,
        color=TRAIN_BD, lw=1.3)

badge(gca_xs[2] + 1.45, gca_y + gca_h/2, "TRAINABLE", TRAIN_BG, TRAIN_BD, fs=7)

# --- Latent bus (dashed horizontal line feeding all GCA) ---
bus_y = gca_y + gca_h / 2
bus_x_start = llm_x + 0.1
ax.plot([bus_x_start, gca_xs[-1]], [bus_y, bus_y],
        color=TRAIN_BD, lw=1.5, linestyle='--', zorder=3)
for gx in gca_xs:
    arr(gx - 0.05, bus_y, gx, bus_y, color=TRAIN_BD, lw=1.3, style='->')

# Label on latent bus
txt(bus_x_start + 0.05, bus_y + 0.35, r"$Z_{\mathrm{latent}}$", fs=10, fc=TRAIN_BD,
    fw='bold', ha='left',
    bbox=dict(boxstyle='round,pad=0.12', fc=TRAIN_BG, ec=TRAIN_BD, lw=0.8))

# --- Connect Perceiver → latent bus ---
# Horizontal out of perceiver, then turn down into LLM
mid_x = px + pw + 0.6
arr(px + pw + 0.05, Y_TOP, mid_x, Y_TOP, color=TRAIN_BD, lw=2)
ax.plot([mid_x, mid_x], [Y_TOP, bus_y], color=TRAIN_BD, lw=1.8, linestyle='--', zorder=2)
arr(mid_x, bus_y + 0.05, bus_x_start + 0.9, bus_y + 0.05, color=TRAIN_BD, lw=1.8, style='->')
dim(mid_x + 0.0, Y_TOP - 0.9, "64 latents\n(K, V)")

# ====================================================================
# 6  GATING MECHANISM CALLOUT
# ====================================================================
gt_x, gt_y, gt_w, gt_h = 14.8, 9.0, 8.2, 1.7
rbox(gt_x, gt_y, gt_w, gt_h, GATE_BG, GATE_BD, lw=1.8, pad=0.22)
txt(gt_x + gt_w/2, gt_y + gt_h - 0.3,
    "Gating Mechanism  (Residual Connection)", fs=11.5, fw='bold', fc='#7D6608')
txt(gt_x + gt_w/2, gt_y + 0.72,
    r"$h' = h + \alpha \cdot \mathrm{CrossAttn}\!\left(Q{=}h,\; K{=}Z_{\mathrm{latent}},\; V{=}Z_{\mathrm{latent}}\right)$",
    fs=12, fc=TEXT_CLR)
txt(gt_x + gt_w/2, gt_y + 0.2,
    r"$\alpha$ initialized to 0  $\longrightarrow$  LLM starts as pure text-only model",
    fs=9.5, fc=DIM_CLR)

# dashed pointer from callout down to blocks
arr(gt_x + gt_w/2, gt_y, blk_xs[1] + blk_w/2, blk_y + blk_h + 0.05,
    color=GATE_BD, lw=1.2, style='->', cs="arc3,rad=-0.12")

# ====================================================================
# 7  TEXT OUTPUT
# ====================================================================
ox, oy, ow, oh = 24.2, Y_TOP - BH_SM/2, 2.3, BH_SM
rbox(ox, oy, ow, oh, OUTPUT_BG, OUTPUT_BD)
txt(ox+ow/2, Y_TOP + 0.12, "Text Output", fs=12, fw='bold', fc=OUTPUT_BD)
txt(ox+ow/2, Y_TOP - 0.2, "Clinical reasoning", fs=9, fc=DIM_CLR, style='italic')

arr(blk_xs[-1] + blk_w, blk_y + blk_h/2, ox - 0.05, Y_TOP, color=LLM_BD, lw=2.2)

# ====================================================================
# 8  TEXT INPUT PATH (bottom row)
# ====================================================================
# Text input source
ti_x, ti_y = 10.2, Y_TXT
txt(ti_x, ti_y, "Text Input\n(metadata: mean, std,\nfreq, duration)",
    fs=9.5, fc=TEXT_CLR,
    bbox=dict(boxstyle='round,pad=0.3', fc=NEUTRAL_BG, ec=NEUTRAL_BD, lw=1.5),
    zorder=3)

# Tokenizer
tk_x, tk_y, tk_w, tk_h = 13.2, Y_TXT - 0.45, 2.5, 0.9
rbox(tk_x, tk_y, tk_w, tk_h, FROZEN_BG, FROZEN_BD)
txt(tk_x+tk_w/2, Y_TXT, "LLM Tokenizer", fs=10, fw='bold', fc=FROZEN_BD)
snow(tk_x+tk_w-0.2, tk_y+tk_h-0.15, size=11)

arr(11.4, ti_y, tk_x - 0.05, ti_y, color=FROZEN_BD, lw=1.5)

# Token stream
ts_x, ts_y, ts_w, ts_h = 16.5, Y_TXT - 0.55, 6.5, 1.1
rbox(ts_x, ts_y, ts_w, ts_h, TOKEN_BG, TOKEN_BD, lw=1.5)
txt(ts_x + ts_w/2, ts_y + ts_h - 0.15, "Token Stream", fs=9, fw='bold', fc=TOKEN_BD)

tokens = ["...", "<TS>", "[ECG latents]", "<endofchunk>", "text tokens", "..."]
tp_xs = np.linspace(ts_x + 0.45, ts_x + ts_w - 0.45, len(tokens))
for tpx, tok_text in zip(tp_xs, tokens):
    is_sp = tok_text in ("<TS>", "<endofchunk>")
    fc_ = "#E1BEE7" if is_sp else "#FAFAFA"
    ec_ = TOKEN_BD if is_sp else "#BDBDBD"
    fw_ = 'bold' if is_sp else 'normal'
    col_ = TOKEN_BD if is_sp else DIM_CLR
    txt(tpx, ti_y - 0.1, tok_text, fs=8, fw=fw_, fc=col_,
        bbox=dict(boxstyle='round,pad=0.13', fc=fc_, ec=ec_, lw=0.9))

arr(tk_x + tk_w + 0.05, ti_y, ts_x - 0.05, ti_y, color=TOKEN_BD, lw=1.5)

# Arrow from token stream up into LLM bottom
arr(ts_x + ts_w/2, ts_y + ts_h + 0.05, llm_x + llm_w/2, llm_y - 0.05,
    color=TOKEN_BD, lw=1.8)
dim(ts_x + ts_w/2 + 1.2, (ts_y + ts_h + llm_y) / 2 + 0.15, "Token\nembeddings")

# ====================================================================
# 9  LEGEND
# ====================================================================
leg_items = [
    ("\u2744  Frozen", FROZEN_BG, FROZEN_BD),
    ("Trainable", TRAIN_BG, TRAIN_BD),
    ("LLM Backbone (Frozen)", BLK_BG, BLK_BD),
    ("Special Tokens", TOKEN_BG, TOKEN_BD),
]
leg_x0, leg_y0 = 0.0, 10.5
for i, (lab, bg, bd) in enumerate(leg_items):
    lx = leg_x0 + i * 3.8
    rbox(lx, leg_y0, 0.45, 0.32, bg, bd, lw=1.5, zorder=4)
    txt(lx + 0.65, leg_y0 + 0.16, lab, fs=9.5, fc=TEXT_CLR, ha='left')

# ====================================================================
# 10  DIMENSION FLOW SUMMARY
# ====================================================================
txt(12.8, -0.5,
    "Dimension flow:   Raw ECG (T \u00d7 12)  \u2192  N_windows \u00d7 d_jepa  "
    "\u2192  N_windows \u00d7 d_model  \u2192  64 \u00d7 d_model  "
    "\u2192  LLM hidden states (d_model)  \u2192  Text tokens",
    fs=9.5, fc=DIM_CLR, style='italic',
    bbox=dict(boxstyle='round,pad=0.22', fc=NEUTRAL_BG, ec='#CFD8DC', lw=1))

# ====================================================================
# SAVE
# ====================================================================
plt.savefig("/home/wangni/notion-figures/qoq-med/fig_001.png",
            dpi=200, bbox_inches='tight', facecolor=WHITE, edgecolor='none',
            pad_inches=0.3)
plt.close()
print("Figure saved: /home/wangni/notion-figures/qoq-med/fig_001.png")
