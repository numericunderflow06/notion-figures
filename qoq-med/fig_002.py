"""
JEPA-Flamingo-DRPO: Multivariate ECG Handling (fig_002)
Left side:  12-lead ECG → 12x ECG-JEPA → 12x Perceiver → 768 latents → Gated Cross-Attention
Right side: Single-channel (EEG/Accel) → Conv1D+Transformer → Perceiver → 64 latents → Gated Cross-Attention
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette (consistent with fig_001) ────────────────────────
FROZEN_BG   = "#DAEAF6"
FROZEN_BD   = "#2471A3"
TRAIN_BG    = "#FDE8D0"
TRAIN_BD    = "#D35400"
SIGNAL_CLR  = "#2C3E50"
ARROW_CLR   = "#444444"
TEXT_CLR    = "#1B1B1B"
DIM_CLR     = "#6C7A89"
LLM_BG      = "#EAF7EA"
LLM_BD      = "#1E8449"
GATE_BG     = "#FFFDE7"
GATE_BD     = "#F9A825"
OUTPUT_BG   = "#E0F2F1"
OUTPUT_BD   = "#00897B"
NEUTRAL_BG  = "#F5F5F5"
NEUTRAL_BD  = "#90A4AE"
WHITE       = "#FFFFFF"
CONCAT_BG   = "#E8EAF6"
CONCAT_BD   = "#3949AB"
PANEL_BG    = "#FAFBFC"
SINGLE_BG   = "#F3E5F5"
SINGLE_BD   = "#7B1FA2"

# ── Figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(32, 17))
ax.set_xlim(-1.5, 31.5)
ax.set_ylim(-2.2, 16.8)
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

def arr(x1, y1, x2, y2, color=ARROW_CLR, lw=2.2, style='->', cs=None, zorder=2):
    props = dict(arrowstyle=style, color=color, lw=lw)
    if cs:
        props['connectionstyle'] = cs
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=zorder)

def dim(x, y, s, fs=10):
    txt(x, y, s, fs=fs, fc=DIM_CLR, style='italic',
        bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.9))

def badge(x, y, s, bg, bd, fs=8):
    txt(x, y, s, fs=fs, fw='bold', fc=bd,
        bbox=dict(boxstyle='round,pad=0.18', fc=bg, ec=bd, lw=1.2))

def snow(x, y, size=13):
    txt(x, y, '\u2744', fs=size, fc=FROZEN_BD, fontfamily='DejaVu Sans', zorder=6)

# ====================================================================
# TITLE
# ====================================================================
txt(15.0, 16.3, "Multivariate ECG Handling: 12-Lead Perceiver Pipeline",
    fs=20, fw='bold')
txt(15.0, 15.7,
    "Left: 12-lead ECG pipeline (12 parallel tracks \u2192 768 latents)     "
    "Right: Single-channel pipeline (1 track \u2192 64 latents)",
    fs=11, fc=DIM_CLR)

# ====================================================================
# PANEL BACKGROUNDS
# ====================================================================
rbox(-1.2, -1.8, 22.0, 18.0, PANEL_BG, '#B0BEC5', lw=1.5, pad=0.3, ls='--', zorder=0)
txt(9.5, 14.8, "12-Lead ECG Pipeline", fs=15, fw='bold', fc=FROZEN_BD)

rbox(21.4, -1.8, 9.8, 18.0, PANEL_BG, '#B0BEC5', lw=1.5, pad=0.3, ls='--', zorder=0)
txt(26.3, 14.8, "Single-Channel Pipeline", fs=15, fw='bold', fc=SINGLE_BD)

# ====================================================================
# LEFT SIDE: 12-LEAD ECG PIPELINE
# ====================================================================

# --- Raw 12-lead ECG input ---
ecg_x, ecg_w, ecg_h = -0.8, 2.0, 2.4
ecg_y = 5.8
rbox(ecg_x, ecg_y, ecg_w, ecg_h, NEUTRAL_BG, NEUTRAL_BD)

# Draw mini ECG waveforms
t = np.linspace(0, 4*np.pi, 120)
for i in range(4):
    yoff = ecg_y + ecg_h - 0.4 - i * 0.5
    ecg_wave = 0.12 * np.sin(t + i * 0.5) + np.where(
        (t > 4.5) & (t < 5.0), 0.2, 0)
    ax.plot(np.linspace(ecg_x + 0.2, ecg_x + ecg_w - 0.2, len(ecg_wave)),
            ecg_wave + yoff, color=SIGNAL_CLR, lw=0.8, zorder=4, alpha=0.7)

txt(ecg_x + ecg_w / 2, ecg_y + 0.2, "Raw 12-Lead\nECG", fs=10, fw='bold')
dim(ecg_x + ecg_w / 2, ecg_y - 0.4, "T samples \u00d7 12 leads")

# --- 12 parallel lead tracks ---
lead_names_top = ["I", "II", "III", "aVR", "aVL", "aVF"]
lead_names_bot = ["V1", "V2", "V3", "V4", "V5", "V6"]
all_leads = lead_names_top + lead_names_bot

n_leads = 12
lead_y_start = 13.0
lead_y_end = 1.0
lead_ys = np.linspace(lead_y_start, lead_y_end, n_leads)

# Column positions
jepa_x = 2.8
jepa_w, jepa_h = 2.2, 0.68

perc_x = 7.5
perc_w, perc_h = 2.2, 0.68

ecg_out_x = ecg_x + ecg_w
ecg_center_y = ecg_y + ecg_h / 2

# Draw 12 lead tracks
for i, (lead_name, ly) in enumerate(zip(all_leads, lead_ys)):
    # Fan-out arrow from ECG block
    arr(ecg_out_x + 0.05, ecg_center_y, jepa_x - 0.1, ly,
        color=NEUTRAL_BD, lw=0.8, style='->')

    # ECG-JEPA box (frozen)
    rbox(jepa_x, ly - jepa_h / 2, jepa_w, jepa_h, FROZEN_BG, FROZEN_BD, lw=1.3)
    txt(jepa_x + jepa_w / 2, ly + 0.07, "ECG-JEPA", fs=8.5, fw='bold', fc=FROZEN_BD)
    txt(jepa_x + jepa_w / 2, ly - 0.17, f"Lead {lead_name}", fs=7.5, fc=DIM_CLR)

    # Arrow JEPA → Perceiver
    arr(jepa_x + jepa_w + 0.05, ly, perc_x - 0.1, ly,
        color=ARROW_CLR, lw=0.8, style='->')

    # Perceiver Resampler box (trainable)
    rbox(perc_x, ly - perc_h / 2, perc_w, perc_h, TRAIN_BG, TRAIN_BD, lw=1.3)
    txt(perc_x + perc_w / 2, ly + 0.07, "Perceiver", fs=8.5, fw='bold', fc=TRAIN_BD)
    txt(perc_x + perc_w / 2, ly - 0.17, "64 latents", fs=7.5, fc=DIM_CLR)

# Column labels above tracks
txt(jepa_x + jepa_w / 2, lead_y_start + 0.7, "ECG-JEPA Encoder", fs=11, fw='bold', fc=FROZEN_BD)
badge(jepa_x + jepa_w / 2, lead_y_start + 0.28, "FROZEN", FROZEN_BG, FROZEN_BD)

txt(perc_x + perc_w / 2, lead_y_start + 0.7, "Perceiver Resampler", fs=11, fw='bold', fc=TRAIN_BD)
badge(perc_x + perc_w / 2, lead_y_start + 0.28, "TRAINABLE", TRAIN_BG, TRAIN_BD)

# Dimension annotations above column labels
dim(jepa_x + jepa_w / 2, lead_y_start + 1.3, "N_win \u00d7 d_jepa per lead")
dim(perc_x + perc_w / 2, lead_y_start + 1.3, "64 \u00d7 d_model per lead")

# Linear projection note — positioned between column headers at mid height
mid_arrow_x = (jepa_x + jepa_w + perc_x) / 2
# Use a single combined label to avoid vertical stacking issues
txt(mid_arrow_x, lead_y_start + 0.55,
    "Linear Proj\n(d_jepa \u2192 d_model)", fs=9.5, fc='#F57F17', fw='bold',
    bbox=dict(boxstyle='round,pad=0.15', fc='#FFF9C4', ec='#FBC02D', lw=1.0, alpha=0.95),
    zorder=7)

# --- Concatenation block ---
concat_x = 11.2
concat_w, concat_h = 2.0, 2.8
concat_cy = (lead_y_start + lead_y_end) / 2
concat_y = concat_cy - concat_h / 2

rbox(concat_x, concat_y, concat_w, concat_h, CONCAT_BG, CONCAT_BD, lw=2.0)
txt(concat_x + concat_w / 2, concat_cy + 0.6, "Concat", fs=12, fw='bold', fc=CONCAT_BD)
txt(concat_x + concat_w / 2, concat_cy + 0.1, "12 \u00d7 64", fs=11, fc=CONCAT_BD)
txt(concat_x + concat_w / 2, concat_cy - 0.35, "= 768", fs=14, fw='bold', fc=CONCAT_BD)
txt(concat_x + concat_w / 2, concat_cy - 0.8, "latent\nvectors", fs=8, fc=DIM_CLR)

# Fan-in arrows: each Perceiver → Concatenation
for ly in lead_ys:
    arr(perc_x + perc_w + 0.05, ly, concat_x - 0.08, concat_cy,
        color=CONCAT_BD, lw=0.7, style='->')

# --- Gated Cross-Attention in LLM ---
gca_x = 14.5
gca_w, gca_h = 3.8, 2.8
gca_cy = concat_cy
gca_y = gca_cy - gca_h / 2

arr(concat_x + concat_w + 0.1, concat_cy, gca_x - 0.1, gca_cy,
    color=CONCAT_BD, lw=2.5, style='->')
dim(concat_x + concat_w + 0.55, concat_cy + 0.5, "768 \u00d7 d_model")

rbox(gca_x, gca_y, gca_w, gca_h, LLM_BG, LLM_BD, lw=2.5, pad=0.2)
txt(gca_x + gca_w / 2, gca_cy + 0.95, "LLM Backbone", fs=12, fw='bold', fc=LLM_BD)
txt(gca_x + gca_w / 2, gca_cy + 0.55, "(LLaMA-3.2-3B)", fs=8.5, fc=DIM_CLR)
snow(gca_x + gca_w - 0.4, gca_cy + 0.95, size=12)

# Inner gated cross-attn blocks
for j, yoff in enumerate([0.0, -0.75]):
    rbox(gca_x + 0.35, gca_cy + yoff - 0.25, 3.0, 0.55, TRAIN_BG, TRAIN_BD, lw=1.3)
    txt(gca_x + 0.35 + 1.5, gca_cy + yoff,
        f"Gated Cross-Attn {j+1}", fs=8.5, fw='bold', fc=TRAIN_BD)

# Gating formula callout below LLM box
txt(gca_x + gca_w / 2, gca_y - 0.55,
    r"$h' = h + \alpha \cdot \mathrm{CrossAttn}(Q{=}h,\; K{=}Z,\; V{=}Z)$",
    fs=11, fc='#7D6608',
    bbox=dict(boxstyle='round,pad=0.18', fc=GATE_BG, ec=GATE_BD, lw=1.0))
txt(gca_x + gca_w / 2, gca_y - 1.05,
    r"$\alpha$ init = 0  $\rightarrow$  LLM starts as text-only", fs=10.5, fc=DIM_CLR)

# --- Text Output (left side) - to the right of LLM box ---
out_x = gca_x + gca_w + 0.6
out_w, out_h = 2.0, 1.0
out_y = gca_cy - out_h / 2

rbox(out_x, out_y, out_w, out_h, OUTPUT_BG, OUTPUT_BD)
txt(out_x + out_w / 2, out_y + out_h / 2 + 0.12, "Text", fs=11, fw='bold', fc=OUTPUT_BD)
txt(out_x + out_w / 2, out_y + out_h / 2 - 0.2, "Output", fs=11, fw='bold', fc=OUTPUT_BD)

arr(gca_x + gca_w + 0.1, gca_cy, out_x - 0.05, gca_cy,
    color=LLM_BD, lw=2.2, style='->')


# ====================================================================
# RIGHT SIDE: SINGLE-CHANNEL PIPELINE (vertical top-to-bottom)
# ====================================================================
right_cx = 26.3
step_gap = 2.6  # vertical spacing between stages

# Stage Y positions (top to bottom)
s1_y = 12.5   # input
s2_y = s1_y - step_gap     # encoder
s3_y = s2_y - step_gap     # perceiver
s4_y = s3_y - step_gap     # gated cross-attn
s5_y = s4_y - step_gap     # text output

box_w = 3.2
box_h = 1.4

# --- Stage 1: Single-channel input ---
si_x = right_cx - box_w / 2
rbox(si_x, s1_y, box_w, box_h, NEUTRAL_BG, NEUTRAL_BD)
t2 = np.linspace(0, 6*np.pi, 150)
wave = 0.2 * np.sin(t2) + 0.1 * np.sin(3 * t2)
ax.plot(np.linspace(si_x + 0.3, si_x + box_w - 0.3, len(wave)),
        wave + s1_y + box_h - 0.3, color=SIGNAL_CLR, lw=1.0, zorder=4)
txt(si_x + box_w / 2, s1_y + 0.25, "EEG / Accelerometer\n(single channel)", fs=9.5, fw='bold')
dim(si_x + box_w / 2, s1_y - 0.4, "T samples \u00d7 1 channel")

# --- Stage 2: Conv1D + TransformerEncoder ---
rbox(si_x, s2_y, box_w, box_h, SINGLE_BG, SINGLE_BD, lw=2.0)
txt(si_x + box_w / 2, s2_y + box_h / 2 + 0.2, "Conv1D +", fs=11, fw='bold', fc=SINGLE_BD)
txt(si_x + box_w / 2, s2_y + box_h / 2 - 0.2, "TransformerEncoder", fs=10, fw='bold', fc=SINGLE_BD)
badge(si_x + box_w / 2, s2_y - 0.35, "TRAINABLE", SINGLE_BG, SINGLE_BD)
dim(si_x + box_w / 2, s2_y - 0.8, "N_windows \u00d7 d_model")

# Annotation: replaces ECG-JEPA
txt(right_cx + box_w / 2 + 1.2, s2_y + box_h / 2, "Replaces\nECG-JEPA", fs=9, fc=SINGLE_BD,
    bbox=dict(boxstyle='round,pad=0.15', fc='#F3E5F5', ec=SINGLE_BD, lw=0.8, alpha=0.85))

arr(si_x + box_w / 2, s1_y - 0.05, si_x + box_w / 2, s2_y + box_h + 0.05,
    color=ARROW_CLR, lw=2.0, style='->')

# --- Stage 3: Perceiver Resampler (shared) ---
rbox(si_x, s3_y, box_w, box_h, TRAIN_BG, TRAIN_BD, lw=2.0)
txt(si_x + box_w / 2, s3_y + box_h / 2 + 0.2, "Perceiver", fs=12, fw='bold', fc=TRAIN_BD)
txt(si_x + box_w / 2, s3_y + box_h / 2 - 0.2, "Resampler", fs=10, fw='bold', fc=TRAIN_BD)
badge(si_x + box_w / 2, s3_y - 0.35, "SHARED / TRAINABLE", TRAIN_BG, TRAIN_BD)
dim(si_x + box_w / 2, s3_y - 0.8, "Output: 64 \u00d7 d_model")

# Annotation: only 64 latents
txt(right_cx + box_w / 2 + 1.2, s3_y + box_h / 2, "Only 64\nlatent vectors\n(1 channel)", fs=9, fc=TRAIN_BD, fw='bold',
    bbox=dict(boxstyle='round,pad=0.18', fc=TRAIN_BG, ec=TRAIN_BD, lw=1.0, alpha=0.85))

arr(si_x + box_w / 2, s2_y - 0.05, si_x + box_w / 2, s3_y + box_h + 0.05,
    color=TRAIN_BD, lw=2.0, style='->')

# --- Stage 4: Gated Cross-Attention in LLM ---
rbox(si_x, s4_y, box_w, box_h + 0.3, LLM_BG, LLM_BD, lw=2.0)
txt(si_x + box_w / 2, s4_y + (box_h + 0.3) / 2 + 0.3, "Gated", fs=11, fw='bold', fc=LLM_BD)
txt(si_x + box_w / 2, s4_y + (box_h + 0.3) / 2 - 0.05, "Cross-Attention", fs=10, fw='bold', fc=LLM_BD)
txt(si_x + box_w / 2, s4_y + (box_h + 0.3) / 2 - 0.4, "in LLM", fs=9, fc=DIM_CLR)
snow(si_x + box_w - 0.3, s4_y + box_h + 0.3 - 0.2, size=11)

arr(si_x + box_w / 2, s3_y - 0.05, si_x + box_w / 2, s4_y + box_h + 0.3 + 0.05,
    color=TRAIN_BD, lw=2.0, style='->')
dim(right_cx + box_w / 2 + 1.0, (s3_y + s4_y + box_h + 0.3) / 2, "64 \u00d7 d_model")

# --- Stage 5: Text Output ---
out2_w, out2_h = 2.0, 0.9
out2_x = right_cx - out2_w / 2
rbox(out2_x, s5_y, out2_w, out2_h, OUTPUT_BG, OUTPUT_BD)
txt(out2_x + out2_w / 2, s5_y + out2_h / 2, "Text Output", fs=10, fw='bold', fc=OUTPUT_BD)

arr(si_x + box_w / 2, s4_y - 0.05, out2_x + out2_w / 2, s5_y + out2_h + 0.05,
    color=LLM_BD, lw=2.0, style='->')


# ====================================================================
# COMPARISON CALLOUT (bottom center)
# ====================================================================
comp_x, comp_w = 3.5, 23.5
comp_y, comp_h = -1.7, 1.0

rbox(comp_x, comp_y, comp_w, comp_h, GATE_BG, GATE_BD, lw=1.5, pad=0.2)
txt(comp_x + comp_w / 2, comp_y + comp_h / 2 + 0.15,
    "Key Difference: 12-lead ECG produces 12 \u00d7 64 = 768 latent vectors   |   "
    "Single-channel produces 1 \u00d7 64 = 64 latent vectors",
    fs=10.5, fw='bold', fc='#7D6608')
txt(comp_x + comp_w / 2, comp_y + comp_h / 2 - 0.25,
    "Perceiver Resampler and gated cross-attention layers are shared across all modalities",
    fs=9, fc=DIM_CLR)


# ====================================================================
# LEGEND
# ====================================================================
leg_items = [
    ("\u2744  Frozen (ECG-JEPA, LLM)", FROZEN_BG, FROZEN_BD),
    ("Trainable (Perceiver, Cross-Attn)", TRAIN_BG, TRAIN_BD),
    ("Non-ECG Encoder (Conv1D+Transformer)", SINGLE_BG, SINGLE_BD),
    ("Concatenation", CONCAT_BG, CONCAT_BD),
]
leg_x0, leg_y0 = -0.5, -0.45
for i, (lab, bg, bd) in enumerate(leg_items):
    lx = leg_x0 + i * 7.0
    rbox(lx, leg_y0, 0.45, 0.32, bg, bd, lw=1.5, zorder=4)
    txt(lx + 0.65, leg_y0 + 0.16, lab, fs=9, fc=TEXT_CLR, ha='left')

# ====================================================================
# VERTICAL DIVIDER
# ====================================================================
ax.plot([21.2, 21.2], [-0.5, 15.0], color='#B0BEC5', lw=1.5,
        linestyle=':', zorder=1, alpha=0.5)
txt(21.2, 0.3, "vs.", fs=14, fw='bold', fc='#78909C', rotation=0,
    bbox=dict(boxstyle='round,pad=0.2', fc=WHITE, ec='#B0BEC5', lw=1))

# ====================================================================
# SAVE
# ====================================================================
plt.savefig("/home/wangni/notion-figures/qoq-med/fig_002.png",
            dpi=200, bbox_inches='tight', facecolor=WHITE, edgecolor='none',
            pad_inches=0.3)
plt.close()
print("Figure saved: /home/wangni/notion-figures/qoq-med/fig_002.png")
