#!/usr/bin/env python3
"""
fig_006: Perceiver Resampler vs Instruction-Guided Resampler (IGR)
Side-by-side comparison showing query-agnostic vs query-conditioned compression.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Color palette ───────────────────────────────────────────────────────────
# Left panel (Perceiver): muted/grey tones
GREY_DARK = "#5a5a5a"
GREY_MED = "#9e9e9e"
GREY_LIGHT = "#d5d5d5"
GREY_BG = "#f0f0f0"
GREY_TEXT = "#4a4a4a"

# Right panel (IGR): vibrant tones
BLUE_DARK = "#1565c0"
BLUE_MED = "#42a5f5"
BLUE_LIGHT = "#bbdefb"
GREEN = "#2e7d32"
GREEN_LIGHT = "#c8e6c9"
ORANGE = "#e65100"
ORANGE_LIGHT = "#ffe0b2"
PURPLE = "#6a1b9a"
PURPLE_LIGHT = "#e1bee7"
TEAL = "#00695c"
TEAL_LIGHT = "#b2dfdb"

WHITE = "#ffffff"
BLACK = "#222222"
DIVIDER_RED = "#c62828"


def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=9,
                     fontweight='normal', textcolor=BLACK, alpha=1.0, linestyle='-',
                     linewidth=1.2, zorder=2, text_wrap=False, ha='center', va='center'):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=zorder + 1,
            wrap=text_wrap, linespacing=1.3)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=BLACK, style='->', lw=1.3,
               connectionstyle="arc3,rad=0", zorder=3, alpha=1.0):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, connectionstyle=connectionstyle,
        zorder=zorder, alpha=alpha,
        mutation_scale=14
    )
    ax.add_patch(arrow)
    return arrow


def draw_time_series_snippet(ax, x_center, y_center, w=0.7, h=0.22, color='#555', seed=42):
    """Draw a small time-series waveform icon."""
    rng = np.random.RandomState(seed)
    n = 40
    t = np.linspace(0, 4 * np.pi, n)
    sig = np.sin(t) + 0.3 * np.sin(3 * t) + 0.15 * rng.randn(n)
    sig = (sig - sig.min()) / (sig.max() - sig.min())  # normalize to [0,1]

    xs = np.linspace(x_center - w / 2, x_center + w / 2, n)
    ys = y_center - h / 2 + sig * h
    ax.plot(xs, ys, color=color, linewidth=1.0, zorder=6, alpha=0.85)


# ─── Figure setup ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(16.5, 11.5))
ax.set_xlim(-0.05, 10.6)
ax.set_ylim(-0.3, 10.8)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)

# ─── Title ───────────────────────────────────────────────────────────────────
ax.text(5.0, 10.55, "Perceiver Resampler  vs  Instruction-Guided Resampler (IGR)",
        ha='center', va='center', fontsize=16, fontweight='bold', color=BLACK)
ax.text(5.0, 10.25, "Query-Agnostic Compression  vs  Query-Conditioned Compression",
        ha='center', va='center', fontsize=11, color=GREY_DARK, style='italic')

# ─── Central divider ────────────────────────────────────────────────────────
div_x = 5.0
ax.plot([div_x, div_x], [0.0, 9.85], color=DIVIDER_RED, linewidth=2.5,
        linestyle='--', alpha=0.7, zorder=1)
# "vs" label
circle = plt.Circle((div_x, 7.0), 0.3, facecolor=WHITE, edgecolor=DIVIDER_RED,
                     linewidth=2.5, zorder=10)
ax.add_patch(circle)
ax.text(div_x, 7.0, "vs", ha='center', va='center', fontsize=13,
        fontweight='bold', color=DIVIDER_RED, zorder=11)

# ─── Panel headers ──────────────────────────────────────────────────────────
# Left
ax.text(2.5, 9.85, "OpenTSLM-Flamingo", ha='center', va='center',
        fontsize=13, fontweight='bold', color=GREY_DARK)
ax.text(2.5, 9.55, "Perceiver Resampler", ha='center', va='center',
        fontsize=11, color=GREY_MED, style='italic')

# Right
ax.text(7.5, 9.85, "OpenTSLM-ITA", ha='center', va='center',
        fontsize=13, fontweight='bold', color=BLUE_DARK)
ax.text(7.5, 9.55, "Instruction-Guided Resampler (IGR)", ha='center', va='center',
        fontsize=11, color=BLUE_MED, style='italic')

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Perceiver Resampler
# ═══════════════════════════════════════════════════════════════════════════════
LX = 2.5  # center x for left panel

# --- Time-Series Input ---
y_ts = 8.7
draw_rounded_box(ax, LX, y_ts, 3.2, 0.55, "",
                 GREY_LIGHT, GREY_MED, fontsize=10, fontweight='bold', textcolor=GREY_DARK)
ax.text(LX, y_ts + 0.14, "Time-Series Patches  H", ha='center', va='center',
        fontsize=10, fontweight='bold', color=GREY_DARK, zorder=7)
draw_time_series_snippet(ax, LX, y_ts - 0.10, w=1.8, h=0.16, color=GREY_DARK, seed=42)

# --- Text Query (shown but ignored) ---
y_query_l = 7.55
draw_rounded_box(ax, LX - 0.9, y_query_l, 1.5, 0.45, 'Text Query',
                 WHITE, GREY_MED, fontsize=9, textcolor=GREY_MED,
                 linestyle='--', linewidth=1.0, alpha=0.6)
# "X" / ignored symbol
ax.text(LX + 0.35, y_query_l, "IGNORED", ha='center', va='center',
        fontsize=8, fontweight='bold', color='#b71c1c', alpha=0.7, zorder=5,
        style='italic')

# --- Fixed Latent Queries ---
y_latent = 6.55
draw_rounded_box(ax, LX, y_latent, 3.2, 0.55, "Fixed Latent Queries  L ∈ R^{M×d}",
                 GREY_LIGHT, GREY_MED, fontsize=10, fontweight='bold', textcolor=GREY_DARK)
ax.text(LX, y_latent - 0.15, "(learnable, query-agnostic)", ha='center', va='center',
        fontsize=8, color=GREY_MED, style='italic')

# Arrow: TS → Perceiver
draw_arrow(ax, LX, y_ts - 0.28, LX, y_latent + 0.55, color=GREY_DARK)

# --- Perceiver Cross-Attention ---
y_perc = 5.45
draw_rounded_box(ax, LX, y_perc, 3.2, 0.60, "Perceiver Cross-Attention\nQ=L, K/V=H",
                 GREY_BG, GREY_DARK, fontsize=9.5, fontweight='bold', textcolor=GREY_DARK)

# Arrow: Latent → Perceiver
draw_arrow(ax, LX, y_latent - 0.28, LX, y_perc + 0.30, color=GREY_DARK)

# --- Compressed Output ---
y_out = 4.25
draw_rounded_box(ax, LX, y_out, 3.2, 0.50, "Compressed Latents  Z",
                 GREY_LIGHT, GREY_DARK, fontsize=10, fontweight='bold', textcolor=GREY_DARK)
ax.text(LX, y_out - 0.13, "(same output regardless of question)",
        ha='center', va='center', fontsize=7.5, color='#b71c1c', style='italic')

# Arrow: Perceiver → Output
draw_arrow(ax, LX, y_perc - 0.30, LX, y_out + 0.25, color=GREY_DARK)

# --- Gated Cross-Attention ---
y_gca = 3.20
draw_rounded_box(ax, LX, y_gca, 3.2, 0.50, "Gated Cross-Attention in LLM",
                 GREY_BG, GREY_DARK, fontsize=9.5, fontweight='bold', textcolor=GREY_DARK)

draw_arrow(ax, LX, y_out - 0.25, LX, y_gca + 0.25, color=GREY_DARK)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Instruction-Guided Resampler (IGR)
# ═══════════════════════════════════════════════════════════════════════════════
RX = 7.5  # center x for right panel

# --- Time-Series Input ---
draw_rounded_box(ax, RX, y_ts, 3.2, 0.55, "",
                 BLUE_LIGHT, BLUE_DARK, fontsize=10, fontweight='bold', textcolor=BLUE_DARK)
ax.text(RX, y_ts + 0.14, "Time-Series Patches  H", ha='center', va='center',
        fontsize=10, fontweight='bold', color=BLUE_DARK, zorder=7)
draw_time_series_snippet(ax, RX, y_ts - 0.10, w=1.8, h=0.16, color=BLUE_DARK, seed=42)

# --- Text Query (active, flows into conditioning) ---
y_query_r = 7.55
draw_rounded_box(ax, RX - 0.9, y_query_r, 1.6, 0.45, "Text Query  Q",
                 ORANGE_LIGHT, ORANGE, fontsize=9.5, fontweight='bold', textcolor=ORANGE)
# Label: active
ax.text(RX + 0.45, y_query_r, "ACTIVE", ha='center', va='center',
        fontsize=8, fontweight='bold', color=GREEN, zorder=5, style='italic')

# --- Query Encoder ---
y_enc = 6.85
draw_rounded_box(ax, RX - 0.9, y_enc, 1.6, 0.40, "Query Encoder",
                 ORANGE_LIGHT, ORANGE, fontsize=9, fontweight='bold', textcolor=ORANGE)
draw_arrow(ax, RX - 0.9, y_query_r - 0.23, RX - 0.9, y_enc + 0.20, color=ORANGE, lw=1.5)

# --- Learnable Instruct Tokens (LIT) ---
y_lit = 6.55
draw_rounded_box(ax, RX + 1.0, y_latent, 1.3, 0.55, "LIT  I\nM=25",
                 PURPLE_LIGHT, PURPLE, fontsize=9.5, fontweight='bold', textcolor=PURPLE)

# --- Self-Attention Conditioning [I; Q_text] ---
y_cond = 5.90
draw_rounded_box(ax, RX, y_cond, 3.2, 0.50, "Self-Attention  [ I ; Q_text ]",
                 ORANGE_LIGHT, PURPLE, fontsize=9.5, fontweight='bold', textcolor=PURPLE)
ax.text(RX, y_cond - 0.13, "→ Query-Conditioned LIT  I'",
        ha='center', va='center', fontsize=8, color=ORANGE, fontweight='bold')

# Arrows into conditioning
draw_arrow(ax, RX - 0.9, y_enc - 0.20, RX - 0.9, y_cond + 0.25, color=ORANGE, lw=1.5)
draw_arrow(ax, RX + 1.0, y_latent - 0.28, RX + 0.5, y_cond + 0.25, color=PURPLE, lw=1.5)

# --- Two-Stage ITA ---
y_ita_a = 5.05
draw_rounded_box(ax, RX, y_ita_a, 3.2, 0.50, "Stage A: Channel Instruct Fusing",
                 TEAL_LIGHT, TEAL, fontsize=9.5, fontweight='bold', textcolor=TEAL)
ax.text(RX, y_ita_a - 0.13, "Cross-attn: I' × channel dim per timestep",
        ha='center', va='center', fontsize=7.5, color=TEAL, style='italic')

draw_arrow(ax, RX, y_cond - 0.25, RX, y_ita_a + 0.25, color=TEAL, lw=1.5)

# Arrow from TS to Stage A
draw_arrow(ax, RX + 1.5, y_ts - 0.28, RX + 1.5, y_ita_a + 0.25,
           color=BLUE_DARK, lw=1.2, connectionstyle="arc3,rad=0.0")

y_ita_b = 4.25
draw_rounded_box(ax, RX, y_ita_b, 3.2, 0.50, "Stage B: Time Instruct Fusing",
                 TEAL_LIGHT, TEAL, fontsize=9.5, fontweight='bold', textcolor=TEAL)
ax.text(RX, y_ita_b - 0.13, "Cross-attn: I' × temporal dim",
        ha='center', va='center', fontsize=7.5, color=TEAL, style='italic')

draw_arrow(ax, RX, y_ita_a - 0.25, RX, y_ita_b + 0.25, color=TEAL, lw=1.5)

# --- Compressed Output ---
y_out_r = 3.35
draw_rounded_box(ax, RX, y_out_r, 3.2, 0.50, "Compressed Latents  Z'",
                 BLUE_LIGHT, BLUE_DARK, fontsize=10, fontweight='bold', textcolor=BLUE_DARK)
ax.text(RX, y_out_r - 0.13, "(different output per question)",
        ha='center', va='center', fontsize=7.5, color=GREEN, fontweight='bold', style='italic')

draw_arrow(ax, RX, y_ita_b - 0.25, RX, y_out_r + 0.25, color=BLUE_DARK, lw=1.5)

# --- Gated Cross-Attention ---
y_gca_r = 2.35
draw_rounded_box(ax, RX, y_gca_r, 3.2, 0.50, "Gated Cross-Attention in LLM",
                 BLUE_LIGHT, BLUE_DARK, fontsize=9.5, fontweight='bold', textcolor=BLUE_DARK)

draw_arrow(ax, RX, y_out_r - 0.25, RX, y_gca_r + 0.25, color=BLUE_DARK, lw=1.5)

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM — Example: Same TS + 2 Questions → Different Outputs
# ═══════════════════════════════════════════════════════════════════════════════

y_ex_top = 1.85
ax.plot([0.3, 9.7], [y_ex_top, y_ex_top], color=GREY_MED, linewidth=0.8,
        linestyle='-', alpha=0.5)
ax.text(5.0, y_ex_top + 0.15, "Example: Same time series, two different questions",
        ha='center', va='center', fontsize=10, fontweight='bold', color=BLACK)

# --- Shared TS icon ---
y_ex = 1.15
ts_x = 1.5
draw_rounded_box(ax, ts_x, y_ex, 1.6, 0.70, "", WHITE, GREY_DARK, linewidth=1.0)
draw_time_series_snippet(ax, ts_x, y_ex + 0.12, w=1.1, h=0.18, color=BLUE_DARK, seed=99)
ax.text(ts_x, y_ex - 0.20, "ECG Signal", ha='center', va='center',
        fontsize=8, color=GREY_DARK, fontweight='bold', zorder=7)

# --- Two questions ---
q1_x = 3.6
q2_x = 3.6
y_q1 = 1.45
y_q2 = 0.80
ax.text(q1_x, y_q1, 'Q1: "Is this arrhythmia?"', ha='center', va='center',
        fontsize=8.5, color=ORANGE, fontweight='bold')
ax.text(q2_x, y_q2, 'Q2: "What is the heart rate?"', ha='center', va='center',
        fontsize=8.5, color=PURPLE, fontweight='bold')

# --- Left: Same output ---
left_out_x = 5.8
draw_rounded_box(ax, left_out_x, y_q1, 1.2, 0.35, "Z (same)",
                 GREY_LIGHT, GREY_MED, fontsize=8, textcolor=GREY_DARK, fontweight='bold')
draw_rounded_box(ax, left_out_x, y_q2, 1.2, 0.35, "Z (same)",
                 GREY_LIGHT, GREY_MED, fontsize=8, textcolor=GREY_DARK, fontweight='bold')

# Draw equals sign between them
ax.text(left_out_x + 0.78, y_ex, "=", ha='center', va='center',
        fontsize=14, fontweight='bold', color='#b71c1c')

# Label
ax.text(left_out_x, y_ex - 0.45, "Perceiver", ha='center', va='center',
        fontsize=8, color=GREY_DARK, style='italic')

# --- Right: Different outputs ---
right_out_x = 8.3
draw_rounded_box(ax, right_out_x, y_q1, 1.5, 0.35, "Z' (rhythm-focused)",
                 ORANGE_LIGHT, ORANGE, fontsize=8, textcolor=ORANGE, fontweight='bold')
draw_rounded_box(ax, right_out_x, y_q2, 1.5, 0.35, "Z' (rate-focused)",
                 PURPLE_LIGHT, PURPLE, fontsize=8, textcolor=PURPLE, fontweight='bold')

# Draw not-equals sign between them
ax.text(right_out_x + 0.85, y_ex, "≠", ha='center', va='center',
        fontsize=14, fontweight='bold', color=GREEN)

# Label
ax.text(right_out_x, y_ex - 0.45, "IGR", ha='center', va='center',
        fontsize=8, color=BLUE_DARK, fontweight='bold', style='italic')

# Arrows from TS to outputs
draw_arrow(ax, ts_x + 0.80, y_q1, left_out_x - 0.95, y_q1, color=GREY_MED, lw=0.8)
draw_arrow(ax, ts_x + 0.80, y_q2, left_out_x - 0.95, y_q2, color=GREY_MED, lw=0.8)

# Arrows: questions connect
draw_arrow(ax, q1_x + 0.90, y_q1, left_out_x - 0.60, y_q1, color=GREY_MED,
           lw=0.8, style='->')
draw_arrow(ax, q2_x + 1.05, y_q2, left_out_x - 0.60, y_q2, color=GREY_MED,
           lw=0.8, style='->')

draw_arrow(ax, left_out_x + 0.60, y_q1, right_out_x - 0.80, y_q1, color=ORANGE, lw=0.8)
draw_arrow(ax, left_out_x + 0.60, y_q2, right_out_x - 0.80, y_q2, color=PURPLE, lw=0.8)

# ─── Key difference annotations (right side) ────────────────────────────────
# Small annotation boxes highlighting the two-stage nature on right panel
# Brace-like annotation for "Two-Stage ITA"
brace_x = RX + 1.85
brace_y_mid = (y_ita_a + y_ita_b) / 2
ax.annotate("", xy=(brace_x + 0.05, y_ita_a + 0.15), xytext=(brace_x + 0.05, y_ita_b - 0.15),
            arrowprops=dict(arrowstyle='-', color=TEAL, lw=1.5))
ax.plot([brace_x, brace_x + 0.15], [y_ita_a + 0.15, y_ita_a + 0.15],
        color=TEAL, lw=1.5)
ax.plot([brace_x, brace_x + 0.15], [y_ita_b - 0.15, y_ita_b - 0.15],
        color=TEAL, lw=1.5)
ax.plot([brace_x + 0.05, brace_x + 0.20], [brace_y_mid, brace_y_mid],
        color=TEAL, lw=1.5)
ax.text(brace_x + 0.30, brace_y_mid, "Two-Stage\nITA", ha='left', va='center',
        fontsize=8, color=TEAL, fontweight='bold', linespacing=1.2)

# ─── Save ────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
fig.savefig("/home/wangni/notion-figures/itformer/fig_006.png",
            dpi=200, bbox_inches='tight', facecolor=WHITE, edgecolor='none')
plt.close()
print("fig_006.png saved successfully to /home/wangni/notion-figures/itformer/fig_006.png")
