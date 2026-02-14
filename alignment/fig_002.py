"""
The Temporal Alignment Problem (fig_002)
Side-by-side comparison: Before TPA (ordinal positional embeddings, no temporal grounding)
vs. With TPA (shared normalized temporal coordinates enabling direct grounding).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette (consistent with fig_001) ─────────────────────────
BLUE_FILL   = '#D6EAF8'
BLUE_EDGE   = '#2980B9'
ORANGE_FILL = '#FEF0DB'
ORANGE_EDGE = '#E67E22'
GREEN_FILL  = '#D5F5E3'
GREEN_EDGE  = '#27AE60'
DARK_TEXT    = '#2C3E50'
ARROW_COLOR = '#5D6D7E'
ANNOT_COLOR = '#7F8C8D'

# Additional colours for this figure
RED_ACCENT    = '#E74C3C'
RED_LIGHT     = '#FADBD8'
GREEN_ACCENT  = '#27AE60'
GREEN_LIGHT   = '#D5F5E3'
PURPLE_ACCENT = '#8E44AD'
PURPLE_LIGHT  = '#E8DAEF'
TEAL_ACCENT   = '#16A085'
LIGHT_GRAY    = '#F2F3F4'
DISCONNECT    = '#C0392B'  # red for disconnect / misalignment

# ── Figure setup ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))

# Create two panels with a small gap
gs = fig.add_gridspec(1, 2, wspace=0.08, left=0.05, right=0.97,
                      top=0.88, bottom=0.05)

# ── Suptitle ─────────────────────────────────────────────────────────
fig.suptitle('The Temporal Alignment Problem',
             fontsize=22, fontweight='bold', color=DARK_TEXT, y=0.96)
fig.text(0.5, 0.925,
         'Why ordinal positional embeddings fail to ground temporal references in text',
         ha='center', fontsize=13, color=ANNOT_COLOR)

# =====================================================================
#  Generate a sample "prediction market" time series
#  Jan 2024 – Dec 2025, with a spike in March 2025
# =====================================================================
np.random.seed(42)
n_months = 24  # Jan 2024 to Dec 2025
t_months = np.arange(n_months)
month_labels = []
for yr in [2024, 2025]:
    for m in ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']:
        month_labels.append(m)

# Build a time series with a clear spike in March 2025 (index 14)
base = 0.3 + 0.015 * t_months + 0.05 * np.sin(t_months * 0.4)
spike_idx = 14  # March 2025
noise = np.random.normal(0, 0.02, n_months)
ts = base + noise
ts[spike_idx] = 0.82
ts[spike_idx - 1] = 0.55
ts[spike_idx + 1] = 0.65
ts[spike_idx + 2] = 0.50
ts = np.clip(ts, 0, 1)

# Number of patches (each patch covers ~3 months)
n_patches = 8
patch_size = n_months // n_patches  # 3 months each

# =====================================================================
#  PANEL HELPER FUNCTIONS
# =====================================================================
def draw_panel_label(ax, label, sublabel, is_problem=True):
    """Draw panel header."""
    color = DISCONNECT if is_problem else GREEN_ACCENT
    bg_color = RED_LIGHT if is_problem else GREEN_LIGHT
    box = FancyBboxPatch((0.02, 0.91), 0.96, 0.08, transform=ax.transAxes,
                          boxstyle="round,pad=0.01", facecolor=bg_color,
                          edgecolor=color, linewidth=2.5, zorder=5)
    ax.add_patch(box)
    ax.text(0.5, 0.955, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=color, zorder=6)
    ax.text(0.5, 0.918, sublabel, transform=ax.transAxes,
            ha='center', va='center', fontsize=10, color=ANNOT_COLOR, zorder=6)


def draw_ts_plot(ax_ts, highlight_patch=None, highlight_color=None,
                 show_ordinal=True, show_time_axis=False):
    """Draw the time series with patch boundaries."""
    ax_ts.plot(t_months, ts, color=BLUE_EDGE, linewidth=2.0, zorder=3)
    ax_ts.fill_between(t_months, ts, alpha=0.1, color=BLUE_EDGE)
    ax_ts.set_xlim(-0.5, n_months - 0.5)
    ax_ts.set_ylim(0, 1.05)
    ax_ts.set_ylabel('Probability', fontsize=10, color=DARK_TEXT)

    # Draw patch boundaries and shade patches
    patch_colors_default = ['#EBF5FB', '#FAFAFA'] * (n_patches // 2)
    for i in range(n_patches):
        x_start = i * patch_size - 0.5
        x_end = (i + 1) * patch_size - 0.5
        fc = patch_colors_default[i]
        if highlight_patch is not None and i == highlight_patch:
            fc = highlight_color if highlight_color else ORANGE_FILL
        ax_ts.axvspan(x_start, x_end, facecolor=fc, alpha=0.4, zorder=0)
        if i > 0:
            ax_ts.axvline(x_start, color='#BDC3C7', linewidth=0.8,
                          linestyle='--', zorder=1)

    # Spike annotation
    ax_ts.annotate('Spike', xy=(spike_idx, ts[spike_idx]),
                   xytext=(spike_idx + 2.5, ts[spike_idx] + 0.05),
                   fontsize=9, color=RED_ACCENT, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=RED_ACCENT, lw=1.2))

    # X-axis
    if show_time_axis:
        ax_ts.set_xticks(t_months)
        ax_ts.set_xticklabels(month_labels, fontsize=7.5, color=DARK_TEXT)
        # Year labels
        ax_ts.text(5.5, -0.18, '2024', transform=ax_ts.get_xaxis_transform(),
                   ha='center', fontsize=10, fontweight='bold', color=DARK_TEXT)
        ax_ts.text(17.5, -0.18, '2025', transform=ax_ts.get_xaxis_transform(),
                   ha='center', fontsize=10, fontweight='bold', color=DARK_TEXT)
    else:
        ax_ts.set_xticks([])

    ax_ts.tick_params(axis='y', labelsize=8, colors=DARK_TEXT)
    ax_ts.spines['top'].set_visible(False)
    ax_ts.spines['right'].set_visible(False)
    ax_ts.spines['bottom'].set_color('#BDC3C7')
    ax_ts.spines['left'].set_color('#BDC3C7')
    return ax_ts


# =====================================================================
#  LEFT PANEL — Before TPA
# =====================================================================
ax_left = fig.add_subplot(gs[0, 0])
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, 10)
ax_left.axis('off')
draw_panel_label(ax_left, 'Before TPA (Current OpenTSLM)',
                 'Ordinal positional embeddings — no temporal grounding',
                 is_problem=True)

# --- Time series plot ---
ax_ts_left = fig.add_axes([0.09, 0.60, 0.38, 0.22])
draw_ts_plot(ax_ts_left, highlight_patch=4, highlight_color=RED_LIGHT,
             show_time_axis=True)
ax_ts_left.set_title('Prediction Market: Jan 2024 – Dec 2025',
                      fontsize=10, color=DARK_TEXT, pad=8)

# Patch index labels below the plot
ax_patch_left = fig.add_axes([0.09, 0.555, 0.38, 0.04])
ax_patch_left.set_xlim(-0.5, n_months - 0.5)
ax_patch_left.set_ylim(0, 1)
ax_patch_left.axis('off')
for i in range(n_patches):
    cx = (i + 0.5) * patch_size - 0.5
    bg = RED_LIGHT if i == 4 else BLUE_FILL
    ec = DISCONNECT if i == 4 else BLUE_EDGE
    rect = FancyBboxPatch((i * patch_size - 0.3, 0.15), patch_size - 0.4, 0.7,
                           boxstyle="round,pad=0.05", facecolor=bg,
                           edgecolor=ec, linewidth=1.5, transform=ax_patch_left.transData)
    ax_patch_left.add_patch(rect)
    ax_patch_left.text(cx, 0.5, f'Patch {i}', ha='center', va='center',
                       fontsize=8.5, fontweight='bold', color=ec)

# Label for patch row
ax_patch_left.text(-2.5, 0.5, 'Encoder\nPositions:', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=DARK_TEXT,
                    transform=ax_patch_left.transData)

# --- Positional Embedding Row ---
ax_pe_left = fig.add_axes([0.09, 0.465, 0.38, 0.06])
ax_pe_left.set_xlim(-0.5, n_months - 0.5)
ax_pe_left.set_ylim(0, 1)
ax_pe_left.axis('off')

ax_pe_left.text(-2.5, 0.5, 'Learnable\nPE:', ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=DARK_TEXT)
for i in range(n_patches):
    cx = (i + 0.5) * patch_size - 0.5
    bg = RED_LIGHT if i == 4 else LIGHT_GRAY
    ec = DISCONNECT if i == 4 else ANNOT_COLOR
    rect = FancyBboxPatch((i * patch_size - 0.3, 0.15), patch_size - 0.4, 0.7,
                           boxstyle="round,pad=0.05", facecolor=bg,
                           edgecolor=ec, linewidth=1.2, transform=ax_pe_left.transData)
    ax_pe_left.add_patch(rect)
    ax_pe_left.text(cx, 0.5, f'PE[{i}]', ha='center', va='center',
                    fontsize=8, color=ec, fontweight='bold')

# --- Disconnect visual ---
# Arrow from PE row trying to connect to text, with X mark
ax_disc = fig.add_axes([0.09, 0.21, 0.38, 0.23])
ax_disc.set_xlim(0, 10)
ax_disc.set_ylim(0, 5)
ax_disc.axis('off')

# "Text prompt" box
text_box = FancyBboxPatch((0.3, 3.2), 9.4, 1.4, boxstyle="round,pad=0.15",
                           facecolor=LIGHT_GRAY, edgecolor=ANNOT_COLOR,
                           linewidth=1.5, zorder=2)
ax_disc.add_patch(text_box)
ax_disc.text(5, 4.2, 'Text Prompt', ha='center', va='center',
             fontsize=10, fontweight='bold', color=DARK_TEXT, zorder=3)
ax_disc.text(5, 3.6, '"The prediction market probability  spiked in March 2025 ..."',
             ha='center', va='center', fontsize=9.5, color=DARK_TEXT,
             style='italic', zorder=3)

# "March 2025" highlight in text
ax_disc.add_patch(FancyBboxPatch((5.9, 3.35), 2.6, 0.55,
                                  boxstyle="round,pad=0.05",
                                  facecolor=PURPLE_LIGHT, edgecolor=PURPLE_ACCENT,
                                  linewidth=1.5, zorder=4, alpha=0.6))

# Disconnection arrows / X marks
# Draw a dashed arrow from "March 2025" up toward patch area, then X it out
arrow_left = FancyArrowPatch((5, 3.2), (5, 1.6),
                              arrowstyle='->', color=DISCONNECT,
                              linewidth=2.0, linestyle='--',
                              mutation_scale=15, zorder=3)
ax_disc.add_patch(arrow_left)

# Big X on the arrow
ax_disc.text(5, 2.3, '✗', ha='center', va='center', fontsize=28,
             fontweight='bold', color=DISCONNECT, zorder=5,
             path_effects=[pe.withStroke(linewidth=3, foreground='white')])

# Explanation box
expl_box = FancyBboxPatch((0.5, 0.1), 9.0, 1.3, boxstyle="round,pad=0.12",
                           facecolor='#FEF9E7', edgecolor='#F4D03F',
                           linewidth=1.5, zorder=2)
ax_disc.add_patch(expl_box)
ax_disc.text(5, 0.95, 'No mechanism to map "March 2025" → Patch 4',
             ha='center', va='center', fontsize=10.5, fontweight='bold',
             color=DISCONNECT, zorder=3)
ax_disc.text(5, 0.45, 'PE[0], PE[1], ... are arbitrary learned vectors —\n'
             'they encode ordinal position, not real-world time.',
             ha='center', va='center', fontsize=9, color=DARK_TEXT, zorder=3)


# =====================================================================
#  RIGHT PANEL — With TPA
# =====================================================================
ax_right = fig.add_subplot(gs[0, 1])
ax_right.set_xlim(0, 10)
ax_right.set_ylim(0, 10)
ax_right.axis('off')
draw_panel_label(ax_right, 'With TPA (Temporal Positional Alignment)',
                 'Shared normalized temporal coordinates — direct grounding',
                 is_problem=False)

# --- Time series plot ---
ax_ts_right = fig.add_axes([0.57, 0.60, 0.38, 0.22])
draw_ts_plot(ax_ts_right, highlight_patch=4, highlight_color=ORANGE_FILL,
             show_time_axis=True)
ax_ts_right.set_title('Prediction Market: Jan 2024 – Dec 2025',
                       fontsize=10, color=DARK_TEXT, pad=8)

# Normalized time axis indicator below plot
ax_norm = fig.add_axes([0.57, 0.555, 0.38, 0.04])
ax_norm.set_xlim(-0.5, n_months - 0.5)
ax_norm.set_ylim(0, 1)
ax_norm.axis('off')

# ATPE-encoded patches with temporal coordinates
for i in range(n_patches):
    cx = (i + 0.5) * patch_size - 0.5
    t_val = (i + 0.5) / n_patches  # midpoint temporal coordinate
    bg = ORANGE_FILL if i == 4 else GREEN_LIGHT
    ec = ORANGE_EDGE if i == 4 else GREEN_ACCENT
    rect = FancyBboxPatch((i * patch_size - 0.3, 0.15), patch_size - 0.4, 0.7,
                           boxstyle="round,pad=0.05", facecolor=bg,
                           edgecolor=ec, linewidth=1.5, transform=ax_norm.transData)
    ax_norm.add_patch(rect)
    ax_norm.text(cx, 0.5, f't={t_val:.2f}', ha='center', va='center',
                 fontsize=8, fontweight='bold', color=ec)

ax_norm.text(-2.5, 0.5, 'ATPE\nCoordinates:', ha='center', va='center',
              fontsize=8.5, fontweight='bold', color=DARK_TEXT)

# --- Shared temporal axis row ---
ax_taxis = fig.add_axes([0.57, 0.465, 0.38, 0.06])
ax_taxis.set_xlim(-0.5, n_months - 0.5)
ax_taxis.set_ylim(0, 1)
ax_taxis.axis('off')

# Continuous time bar
bar_left = -0.3
bar_right = n_months - 0.7
bar = FancyBboxPatch((bar_left, 0.25), bar_right - bar_left, 0.5,
                      boxstyle="round,pad=0.03",
                      facecolor='#EBF5FB', edgecolor=BLUE_EDGE,
                      linewidth=1.5, transform=ax_taxis.transData)
ax_taxis.add_patch(bar)
ax_taxis.text(-2.5, 0.5, 'Shared\nTime Axis:', ha='center', va='center',
               fontsize=8.5, fontweight='bold', color=DARK_TEXT)

# Tick marks at 0.0, 0.25, 0.5, 0.75, 1.0
for val, label in [(0.0, '0.0'), (0.25, '0.25'), (0.5, '0.5'),
                    (0.75, '0.75'), (1.0, '1.0')]:
    x_pos = bar_left + val * (bar_right - bar_left)
    ax_taxis.plot([x_pos, x_pos], [0.2, 0.8], color=BLUE_EDGE, linewidth=1.2)
    ax_taxis.text(x_pos, 0.05, label, ha='center', va='center',
                  fontsize=8, color=BLUE_EDGE, fontweight='bold')

# Mark "March 2025" position on the bar
march_t = (spike_idx + 0.5) / n_months  # ≈ 0.604
march_x = bar_left + march_t * (bar_right - bar_left)
ax_taxis.plot(march_x, 0.5, 'o', color=ORANGE_EDGE, markersize=10, zorder=5)
ax_taxis.text(march_x, 0.92, f't≈{march_t:.2f}', ha='center', va='center',
              fontsize=8, fontweight='bold', color=ORANGE_EDGE)

# --- Connection visual ---
ax_conn = fig.add_axes([0.57, 0.21, 0.38, 0.23])
ax_conn.set_xlim(0, 10)
ax_conn.set_ylim(0, 5)
ax_conn.axis('off')

# "Text prompt" box
text_box2 = FancyBboxPatch((0.3, 3.2), 9.4, 1.4, boxstyle="round,pad=0.15",
                            facecolor=LIGHT_GRAY, edgecolor=ANNOT_COLOR,
                            linewidth=1.5, zorder=2)
ax_conn.add_patch(text_box2)
ax_conn.text(5, 4.2, 'Text Prompt', ha='center', va='center',
              fontsize=10, fontweight='bold', color=DARK_TEXT, zorder=3)
ax_conn.text(5, 3.6, '"The prediction market probability  spiked in March 2025 ..."',
              ha='center', va='center', fontsize=9.5, color=DARK_TEXT,
              style='italic', zorder=3)

# "March 2025" highlight with temporal coordinate
ax_conn.add_patch(FancyBboxPatch((5.9, 3.35), 2.6, 0.55,
                                  boxstyle="round,pad=0.05",
                                  facecolor=ORANGE_FILL, edgecolor=ORANGE_EDGE,
                                  linewidth=1.5, zorder=4, alpha=0.7))

# Temporal anchor label
ax_conn.text(7.2, 2.9, f'Anchor: t≈{march_t:.2f}', ha='center', va='center',
              fontsize=9.5, fontweight='bold', color=ORANGE_EDGE, zorder=5,
              bbox=dict(boxstyle='round,pad=0.2', fc=ORANGE_FILL, ec=ORANGE_EDGE,
                        lw=1.2, alpha=0.9))

# Successful alignment arrows
# Arrow from text "March 2025" down to shared time coordinate
arrow_down = FancyArrowPatch((5.5, 3.2), (3.5, 1.8),
                              arrowstyle='->', color=ORANGE_EDGE,
                              linewidth=2.0, linestyle='-',
                              mutation_scale=15, zorder=3,
                              connectionstyle='arc3,rad=0.15')
ax_conn.add_patch(arrow_down)

# Arrow from ATPE patch coordinate down to shared coordinate
arrow_down2 = FancyArrowPatch((7.2, 2.7), (6.5, 1.8),
                               arrowstyle='->', color=GREEN_ACCENT,
                               linewidth=2.0, linestyle='-',
                               mutation_scale=15, zorder=3,
                               connectionstyle='arc3,rad=-0.15')
ax_conn.add_patch(arrow_down2)

# Shared coordinate meeting point
ax_conn.plot(5, 1.6, 'o', color=TEAL_ACCENT, markersize=14, zorder=5)
ax_conn.text(5, 1.6, '✓', ha='center', va='center', fontsize=14,
              fontweight='bold', color='white', zorder=6)

# Match label
ax_conn.text(5, 1.05, f'Same temporal coordinate: t ≈ {march_t:.2f}',
              ha='center', va='center', fontsize=10.5, fontweight='bold',
              color=TEAL_ACCENT, zorder=3)

# Labels on the arrows
ax_conn.text(3.5, 2.8, 'Text\ntoken', ha='center', va='center',
              fontsize=8, color=ORANGE_EDGE, fontweight='bold')
ax_conn.text(7.7, 2.2, 'TS\npatch', ha='center', va='center',
              fontsize=8, color=GREEN_ACCENT, fontweight='bold')

# Explanation box
expl_box2 = FancyBboxPatch((0.5, 0.05), 9.0, 0.8, boxstyle="round,pad=0.12",
                            facecolor='#EAFAF1', edgecolor=GREEN_ACCENT,
                            linewidth=1.5, zorder=2)
ax_conn.add_patch(expl_box2)
ax_conn.text(5, 0.45, 'ATPE encodes t≈0.60 for Patch 4.  Temporal Anchor maps "March 2025" → t≈0.60.\n'
             'Cross-attention bridges them via temporal proximity.',
             ha='center', va='center', fontsize=8.5, color=DARK_TEXT, zorder=3)


# =====================================================================
#  Bottom annotation — how TPA mechanism labels
# =====================================================================
fig.text(0.5, 0.08,
         'Timestamps normalized to [0, 1] within each sample\'s time window  ·  '
         'ATPE: sinusoidal encoding of real-world midpoint timestamps  ·  '
         'Temporal Anchor: same ATPE applied to text references',
         ha='center', fontsize=9.5, color=ANNOT_COLOR, style='italic')

# ── Vertical separator ───────────────────────────────────────────────
fig.patches.append(Rectangle((0.497, 0.10), 0.006, 0.80,
                              transform=fig.transFigure,
                              facecolor='#D5D8DC', edgecolor='none',
                              zorder=10))

# ── Save ─────────────────────────────────────────────────────────────
fig.savefig('/home/wangni/notion-figures/alignment/fig_002.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none',
            pad_inches=0.3)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/alignment/fig_002.png")
