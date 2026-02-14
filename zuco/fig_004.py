"""
Figure 004: Cross-Modality Transfer: Sleep EEG to Eye-Tracking
Two-panel design:
  Left: Bar chart comparing Sleep-Flamingo (68.29%) vs Scratch (62.25%) on ZuCo 2.0
  Right: Conceptual bridge diagram showing Sleep EEG → Shared Representations → Eye-Tracking
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Data ──
sleep_acc = 68.29
scratch_acc = 62.25
delta = sleep_acc - scratch_acc  # +6.04

# ── Colors ──
COL_SLEEP = "#2E86AB"      # Strong teal-blue
COL_SCRATCH = "#A3A3A3"    # Neutral gray
COL_DELTA = "#E8533F"      # Warm red for delta highlight
COL_EEG = "#6A4C93"        # Purple for Sleep EEG modality
COL_SHARED = "#F4A261"     # Warm amber for shared representations
COL_ET = "#2A9D8F"         # Teal-green for Eye-Tracking modality
COL_BG_PANEL = "#FAFAFA"   # Very subtle panel background

fig, (ax_bar, ax_bridge) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         gridspec_kw={'width_ratios': [1, 1.3]})
fig.patch.set_facecolor('white')

# ════════════════════════════════════════════════════════════════
# LEFT PANEL: Bar chart — Sleep vs Scratch on ZuCo 2.0
# ════════════════════════════════════════════════════════════════
ax_bar.set_facecolor(COL_BG_PANEL)

bars = ax_bar.bar(
    [0, 1],
    [scratch_acc, sleep_acc],
    width=0.55,
    color=[COL_SCRATCH, COL_SLEEP],
    edgecolor='white',
    linewidth=1.5,
    zorder=3,
)

# Value labels on bars
for bar, val, col in zip(bars, [scratch_acc, sleep_acc], [COL_SCRATCH, COL_SLEEP]):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.6,
        f"{val:.2f}%",
        ha='center', va='bottom',
        fontsize=14, fontweight='bold', color='#222222',
    )

# Delta annotation arrow
ax_bar.annotate(
    '',
    xy=(1, sleep_acc),
    xytext=(1, scratch_acc),
    arrowprops=dict(
        arrowstyle='<->',
        color=COL_DELTA,
        lw=2.2,
        shrinkA=2, shrinkB=2,
    ),
    zorder=5,
)

# Delta label
mid_y = (sleep_acc + scratch_acc) / 2
ax_bar.text(
    1.32, mid_y,
    f"+{delta:.2f} pp",
    ha='left', va='center',
    fontsize=13, fontweight='bold', color=COL_DELTA,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              edgecolor=COL_DELTA, linewidth=1.5, alpha=0.95),
    zorder=5,
)

# Axis formatting
ax_bar.set_xticks([0, 1])
ax_bar.set_xticklabels(['Scratch\n(from random init)', 'Sleep-Flamingo\n(pretrained on Sleep EEG)'],
                        fontsize=11, fontweight='medium')
ax_bar.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='medium')
ax_bar.set_ylim(50, 75)
ax_bar.set_xlim(-0.6, 1.85)
ax_bar.set_yticks(np.arange(50, 76, 5))
ax_bar.tick_params(axis='y', labelsize=10)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_linewidth(0.8)
ax_bar.spines['bottom'].set_linewidth(0.8)
ax_bar.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Chance baseline
ax_bar.axhline(y=50, color='#999999', linestyle=':', linewidth=1, zorder=1)
ax_bar.text(-0.55, 50.4, 'chance', fontsize=8.5, color='#999999', va='bottom')

ax_bar.set_title('ZuCo 2.0 Classification Accuracy', fontsize=13, fontweight='bold',
                  pad=12, color='#222222')

# ════════════════════════════════════════════════════════════════
# RIGHT PANEL: Cross-modality bridge diagram
# ════════════════════════════════════════════════════════════════
ax_bridge.set_facecolor(COL_BG_PANEL)
ax_bridge.set_xlim(0, 10)
ax_bridge.set_ylim(0, 7)
ax_bridge.axis('off')

ax_bridge.set_title('Cross-Modality Transfer Bridge', fontsize=13, fontweight='bold',
                     pad=12, color='#222222')


def draw_rounded_box(ax, xy, width, height, color, text_lines, text_color='white',
                     fontsize=11, alpha=0.95, border_color=None):
    """Draw a rounded rectangle with centered multi-line text."""
    x, y = xy
    if border_color is None:
        border_color = color
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=border_color,
        linewidth=2, alpha=alpha, zorder=3,
    )
    ax.add_patch(box)
    cx = x + width / 2
    cy = y + height / 2
    n = len(text_lines)
    line_spacing = fontsize * 1.6 / 72  # approx in data coords — adjusted below
    # Use axis transform for text
    for i, line in enumerate(text_lines):
        fw = 'bold' if i == 0 else 'normal'
        fs = fontsize if i == 0 else fontsize - 1
        offset = (i - (n - 1) / 2) * 0.35
        ax.text(cx, cy - offset, line, ha='center', va='center',
                fontsize=fs, fontweight=fw, color=text_color, zorder=4)


# ── Source modality: Sleep EEG ──
draw_rounded_box(ax_bridge, (0.3, 3.8), 2.6, 1.8, COL_EEG,
                 ['Sleep EEG', '(Polysomnography)', 'Source Modality'],
                 text_color='white', fontsize=12)

# ── Shared Flamingo representations ──
draw_rounded_box(ax_bridge, (3.8, 3.8), 2.6, 1.8, COL_SHARED,
                 ['OpenTSLM', 'Flamingo', 'Shared Repr.'],
                 text_color='#333333', fontsize=12)

# ── Target modality: Eye-Tracking ──
draw_rounded_box(ax_bridge, (7.3, 3.8), 2.4, 1.8, COL_ET,
                 ['Eye-Tracking', '(ZuCo 2.0)', 'Target Modality'],
                 text_color='white', fontsize=12)

# ── Arrows between boxes ──
arrow_kw = dict(
    arrowstyle='-|>',
    color='#444444',
    lw=2.5,
    mutation_scale=18,
    connectionstyle='arc3,rad=0.0',
)

ax_bridge.annotate('', xy=(3.75, 4.7), xytext=(2.95, 4.7),
                    arrowprops=arrow_kw, zorder=5)
ax_bridge.annotate('', xy=(7.25, 4.7), xytext=(6.45, 4.7),
                    arrowprops=arrow_kw, zorder=5)

# ── Architecture components below the bridge ──
comp_y = 2.1
comp_h = 0.9
comp_w = 2.0
comp_fontsize = 9
comp_color = '#E0E0E0'

# CNN Encoder
draw_rounded_box(ax_bridge, (0.6, comp_y), comp_w, comp_h, comp_color,
                 ['CNN Encoder'],
                 text_color='#333333', fontsize=comp_fontsize, border_color='#BBBBBB')

# Perceiver Resampler
draw_rounded_box(ax_bridge, (3.1, comp_y), 2.7, comp_h, comp_color,
                 ['Perceiver Resampler'],
                 text_color='#333333', fontsize=comp_fontsize, border_color='#BBBBBB')

# Gated Cross-Attention
draw_rounded_box(ax_bridge, (6.6, comp_y), 2.7, comp_h, comp_color,
                 ['Gated Cross-Attn', '+ Llama 3.2 3B'],
                 text_color='#333333', fontsize=comp_fontsize, border_color='#BBBBBB')

# Small arrows connecting architecture to main bridge
for cx in [1.6, 4.45, 7.95]:
    ax_bridge.annotate('', xy=(cx, 3.75), xytext=(cx, 3.05),
                        arrowprops=dict(arrowstyle='->', color='#999999',
                                        lw=1.2, connectionstyle='arc3,rad=0'),
                        zorder=2)

# Label for architecture strip
ax_bridge.text(5.0, 1.55, 'Flamingo Architecture Components (frozen weights transfer)',
               ha='center', va='top', fontsize=9, fontstyle='italic', color='#777777')

# ── Key insight annotation at top ──
ax_bridge.text(5.0, 6.55,
               'First evidence of cross-domain generalization beyond original 5 training datasets',
               ha='center', va='center', fontsize=10.5, fontstyle='italic',
               color=COL_DELTA, fontweight='medium',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF5F3',
                         edgecolor=COL_DELTA, linewidth=1.2, alpha=0.9),
               zorder=5)

# ── Bridge label above arrows ──
ax_bridge.text(3.35, 5.85, 'Pre-train', ha='center', va='center',
               fontsize=10, fontweight='bold', color=COL_EEG)
ax_bridge.text(6.85, 5.85, 'Fine-tune', ha='center', va='center',
               fontsize=10, fontweight='bold', color=COL_ET)

# Curved bridge line over top
from matplotlib.patches import Arc
bridge_arc = Arc((5.1, 5.2), 6.8, 2.2, angle=0, theta1=0, theta2=180,
                 color='#444444', linewidth=1.5, linestyle='--', zorder=2)
ax_bridge.add_patch(bridge_arc)

# ── Overall title ──
fig.suptitle('Cross-Modality Transfer: Sleep EEG → Eye-Tracking',
             fontsize=16, fontweight='bold', color='#1A1A1A', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('/home/wangni/notion-figures/zuco/fig_004.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()

print("Figure saved to /home/wangni/notion-figures/zuco/fig_004.png")
