"""
Fig 008: Training Parameter Configuration
Visual summary of trainable vs frozen parameters in the OpenTSLM-Flamingo model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color Palette ──────────────────────────────────────────────────────────
BLUE_TRAIN = '#3B82F6'       # Trainable component fill
BLUE_TRAIN_EDGE = '#1D4ED8'  # Trainable component border
BLUE_TRAIN_LIGHT = '#DBEAFE' # Trainable component lighter fill
GRAY_FROZEN = '#9CA3AF'      # Frozen component fill
GRAY_FROZEN_EDGE = '#6B7280' # Frozen component border
GRAY_FROZEN_LIGHT = '#F3F4F6'# Frozen component lighter fill
LR_COLOR = '#059669'         # Learning rate label color
ARROW_COLOR = '#374151'      # Arrow color
BG_SECTION = '#F9FAFB'       # Section background

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(7, 7.6, 'Training Parameter Configuration',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color='#111827')
ax.text(7, 7.15, 'OpenTSLM-Flamingo: Trainable vs Frozen Components',
        fontsize=11, ha='center', va='center', color='#6B7280')

# ── Helper Functions ───────────────────────────────────────────────────────
def draw_component_box(ax, x, y, w, h, label, sublabel, is_trainable,
                       lr_text=None, detail_lines=None):
    """Draw a component box with training status coloring."""
    fill = BLUE_TRAIN_LIGHT if is_trainable else GRAY_FROZEN_LIGHT
    edge = BLUE_TRAIN_EDGE if is_trainable else GRAY_FROZEN_EDGE
    header_fill = BLUE_TRAIN if is_trainable else GRAY_FROZEN

    # Main box
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         facecolor=fill, edgecolor=edge, linewidth=1.8)
    ax.add_patch(box)

    # Header bar
    header_h = 0.38
    header = FancyBboxPatch((x + 0.04, y + h - header_h - 0.04), w - 0.08, header_h,
                            boxstyle="round,pad=0.04",
                            facecolor=header_fill, edgecolor='none')
    ax.add_patch(header)

    # Component name (in header)
    ax.text(x + w / 2, y + h - header_h / 2 - 0.04, label,
            fontsize=10.5, fontweight='bold', ha='center', va='center',
            color='white')

    # Status badge
    status = 'TRAINABLE' if is_trainable else 'FROZEN'
    badge_color = '#16A34A' if is_trainable else '#DC2626'
    badge_y = y + h - header_h - 0.35
    ax.text(x + w / 2, badge_y, status,
            fontsize=7.5, fontweight='bold', ha='center', va='center',
            color=badge_color,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor=badge_color, linewidth=0.8, alpha=0.9))

    # Sub-label / description
    if sublabel:
        ax.text(x + w / 2, badge_y - 0.35, sublabel,
                fontsize=8.5, ha='center', va='center', color='#374151')

    # Detail lines
    if detail_lines:
        for i, line in enumerate(detail_lines):
            ax.text(x + w / 2, badge_y - 0.35 - 0.28 * (i + 1), line,
                    fontsize=7.5, ha='center', va='center', color='#6B7280',
                    fontstyle='italic')

    # Learning rate label
    if lr_text:
        ax.text(x + w / 2, y + 0.18, f'lr = {lr_text}',
                fontsize=9, fontweight='bold', ha='center', va='center',
                color=LR_COLOR,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='#ECFDF5',
                          edgecolor=LR_COLOR, linewidth=0.8))


def draw_arrow(ax, x1, y1, x2, y2):
    """Draw a connection arrow between components."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=14,
                            color=ARROW_COLOR, linewidth=1.8,
                            connectionstyle='arc3,rad=0')
    ax.add_patch(arrow)


# ── Layout: Horizontal pipeline ───────────────────────────────────────────
# Components left to right:
# 1. CNN Encoder (trainable, lr=2e-4)
# 2. Perceiver Resampler (trainable, lr=1e-4)
# 3. Gated Cross-Attention (trainable, lr=1e-4)
# 4. Qwen2.5-0.5B LLM (frozen)

box_w = 2.7
box_h = 3.2
y_base = 2.1
gap = 0.55

x_positions = []
x_start = 0.6
for i in range(4):
    x_positions.append(x_start + i * (box_w + gap))

# ── Draw Components ────────────────────────────────────────────────────────

# 1. CNN Encoder
draw_component_box(ax, x_positions[0], y_base, box_w, box_h,
                   'CNN Encoder', 'Conv1D + PosEmbed + LN',
                   is_trainable=True, lr_text='2e-4',
                   detail_lines=['patch_size=4, embed_dim=128',
                                 'Output: [B, 1024, 128]'])

# 2. Perceiver Resampler
draw_component_box(ax, x_positions[1], y_base, box_w, box_h,
                   'Perceiver Resampler', '6 layers, 64 latent tokens',
                   is_trainable=True, lr_text='1e-4',
                   detail_lines=['8 heads × 64 dim/head',
                                 'Output: [B, 64, 512]'])

# 3. Gated Cross-Attention
draw_component_box(ax, x_positions[2], y_base, box_w, box_h,
                   'Gated Cross-Attn', 'Inserted in each LLM layer',
                   is_trainable=True, lr_text='1e-4',
                   detail_lines=['tanh gate (α), learned',
                                 'Q: LLM states, K/V: Perceiver'])

# 4. Frozen LLM
draw_component_box(ax, x_positions[3], y_base, box_w, box_h,
                   'Qwen2.5-0.5B LLM', 'Decoder-only backbone',
                   is_trainable=False, lr_text=None,
                   detail_lines=['Self-attn + FFN: frozen',
                                 'All weights fixed'])

# ── Draw Arrows ────────────────────────────────────────────────────────────
arrow_y = y_base + box_h / 2
for i in range(3):
    draw_arrow(ax, x_positions[i] + box_w + 0.03, arrow_y,
               x_positions[i + 1] - 0.03, arrow_y)

# ── Input / Output labels ─────────────────────────────────────────────────
# Input label (left of CNN)
ax.annotate('DNA\n5-channel\nEncoding',
            xy=(x_positions[0] - 0.03, arrow_y),
            xytext=(x_positions[0] - 0.65, arrow_y + 1.0),
            fontsize=8.5, ha='center', va='center', color='#374151',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.3),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF3C7',
                      edgecolor='#D97706', linewidth=0.8))

# Output label (right of LLM)
ax.annotate('Task\nPrediction',
            xy=(x_positions[3] + box_w + 0.03, arrow_y),
            xytext=(x_positions[3] + box_w + 0.65, arrow_y + 1.0),
            fontsize=8.5, ha='center', va='center', color='#374151',
            fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.3),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FEF3C7',
                      edgecolor='#D97706', linewidth=0.8))

# ── Input Embeddings annotation ───────────────────────────────────────────
# Input embeddings are trainable — show as a small note under the LLM box
embed_y = y_base - 0.45
embed_box = FancyBboxPatch((x_positions[3] + 0.15, embed_y - 0.22),
                           box_w - 0.3, 0.44,
                           boxstyle="round,pad=0.05",
                           facecolor=BLUE_TRAIN_LIGHT,
                           edgecolor=BLUE_TRAIN_EDGE, linewidth=1.2,
                           linestyle='--')
ax.add_patch(embed_box)
ax.text(x_positions[3] + box_w / 2, embed_y,
        'Input Embeddings: TRAINABLE',
        fontsize=7.5, fontweight='bold', ha='center', va='center',
        color=BLUE_TRAIN_EDGE)

# Arrow from input embeddings to LLM
ax.annotate('', xy=(x_positions[3] + box_w / 2, y_base),
            xytext=(x_positions[3] + box_w / 2, embed_y + 0.22),
            arrowprops=dict(arrowstyle='->', color=BLUE_TRAIN_EDGE,
                            lw=1.2, linestyle='--'))

# ── Legend ─────────────────────────────────────────────────────────────────
legend_y = 0.65
legend_x = 1.0

# Trainable legend
train_patch = FancyBboxPatch((legend_x, legend_y - 0.15), 0.4, 0.3,
                              boxstyle="round,pad=0.03",
                              facecolor=BLUE_TRAIN, edgecolor=BLUE_TRAIN_EDGE,
                              linewidth=1.2)
ax.add_patch(train_patch)
ax.text(legend_x + 0.6, legend_y, 'Trainable (optimized during training)',
        fontsize=9, va='center', color='#374151')

# Frozen legend
frozen_patch = FancyBboxPatch((legend_x + 6.5, legend_y - 0.15), 0.4, 0.3,
                               boxstyle="round,pad=0.03",
                               facecolor=GRAY_FROZEN, edgecolor=GRAY_FROZEN_EDGE,
                               linewidth=1.2)
ax.add_patch(frozen_patch)
ax.text(legend_x + 7.1, legend_y, 'Frozen (weights fixed, no gradient)',
        fontsize=9, va='center', color='#374151')

# ── Parameter count summary ────────────────────────────────────────────────
summary_x = 7.0
summary_y = 0.18
ax.text(summary_x, summary_y,
        'Training config:  batch_size=4  |  early_stop=5 epochs  |  Encoder lr=2e-4  |  Projector lr=1e-4',
        fontsize=8.5, ha='center', va='center', color='#6B7280',
        fontstyle='italic')

# ── Save ───────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/genomics/fig_008.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Figure saved to /home/wangni/notion-figures/genomics/fig_008.png")
