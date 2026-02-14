#!/usr/bin/env python3
"""
fig_005: Prompt Structure and Information Flow
Shows the interleaved prompt structure with <image> tokens and text descriptions,
illustrating how information flows differently under masked vs unmasked attention.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = {
    'ch1': '#4C72B0',   # Steel blue
    'ch2': '#DD8452',   # Warm orange
    'ch3': '#55A868',   # Sage green
    'ch4': '#C44E52',   # Muted red
    'post': '#8172B3',  # Purple
    'bg_masked': '#FFF8F0',
    'bg_unmasked': '#F0F4FF',
    'text_dark': '#2D2D2D',
    'text_mid': '#555555',
    'media_token': '#E8D44D',  # Gold for <image> tokens
}

CH_COLORS = [COLORS['ch1'], COLORS['ch2'], COLORS['ch3'], COLORS['ch4']]
CH_LABELS = ['Voltage', 'Current', 'Temperature', 'RPM']

fig = plt.figure(figsize=(19, 16.5), facecolor='white')

# ─── SECTION 1: Prompt Structure (top) ───────────────────────────────────────
ax_prompt = fig.add_axes([0.03, 0.76, 0.94, 0.20])
ax_prompt.set_xlim(-0.5, 23)
ax_prompt.set_ylim(-1.8, 4.5)
ax_prompt.axis('off')

ax_prompt.text(11.5, 4.3, 'Prompt Structure: Interleaved Channel Blocks',
               fontsize=16, fontweight='bold', ha='center', va='top',
               color=COLORS['text_dark'])

# Draw the prompt sequence as blocks
block_specs = []
x_cursor = 0.0
block_width_img = 1.0
block_width_desc = 2.8
block_gap = 0.15
eoc_width = 0.6

channel_block_centers = []

for i in range(4):
    ch_x_start = x_cursor
    # <image> token
    block_specs.append({
        'x': x_cursor, 'w': block_width_img, 'label': '<image>',
        'color': COLORS['media_token'], 'type': 'image', 'ch': i
    })
    x_cursor += block_width_img + block_gap

    # Channel description text
    desc = f'channel {i+1}:\n{CH_LABELS[i]}\ndescription'
    block_specs.append({
        'x': x_cursor, 'w': block_width_desc, 'label': desc,
        'color': CH_COLORS[i], 'type': 'desc', 'ch': i
    })
    x_cursor += block_width_desc + block_gap

    # <|endofchunk|>
    block_specs.append({
        'x': x_cursor, 'w': eoc_width, 'label': '<|eoc|>',
        'color': '#DDDDDD', 'type': 'eoc', 'ch': i
    })
    x_cursor += eoc_width + block_gap * 2.5

    ch_x_end = x_cursor - block_gap * 2.5
    channel_block_centers.append((ch_x_start, ch_x_end, (ch_x_start + ch_x_end) / 2))

# Post-prompt region
post_x_start = x_cursor
block_specs.append({
    'x': x_cursor, 'w': 3.0, 'label': 'Post-prompt\n(diagnosis\ngeneration)',
    'color': COLORS['post'], 'type': 'post', 'ch': -1
})

# Draw blocks
block_y = 0.2
block_h = 1.6
for spec in block_specs:
    alpha = 0.25 if spec['type'] in ('desc', 'post') else 0.4
    edgecolor = spec['color'] if spec['type'] not in ('eoc',) else '#999999'
    box = FancyBboxPatch(
        (spec['x'], block_y), spec['w'], block_h,
        boxstyle="round,pad=0.08",
        facecolor=spec['color'], alpha=alpha,
        edgecolor=edgecolor, linewidth=1.5
    )
    ax_prompt.add_patch(box)
    fontsize = 8.5 if spec['type'] == 'desc' else (7.5 if spec['type'] == 'eoc' else 9)
    fontweight = 'bold' if spec['type'] == 'image' else 'normal'
    ax_prompt.text(
        spec['x'] + spec['w'] / 2, block_y + block_h / 2,
        spec['label'], fontsize=fontsize, ha='center', va='center',
        fontweight=fontweight, color=COLORS['text_dark'],
        linespacing=1.2
    )

# Draw channel grouping brackets
for i in range(4):
    xs, xe, xc = channel_block_centers[i]
    bracket_y = block_y + block_h + 0.2
    ax_prompt.annotate('', xy=(xs, bracket_y + 0.4), xytext=(xe, bracket_y + 0.4),
                       arrowprops=dict(arrowstyle='-', color=CH_COLORS[i], lw=2.0))
    ax_prompt.plot([xs, xs], [bracket_y + 0.1, bracket_y + 0.4], color=CH_COLORS[i], lw=2.0)
    ax_prompt.plot([xe, xe], [bracket_y + 0.1, bracket_y + 0.4], color=CH_COLORS[i], lw=2.0)
    ax_prompt.text(xc, bracket_y + 0.6, f'Channel {i+1} Block',
                   fontsize=9, ha='center', va='bottom', color=CH_COLORS[i],
                   fontweight='bold')

# Label post-prompt bracket
pp_bracket_y = block_y + block_h + 0.2
ax_prompt.annotate('', xy=(post_x_start, pp_bracket_y + 0.4),
                   xytext=(post_x_start + 3.0, pp_bracket_y + 0.4),
                   arrowprops=dict(arrowstyle='-', color=COLORS['post'], lw=2.0))
ax_prompt.plot([post_x_start, post_x_start],
               [pp_bracket_y + 0.1, pp_bracket_y + 0.4],
               color=COLORS['post'], lw=2.0)
ax_prompt.plot([post_x_start + 3.0, post_x_start + 3.0],
               [pp_bracket_y + 0.1, pp_bracket_y + 0.4],
               color=COLORS['post'], lw=2.0)
ax_prompt.text(post_x_start + 1.5, pp_bracket_y + 0.6, 'Post-prompt',
               fontsize=9, ha='center', va='bottom', color=COLORS['post'], fontweight='bold')

# Sequence flow arrow along bottom
ax_prompt.annotate('', xy=(post_x_start + 3.0, block_y - 0.4),
                   xytext=(0.0, block_y - 0.4),
                   arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))
ax_prompt.text((post_x_start + 3.0) / 2, block_y - 0.85,
               'Token sequence direction  \u2192', fontsize=9.5, ha='center',
               color='#888888', style='italic')


# ─── Helper to draw attention flow sections ──────────────────────────────────
def draw_attention_section(ax, title, bg_color, edge_color, is_masked):
    ax.set_xlim(-1.5, 22.5)
    ax.set_ylim(-1.2, 7.0)
    ax.axis('off')

    # Background panel
    bg = FancyBboxPatch((-1.2, -1.0), 23.2, 7.8, boxstyle="round,pad=0.2",
                         facecolor=bg_color, alpha=0.4,
                         edgecolor=edge_color, linewidth=1.5)
    ax.add_patch(bg)

    ax.text(10.5, 6.4, title,
            fontsize=14, fontweight='bold', ha='center', va='center',
            color=COLORS['text_dark'])

    # Layout
    text_row_y = 4.3
    media_row_y = 0.8
    box_w = 3.4
    box_h = 1.1
    gap = 1.2

    text_positions = []
    media_positions = []

    for i in range(4):
        tx = i * (box_w + gap) + 0.0
        text_positions.append(tx + box_w / 2)

        # Text block (query)
        rect = FancyBboxPatch((tx, text_row_y), box_w, box_h,
                               boxstyle="round,pad=0.06",
                               facecolor=CH_COLORS[i], alpha=0.3,
                               edgecolor=CH_COLORS[i], linewidth=1.8)
        ax.add_patch(rect)
        ax.text(tx + box_w / 2, text_row_y + box_h / 2,
                f'Text Block {i+1}\n({CH_LABELS[i]})',
                fontsize=10, ha='center', va='center',
                fontweight='bold', color=COLORS['text_dark'])

        # Media block (key/value)
        mx = tx
        media_positions.append(mx + box_w / 2)
        rect_m = FancyBboxPatch((mx, media_row_y), box_w, box_h,
                                 boxstyle="round,pad=0.06",
                                 facecolor=CH_COLORS[i], alpha=0.2,
                                 edgecolor=CH_COLORS[i], linewidth=1.5,
                                 linestyle='--')
        ax.add_patch(rect_m)
        ax.text(mx + box_w / 2, media_row_y + box_h / 2,
                f'Media {i+1}\n(time series)',
                fontsize=9, ha='center', va='center', color=COLORS['text_mid'])

    # Post-prompt block
    pp_x = 4 * (box_w + gap) + 0.0
    text_positions.append(pp_x + box_w / 2)
    rect_pp = FancyBboxPatch((pp_x, text_row_y), box_w, box_h,
                              boxstyle="round,pad=0.06",
                              facecolor=COLORS['post'], alpha=0.3,
                              edgecolor=COLORS['post'], linewidth=1.8)
    ax.add_patch(rect_pp)
    ax.text(pp_x + box_w / 2, text_row_y + box_h / 2,
            'Post-prompt\n(diagnosis)', fontsize=10,
            ha='center', va='center', fontweight='bold', color=COLORS['text_dark'])

    # Row labels
    ax.text(-1.1, text_row_y + box_h / 2, 'Text\n(query)',
            fontsize=9.5, ha='center', va='center', color='#777777', fontweight='bold')
    ax.text(-1.1, media_row_y + box_h / 2, 'Media\n(key/value)',
            fontsize=9.5, ha='center', va='center', color='#777777', fontweight='bold')

    if is_masked:
        _draw_masked_arrows(ax, text_positions, media_positions,
                            text_row_y, media_row_y, box_h)
    else:
        _draw_unmasked_arrows(ax, text_positions, media_positions,
                              text_row_y, media_row_y, box_h)


def _draw_masked_arrows(ax, text_pos, media_pos, ty, my, bh):
    """Masked: each block → own channel only; post-prompt → last channel only."""
    # Diagonal arrows: block i → media i
    for i in range(4):
        ax.annotate(
            '', xy=(media_pos[i], my + bh),
            xytext=(text_pos[i], ty),
            arrowprops=dict(arrowstyle='->', color=CH_COLORS[i],
                            lw=2.8, shrinkA=4, shrinkB=4)
        )

    # X marks for blocked connections
    for i in range(4):
        for j in range(4):
            if j != i:
                # Compute midpoint along hypothetical arrow
                mx = (text_pos[i] * 0.45 + media_pos[j] * 0.55)
                mid_y = (ty + my + bh) / 2
                ax.text(mx, mid_y, '\u00d7', fontsize=10,
                        ha='center', va='center', color='#CCCCCC', fontweight='bold')

    # Post-prompt → Media 4 ONLY
    ax.annotate(
        '', xy=(media_pos[3], my + bh),
        xytext=(text_pos[4], ty),
        arrowprops=dict(arrowstyle='->', color=COLORS['post'],
                        lw=2.8, shrinkA=4, shrinkB=4)
    )

    # Red X for post-prompt → media 1-3
    for j in range(3):
        mx = (text_pos[4] * 0.4 + media_pos[j] * 0.6)
        mid_y = (ty + my + bh) / 2
        ax.text(mx, mid_y, '\u00d7', fontsize=13,
                ha='center', va='center', color='#DD6666', fontweight='bold')

    # Annotation callout
    ax.text(text_pos[4], ty - 0.35,
            'Post-prompt sees\nONLY last channel!',
            fontsize=11, fontweight='bold', color='#C44E52',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFEEEE',
                      edgecolor='#C44E52', linewidth=1.8))


def _draw_unmasked_arrows(ax, text_pos, media_pos, ty, my, bh):
    """Unmasked causal: block i → media 1..i; post-prompt → all media."""
    # Cumulative access labels above text blocks
    for i in range(4):
        ax.text(text_pos[i], ty + bh + 0.18,
                f'sees {i+1} channel{"s" if i > 0 else ""}',
                fontsize=8.5, ha='center', va='bottom',
                color=CH_COLORS[i], style='italic', fontweight='bold')

    ax.text(text_pos[4], ty + bh + 0.18,
            'sees ALL 4 channels',
            fontsize=9, ha='center', va='bottom',
            color=COLORS['post'], style='italic', fontweight='bold')

    # Block i → media 1..i with increasing fan
    for i in range(4):
        for j in range(i + 1):
            # Direct channel connection is most opaque
            is_direct = (j == i)
            alpha_val = 0.85 if is_direct else (0.35 + 0.1 * j)
            lw_val = 2.8 if is_direct else 1.8

            # Curve to separate overlapping arrows
            curve_rad = 0.0 if is_direct else 0.06 * (i - j) * ((-1) ** j)

            ax.annotate(
                '', xy=(media_pos[j], my + bh),
                xytext=(text_pos[i], ty),
                arrowprops=dict(
                    arrowstyle='->', color=CH_COLORS[i],
                    lw=lw_val, alpha=alpha_val,
                    shrinkA=4, shrinkB=4,
                    connectionstyle=f'arc3,rad={curve_rad}'
                )
            )

    # Post-prompt → ALL media (thick, prominent)
    for j in range(4):
        curve_rad = 0.05 * (j - 1.5)
        ax.annotate(
            '', xy=(media_pos[j], my + bh),
            xytext=(text_pos[4], ty),
            arrowprops=dict(
                arrowstyle='->', color=COLORS['post'],
                lw=3.0, alpha=0.75,
                shrinkA=4, shrinkB=4,
                connectionstyle=f'arc3,rad={curve_rad}'
            )
        )

    # Annotation callout
    ax.text(text_pos[4], ty - 0.35,
            'Post-prompt attends to\nALL channels \u2192 cross-channel fusion!',
            fontsize=11, fontweight='bold', color='#2B6CB0',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#EBF4FF',
                      edgecolor='#4C72B0', linewidth=1.8))


# ─── SECTION 2: Masked Attention ─────────────────────────────────────────────
ax_masked = fig.add_axes([0.03, 0.39, 0.94, 0.34])
draw_attention_section(
    ax_masked,
    'Masked Attention (torch.eq): Each Block \u2192 Own Channel Only',
    COLORS['bg_masked'], '#D4A574', is_masked=True
)

# ─── SECTION 3: Causal Unmasked Attention ─────────────────────────────────────
ax_unmasked = fig.add_axes([0.03, 0.02, 0.94, 0.34])
draw_attention_section(
    ax_unmasked,
    'Causal Unmasked Attention (torch.ge): Cumulative Channel Access',
    COLORS['bg_unmasked'], '#7494C4', is_masked=False
)

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = '/home/wangni/notion-figures/nomask/fig_005.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f'Saved: {out_path}')
