"""
fig_002: Masked vs. Causal Unmasked vs. Full Unmasked Attention Patterns
Side-by-side heatmap grids comparing three attention conditions for a 4-channel example.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# ---------- Data from spec (Section 4.2 / 5.3) ----------
# Rows: Pre-prompt, Block 1, Block 2, Block 3, Block 4, Post-prompt
# Cols: Media 1, Media 2, Media 3, Media 4
# 1 = can attend, 0 = blocked

masked = np.array([
    [0, 0, 0, 0],  # Pre-prompt (zeroed)
    [1, 0, 0, 0],  # Block 1 (eq 1)
    [0, 1, 0, 0],  # Block 2 (eq 2)
    [0, 0, 1, 0],  # Block 3 (eq 3)
    [0, 0, 0, 1],  # Block 4 (eq 4)
    [0, 0, 0, 1],  # Post-prompt (text_time=4, eq only last)
])

causal_unmasked = np.array([
    [0, 0, 0, 0],  # Pre-prompt (zeroed)
    [1, 0, 0, 0],  # Block 1 (ge 1)
    [1, 1, 0, 0],  # Block 2 (ge 2)
    [1, 1, 1, 0],  # Block 3 (ge 3)
    [1, 1, 1, 1],  # Block 4 (ge 4)
    [1, 1, 1, 1],  # Post-prompt (ge all)
])

full_unmasked = np.array([
    [1, 1, 1, 1],  # Pre-prompt (no mask)
    [1, 1, 1, 1],  # Block 1
    [1, 1, 1, 1],  # Block 2
    [1, 1, 1, 1],  # Block 3
    [1, 1, 1, 1],  # Block 4
    [1, 1, 1, 1],  # Post-prompt
])

row_labels = ['Pre-prompt', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Post-prompt']
col_labels = ['Media 1\n(volt1)', 'Media 2\n(amp1)', 'Media 3\n(OilT)', 'Media 4\n(RPM)']
panel_titles = [
    'Masked\n(torch.eq — current)',
    'Causal Unmasked\n(torch.ge — proposed)',
    'Full Unmasked\n(no mask — ablation)',
]
matrices = [masked, causal_unmasked, full_unmasked]

# ---------- Color scheme ----------
color_blocked = '#E8E8E8'   # light gray
color_allowed = '#4CAF50'   # green
cmap = ListedColormap([color_blocked, color_allowed])

# ---------- Figure setup ----------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0))
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.09, right=0.91, top=0.78, bottom=0.08, wspace=0.35)

nrows, ncols = 6, 4

for ax_idx, (ax, mat, title) in enumerate(zip(axes, matrices, panel_titles)):
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=14)

    # Draw cells
    for i in range(nrows):
        for j in range(ncols):
            val = mat[i, j]
            fc = color_allowed if val == 1 else color_blocked
            ec = '#AAAAAA'
            lw = 0.8

            rect = mpatches.FancyBboxPatch(
                (j - 0.45, i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.04",
                facecolor=fc, edgecolor=ec, linewidth=lw,
            )
            ax.add_patch(rect)

            # Checkmark or cross
            if val == 1:
                ax.text(j, i, '✓', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='white')
            else:
                ax.text(j, i, '✗', ha='center', va='center',
                        fontsize=13, fontweight='bold', color='#BBBBBB')

    # Bold border around the post-prompt row (row index 5)
    post_row = nrows - 1
    rect_border = mpatches.FancyBboxPatch(
        (-0.5, post_row - 0.5), ncols, 1.0,
        boxstyle="round,pad=0.02",
        facecolor='none', edgecolor='#D32F2F', linewidth=2.5,
        linestyle='-', zorder=5,
    )
    ax.add_patch(rect_border)

    # Y-axis labels (row labels) — only for the leftmost panel
    ax.set_yticks(range(nrows))
    if ax_idx == 0:
        formatted_labels = []
        for lbl in row_labels:
            if lbl == 'Post-prompt':
                formatted_labels.append(lbl)
            else:
                formatted_labels.append(lbl)
        ax.set_yticklabels(row_labels, fontsize=10.5)
        # Bold the post-prompt label
        ax.get_yticklabels()[-1].set_fontweight('bold')
        ax.get_yticklabels()[-1].set_color('#D32F2F')
    else:
        ax.set_yticklabels([])

    # X-axis labels (column labels)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=9.5, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

# ---------- Annotation arrow pointing to post-prompt row ----------
# Place on the rightmost panel
ax_right = axes[2]
# Arrow from outside the panel pointing to the post-prompt row
ax_right.annotate(
    'Critical:\ndiagnosis\ngeneration\npoint',
    xy=(ncols - 0.5 + 0.15, nrows - 1),  # arrow tip at post-prompt row, right edge
    xytext=(ncols + 0.9, nrows - 1.8),     # text position
    fontsize=9.5, fontweight='bold', color='#D32F2F',
    ha='left', va='center',
    arrowprops=dict(
        arrowstyle='->', color='#D32F2F', lw=2.0,
        connectionstyle='arc3,rad=-0.15',
    ),
    annotation_clip=False,
    zorder=10,
)

# ---------- Legend ----------
legend_elements = [
    mpatches.Patch(facecolor=color_allowed, edgecolor='#888', label='Attends (allowed)'),
    mpatches.Patch(facecolor=color_blocked, edgecolor='#888', label='Blocked'),
    mpatches.Patch(facecolor='none', edgecolor='#D32F2F', linewidth=2,
                   label='Post-prompt row (diagnosis)'),
]
fig.legend(
    handles=legend_elements, loc='lower center', ncol=3,
    fontsize=10, frameon=True, fancybox=True, edgecolor='#CCCCCC',
    bbox_to_anchor=(0.48, -0.01),
)

# ---------- Main title ----------
fig.suptitle(
    'Masked vs. Causal Unmasked Attention Patterns',
    fontsize=15, fontweight='bold', y=0.97,
)

# ---------- Save ----------
out_path = '/home/wangni/notion-figures/nomask/fig_002.png'
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Figure saved to {out_path}')
