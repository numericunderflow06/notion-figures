"""
Figure 002: DNA to Time Series Encoding
Shows how a short DNA sequence (ACGTACGT) is encoded into 5 parallel channels
representing biophysical properties.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Data from codebase (GenomicsLRBDataset.py) ---
sequence = "ACGTACGT"

# Encoding values per channel, per base
# Channel 0: Base Identity  (A=1, C=2, G=3, T=4)
# Channel 1: Purine/Pyrimidine (Purine=1, Pyrimidine=0)
# Channel 2: Amino/Keto (Amino=1, Keto=0)
# Channel 3: Hydrogen Bonds (A-T=2, G-C=3)
# Channel 4: GC Content (G/C=1, A/T=0)

encoding = {
    'A': [1.0, 1.0, 1.0, 2.0, 0.0],
    'C': [2.0, 0.0, 1.0, 3.0, 1.0],
    'G': [3.0, 1.0, 0.0, 3.0, 1.0],
    'T': [4.0, 0.0, 0.0, 2.0, 0.0],
}

channel_names = [
    "Ch 0: Base Identity",
    "Ch 1: Purine / Pyrimidine",
    "Ch 2: Amino / Keto",
    "Ch 3: Hydrogen Bonds",
    "Ch 4: GC Content",
]

channel_value_labels = [
    {1.0: "A=1", 2.0: "C=2", 3.0: "G=3", 4.0: "T=4"},
    {1.0: "Pur=1", 0.0: "Pyr=0"},
    {1.0: "Amino=1", 0.0: "Keto=0"},
    {2.0: "2 (A-T)", 3.0: "3 (G-C)"},
    {1.0: "G/C=1", 0.0: "A/T=0"},
]

# Build encoding matrix: shape (5, 8)
n_channels = 5
n_bases = len(sequence)
matrix = np.zeros((n_channels, n_bases))
for j, base in enumerate(sequence):
    vals = encoding[base]
    for i in range(n_channels):
        matrix[i, j] = vals[i]

# --- Colors ---
base_colors = {'A': '#2ca02c', 'C': '#1f77b4', 'G': '#bcbd22', 'T': '#d62728'}

# Channel color palettes (each channel gets its own 2-color or multi-color map)
channel_palettes = [
    ['#f0f0f0', '#1b4f72'],  # Base Identity: light to dark blue
    ['#fce4ec', '#6a1b9a'],  # Purine/Pyr: light pink to purple
    ['#e8f5e9', '#1b5e20'],  # Amino/Keto: light to dark green
    ['#fff3e0', '#e65100'],  # H-bonds: light to dark orange
    ['#e3f2fd', '#0d47a1'],  # GC Content: light to dark navy
]

# --- Figure setup ---
fig = plt.figure(figsize=(12, 9), facecolor='white')

# Use gridspec for layout: DNA row on top, then 5 channel rows
gs = fig.add_gridspec(
    nrows=7, ncols=1,
    height_ratios=[1.4, 0.25, 1, 1, 1, 1, 1],
    hspace=0.30,
    left=0.18, right=0.92, top=0.92, bottom=0.04
)

# --- Title ---
fig.suptitle("DNA to Time Series Encoding", fontsize=16, fontweight='bold',
             y=0.97, color='#222222')

# --- Top panel: DNA sequence as colored letters ---
ax_dna = fig.add_subplot(gs[0])
ax_dna.set_xlim(-0.5, n_bases - 0.5)
ax_dna.set_ylim(0, 1)
ax_dna.axis('off')

# Draw each base as a colored box with letter
box_width = 0.85
for j, base in enumerate(sequence):
    color = base_colors[base]
    rect = mpatches.FancyBboxPatch(
        (j - box_width / 2, 0.15), box_width, 0.7,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor='#333333', linewidth=1.2, alpha=0.85
    )
    ax_dna.add_patch(rect)
    ax_dna.text(j, 0.50, base, ha='center', va='center',
                fontsize=22, fontweight='bold', color='white',
                fontfamily='monospace')

# Position labels
for j in range(n_bases):
    ax_dna.text(j, -0.05, str(j), ha='center', va='top',
                fontsize=9, color='#666666')

ax_dna.text(-0.5, 0.50, "DNA\nSequence", ha='right', va='center',
            fontsize=11, fontweight='bold', color='#333333',
            transform=ax_dna.transData)

# Arrow between DNA and channels
ax_arrow = fig.add_subplot(gs[1])
ax_arrow.axis('off')
ax_arrow.annotate(
    '', xy=(0.5, 0.0), xytext=(0.5, 1.0),
    arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0),
    xycoords='axes fraction'
)
ax_arrow.text(0.5, 0.5, "encode", ha='center', va='center',
              fontsize=10, fontstyle='italic', color='#555555',
              transform=ax_arrow.transAxes)

# --- Channel panels: step plots with heatmap coloring ---
for ch_idx in range(n_channels):
    ax = fig.add_subplot(gs[ch_idx + 2])
    values = matrix[ch_idx]
    vmin = values.min()
    vmax = values.max()

    # Create colormap for this channel
    cmap = LinearSegmentedColormap.from_list(
        f'ch{ch_idx}', channel_palettes[ch_idx], N=256
    )

    # Draw colored bars for each position
    for j in range(n_bases):
        val = values[j]
        # Normalize value for color mapping
        if vmax > vmin:
            norm_val = (val - vmin) / (vmax - vmin)
        else:
            norm_val = 0.5
        color = cmap(norm_val)

        rect = mpatches.FancyBboxPatch(
            (j - 0.42, 0.05), 0.84, 0.9,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor='#aaaaaa', linewidth=0.8
        )
        ax.add_patch(rect)

        # Value text inside each cell
        # Choose text color for contrast
        text_color = 'white' if norm_val > 0.55 else '#222222'
        ax.text(j, 0.50, f"{val:.0f}", ha='center', va='center',
                fontsize=13, fontweight='bold', color=text_color,
                fontfamily='monospace')

    ax.set_xlim(-0.5, n_bases - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Channel label on the left
    ax.text(-0.5, 0.50, channel_names[ch_idx], ha='right', va='center',
            fontsize=10.5, fontweight='bold', color='#333333',
            transform=ax.transData)

    # Value legend on the right
    unique_vals = sorted(set(values))
    legend_parts = []
    for uv in unique_vals:
        if uv in channel_value_labels[ch_idx]:
            legend_parts.append(channel_value_labels[ch_idx][uv])
    legend_text = "  ".join(legend_parts)
    ax.text(n_bases - 0.3, 0.50, legend_text, ha='left', va='center',
            fontsize=9, color='#555555', transform=ax.transData)

# --- Base color legend at bottom right ---
legend_handles = [
    mpatches.Patch(facecolor=base_colors['A'], edgecolor='#333', label='A (Adenine)'),
    mpatches.Patch(facecolor=base_colors['C'], edgecolor='#333', label='C (Cytosine)'),
    mpatches.Patch(facecolor=base_colors['G'], edgecolor='#333', label='G (Guanine)'),
    mpatches.Patch(facecolor=base_colors['T'], edgecolor='#333', label='T (Thymine)'),
]
fig.legend(
    handles=legend_handles, loc='lower center', ncol=4,
    fontsize=10, frameon=True, edgecolor='#cccccc',
    fancybox=True, shadow=False,
    bbox_to_anchor=(0.55, -0.005)
)

# --- Save ---
out_path = "/home/wangni/notion-figures/genomics/fig_002.png"
fig.savefig(out_path, dpi=200, facecolor='white', bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to {out_path}")
