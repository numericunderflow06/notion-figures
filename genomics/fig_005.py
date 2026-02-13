import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data from spec
models = ['Transformer\nAttention', 'Mamba', 'Caduceus', 'OpenTSLM-\nFlamingo']
accuracies = [0.60, 0.61, 0.65, 0.71]

# Colors: baselines in gray tones, OpenTSLM-Flamingo highlighted in blue
colors = ['#9E9E9E', '#8A8A8A', '#757575', '#1565C0']
edge_colors = ['#757575', '#6B6B6B', '#5A5A5A', '#0D47A1']

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars = ax.bar(models, accuracies, width=0.55, color=colors, edgecolor=edge_colors,
              linewidth=1.2, zorder=3)

# Value labels on each bar
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f'{acc:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='#212121')

# Axis formatting
ax.set_ylim(0.50, 0.76)
ax.set_ylabel('Accuracy', fontsize=13, fontweight='medium', labelpad=10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=11)

# Grid
ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Title
ax.set_title('ClinVar Variant Pathogenicity — Accuracy', fontsize=15,
             fontweight='bold', pad=14, color='#212121')

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#BDBDBD')
ax.spines['bottom'].set_color('#BDBDBD')

# Subtitle / task label
ax.text(0.5, -0.13, 'Task: variant_effect_pathogenic_clinvar (binary classification)',
        ha='center', va='top', transform=ax.transAxes, fontsize=10,
        fontstyle='italic', color='#616161')

plt.tight_layout()
fig.savefig('/home/wangni/notion-figures/genomics/fig_005.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: /home/wangni/notion-figures/genomics/fig_005.png')
