import matplotlib.pyplot as plt
import numpy as np

# Data from Section 7.1
strategies = ['Full Curriculum', 'HAR', 'Sleep', 'Scratch']
zuco1 = [70.51, 69.65, 69.25, 68.63]
zuco2 = [59.95, 59.59, 68.29, 62.25]

# Best per dataset
best_zuco1_idx = np.argmax(zuco1)  # Full Curriculum (70.51)
best_zuco2_idx = np.argmax(zuco2)  # Sleep (68.29)

x = np.arange(len(strategies))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(9, 5.5))

# Colors: warm for ZuCo 1.0, cool for ZuCo 2.0
color_zuco1 = '#E07B54'  # warm coral/orange
color_zuco2 = '#4A90D9'  # cool blue

bars1 = ax.bar(x - bar_width/2, zuco1, bar_width, label='ZuCo 1.0',
               color=color_zuco1, edgecolor='white', linewidth=0.8, zorder=3)
bars2 = ax.bar(x + bar_width/2, zuco2, bar_width, label='ZuCo 2.0',
               color=color_zuco2, edgecolor='white', linewidth=0.8, zorder=3)

# Label each bar with accuracy
for i, (bar, val) in enumerate(zip(bars1, zuco1)):
    fontweight = 'bold' if i == best_zuco1_idx else 'normal'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10,
            fontweight=fontweight, color='#333333')

for i, (bar, val) in enumerate(zip(bars2, zuco2)):
    fontweight = 'bold' if i == best_zuco2_idx else 'normal'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10,
            fontweight=fontweight, color='#333333')

# 50% random baseline
ax.axhline(y=50, color='#888888', linestyle='--', linewidth=1.2, zorder=2,
           label='Random baseline (50%)')

# Axes
ax.set_xlabel('Initialization Strategy', fontsize=12, labelpad=8)
ax.set_ylabel('Classification Accuracy (%)', fontsize=12, labelpad=8)
ax.set_title('ZuCo Classification Accuracy: Initialization Strategy Comparison',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11)
ax.set_ylim(45, 78)
ax.tick_params(axis='y', labelsize=10)

# Grid
ax.yaxis.grid(True, linestyle=':', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(fontsize=10, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

# Clean up spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#cccccc')
ax.spines['bottom'].set_color('#cccccc')

fig.set_facecolor('white')
ax.set_facecolor('white')

plt.tight_layout()
plt.savefig('/home/wangni/notion-figures/zuco/fig_003.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print('Figure saved to /home/wangni/notion-figures/zuco/fig_003.png')
