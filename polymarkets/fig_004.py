import matplotlib.pyplot as plt
import numpy as np

# Data from verified facts
tasks = ['Resolution', 'Volatility', 'Conf.\nEvolution', 'Past\nTrend', 'Future\nTrend', 'Overall']
flamingo_acc = [100, 100, 80, 67, 53, 76]
baseline_acc = [70, 70, 60, 33, 33, 50]

x = np.arange(len(tasks))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(10, 5.5))

# Bars
bars_flamingo = ax.bar(x - bar_width/2, flamingo_acc, bar_width,
                       label='Trained Flamingo', color='#2563EB', zorder=3)
bars_baseline = ax.bar(x + bar_width/2, baseline_acc, bar_width,
                       label='Text-Only Baseline', color='#C8C8C8', zorder=3)

# 50% random baseline reference line
ax.axhline(y=50, color='#888888', linestyle='--', linewidth=0.9, zorder=2, label='Random baseline (50%)')

# Exact percentage labels on each bar
for bar in bars_flamingo:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
            f'{int(height)}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#1e40af')

for bar in bars_baseline:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
            f'{int(height)}%', ha='center', va='bottom',
            fontsize=10, color='#555555')

# Axes
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='medium')
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10.5)
ax.tick_params(axis='y', labelsize=10)

# Title
ax.set_title('Task-Wise Accuracy: Trained Flamingo vs Text-Only Baseline',
             fontsize=13.5, fontweight='bold', pad=14)

# Grid
ax.yaxis.grid(True, linestyle=':', alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Legend
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.set_facecolor('white')
ax.set_facecolor('white')

# Separator line before "Overall"
sep_x = x[-2] + 0.5 + (x[-1] - x[-2]) * 0.0
ax.axvline(x=sep_x, color='#aaaaaa', linestyle='-', linewidth=0.7, ymin=0, ymax=0.92, zorder=1)

plt.tight_layout()
plt.savefig('/home/wangni/notion-figures/polymarkets/fig_004.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()

print('Figure saved to /home/wangni/notion-figures/polymarkets/fig_004.png')
