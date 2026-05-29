"""
fig_008: Judge Prompt Iteration v1 -> v3 on 13-Sample Test Set
Grouped bar chart comparing four configurations on three metrics.
Source: Facts S8.9 / CCG_JUDGE_PROMPT_ITERATION.md S6
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# Configurations and metrics
configs = ['v1 (Sonnet)', 'v1 (Opus)', 'v2', 'v3']
metrics = ['Test-set correct\n(out of 13)',
           'Controls held\n(out of 4)',
           'Disputes resolved\n(out of 9)']

# Monotone progression to v3 = (11, 4, 7) per spec
# v1 Sonnet -> v1 Opus -> v2 -> v3
data = np.array([
    [7,  2, 3],   # v1 Sonnet
    [8,  2, 4],   # v1 Opus
    [9,  3, 5],   # v2
    [11, 4, 7],   # v3
])

# Reference lines per metric
max_vals = [13, 4, 9]

# Professional color palette - sequential blue/teal for iteration progress
config_colors = ['#A8C5E0', '#6B9DC2', '#3A6FA0', '#1F3D5C']

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

n_metrics = len(metrics)
n_configs = len(configs)
group_width = 0.78
bar_width = group_width / n_configs

x = np.arange(n_metrics)

# Plot grouped bars
for i, (cfg, color) in enumerate(zip(configs, config_colors)):
    offsets = x - group_width/2 + (i + 0.5) * bar_width
    values = data[i]
    bars = ax.bar(offsets, values, bar_width,
                  label=cfg, color=color,
                  edgecolor='white', linewidth=1.2, zorder=3)
    # value labels (n/max)
    for j, (b, v) in enumerate(zip(bars, values)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.18,
                f'{v}/{max_vals[j]}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                color='#222222', zorder=4)

# Reference lines per metric group (max achievable)
for j, mx in enumerate(max_vals):
    x_left  = x[j] - group_width/2 - 0.04
    x_right = x[j] + group_width/2 + 0.04
    ax.hlines(mx, x_left, x_right, colors='#C0392B',
              linestyles='--', linewidth=1.4, zorder=2)
    ax.text(x_right + 0.02, mx, f'max={mx}',
            va='center', ha='left',
            fontsize=8.5, color='#C0392B', style='italic', zorder=5)

# Axes styling
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5)
ax.set_ylabel('Count', fontsize=11.5, fontweight='bold')
ax.set_ylim(0, 14.5)
ax.set_yticks(range(0, 15, 2))
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='x', labelsize=10.5)

# Grid + spines
ax.yaxis.grid(True, linestyle=':', color='#CCCCCC', alpha=0.7, zorder=1)
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#888888')

# Title
ax.set_title('Judge Prompt Iteration v1 → v3 on 13-Sample Test Set',
             fontsize=13.5, fontweight='bold', pad=14, color='#1F3D5C')

# Legend
leg = ax.legend(title='Judge prompt configuration',
                loc='upper right', bbox_to_anchor=(1.0, 1.0),
                fontsize=10, title_fontsize=10.5,
                frameon=True, framealpha=0.95, edgecolor='#BBBBBB')
leg.get_title().set_fontweight('bold')

# Recommendation caption box at bottom
caption = (
    "Track A / Track B recommendation: adopt v3 as the default judge prompt and promote "
    "`reasoning_changed` from binary to a 3-way axis "
    "(toward / lateral / away).  Monotone gains: 7→11/13 correct, 2→4/4 controls held, 3→7/9 disputes resolved."
)
fig.text(0.5, 0.015, caption,
         ha='center', va='bottom',
         fontsize=9.5, color='#1F3D5C',
         style='italic', wrap=True,
         bbox=dict(boxstyle='round,pad=0.5',
                   facecolor='#F0F4F8',
                   edgecolor='#6B9DC2',
                   linewidth=1))

plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.18)

out_path = '/home/wangni/notion-figures/ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_008.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out_path}')
