#!/usr/bin/env python3
"""
fig_008: Trainable Parameter Breakdown
Grouped bar chart showing trainable parameter counts for each component
across Llama-3.2-1B and Llama-3.2-3B scales.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Data from architecture_plan.md Section 5.2 ---
components = ['Patch\nEncoder', 'TPE\n(Channel Emb.)', 'IGR\n(LIT+ITA+Proj)', 'Gated\nCross-Attention']

params_1b = [0.5e6, 3e3, 1.2e6, 12e6]   # Llama-3.2-1B
params_3b = [0.5e6, 3e3, 2.8e6, 36e6]    # Llama-3.2-3B

total_trainable_1b = 13.7e6
total_trainable_3b = 39.3e6
total_model_1b = 1.0e9
total_model_3b = 3.0e9
pct_1b = 1.4
pct_3b = 1.3

# --- Color palette ---
color_1b = '#4878CF'   # blue
color_3b = '#D65F5F'   # red-coral

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(11, 7.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x = np.arange(len(components))
bar_width = 0.30

bars_1b = ax.bar(x - bar_width/2, params_1b, bar_width, label='Llama-3.2-1B',
                 color=color_1b, edgecolor='white', linewidth=0.8, zorder=3)
bars_3b = ax.bar(x + bar_width/2, params_3b, bar_width, label='Llama-3.2-3B',
                 color=color_3b, edgecolor='white', linewidth=0.8, zorder=3)

# --- Log scale y-axis ---
ax.set_yscale('log')
ax.set_ylim(1e3, 3e8)

# Custom y-tick labels
ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda val, pos: f'{val/1e6:.0f}M' if val >= 1e6 else (f'{val/1e3:.0f}K' if val >= 1e3 else f'{val:.0f}')
))

# --- Grid ---
ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
ax.grid(axis='x', visible=False)

# --- Annotate each bar with exact count ---
def format_count(val):
    if val >= 1e6:
        return f'{val/1e6:.1f}M'
    elif val >= 1e3:
        return f'{val/1e3:.0f}K'
    else:
        return f'{int(val)}'

for bar_set, color in [(bars_1b, color_1b), (bars_3b, color_3b)]:
    for bar in bar_set:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.45,
                format_count(height),
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=color)

# --- Title ---
fig.suptitle('Trainable Parameter Breakdown by Component',
             fontsize=15, fontweight='bold', y=0.97)

# --- Summary annotation boxes below the title ---
fig.text(0.28, 0.915,
         f'1B Scale:  {total_trainable_1b/1e6:.1f}M trainable  /  {total_model_1b/1e9:.0f}B total  ({pct_1b}%)',
         fontsize=10.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.45', facecolor=color_1b, alpha=0.10,
                   edgecolor=color_1b, linewidth=1.3),
         color=color_1b, fontweight='bold')

fig.text(0.72, 0.915,
         f'3B Scale:  {total_trainable_3b/1e6:.1f}M trainable  /  {total_model_3b/1e9:.0f}B total  ({pct_3b}%)',
         fontsize=10.5, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.45', facecolor=color_3b, alpha=0.10,
                   edgecolor=color_3b, linewidth=1.3),
         color=color_3b, fontweight='bold')

# --- Insight annotation at bottom of plot area ---
ax.text(0.5, 0.03,
        'Gated Cross-Attention dominates trainable parameters;  '
        'core IGR contribution is lightweight (~1.2\u20132.8M)',
        transform=ax.transAxes,
        fontsize=9.5, ha='center', va='bottom', fontstyle='italic',
        color='#666666')

# --- Labels ---
ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=11)
ax.set_ylabel('Trainable Parameters (log scale)', fontsize=12, labelpad=10)

# --- Legend ---
ax.legend(fontsize=11, loc='upper left', framealpha=0.95, edgecolor='#cccccc',
          bbox_to_anchor=(0.0, 1.0))

# --- Spine styling ---
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax.spines[spine].set_color('#aaaaaa')
    ax.spines[spine].set_linewidth(0.8)

ax.tick_params(axis='both', which='both', labelsize=10, colors='#333333')

plt.subplots_adjust(top=0.86, bottom=0.13, left=0.09, right=0.96)
plt.savefig('/home/wangni/notion-figures/itformer/fig_008.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Figure saved to /home/wangni/notion-figures/itformer/fig_008.png")
