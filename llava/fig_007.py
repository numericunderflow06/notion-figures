import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Data from Section 5.7 / architecture_plan.md
# ECG-QA with LLaMA-3B unless noted
models = [
    'LLaVA-TSM\n(3B)',
    'OpenTSLM\nSoftPrompt (3B)',
    'OpenTSLM\nFlamingo (3B)',
    'LLaVA-TSM\n(1B)',
]
memory_gb = [60, 110, 40, 30]

# Color palette – distinct per architecture family
colors = ['#2563EB', '#E11D48', '#7C3AED', '#60A5FA']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    np.arange(len(models)),
    memory_gb,
    width=0.55,
    color=colors,
    edgecolor='white',
    linewidth=1.2,
    zorder=3,
)

# Exact GB annotations on each bar
for bar, mem in zip(bars, memory_gb):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f'{mem} GB',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold',
        color='#1e293b',
    )

# Additional note for LLaVA-TSM 3B (multi-GPU)
ax.text(
    bars[0].get_x() + bars[0].get_width() / 2,
    bars[0].get_height() / 2,
    '4×A100\nw/ ZeRO-3',
    ha='center',
    va='center',
    fontsize=8.5,
    color='white',
    fontweight='bold',
)

# Horizontal dashed line at 80 GB (single A100 capacity)
ax.axhline(
    y=80,
    color='#64748b',
    linestyle='--',
    linewidth=1.3,
    zorder=2,
)
ax.text(
    len(models) - 0.5,
    82,
    'Single A100 capacity (80 GB)',
    ha='right',
    va='bottom',
    fontsize=9,
    color='#64748b',
    fontstyle='italic',
)

# Axes
ax.set_xticks(np.arange(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('Estimated GPU Memory (GB)', fontsize=11, labelpad=8)
ax.set_ylim(0, 130)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

# Title
ax.set_title(
    'GPU Memory Comparison Across Architectures\n(ECG-QA, LLaMA backbone)',
    fontsize=13,
    fontweight='bold',
    pad=12,
)

# Grid – light horizontal only
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='#cbd5e1', zorder=0)
ax.set_axisbelow(True)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#94a3b8')
ax.spines['bottom'].set_color('#94a3b8')
ax.tick_params(axis='both', colors='#334155')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.tight_layout()
plt.savefig(
    '/home/wangni/notion-figures/llava/fig_007.png',
    dpi=200,
    bbox_inches='tight',
    facecolor='white',
)
plt.close()

print('Saved: /home/wangni/notion-figures/llava/fig_007.png')
