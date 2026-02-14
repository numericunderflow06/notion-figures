import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from Section 7.1
strategies = ['Sleep-flamingo', 'Scratch', 'HAR', 'Full Curriculum']
zuco1 = [69.25, 68.63, 69.65, 70.51]
zuco2 = [68.29, 62.25, 59.59, 59.95]
deltas = [z2 - z1 for z1, z2 in zip(zuco1, zuco2)]
# deltas: [-0.96, -6.38, -10.06, -10.56]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Color scheme: green for resilient (small drop), red gradient for large drops
def get_color(delta):
    if abs(delta) < 2:
        return '#2ca02c'  # green — resilient
    elif abs(delta) < 7:
        return '#e8833a'  # orange — moderate drop
    else:
        return '#d62728'  # red — large drop

colors = [get_color(d) for d in deltas]

y_pos = np.arange(len(strategies))
bars = ax.barh(y_pos, deltas, height=0.55, color=colors, edgecolor='#333333',
               linewidth=0.8, zorder=3)

# Zero line
ax.axvline(x=0, color='#555555', linewidth=1.0, zorder=2)

# Annotate each bar with delta value and ZuCo 1.0 → ZuCo 2.0 scores
for i, (delta, z1, z2) in enumerate(zip(deltas, zuco1, zuco2)):
    # Delta label inside or beside bar
    if abs(delta) > 3:
        ax.text(delta / 2, i, f'{delta:+.2f} pp',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white', zorder=4)
    else:
        ax.text(delta - 0.3, i, f'{delta:+.2f} pp',
                ha='right', va='center', fontsize=11, fontweight='bold',
                color=colors[i], zorder=4)

    # Score breakdown to the right of zero line
    ax.text(0.4, i, f'{z1:.2f}%  →  {z2:.2f}%',
            ha='left', va='center', fontsize=9.5, color='#444444',
            fontstyle='italic', zorder=4)

ax.set_yticks(y_pos)
ax.set_yticklabels(strategies, fontsize=12, fontweight='bold')
ax.set_xlabel('Accuracy Change (percentage points)', fontsize=11, labelpad=8)
ax.set_title('ZuCo 2.0 Performance Gap Analysis\nAccuracy Drop from ZuCo 1.0 → ZuCo 2.0 by Initialization Strategy',
             fontsize=13, fontweight='bold', pad=14)

# Axis formatting
ax.set_xlim(-13, 6)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.tick_params(axis='x', labelsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=1)
ax.invert_yaxis()

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Highlight box around Sleep-flamingo
bbox = mpatches.FancyBboxPatch(
    (-12.8, -0.42), 21.3, 0.84,
    boxstyle="round,pad=0.1", linewidth=1.8,
    edgecolor='#2ca02c', facecolor='#2ca02c', alpha=0.07, zorder=1
)
ax.add_patch(bbox)

# Callout annotation for Sleep-flamingo
ax.annotate('Most robust: only −0.96 pp drop',
            xy=(-0.96, 0), xytext=(-7.5, 0.42),
            fontsize=9.5, fontweight='bold', color='#1a7a1a',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
            ha='center', va='bottom', zorder=5)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ca02c', edgecolor='#333', label='Resilient (< 2 pp drop)'),
    mpatches.Patch(facecolor='#e8833a', edgecolor='#333', label='Moderate (2–7 pp drop)'),
    mpatches.Patch(facecolor='#d62728', edgecolor='#333', label='Collapse (> 7 pp drop)'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
          framealpha=0.9, edgecolor='#cccccc', fancybox=True)

# Subtitle note
fig.text(0.5, 0.01,
         'ZuCo 2.0 has 18 subjects (vs 12) and ~2217 test samples (vs ~1275), making it a harder benchmark.',
         ha='center', fontsize=8.5, color='#666666', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('/home/wangni/notion-figures/zuco/fig_008.png', dpi=200,
            facecolor='white', bbox_inches='tight')
plt.close()
print('Figure saved to /home/wangni/notion-figures/zuco/fig_008.png')
