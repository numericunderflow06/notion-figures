import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(8, 12), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# Color scheme
colors = {
    'generation': '#B3D9FF',   # light blue
    'forward':    '#4A90D9',   # blue
    'loss':       '#F5A623',   # orange
    'update':     '#7BC67E',   # green
    'title_bg':   '#2C3E50',   # dark blue-gray
}
text_colors = {
    'generation': '#1A3A5C',
    'forward':    '#FFFFFF',
    'loss':       '#5C3A00',
    'update':     '#1A3A1A',
}
border_colors = {
    'generation': '#6AAED6',
    'forward':    '#2C6FAC',
    'loss':       '#D4891A',
    'update':     '#4A9E4D',
}

# Title
ax.text(5, 13.3, 'SDFT Training Loop: Step-by-Step Process',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color='#2C3E50', family='sans-serif')
ax.plot([1.5, 8.5], [12.95, 12.95], color='#2C3E50', linewidth=1.5)

# Step definitions
steps = [
    {
        'num': '1',
        'title': 'On-Policy Generation by Student',
        'details': [
            'Student generates CoT from time series input',
            'Teacher receives golden CoT as demonstration',
            'torch.no_grad()  \u2014  no gradients tracked',
        ],
        'color': 'generation',
        'y_center': 11.3,
    },
    {
        'num': '2',
        'title': 'Forward Pass: Student & Teacher Logits',
        'details': [
            'Student forward pass  (with grad)',
            'Teacher forward pass  (torch.no_grad())',
            'Compute token-level logit distributions',
        ],
        'color': 'forward',
        'y_center': 8.7,
    },
    {
        'num': '3',
        'title': 'KL Divergence Loss Computation',
        'details': [
            'Forward KL: D_KL(teacher \u2016 student)',
            'Sum over vocabulary dimension',
            'Mask padding tokens  \u2022  Average over sequence',
        ],
        'color': 'loss',
        'y_center': 6.1,
    },
    {
        'num': '4',
        'title': 'Backward Pass + EMA Teacher Update',
        'details': [
            'Backward pass  \u2192  gradient clipping',
            'Optimizer step (student parameters)',
            'EMA update: \u03B8_teacher \u2190 \u03B1\u00b7\u03B8_student + (1\u2212\u03B1)\u00b7\u03B8_teacher',
        ],
        'color': 'update',
        'y_center': 3.5,
    },
]

box_width = 6.6
box_height = 1.9
box_x = 5 - box_width / 2

for step in steps:
    y = step['y_center']
    col_key = step['color']
    bg = colors[col_key]
    tc = text_colors[col_key]
    bc = border_colors[col_key]

    # Main rounded box
    rect = FancyBboxPatch(
        (box_x, y - box_height / 2), box_width, box_height,
        boxstyle="round,pad=0.15", linewidth=2,
        edgecolor=bc, facecolor=bg, zorder=2
    )
    ax.add_patch(rect)

    # Step number circle
    circle = plt.Circle((box_x + 0.55, y + box_height / 2 - 0.45), 0.3,
                         color=bc, zorder=3)
    ax.add_patch(circle)
    ax.text(box_x + 0.55, y + box_height / 2 - 0.45, step['num'],
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white', zorder=4)

    # Title text
    ax.text(box_x + 1.1, y + box_height / 2 - 0.45, step['title'],
            ha='left', va='center', fontsize=12.5, fontweight='bold',
            color=tc, zorder=3)

    # Detail lines
    for i, detail in enumerate(step['details']):
        ax.text(box_x + 0.55, y + box_height / 2 - 1.0 - i * 0.3, detail,
                ha='left', va='center', fontsize=10, color=tc,
                family='sans-serif', zorder=3, style='italic' if 'no_grad' in detail or 'with grad' in detail else 'normal')

# Arrows between steps (straight down)
arrow_props = dict(
    arrowstyle='->', color='#555555', lw=2.0,
    connectionstyle='arc3,rad=0', mutation_scale=18,
    shrinkA=0, shrinkB=0,
)

for i in range(3):
    y_start = steps[i]['y_center'] - box_height / 2
    y_end = steps[i + 1]['y_center'] + box_height / 2
    ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=2.2, mutation_scale=18))

# Cyclic arrow from Step 4 back to Step 1
# Path: down from step 4, right, up along the right side, then left into step 1
cycle_color = '#2C3E50'

# Down from bottom of step 4
y4_bot = steps[3]['y_center'] - box_height / 2
y1_top = steps[0]['y_center'] + box_height / 2

right_x = box_x + box_width + 0.6
mid_x = 5

# Draw the cyclic path with individual segments
# Segment 1: down from step 4 bottom center
ax.annotate('', xy=(mid_x, y4_bot - 0.35), xytext=(mid_x, y4_bot),
            arrowprops=dict(arrowstyle='-', color=cycle_color, lw=2.2))

# Segment 2: right turn
ax.annotate('', xy=(right_x, y4_bot - 0.35), xytext=(mid_x, y4_bot - 0.35),
            arrowprops=dict(arrowstyle='-', color=cycle_color, lw=2.2))

# Segment 3: up the right side
ax.annotate('', xy=(right_x, y1_top + 0.35), xytext=(right_x, y4_bot - 0.35),
            arrowprops=dict(arrowstyle='-', color=cycle_color, lw=2.2))

# Segment 4: left turn back to step 1 with arrow
ax.annotate('', xy=(mid_x + 0.05, y1_top + 0.35), xytext=(right_x, y1_top + 0.35),
            arrowprops=dict(arrowstyle='->', color=cycle_color, lw=2.2,
                            mutation_scale=20))

# "Repeat" label on cycle arrow
ax.text(right_x + 0.35, (y4_bot + y1_top) / 2, 'Iterate',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=cycle_color, rotation=90, family='sans-serif')

# Legend
legend_y = 1.3
legend_items = [
    ('Generation', colors['generation'], border_colors['generation']),
    ('Forward Pass', colors['forward'], border_colors['forward']),
    ('Loss Computation', colors['loss'], border_colors['loss']),
    ('Parameter Update', colors['update'], border_colors['update']),
]
legend_x_start = 1.3
spacing = 2.0

for i, (label, bg, ec) in enumerate(legend_items):
    x = legend_x_start + i * spacing
    rect = FancyBboxPatch((x, legend_y - 0.18), 0.35, 0.35,
                           boxstyle="round,pad=0.05", facecolor=bg,
                           edgecolor=ec, linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + 0.55, legend_y, label, ha='left', va='center',
            fontsize=9.5, color='#333333', family='sans-serif')

# Note at bottom
ax.text(5, 0.55, 'mixup_alpha = 0.01  |  EMA sync every step  |  On-policy training loop',
        ha='center', va='center', fontsize=9, color='#777777',
        family='sans-serif', style='italic')

plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/sdft/fig_006.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print('Figure saved to /home/wangni/notion-figures/sdft/fig_006.png')
