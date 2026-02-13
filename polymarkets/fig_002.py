"""
fig_002: Data Pipeline: From Raw Markets to Evaluation
Horizontal flow diagram showing the 7-step data pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Color palette (muted professional) ---
COLORS = {
    'bg': '#FFFFFF',
    'step1': '#4A7C9B',   # Steel blue - data acquisition
    'step2': '#5B8C5A',   # Sage green - generation
    'step3': '#8B7355',   # Warm brown - splitting
    'step4': '#7B6B8D',   # Muted purple - balancing
    'step5': '#C47A5A',   # Terracotta - training
    'step6': '#5A8B8B',   # Teal - evaluation
    'step7': '#8B5A5A',   # Dusty rose - comparison
    'arrow': '#555555',
    'text_dark': '#2C2C2C',
    'text_light': '#FFFFFF',
    'annotation': '#666666',
    'branch_trained': '#C47A5A',
    'branch_baseline': '#8B5A5A',
}

fig, ax = plt.subplots(figsize=(20, 9), facecolor=COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.set_xlim(-0.5, 19.5)
ax.set_ylim(-1.8, 8.0)
ax.axis('off')

# --- Title ---
ax.text(9.5, 7.4, 'Data Pipeline: From Raw Markets to Evaluation',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=COLORS['text_dark'], fontfamily='sans-serif')

# --- Helper: draw a rounded box ---
def draw_step_box(ax, x, y, w, h, color, step_num, title, subtitle='', annotation=''):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=color, edgecolor='none',
                         alpha=0.92, zorder=3)
    ax.add_patch(box)

    # Step number circle
    circle = plt.Circle((x + 0.40, y + h - 0.38), 0.24,
                         color='white', alpha=0.3, zorder=4)
    ax.add_patch(circle)
    ax.text(x + 0.40, y + h - 0.38, str(step_num),
            fontsize=10.5, fontweight='bold', ha='center', va='center',
            color=COLORS['text_light'], zorder=5)

    # Title (shifted right to account for number circle)
    title_y = y + h - 0.38 if not subtitle else y + h - 0.32
    ax.text(x + w / 2 + 0.15, title_y, title,
            fontsize=11.5, fontweight='bold', ha='center', va='center',
            color=COLORS['text_light'], zorder=5, fontfamily='sans-serif')

    # Subtitle
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.18, subtitle,
                fontsize=9.5, ha='center', va='center',
                color=COLORS['text_light'], alpha=0.9, zorder=5,
                fontfamily='sans-serif', style='italic')

    # Annotation below box
    if annotation:
        ax.text(x + w / 2, y - 0.30, annotation,
                fontsize=9, ha='center', va='top',
                color=COLORS['annotation'], zorder=5,
                fontfamily='sans-serif', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#F0F0F0',
                          edgecolor='#CCCCCC', alpha=0.8))

def draw_arrow(ax, x1, y1, x2, y2, color=None, style='->', lw=2, ls='-'):
    if color is None:
        color = COLORS['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls,
                                connectionstyle='arc3,rad=0'),
                zorder=2)


# ==========================================
# Layout:
# Row 1 (top, y~4.5): Steps 1-4 left to right
# Row 2 (bottom, y~1.0): Steps 5,6,7 right to left
# ==========================================

box_w = 3.3
box_h = 1.55
row1_y = 4.3
row2_y = 0.8
gap_x = 0.7

# --- ROW 1: Steps 1-4 (left to right, starting at x=1) ---
r1_start_x = 1.0
x_positions_r1 = []
for i in range(4):
    x = r1_start_x + i * (box_w + gap_x)
    x_positions_r1.append(x)

# Step 1: Download markets
draw_step_box(ax, x_positions_r1[0], row1_y, box_w, box_h,
              COLORS['step1'], 1,
              'Download Markets',
              'Polymarket CLOB API',
              '5,000 active markets (426 MB)')

# Step 2: Generate questions
draw_step_box(ax, x_positions_r1[1], row1_y, box_w, box_h,
              COLORS['step2'], 2,
              'Generate Questions',
              'Rule-based ground truth',
              '22,521 questions (5 types)')

# Step 3: Split by resolution
draw_step_box(ax, x_positions_r1[2], row1_y, box_w, box_h,
              COLORS['step3'], 3,
              'Split by Resolution',
              'Group by outcomes',
              'Equal type/answer split')

# Step 4: Create balanced dataset
draw_step_box(ax, x_positions_r1[3], row1_y, box_w, box_h,
              COLORS['step4'], 4,
              'Balance Dataset',
              'Equal representation',
              '1,000 questions')

# Arrows Row 1: step 1->2, 2->3, 3->4
for i in range(3):
    x_start = x_positions_r1[i] + box_w
    x_end = x_positions_r1[i + 1]
    y_mid = row1_y + box_h / 2
    draw_arrow(ax, x_start + 0.06, y_mid, x_end - 0.06, y_mid)


# --- ROW 2: Steps 5-7 (right to left under row 1) ---
# Step 5 under step 4, step 6 under step 3, step 7 under step 2
r2_positions = [x_positions_r1[3], x_positions_r1[2], x_positions_r1[1]]

# Step 5: Train Flamingo (under step 4)
draw_step_box(ax, r2_positions[0], row2_y, box_w, box_h,
              COLORS['step5'], 5,
              'Train Flamingo',
              '1 epoch, ~45 min on A100',
              'Balanced 1k dataset')

# Step 6: Evaluate (under step 3)
draw_step_box(ax, r2_positions[1], row2_y, box_w, box_h,
              COLORS['step6'], 6,
              'Evaluate',
              '5 per category',
              'Trained model inference')

# Step 7: Compare (under step 2)
draw_step_box(ax, r2_positions[2], row2_y, box_w, box_h,
              COLORS['step7'], 7,
              'Compare Results',
              'Trained vs Baseline',
              '76% vs 50% accuracy')


# --- Connector: Step 4 down to Step 5 ---
s4_cx = x_positions_r1[3] + box_w / 2
# Vertical arrow from bottom of step 4 to top of step 5
draw_arrow(ax, s4_cx, row1_y - 0.06, s4_cx, row2_y + box_h + 0.06)

# --- Arrows Row 2: step 5->6, 6->7 (right to left) ---
draw_arrow(ax, r2_positions[0] - 0.06, row2_y + box_h / 2,
           r2_positions[1] + box_w + 0.06, row2_y + box_h / 2)

draw_arrow(ax, r2_positions[1] - 0.06, row2_y + box_h / 2,
           r2_positions[2] + box_w + 0.06, row2_y + box_h / 2)


# --- "Trained Model Path" label ---
trained_label_x = (r2_positions[0] + r2_positions[1] + box_w) / 2
ax.text(trained_label_x, row2_y + box_h + 0.28,
        'Trained Model Path',
        fontsize=9, ha='center', va='bottom',
        color=COLORS['branch_trained'], fontweight='bold',
        fontfamily='sans-serif')


# --- Baseline path: dashed from step 4 area to step 7 ---
# The baseline path bypasses training: balanced dataset -> text-only baseline -> compare
baseline_start_x = x_positions_r1[1] + box_w / 2  # Under step 2 area
baseline_mid_y = row2_y + box_h + 1.5

# "Baseline Path" box
bl_box_w = 2.8
bl_box_h = 1.1
bl_box_x = x_positions_r1[0] + box_w / 2 - bl_box_w / 2 - 0.3
bl_box_y = baseline_mid_y - bl_box_h / 2 - 0.15

# Place baseline info box between rows, roughly between step 4 and step 7
bl_info_x = (x_positions_r1[1] + box_w / 2)
bl_info_y = (row1_y + row2_y + box_h) / 2

# Instead of a separate box, draw a dashed path from step 4's output side down and left to step 7
# Path: from step 4 bottom-left area -> angle down-left -> step 7 top

# Dashed connector from Step 4 (bottom-left) curving to Step 7 (top)
# We'll use a simple two-segment path with text annotation

# Midpoint for the baseline annotation
mid_x = (x_positions_r1[3] + r2_positions[2] + box_w) / 2
mid_y = (row1_y + row2_y + box_h) / 2 + 0.15

ax.text(mid_x - 1.3, mid_y + 0.15, 'Baseline Path',
        fontsize=9, fontweight='bold', ha='center', va='bottom',
        color=COLORS['branch_baseline'], fontfamily='sans-serif')
ax.text(mid_x - 1.3, mid_y - 0.15, 'Text-only Qwen 2.5-7B\n(50-point downsampled)',
        fontsize=8.5, ha='center', va='top',
        color=COLORS['annotation'], fontfamily='sans-serif', style='italic')

# Dashed line from bottom of step 4 area going left-down to top of step 7
# Segment 1: from step 4 bottom-center down to midpoint
s4_bx = x_positions_r1[3] + box_w * 0.25
s7_cx = r2_positions[2] + box_w / 2

ax.annotate('',
            xy=(mid_x - 1.3, mid_y - 0.55),
            xytext=(s4_bx, row1_y - 0.06),
            arrowprops=dict(arrowstyle='-', color=COLORS['branch_baseline'],
                            lw=1.8, linestyle='dashed',
                            connectionstyle='arc3,rad=-0.15'),
            zorder=1)

ax.annotate('',
            xy=(s7_cx, row2_y + box_h + 0.06),
            xytext=(mid_x - 1.3, mid_y - 0.75),
            arrowprops=dict(arrowstyle='->', color=COLORS['branch_baseline'],
                            lw=1.8, linestyle='dashed',
                            connectionstyle='arc3,rad=-0.15'),
            zorder=1)


# --- Data volume flow annotation along top ---
flow_y = 6.5
ax.text(x_positions_r1[0] + box_w / 2, flow_y, '5,000',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=COLORS['step1'], fontfamily='sans-serif')
ax.text(x_positions_r1[0] + box_w / 2, flow_y - 0.35, 'markets',
        fontsize=9.5, ha='center', va='center',
        color=COLORS['annotation'], fontfamily='sans-serif')

ax.annotate('', xy=(x_positions_r1[1] + box_w / 2 - 0.7, flow_y),
            xytext=(x_positions_r1[0] + box_w / 2 + 0.65, flow_y),
            arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=1.5),
            zorder=1)

ax.text(x_positions_r1[1] + box_w / 2, flow_y, '22,521',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=COLORS['step2'], fontfamily='sans-serif')
ax.text(x_positions_r1[1] + box_w / 2, flow_y - 0.35, 'questions',
        fontsize=9.5, ha='center', va='center',
        color=COLORS['annotation'], fontfamily='sans-serif')

ax.annotate('', xy=(x_positions_r1[3] + box_w / 2 - 0.55, flow_y),
            xytext=(x_positions_r1[1] + box_w / 2 + 0.75, flow_y),
            arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=1.5),
            zorder=1)

ax.text(x_positions_r1[3] + box_w / 2, flow_y, '1,000',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=COLORS['step4'], fontfamily='sans-serif')
ax.text(x_positions_r1[3] + box_w / 2, flow_y - 0.35, 'balanced',
        fontsize=9.5, ha='center', va='center',
        color=COLORS['annotation'], fontfamily='sans-serif')


# --- Legend ---
legend_x, legend_y = 15.5, -1.1
# Solid arrow
ax.plot([legend_x, legend_x + 0.8], [legend_y, legend_y],
        color=COLORS['arrow'], lw=2, solid_capstyle='round')
ax.annotate('', xy=(legend_x + 1.0, legend_y),
            xytext=(legend_x + 0.8, legend_y),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2))
ax.text(legend_x + 1.2, legend_y, 'Pipeline flow',
        fontsize=9, va='center', color=COLORS['text_dark'],
        fontfamily='sans-serif')

# Dashed arrow
ax.plot([legend_x, legend_x + 0.8], [legend_y - 0.5, legend_y - 0.5],
        color=COLORS['branch_baseline'], lw=1.8, linestyle='dashed')
ax.annotate('', xy=(legend_x + 1.0, legend_y - 0.5),
            xytext=(legend_x + 0.8, legend_y - 0.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['branch_baseline'],
                            lw=1.8, linestyle='dashed'))
ax.text(legend_x + 1.2, legend_y - 0.5, 'Baseline path',
        fontsize=9, va='center', color=COLORS['text_dark'],
        fontfamily='sans-serif')


plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/polymarkets/fig_002.png',
            dpi=200, bbox_inches='tight', facecolor=COLORS['bg'],
            edgecolor='none')
plt.close()
print("Figure saved to /home/wangni/notion-figures/polymarkets/fig_002.png")
