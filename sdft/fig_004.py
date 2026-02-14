"""
fig_004: SDFT-OpenTSLM Integration: Student vs Teacher Data Flow
Two parallel vertical tracks (student=blue, teacher=orange/gold) converging at KL loss.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis('off')
fig.patch.set_facecolor('white')

# --- Color Palette ---
STUDENT_BG = '#D6E4F0'
STUDENT_BORDER = '#2B6CB0'
STUDENT_DARK = '#1A4971'
TEACHER_BG = '#FDE8CD'
TEACHER_BORDER = '#C77C1A'
TEACHER_DARK = '#8B5A00'
SHARED_BG = '#E8E8E8'
SHARED_BORDER = '#555555'
LOSS_BG = '#F0D0D0'
LOSS_BORDER = '#B03030'
LOSS_DARK = '#8B0000'
EMA_COLOR = '#6B21A8'
TS_BG = '#D4EDDA'
TS_BORDER = '#28774E'

# --- Helper: rounded box with centered text ---
def draw_box(ax, cx, cy, w, h, text, bg, border, fontsize=11, fontweight='bold',
             textcolor='black', linestyle='-', linewidth=2.0, alpha=1.0, zorder=3,
             multiline=False):
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=bg, edgecolor=border,
        linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    if multiline:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=zorder+1,
                linespacing=1.4)
    else:
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=textcolor, zorder=zorder+1)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#333', lw=2.0, style='->', linestyle='-',
               connectionstyle='arc3,rad=0', zorder=2, shrinkA=8, shrinkB=8):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder, mutation_scale=18
    )
    ax.add_patch(arrow)
    return arrow

# ============================================================
# Layout constants
# ============================================================
LEFT_X = 4.0     # Student column center
RIGHT_X = 10.0   # Teacher column center
CENTER_X = 7.0   # Center for shared elements
BOX_W = 3.6
BOX_H = 0.8
PROMPT_H = 1.3

# Y positions (top to bottom)
Y_TITLE = 15.3
Y_LABELS = 14.5
Y_PROMPT = 13.0
Y_TS = 11.3
Y_ENCODER = 10.0
Y_LLM = 8.6
Y_LOGITS = 7.3
Y_LOSS = 5.7

# ============================================================
# Title
# ============================================================
ax.text(CENTER_X, Y_TITLE, 'SDFT-OpenTSLM Integration: Student vs Teacher Data Flow',
        ha='center', va='center', fontsize=15, fontweight='bold', color='#222')

# ============================================================
# Column labels
# ============================================================
ax.text(LEFT_X, Y_LABELS, 'Student Path', ha='center', va='center',
        fontsize=13, fontweight='bold', color=STUDENT_DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=STUDENT_BG, edgecolor=STUDENT_BORDER, linewidth=1.5))
ax.text(RIGHT_X, Y_LABELS, 'Teacher Path', ha='center', va='center',
        fontsize=13, fontweight='bold', color=TEACHER_DARK,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=TEACHER_BG, edgecolor=TEACHER_BORDER, linewidth=1.5))

# ============================================================
# Student Path boxes
# ============================================================
# Prompt
draw_box(ax, LEFT_X, Y_PROMPT, BOX_W, PROMPT_H,
         'Standard Prompt\n(task instruction only)',
         STUDENT_BG, STUDENT_BORDER, fontsize=11, textcolor=STUDENT_DARK, multiline=True)

# Encoder
draw_box(ax, LEFT_X, Y_ENCODER, BOX_W, BOX_H,
         'Encoder + Projector',
         STUDENT_BG, STUDENT_BORDER, fontsize=11, textcolor=STUDENT_DARK)

# LLM
draw_box(ax, LEFT_X, Y_LLM, BOX_W, BOX_H,
         'LLM (Frozen)',
         STUDENT_BG, STUDENT_BORDER, fontsize=11, textcolor=STUDENT_DARK)

# Logits P
draw_box(ax, LEFT_X, Y_LOGITS, BOX_W, BOX_H,
         'Logits  P  (on-policy)',
         STUDENT_BG, STUDENT_BORDER, fontsize=12, textcolor=STUDENT_DARK)

# ============================================================
# Teacher Path boxes
# ============================================================
# Prompt (augmented)
draw_box(ax, RIGHT_X, Y_PROMPT, BOX_W, PROMPT_H,
         'Augmented Prompt\n(task instruction +\ngolden CoT demo)',
         TEACHER_BG, TEACHER_BORDER, fontsize=11, textcolor=TEACHER_DARK, multiline=True)

# Encoder
draw_box(ax, RIGHT_X, Y_ENCODER, BOX_W, BOX_H,
         'Encoder + Projector',
         TEACHER_BG, TEACHER_BORDER, fontsize=11, textcolor=TEACHER_DARK)

# LLM
draw_box(ax, RIGHT_X, Y_LLM, BOX_W, BOX_H,
         'LLM (Frozen)',
         TEACHER_BG, TEACHER_BORDER, fontsize=11, textcolor=TEACHER_DARK)

# Logits Q
draw_box(ax, RIGHT_X, Y_LOGITS, BOX_W, BOX_H,
         'Logits  Q  (demo-conditioned)',
         TEACHER_BG, TEACHER_BORDER, fontsize=12, textcolor=TEACHER_DARK)

# ============================================================
# Shared Time Series Data block (between columns)
# ============================================================
draw_box(ax, CENTER_X, Y_TS, 3.0, 0.8,
         'Time Series Data',
         TS_BG, TS_BORDER, fontsize=11, textcolor='#1B4332')

# Arrows from time series to both encoders
draw_arrow(ax, CENTER_X - 0.8, Y_TS - 0.4, LEFT_X + 0.3, Y_ENCODER + 0.4,
           color=TS_BORDER, lw=2.0, connectionstyle='arc3,rad=0.15')
draw_arrow(ax, CENTER_X + 0.8, Y_TS - 0.4, RIGHT_X - 0.3, Y_ENCODER + 0.4,
           color=TS_BORDER, lw=2.0, connectionstyle='arc3,rad=-0.15')

# ============================================================
# Vertical flow arrows (Student)
# ============================================================
draw_arrow(ax, LEFT_X, Y_PROMPT - PROMPT_H/2, LEFT_X, Y_ENCODER + BOX_H/2,
           color=STUDENT_BORDER, lw=2.2)
draw_arrow(ax, LEFT_X, Y_ENCODER - BOX_H/2, LEFT_X, Y_LLM + BOX_H/2,
           color=STUDENT_BORDER, lw=2.2)
draw_arrow(ax, LEFT_X, Y_LLM - BOX_H/2, LEFT_X, Y_LOGITS + BOX_H/2,
           color=STUDENT_BORDER, lw=2.2)

# ============================================================
# Vertical flow arrows (Teacher)
# ============================================================
draw_arrow(ax, RIGHT_X, Y_PROMPT - PROMPT_H/2, RIGHT_X, Y_ENCODER + BOX_H/2,
           color=TEACHER_BORDER, lw=2.2)
draw_arrow(ax, RIGHT_X, Y_ENCODER - BOX_H/2, RIGHT_X, Y_LLM + BOX_H/2,
           color=TEACHER_BORDER, lw=2.2)
draw_arrow(ax, RIGHT_X, Y_LLM - BOX_H/2, RIGHT_X, Y_LOGITS + BOX_H/2,
           color=TEACHER_BORDER, lw=2.2)

# ============================================================
# KL Divergence Loss block (centered, bottom)
# ============================================================
draw_box(ax, CENTER_X, Y_LOSS, 4.4, 1.1,
         'KL Divergence Loss\nKL( P  ||  Q )',
         LOSS_BG, LOSS_BORDER, fontsize=13, textcolor=LOSS_DARK, linewidth=2.5, multiline=True)

# Arrows from logits to KL loss
draw_arrow(ax, LEFT_X, Y_LOGITS - BOX_H/2, CENTER_X - 0.5, Y_LOSS + 0.55,
           color=STUDENT_BORDER, lw=2.5, connectionstyle='arc3,rad=0.2')
draw_arrow(ax, RIGHT_X, Y_LOGITS - BOX_H/2, CENTER_X + 0.5, Y_LOSS + 0.55,
           color=TEACHER_BORDER, lw=2.5, connectionstyle='arc3,rad=-0.2')

# P and Q labels near arrows going into KL block
ax.text(LEFT_X + 0.6, Y_LOGITS - 0.85, 'P', fontsize=13, fontweight='bold',
        color=STUDENT_DARK, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=STUDENT_BORDER, linewidth=1))
ax.text(RIGHT_X - 0.6, Y_LOGITS - 0.85, 'Q', fontsize=13, fontweight='bold',
        color=TEACHER_DARK, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor=TEACHER_BORDER, linewidth=1))

# ============================================================
# EMA arrow (student -> teacher), dotted, curved
# ============================================================
ema_y = Y_LLM
draw_arrow(ax, LEFT_X + BOX_W/2, ema_y + 0.15, RIGHT_X - BOX_W/2, ema_y + 0.15,
           color=EMA_COLOR, lw=2.5, linestyle='--',
           connectionstyle='arc3,rad=-0.35', style='->')

# EMA label
ax.text(CENTER_X, ema_y + 1.15, 'EMA Update',
        ha='center', va='center', fontsize=11, fontweight='bold', color=EMA_COLOR,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E8FF', edgecolor=EMA_COLOR,
                  linewidth=1.5, linestyle='--'))
ax.text(CENTER_X, ema_y + 0.7, r'$\theta_T \leftarrow \alpha\,\theta_T + (1-\alpha)\,\theta_S$',
        ha='center', va='center', fontsize=10, color=EMA_COLOR, style='italic')

# ============================================================
# Prompt composition callouts
# ============================================================
# Student callout
callout_y_s = Y_PROMPT + 0.05
ax.annotate(
    'Input:\n"Classify the following\n  time series signal..."',
    xy=(LEFT_X - BOX_W/2, callout_y_s),
    xytext=(LEFT_X - BOX_W/2 - 1.6, callout_y_s + 0.9),
    fontsize=8.5, color=STUDENT_DARK, ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#EBF2FA', edgecolor=STUDENT_BORDER, linewidth=1, alpha=0.9),
    arrowprops=dict(arrowstyle='->', color=STUDENT_BORDER, lw=1.2, connectionstyle='arc3,rad=0.2'),
    zorder=5
)

# Teacher callout
callout_y_t = Y_PROMPT + 0.05
ax.annotate(
    'Input:\n"Here is a worked example:\n  [golden CoT rationale]\n  Now classify..."',
    xy=(RIGHT_X + BOX_W/2, callout_y_t),
    xytext=(RIGHT_X + BOX_W/2 + 1.8, callout_y_t + 1.0),
    fontsize=8.5, color=TEACHER_DARK, ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5E6', edgecolor=TEACHER_BORDER, linewidth=1, alpha=0.9),
    arrowprops=dict(arrowstyle='->', color=TEACHER_BORDER, lw=1.2, connectionstyle='arc3,rad=-0.2'),
    zorder=5
)

# ============================================================
# Loss output arrow (downward from KL block)
# ============================================================
draw_arrow(ax, CENTER_X, Y_LOSS - 0.55, CENTER_X, Y_LOSS - 1.4,
           color=LOSS_BORDER, lw=2.5, style='->', shrinkB=2)
ax.text(CENTER_X, Y_LOSS - 1.65, 'Backprop to Student',
        ha='center', va='center', fontsize=10, fontweight='bold', color=LOSS_DARK,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FAEAEA', edgecolor=LOSS_BORDER, linewidth=1.2))

# ============================================================
# Architecture note at bottom
# ============================================================
note_text = (
    "Architecture: Encoder + Projector + Flamingo Cross-Attention + Frozen LLM\n"
    "EMA sync: mixup_alpha = 0.01, updated every step"
)
ax.text(CENTER_X, 3.3, note_text,
        ha='center', va='center', fontsize=9, color='#444',
        style='italic', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#CCCCCC', linewidth=1))

# ============================================================
# Light background panels for each column
# ============================================================
# Student panel
student_panel = FancyBboxPatch(
    (LEFT_X - BOX_W/2 - 0.3, Y_LOGITS - BOX_H/2 - 0.25),
    BOX_W + 0.6, (Y_LABELS + 0.35) - (Y_LOGITS - BOX_H/2 - 0.25),
    boxstyle="round,pad=0.2", facecolor=STUDENT_BG, edgecolor=STUDENT_BORDER,
    linewidth=1.0, alpha=0.15, zorder=0
)
ax.add_patch(student_panel)

# Teacher panel
teacher_panel = FancyBboxPatch(
    (RIGHT_X - BOX_W/2 - 0.3, Y_LOGITS - BOX_H/2 - 0.25),
    BOX_W + 0.6, (Y_LABELS + 0.35) - (Y_LOGITS - BOX_H/2 - 0.25),
    boxstyle="round,pad=0.2", facecolor=TEACHER_BG, edgecolor=TEACHER_BORDER,
    linewidth=1.0, alpha=0.15, zorder=0
)
ax.add_patch(teacher_panel)

# ============================================================
# Save
# ============================================================
plt.tight_layout()
fig.savefig('/home/wangni/notion-figures/sdft/fig_004.png', dpi=200,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved: /home/wangni/notion-figures/sdft/fig_004.png")
