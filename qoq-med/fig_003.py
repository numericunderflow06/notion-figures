#!/usr/bin/env python3
"""
fig_003: Training Pipeline: Supervised Curriculum → DRPO
Two-phase training timeline showing 5 stages.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(18, 7.5), dpi=200)
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-1.5, 7.5)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ──
# Phase 1: gradient blues
blues = ['#1a5276', '#1f6fa0', '#2e86c1', '#5dade2']
blue_text = 'white'
# Phase 2: warm red/orange
phase2_color = '#e74c3c'
phase2_text = 'white'
# Phase labels
phase1_bg = '#d4e6f1'
phase2_bg = '#fadbd8'
# Arrow color
arrow_color = '#5d6d7e'

# ── Helper: draw a stage box ──
def draw_stage_box(ax, x_center, y_center, width, height, color, stage_num,
                   dataset, task, detail, text_color='white'):
    box = FancyBboxPatch(
        (x_center - width / 2, y_center - height / 2), width, height,
        boxstyle="round,pad=0.15", linewidth=1.5,
        edgecolor='white', facecolor=color, zorder=3
    )
    ax.add_patch(box)

    # Stage number badge
    badge_r = 0.28
    badge = plt.Circle((x_center - width / 2 + 0.45, y_center + height / 2 - 0.35),
                        badge_r, color='white', alpha=0.3, zorder=4)
    ax.add_patch(badge)
    ax.text(x_center - width / 2 + 0.45, y_center + height / 2 - 0.35,
            str(stage_num), fontsize=11, fontweight='bold', color=text_color,
            ha='center', va='center', zorder=5)

    # Dataset name (bold, large)
    ax.text(x_center, y_center + 0.55, dataset,
            fontsize=11.5, fontweight='bold', color=text_color,
            ha='center', va='center', zorder=5)

    # Task type
    ax.text(x_center, y_center + 0.0, task,
            fontsize=9.5, color=text_color, alpha=0.92,
            ha='center', va='center', zorder=5,
            style='italic')

    # Key detail (smaller)
    ax.text(x_center, y_center - 0.55, detail,
            fontsize=8.5, color=text_color, alpha=0.82,
            ha='center', va='center', zorder=5,
            wrap=True)


# ── Phase background panels ──
phase1_rect = FancyBboxPatch(
    (0.1, 1.0), 13.2, 5.2,
    boxstyle="round,pad=0.25", linewidth=1.2,
    edgecolor='#aed6f1', facecolor=phase1_bg, alpha=0.45, zorder=1
)
ax.add_patch(phase1_rect)

phase2_rect = FancyBboxPatch(
    (14.0, 1.0), 4.2, 5.2,
    boxstyle="round,pad=0.25", linewidth=1.2,
    edgecolor='#f5b7b1', facecolor=phase2_bg, alpha=0.45, zorder=1
)
ax.add_patch(phase2_rect)

# Phase labels
ax.text(6.7, 6.55, 'Phase 1: Supervised Curriculum Learning',
        fontsize=14, fontweight='bold', color='#1a5276',
        ha='center', va='center', zorder=5)
ax.text(16.1, 6.55, 'Phase 2: DRPO RL',
        fontsize=14, fontweight='bold', color='#c0392b',
        ha='center', va='center', zorder=5)

# ── Stage boxes (Phase 1) ──
box_w = 2.8
box_h = 2.8
y_pos = 3.6

stages_x = [1.8, 4.95, 8.1, 11.25]

draw_stage_box(ax, stages_x[0], y_pos, box_w, box_h, blues[0], 1,
               'TSQA', 'Multiple-Choice QA', 'Synthetic TS\nEncoder warmup')

draw_stage_box(ax, stages_x[1], y_pos, box_w, box_h, blues[1], 2,
               'M4 Captions', 'TS Captioning', 'Representation\nalignment')

draw_stage_box(ax, stages_x[2], y_pos, box_w, box_h, blues[2], 3,
               'HAR-CoT', 'Activity Recognition', 'Accelerometer\nNon-ECG validation')

draw_stage_box(ax, stages_x[3], y_pos, box_w, box_h, blues[3], 4,
               'Multi-Domain', 'Merged Clinical Tasks', 'Sleep + ECG-QA\nSimultaneous training')

# ── Stage box (Phase 2) ──
draw_stage_box(ax, 16.1, y_pos, box_w, box_h, phase2_color, 5,
               'DRPO', 'Reinforcement Learning', 'Domain-aware\ntemperature scaling')

# ── Arrows between stages ──
arrow_style = "Simple,tail_width=5,head_width=14,head_length=8"

for i in range(3):
    x_start = stages_x[i] + box_w / 2
    x_end = stages_x[i + 1] - box_w / 2
    arrow = FancyArrowPatch(
        (x_start + 0.05, y_pos), (x_end - 0.05, y_pos),
        arrowstyle=arrow_style,
        color=arrow_color, alpha=0.6, zorder=4
    )
    ax.add_patch(arrow)

# ── Checkpoint icon + arrow from Stage 4 to Stage 5 ──
ckpt_x = 13.55
ckpt_y = y_pos

# Checkpoint diamond
diamond_size = 0.32
diamond = mpatches.RegularPolygon((ckpt_x, ckpt_y), numVertices=4,
                                   radius=diamond_size, orientation=0,
                                   facecolor='#f39c12', edgecolor='#d68910',
                                   linewidth=1.5, zorder=6)
ax.add_patch(diamond)
ax.text(ckpt_x, ckpt_y - 0.55, 'ckpt', fontsize=7.5, fontweight='bold',
        color='#7d6608', ha='center', va='center', zorder=6)

# Arrow Stage 4 → checkpoint
arrow_4_ckpt = FancyArrowPatch(
    (stages_x[3] + box_w / 2 + 0.05, y_pos),
    (ckpt_x - diamond_size - 0.08, y_pos),
    arrowstyle=arrow_style,
    color=arrow_color, alpha=0.6, zorder=4
)
ax.add_patch(arrow_4_ckpt)

# Arrow checkpoint → Stage 5
arrow_ckpt_5 = FancyArrowPatch(
    (ckpt_x + diamond_size + 0.08, y_pos),
    (16.1 - box_w / 2 - 0.05, y_pos),
    arrowstyle=arrow_style,
    color='#c0392b', alpha=0.7, zorder=4
)
ax.add_patch(arrow_ckpt_5)

# ── Progression annotation (simple → complex) ──
ax.annotate('', xy=(12.5, 1.25), xytext=(1.0, 1.25),
            arrowprops=dict(arrowstyle='->', color='#85929e',
                            lw=1.5, ls='--'))
ax.text(6.7, 0.85, 'Simple tasks  →  Complex multi-domain tasks',
        fontsize=10, color='#5d6d7e', ha='center', va='center',
        style='italic')

# ── Comparison note: OpenTSLM 11 stages → 5 ──
note_x = 9.5
note_y = -0.5
note_box = FancyBboxPatch(
    (note_x - 3.5, note_y - 0.45), 7.0, 0.9,
    boxstyle="round,pad=0.15", linewidth=1.0,
    edgecolor='#aab7b8', facecolor='#f8f9f9', alpha=0.9, zorder=3
)
ax.add_patch(note_box)
ax.text(note_x, note_y, "OpenTSLM's 11-stage curriculum  →  condensed to 5 stages",
        fontsize=10, color='#5d6d7e', ha='center', va='center',
        fontweight='bold', zorder=5)

# ── Loss type annotations below each phase ──
ax.text(6.7, 1.55, 'Cross-Entropy Loss (next-token prediction)',
        fontsize=9, color='#1a5276', ha='center', va='center',
        fontweight='bold', alpha=0.75, zorder=5)
ax.text(16.1, 1.55, 'DRPO Loss (domain-weighted)',
        fontsize=9, color='#c0392b', ha='center', va='center',
        fontweight='bold', alpha=0.75, zorder=5)

# ── Title ──
ax.text(9.25, 7.2, 'Training Pipeline: Supervised Curriculum → DRPO',
        fontsize=16, fontweight='bold', color='#2c3e50',
        ha='center', va='center')

plt.tight_layout()
plt.savefig('/home/wangni/notion-figures/qoq-med/fig_003.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Saved: /home/wangni/notion-figures/qoq-med/fig_003.png")
