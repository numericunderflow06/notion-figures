#!/usr/bin/env python3
"""
fig_007: Three-Stage Training Curriculum
Visual representation of the three-stage training procedure for OpenTSLM-ITA.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8.5), dpi=200)
ax.set_xlim(0, 18)
ax.set_ylim(0, 8.5)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ─── Color Palette ─────────────────────────────────────────────────────────────
# Increasing intensity from left to right (blues)
stage_colors = ['#DBEAFE', '#93C5FD', '#3B82F6']
stage_border = ['#60A5FA', '#3B82F6', '#1D4ED8']
stage_header_bg = ['#BFDBFE', '#60A5FA', '#2563EB']
stage_header_fg = ['#1E3A5F', '#FFFFFF', '#FFFFFF']

frozen_color = '#E5E7EB'
frozen_border = '#9CA3AF'
frozen_text = '#4B5563'

trainable_highlight = '#FEF3C7'
trainable_border = '#F59E0B'

arrow_color = '#6B7280'
annotation_color = '#6B7280'

# ─── Helper Functions ─────────────────────────────────────────────────────────
def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, lw=1.5, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_header_bar(ax, x, y, w, h, facecolor, edgecolor, lw=1.2, zorder=3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def draw_lock_icon(ax, cx, cy, size=0.18, color='#6B7280'):
    """Draw a simple lock icon using matplotlib patches."""
    # Lock body (rectangle)
    body_w = size * 0.8
    body_h = size * 0.6
    body = FancyBboxPatch((cx - body_w/2, cy - body_h/2 - size*0.1),
                          body_w, body_h,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=color,
                          linewidth=1, zorder=5)
    ax.add_patch(body)
    # Lock shackle (arc)
    theta = np.linspace(0, np.pi, 30)
    r = size * 0.3
    sx = cx + r * np.cos(theta)
    sy = cy + body_h/2 - size*0.1 + r * np.sin(theta)
    ax.plot(sx, sy, color=color, lw=2.5, solid_capstyle='round', zorder=5)

# ─── Stage Definitions ────────────────────────────────────────────────────────
stages = [
    {
        'name': 'Stage 0',
        'title': 'Encoder Pre-Training',
        'data_source': 'Unlabeled TS data\n(HAR, EEG, ECG)',
        'trainable': 'Patch Encoder',
        'objective': 'Masked Patch Modeling\n(MSE loss, 40% mask ratio)',
        'hp': 'Epochs: 10',
    },
    {
        'name': 'Stage 1',
        'title': 'Encoder Warmup',
        'data_source': 'TSQA +\nM4-Captions',
        'trainable': 'Patch Encoder, TPE,\nIGR (LIT + ITA + proj),\nGated Cross-Attn (\u03b3)',
        'objective': 'Autoregressive LM\n(cross-entropy loss)',
        'hp': 'LR: 1e-4',
    },
    {
        'name': 'Stage 2',
        'title': 'Task-Specific CoT Fine-Tuning',
        'data_source': 'Target CoT datasets\n(HAR / Sleep / ECG-QA)',
        'trainable': 'Patch Encoder, TPE,\nIGR (LIT + ITA + proj),\nGated Cross-Attn (\u03b3)',
        'objective': 'Autoregressive CoT\n(cross-entropy loss)',
        'hp': 'LR: 5e-5',
    },
]

# ─── Layout Parameters ────────────────────────────────────────────────────────
box_w = 4.6
box_h = 5.2
box_y = 2.4
spacing = 5.35  # center-to-center distance between boxes
start_x = 0.6

# ─── Draw Stages ──────────────────────────────────────────────────────────────
stage_centers = []
for i, stage in enumerate(stages):
    x = start_x + i * spacing
    y = box_y
    stage_centers.append(x + box_w / 2)

    # Main stage box
    draw_rounded_box(ax, x, y, box_w, box_h,
                     facecolor=stage_colors[i],
                     edgecolor=stage_border[i], lw=2.2)

    # Header bar
    hdr_h = 0.65
    hdr_y = y + box_h - hdr_h - 0.15
    draw_header_bar(ax, x + 0.15, hdr_y, box_w - 0.3, hdr_h,
                    facecolor=stage_header_bg[i],
                    edgecolor=stage_border[i], lw=1.2)

    # Header text — use smaller font if title is long
    header_text = f"{stage['name']}: {stage['title']}"
    hdr_fs = 10.5 if len(header_text) > 30 else 11.5
    ax.text(x + box_w / 2, hdr_y + hdr_h / 2,
            header_text,
            ha='center', va='center', fontsize=hdr_fs, fontweight='bold',
            color=stage_header_fg[i], zorder=4)

    # ── Section: Data Source ──
    sec_y = hdr_y - 0.32
    ax.text(x + 0.4, sec_y, "Data Source", fontsize=9.5, fontweight='bold',
            color='#374151', va='top', zorder=4)
    sec_y -= 0.18
    ax.text(x + 0.4, sec_y, stage['data_source'], fontsize=9.5,
            color='#4B5563', va='top', zorder=4, linespacing=1.3)

    # ── Section: Trainable Components ──
    sec_y -= 0.7
    ax.text(x + 0.4, sec_y, "Trainable Components", fontsize=9.5, fontweight='bold',
            color='#374151', va='top', zorder=4)

    # Highlight box for trainable components
    tc_lines = stage['trainable'].count('\n') + 1
    tc_box_h = 0.33 * tc_lines + 0.22
    tc_box_y = sec_y - tc_box_h - 0.12
    draw_rounded_box(ax, x + 0.25, tc_box_y, box_w - 0.5, tc_box_h,
                     facecolor=trainable_highlight,
                     edgecolor=trainable_border, lw=1.2, zorder=3)
    ax.text(x + box_w / 2, tc_box_y + tc_box_h / 2,
            stage['trainable'], fontsize=9.5, color='#92400E',
            ha='center', va='center', zorder=4, linespacing=1.4,
            fontweight='medium')

    # ── Section: Training Objective ──
    sec_y = tc_box_y - 0.32
    ax.text(x + 0.4, sec_y, "Training Objective", fontsize=9.5, fontweight='bold',
            color='#374151', va='top', zorder=4)
    sec_y -= 0.18
    ax.text(x + 0.4, sec_y, stage['objective'], fontsize=9.5,
            color='#4B5563', va='top', zorder=4, linespacing=1.3)

    # ── Hyperparameter annotation (bottom of box) ──
    ax.text(x + box_w / 2, y + 0.28, stage['hp'],
            ha='center', va='center', fontsize=9, fontstyle='italic',
            color=annotation_color, zorder=4,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#D1D5DB', alpha=0.9))

# ─── Draw Arrows Between Stages ──────────────────────────────────────────────
for i in range(2):
    x1 = start_x + i * spacing + box_w + 0.05
    x2 = start_x + (i + 1) * spacing - 0.05
    mid_y = box_y + box_h / 2

    arrow = FancyArrowPatch(
        (x1, mid_y), (x2, mid_y),
        arrowstyle='->,head_length=8,head_width=5',
        color=arrow_color, lw=2.5, zorder=5,
        connectionstyle='arc3,rad=0'
    )
    ax.add_patch(arrow)

# ─── Draw Frozen LLM Bar (below stages) ──────────────────────────────────────
llm_x = start_x
llm_y = 1.1
llm_h = 0.85
llm_w = 2 * spacing + box_w  # span all three stages

draw_rounded_box(ax, llm_x, llm_y, llm_w, llm_h,
                 facecolor=frozen_color, edgecolor=frozen_border, lw=2)

# Lock icon
draw_lock_icon(ax, llm_x + 0.55, llm_y + llm_h / 2 + 0.02, size=0.22, color='#6B7280')

# Frozen LLM text
ax.text(llm_x + llm_w / 2, llm_y + llm_h / 2 + 0.12,
        "Frozen LLM  (Llama-3.2-1B / 3B)",
        ha='center', va='center', fontsize=13, fontweight='bold',
        color=frozen_text, zorder=4)
ax.text(llm_x + llm_w / 2, llm_y + llm_h / 2 - 0.22,
        "Weights frozen across all stages  \u00b7  AdamW (\u03b2\u2081=0.9, \u03b2\u2082=0.999)  \u00b7  Batch size 32  \u00b7  Weight decay 0.01",
        ha='center', va='center', fontsize=8.5, color='#6B7280', zorder=4)

# ─── Dashed lines from each stage down to frozen LLM ─────────────────────────
for i in range(3):
    x_mid = start_x + i * spacing + box_w / 2
    ax.plot([x_mid, x_mid], [box_y - 0.05, llm_y + llm_h + 0.05],
            ls=':', color=frozen_border, lw=1.3, alpha=0.5, zorder=1)

# ─── Title ────────────────────────────────────────────────────────────────────
title_cx = llm_x + llm_w / 2
ax.text(title_cx, 8.15, "Three-Stage Training Curriculum",
        ha='center', va='center', fontsize=17, fontweight='bold',
        color='#111827', zorder=4)
ax.text(title_cx, 7.8,
        "OpenTSLM-ITA: Progressive training from self-supervised pre-training to task-specific CoT fine-tuning",
        ha='center', va='center', fontsize=10, color='#6B7280', zorder=4)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_x = llm_x + llm_w - 3.8
legend_y = 0.35

# Trainable legend
draw_rounded_box(ax, legend_x, legend_y, 0.35, 0.22,
                 facecolor=trainable_highlight, edgecolor=trainable_border, lw=1)
ax.text(legend_x + 0.5, legend_y + 0.11, "= Trainable components",
        va='center', fontsize=8.5, color='#374151')

# Frozen legend
draw_rounded_box(ax, legend_x + 2.8, legend_y, 0.35, 0.22,
                 facecolor=frozen_color, edgecolor=frozen_border, lw=1)
ax.text(legend_x + 3.3, legend_y + 0.11, "= Frozen components",
        va='center', fontsize=8.5, color='#374151')

# ─── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig('/home/wangni/notion-figures/itformer/fig_007.png',
            dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print("Done: /home/wangni/notion-figures/itformer/fig_007.png")
