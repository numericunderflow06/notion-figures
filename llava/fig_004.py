#!/usr/bin/env python3
"""
fig_004: Three-Stage Training Pipeline for LLaVA-TSM
Shows Stage 0 (encoder pre-training), Stage 1 (alignment), Stage 2 (full fine-tuning)
with frozen/trainable component visualization, data sources, and hyperparameters.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ──────────────────────────────────────────────────────────
FROZEN_BG    = "#D5D8DC"
FROZEN_EDGE  = "#95A5A6"
TRAIN_BLUE   = "#5DADE2"
TRAIN_BLUE_E = "#2E86C1"
TRAIN_GREEN  = "#58D68D"
TRAIN_GREEN_E= "#28B463"
TRAIN_AMBER  = "#F5B041"
TRAIN_AMBER_E= "#D4930D"
TRAIN_PURPLE = "#AF7AC5"
TRAIN_PURPLE_E="#7D3C98"
STAGE_BG     = "#F8F9F9"
STAGE_EDGE   = "#BDC3C7"
DATA_BG      = "#FADBD8"
DATA_EDGE    = "#E74C3C"
HYPER_BG     = "#FCF3CF"
HYPER_EDGE   = "#D4AC0D"
ARROW_COLOR  = "#5D6D7E"
TEXT_COLOR   = "#2C3E50"
TITLE_COLOR  = "#1B2631"

fig, axes = plt.subplots(1, 3, figsize=(22, 10))
fig.patch.set_facecolor("white")

# ── Shared layout constants ─────────────────────────────────────────────────
BOX_X = 0.10          # left margin for component boxes
BOX_W = 0.80          # width of component boxes
BOX_H = 0.105         # height of each component box
GAP   = 0.035         # gap between boxes


def draw_box(ax, x, y, w, h, label, frozen, sublabel=None,
             bg=None, ec=None, fontsize=11.5):
    """Draw a rounded component box with frozen/trainable styling."""
    if frozen:
        bg = bg or FROZEN_BG
        ec = ec or FROZEN_EDGE
        lw = 1.4
    else:
        bg = bg or TRAIN_BLUE
        ec = ec or TRAIN_BLUE_E
        lw = 2.0
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04",
                         facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    cy = y + h * 0.58 if sublabel else y + h * 0.5
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=TEXT_COLOR, zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.22, sublabel,
                ha="center", va="center", fontsize=9,
                color="#566573", style="italic", zorder=3)
    # Frozen / trainable badge
    tag = "FROZEN" if frozen else "TRAIN"
    tag_bg = "#ABB2B9" if frozen else "#82E0AA"
    tag_ec = "#808B96" if frozen else "#27AE60"
    tag_w, tag_h = 0.135, 0.03
    tag_x = x + w - tag_w - 0.015
    tag_y = y + h - tag_h - 0.01
    tag_box = FancyBboxPatch((tag_x, tag_y), tag_w, tag_h,
                             boxstyle="round,pad=0.012",
                             facecolor=tag_bg, edgecolor=tag_ec,
                             linewidth=0.8, zorder=4)
    ax.add_patch(tag_box)
    ax.text(tag_x + tag_w / 2, tag_y + tag_h / 2, tag,
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="white" if frozen else "#1B4332", zorder=5)


def draw_info_box(ax, x, y, w, h, lines, bg=DATA_BG, ec=DATA_EDGE, fontsize=8.5):
    """Draw a small info box (data or hyperparams)."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.03",
                         facecolor=bg, edgecolor=ec, linewidth=1.2, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, "\n".join(lines),
            ha="center", va="center", fontsize=fontsize,
            color=TEXT_COLOR, linespacing=1.3, zorder=3,
            family="monospace")


def draw_arrow(ax, x, y1, y2):
    """Draw a downward arrow at x from y1 to y2."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                                lw=2, mutation_scale=14),
                zorder=1)


def draw_input_label(ax, x, y, text):
    """Draw an input label with a rounded border."""
    ax.text(x, y, text,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=TEXT_COLOR, zorder=3,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ARROW_COLOR,
                      lw=1.2, alpha=0.95))


def setup_panel(ax, title, subtitle):
    """Configure a panel axis."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    bg = FancyBboxPatch((-0.02, -0.02), 1.04, 1.04,
                        boxstyle="round,pad=0.02",
                        facecolor=STAGE_BG, edgecolor=STAGE_EDGE,
                        linewidth=1.8, zorder=0)
    ax.add_patch(bg)
    ax.text(0.50, 0.99, title,
            ha="center", va="top", fontsize=16,
            fontweight="bold", color=TITLE_COLOR, zorder=5)
    ax.text(0.50, 0.94, subtitle,
            ha="center", va="top", fontsize=11,
            color="#566573", style="italic", zorder=5)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 0: Encoder Pre-training
# ═══════════════════════════════════════════════════════════════════════════
ax0 = axes[0]
setup_panel(ax0, "Stage 0", "Encoder Pre-training (MAE)")

# -- Component positions for Stage 0 (only 2 components, centered vertically)
y_top_0 = 0.62
y_bot_0 = y_top_0 - BOX_H - GAP

draw_input_label(ax0, 0.50, 0.86, "Unlabeled Time Series")
draw_arrow(ax0, 0.50, 0.83, y_top_0 + BOX_H)

draw_box(ax0, BOX_X, y_top_0, BOX_W, BOX_H, "Patch Encoder",
         frozen=False, sublabel="Conv1D + 4-layer Transformer",
         bg=TRAIN_BLUE, ec=TRAIN_BLUE_E)

draw_arrow(ax0, 0.50, y_top_0, y_top_0 - GAP)

draw_box(ax0, BOX_X, y_bot_0, BOX_W, BOX_H, "Lightweight Decoder",
         frozen=False, sublabel="(discarded after Stage 0)",
         bg=TRAIN_BLUE, ec=TRAIN_BLUE_E)

# Objective text
obj_y = y_bot_0 - 0.04
draw_arrow(ax0, 0.50, y_bot_0, obj_y)
ax0.text(0.50, obj_y - 0.025, "Masked Patch Prediction",
         ha="center", va="center", fontsize=11,
         fontweight="bold", color="#943126", zorder=3)
ax0.text(0.50, obj_y - 0.06, "Mask 75% patches, reconstruct originals",
         ha="center", va="center", fontsize=9,
         color="#7B241C", zorder=3)

# Data sources
draw_info_box(ax0, 0.04, 0.03, 0.50, 0.18,
              ["  Data Sources:", "  PTB-XL ECGs (21.8K)",
               "  Chapman ECGs (45.2K)", "  SleepEDF EEGs",
               "  HAR accelerometer"],
              bg=DATA_BG, ec=DATA_EDGE, fontsize=8)

# Hyperparameters
draw_info_box(ax0, 0.57, 0.03, 0.39, 0.18,
              ["Hyperparams:", "mask = 75%", "epochs = 100",
               "d_enc = 512", "4 layers, 8 heads"],
              bg=HYPER_BG, ec=HYPER_EDGE, fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Alignment Pre-training
# ═══════════════════════════════════════════════════════════════════════════
ax1 = axes[1]
setup_panel(ax1, "Stage 1", "Alignment Pre-training")

# 4 stacked components
y3 = 0.70                           # top component
y2 = y3 - BOX_H - GAP
y1 = y2 - BOX_H - GAP
y0 = y1 - BOX_H - GAP

draw_input_label(ax1, 0.50, 0.86, "TS-Caption Pairs")
draw_arrow(ax1, 0.50, 0.83, y3 + BOX_H)

draw_box(ax1, BOX_X, y3, BOX_W, BOX_H, "Patch Encoder",
         frozen=True, sublabel="(pre-trained from Stage 0)")
draw_arrow(ax1, 0.50, y3, y3 - GAP)

draw_box(ax1, BOX_X, y2, BOX_W, BOX_H, "Perceiver Pooling",
         frozen=False, sublabel="K = 64 learned queries",
         bg=TRAIN_GREEN, ec=TRAIN_GREEN_E)
draw_arrow(ax1, 0.50, y2, y2 - GAP)

draw_box(ax1, BOX_X, y1, BOX_W, BOX_H, "MLP Projector",
         frozen=False, sublabel="2-layer MLP + GELU",
         bg=TRAIN_AMBER, ec=TRAIN_AMBER_E)
draw_arrow(ax1, 0.50, y1, y1 - GAP)

draw_box(ax1, BOX_X, y0, BOX_W, BOX_H, "LLaMA 3.2 (LLM)",
         frozen=True, sublabel="(frozen)")

# Data sources
draw_info_box(ax1, 0.04, 0.03, 0.48, 0.11,
              ["Data Sources:", "M4-Captions (80K)", "TSQA (38K)"],
              bg=DATA_BG, ec=DATA_EDGE, fontsize=8.5)

# Hyperparameters
draw_info_box(ax1, 0.55, 0.03, 0.41, 0.11,
              ["Hyperparams:", "lr = 1e-3, cosine",
               "epochs = 1-2", "batch = 256"],
              bg=HYPER_BG, ec=HYPER_EDGE, fontsize=8.5)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: End-to-End Instruction Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════
ax2 = axes[2]
setup_panel(ax2, "Stage 2", "End-to-End Fine-tuning")

draw_input_label(ax2, 0.50, 0.86, "Task-Specific CoT Data")
draw_arrow(ax2, 0.50, 0.83, y3 + BOX_H)

draw_box(ax2, BOX_X, y3, BOX_W, BOX_H, "Patch Encoder",
         frozen=True, sublabel="(pre-trained, frozen)")
draw_arrow(ax2, 0.50, y3, y3 - GAP)

draw_box(ax2, BOX_X, y2, BOX_W, BOX_H, "Perceiver Pooling",
         frozen=False, sublabel="K = 64 learned queries",
         bg=TRAIN_GREEN, ec=TRAIN_GREEN_E)
draw_arrow(ax2, 0.50, y2, y2 - GAP)

draw_box(ax2, BOX_X, y1, BOX_W, BOX_H, "MLP Projector",
         frozen=False, sublabel="2-layer MLP + GELU",
         bg=TRAIN_AMBER, ec=TRAIN_AMBER_E)
draw_arrow(ax2, 0.50, y1, y1 - GAP)

draw_box(ax2, BOX_X, y0, BOX_W, BOX_H, "LLaMA 3.2 (LLM)",
         frozen=False, sublabel="full fine-tuning (no LoRA)",
         bg=TRAIN_PURPLE, ec=TRAIN_PURPLE_E)

# Data sources
draw_info_box(ax2, 0.04, 0.03, 0.48, 0.11,
              ["Data Sources:", "ECG-QA-CoT (159K)",
               "HAR-CoT (68K)", "Sleep-CoT (7.4K)"],
              bg=DATA_BG, ec=DATA_EDGE, fontsize=8.5)

# Hyperparameters
draw_info_box(ax2, 0.55, 0.03, 0.41, 0.11,
              ["Hyperparams:", "lr = 2e-5, cosine+warmup",
               "epochs<=50 (ES p=5)", "batch=128, DeepSpeed"],
              bg=HYPER_BG, ec=HYPER_EDGE, fontsize=8.5)


# ── Legend ──────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=FROZEN_BG, edgecolor=FROZEN_EDGE, linewidth=1.5,
                   label="Frozen"),
    mpatches.Patch(facecolor=TRAIN_BLUE, edgecolor=TRAIN_BLUE_E, linewidth=1.5,
                   label="Trainable (encoder)"),
    mpatches.Patch(facecolor=TRAIN_GREEN, edgecolor=TRAIN_GREEN_E, linewidth=1.5,
                   label="Trainable (pooling)"),
    mpatches.Patch(facecolor=TRAIN_AMBER, edgecolor=TRAIN_AMBER_E, linewidth=1.5,
                   label="Trainable (projector)"),
    mpatches.Patch(facecolor=TRAIN_PURPLE, edgecolor=TRAIN_PURPLE_E, linewidth=1.5,
                   label="Trainable (LLM)"),
    mpatches.Patch(facecolor=DATA_BG, edgecolor=DATA_EDGE, linewidth=1.5,
                   label="Data sources"),
    mpatches.Patch(facecolor=HYPER_BG, edgecolor=HYPER_EDGE, linewidth=1.5,
                   label="Hyperparameters"),
]

fig.legend(handles=legend_elements, loc="lower center", ncol=7,
           fontsize=10, frameon=True, fancybox=True,
           edgecolor=STAGE_EDGE, facecolor="white",
           bbox_to_anchor=(0.5, -0.005))

fig.suptitle("LLaVA-TSM: Three-Stage Training Pipeline",
             fontsize=19, fontweight="bold", color=TITLE_COLOR,
             y=0.995)

plt.tight_layout(rect=[0, 0.04, 1, 0.955])

outpath = "/home/wangni/notion-figures/llava/fig_004.png"
fig.savefig(outpath, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Figure saved to {outpath}")
