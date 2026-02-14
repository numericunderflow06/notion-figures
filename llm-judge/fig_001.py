"""
Figure 001: OpenTSLM + RL Architecture Overview
High-level architecture diagram showing the OpenTSLM-SP model extended with the GRPO RL training loop.
Two-phase approach: SFT pre-training followed by GRPO RL fine-tuning.
Color coding: blue=frozen, orange=trainable, green=RL-specific additions.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────
FROZEN_BLUE = "#4A90D9"       # frozen components
FROZEN_BLUE_LIGHT = "#D6E4F0" # frozen fill
TRAIN_ORANGE = "#E8732A"      # trainable components
TRAIN_ORANGE_LIGHT = "#FDEBD0" # trainable fill
RL_GREEN = "#27AE60"          # RL-specific additions
RL_GREEN_LIGHT = "#D5F5E3"    # RL fill
BG_GRAY = "#F7F9FC"           # section background
ARROW_GRAY = "#4A4A4A"        # arrow color
TEXT_DARK = "#1A1A1A"         # main text
TEXT_MED = "#3A3A3A"          # secondary text
PHASE_BG_1 = "#EBF0F7"       # Phase 1 background
PHASE_BG_2 = "#E8F8EE"       # Phase 2 background

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper functions ───────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label, fill_color, edge_color, fontsize=10,
             fontstyle="normal", fontweight="normal", sublabel=None, alpha=1.0,
             text_color=TEXT_DARK, lw=1.8, zorder=3):
    """Draw a rounded-rectangle component box."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=fill_color, edgecolor=edge_color,
        linewidth=lw, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, fontstyle=fontstyle,
                color=text_color, zorder=zorder + 1)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel,
                ha="center", va="center", fontsize=8,
                fontstyle="italic", color=TEXT_MED, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, fontstyle=fontstyle,
                color=text_color, zorder=zorder + 1)


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_GRAY, style="-|>",
               lw=1.5, connectionstyle="arc3,rad=0", zorder=2):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        lw=lw, connectionstyle=connectionstyle,
        mutation_scale=14, zorder=zorder
    )
    ax.add_patch(arrow)


def draw_section_bg(ax, x, y, w, h, color, label=None, label_pos="top"):
    """Draw a section background rectangle with optional label."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="none",
        linewidth=0, alpha=0.5, zorder=0
    )
    ax.add_patch(rect)
    if label:
        if label_pos == "top":
            ax.text(x + w / 2, y + h - 0.15, label,
                    ha="center", va="top", fontsize=11,
                    fontweight="bold", color=TEXT_MED, zorder=1)


# ── Title ──────────────────────────────────────────────────────────────
ax.text(8, 10.65, "OpenTSLM + RL Architecture Overview",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=TEXT_DARK, zorder=10)
ax.text(8, 10.3, "Two-Phase Training: SFT Pre-training \u2192 GRPO RL Fine-tuning",
        ha="center", va="center", fontsize=11, color=TEXT_MED, zorder=10)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: OpenTSLM-SP Model (left portion, SFT)
# ══════════════════════════════════════════════════════════════════════
draw_section_bg(ax, 0.3, 1.2, 8.7, 8.7, PHASE_BG_1,
                label="Phase 1: SFT Pre-training (OpenTSLM-SP)", label_pos="top")

# ── Input ──────────────────────────────────────────────────────────────
# Time-series input
draw_box(ax, 0.7, 7.0, 2.5, 0.75, "Time-Series Input", "#FAFAFA", "#888888",
         fontsize=10, fontweight="bold")

# Text prompt input
draw_box(ax, 0.7, 5.7, 2.5, 0.75, "Text Prompt", "#FAFAFA", "#888888",
         fontsize=10, fontweight="bold",
         sublabel="(pre_prompt + post_prompt)")

# ── Encoder (trainable) ───────────────────────────────────────────────
draw_box(ax, 4.0, 7.0, 2.2, 0.75, "Time-Series Encoder",
         TRAIN_ORANGE_LIGHT, TRAIN_ORANGE, fontsize=10, fontweight="bold")

# ── Projector (trainable) ─────────────────────────────────────────────
draw_box(ax, 4.0, 5.7, 2.2, 0.75, "Projector",
         TRAIN_ORANGE_LIGHT, TRAIN_ORANGE, fontsize=10, fontweight="bold",
         sublabel="(soft-prompt mapping)")

# ── LLM backbone (frozen) ─────────────────────────────────────────────
# Large box for the LLM
llm_x, llm_y, llm_w, llm_h = 3.0, 2.6, 5.5, 2.5
llm_box = FancyBboxPatch(
    (llm_x, llm_y), llm_w, llm_h,
    boxstyle="round,pad=0.18",
    facecolor=FROZEN_BLUE_LIGHT, edgecolor=FROZEN_BLUE,
    linewidth=2.2, alpha=0.85, zorder=2
)
ax.add_patch(llm_box)
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.3, "Large Language Model (Frozen)",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=FROZEN_BLUE, zorder=4)

# Inside LLM: Frozen weights indicator
draw_box(ax, 3.4, 3.0, 2.2, 0.7, "Frozen Weights",
         FROZEN_BLUE_LIGHT, FROZEN_BLUE, fontsize=9,
         sublabel="(base parameters)")

# Inside LLM: LoRA adapters (trainable)
draw_box(ax, 6.0, 3.0, 2.1, 0.7, "LoRA Adapters",
         TRAIN_ORANGE_LIGHT, TRAIN_ORANGE, fontsize=9, fontweight="bold",
         sublabel="(trainable)")

# Combined token input arrow area
ax.text(5.1, 5.28, "Token\nEmbeddings", ha="center", va="center",
        fontsize=8, fontstyle="italic", color=TEXT_MED, zorder=5)

# ── Output ─────────────────────────────────────────────────────────────
draw_box(ax, 3.8, 1.4, 3.9, 0.75, "CoT Response + Answer",
         "#FAFAFA", "#888888", fontsize=10, fontweight="bold",
         sublabel="<reasoning>... Answer: <label>")

# ── Arrows for Phase 1 ────────────────────────────────────────────────
# Time-series -> Encoder
draw_arrow(ax, 3.2, 7.375, 4.0, 7.375, color=ARROW_GRAY)

# Encoder -> Projector
draw_arrow(ax, 5.1, 7.0, 5.1, 6.45 + 0.05, color=ARROW_GRAY)

# Projector -> LLM (soft prompt embeddings)
draw_arrow(ax, 5.1, 5.7, 5.1, 5.1 + 0.05, color=ARROW_GRAY)

# Text Prompt -> LLM (tokenized text)
draw_arrow(ax, 3.2, 6.075, 3.65, 6.075, color=ARROW_GRAY)
draw_arrow(ax, 3.65, 6.075, 3.65, 5.15, color=ARROW_GRAY,
           connectionstyle="arc3,rad=0")

# Merged embeddings -> LLM
draw_arrow(ax, 5.1, 5.1, 5.1, 4.55, color=ARROW_GRAY,
           connectionstyle="arc3,rad=0", style="-|>", lw=2.0)
ax.plot([3.65, 5.1], [5.1, 5.1], color=ARROW_GRAY, lw=1.5, zorder=2)

# LLM -> Output
draw_arrow(ax, 5.75, 2.6, 5.75, 2.15, color=ARROW_GRAY)


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: GRPO RL Fine-tuning (right portion)
# ══════════════════════════════════════════════════════════════════════
draw_section_bg(ax, 9.3, 1.2, 6.4, 8.7, PHASE_BG_2,
                label="Phase 2: GRPO RL Fine-tuning", label_pos="top")

# ── Generate K responses ──────────────────────────────────────────────
draw_box(ax, 9.8, 7.8, 2.6, 0.75, "Generate K Responses",
         RL_GREEN_LIGHT, RL_GREEN, fontsize=10, fontweight="bold",
         sublabel="(per time-series input)")

# ── Candidate responses ───────────────────────────────────────────────
# Show stacked response boxes to indicate K candidates
for i, offset in enumerate([0.08, 0.04, 0]):
    alpha_val = 0.4 + i * 0.3
    draw_box(ax, 9.8 + offset, 6.55 + offset, 2.5, 0.7,
             "" if i < 2 else "K Candidate CoTs",
             RL_GREEN_LIGHT, RL_GREEN, fontsize=9, fontweight="bold",
             alpha=alpha_val, lw=1.2)

# ── LLM Judge (frozen, external) ──────────────────────────────────────
draw_box(ax, 12.9, 6.55, 2.5, 0.7, "LLM Judge",
         RL_GREEN_LIGHT, RL_GREEN, fontsize=10, fontweight="bold",
         sublabel="(GPT-4o / Llama-70B)")
# Frozen indicator
ax.text(14.15, 6.38, "frozen, external",
        ha="center", va="center", fontsize=7, fontstyle="italic",
        color=RL_GREEN, zorder=5)

# ── Reward scores ──────────────────────────────────────────────────────
draw_box(ax, 12.9, 5.25, 2.5, 0.85, "Reward Scores",
         RL_GREEN_LIGHT, RL_GREEN, fontsize=10, fontweight="bold",
         sublabel="r = 0.5\u00b7corr + 0.3\u00b7reas + 0.2\u00b7cons")

# ── Group normalization ───────────────────────────────────────────────
draw_box(ax, 9.8, 5.25, 2.5, 0.85, "Group Normalization",
         RL_GREEN_LIGHT, RL_GREEN, fontsize=10, fontweight="bold",
         sublabel="(mean-sub, std-div)")

# ── GRPO Loss ─────────────────────────────────────────────────────────
draw_box(ax, 10.5, 3.8, 3.5, 0.85, "GRPO Loss Computation",
         RL_GREEN_LIGHT, RL_GREEN, fontsize=10, fontweight="bold",
         sublabel="clipped ratios (\u03b5=0.2) + KL penalty (\u03b2=0.04)")

# ── Reference policy (frozen SFT checkpoint) ──────────────────────────
draw_box(ax, 10.0, 2.55, 2.3, 0.7, "Reference Policy",
         FROZEN_BLUE_LIGHT, FROZEN_BLUE, fontsize=9, fontweight="bold",
         sublabel="(frozen SFT ckpt)")

# ── Update trainable params ───────────────────────────────────────────
draw_box(ax, 12.7, 2.55, 2.7, 0.7, "Update Parameters",
         TRAIN_ORANGE_LIGHT, TRAIN_ORANGE, fontsize=10, fontweight="bold",
         sublabel="encoder + projector + LoRA")

# ── Phase transition arrow ─────────────────────────────────────────────
# Curved arrow from Phase 1 output to Phase 2 input
draw_arrow(ax, 7.7, 1.77, 9.5, 1.77, color=RL_GREEN, lw=2.5,
           style="-|>")
ax.text(8.6, 1.98, "SFT checkpoint\ninitializes RL",
        ha="center", va="bottom", fontsize=8, fontweight="bold",
        color=RL_GREEN, zorder=5)

# ── Arrows for Phase 2 ────────────────────────────────────────────────
# Generate -> K candidates
draw_arrow(ax, 11.05, 7.8, 11.05, 7.33, color=RL_GREEN, lw=1.5)

# K candidates -> Judge
draw_arrow(ax, 12.38, 6.9, 12.9, 6.9, color=RL_GREEN, lw=1.5)

# Judge -> Reward scores
draw_arrow(ax, 14.15, 6.55, 14.15, 6.1, color=RL_GREEN, lw=1.5)

# Reward scores -> Group normalization
draw_arrow(ax, 12.9, 5.67, 12.3, 5.67, color=RL_GREEN, lw=1.5)

# Group normalization -> GRPO Loss
draw_arrow(ax, 11.05, 5.25, 11.05, 4.65, color=RL_GREEN, lw=1.5)
# Also reward info feeds into GRPO
draw_arrow(ax, 14.15, 5.25, 14.15, 4.65, color=RL_GREEN, lw=1.5,
           connectionstyle="arc3,rad=0")
ax.plot([14.15, 14.0], [4.65, 4.65], color=RL_GREEN, lw=1.5, zorder=2)

# Reference policy -> GRPO Loss (for KL penalty)
draw_arrow(ax, 11.15, 3.25, 11.55, 3.8, color=FROZEN_BLUE, lw=1.5,
           connectionstyle="arc3,rad=-0.15")
ax.text(10.8, 3.55, "KL", ha="center", va="center", fontsize=8,
        fontweight="bold", color=FROZEN_BLUE, zorder=5)

# GRPO Loss -> Update
draw_arrow(ax, 13.3, 3.8, 13.8, 3.25, color=TRAIN_ORANGE, lw=1.5,
           connectionstyle="arc3,rad=0.15")

# Update -> feeds back to model (curved arrow going left and up)
draw_arrow(ax, 14.05, 2.55, 14.05, 1.6, color=TRAIN_ORANGE, lw=1.5,
           connectionstyle="arc3,rad=0")
ax.annotate("", xy=(5.1, 1.6), xytext=(14.05, 1.6),
            arrowprops=dict(arrowstyle="-|>", color=TRAIN_ORANGE,
                            lw=1.5, connectionstyle="arc3,rad=0"),
            zorder=2)
ax.annotate("", xy=(5.1, 2.6), xytext=(5.1, 1.6),
            arrowprops=dict(arrowstyle="-|>", color=TRAIN_ORANGE,
                            lw=1.5, connectionstyle="arc3,rad=0"),
            zorder=2)
ax.text(9.5, 1.38, "gradient update to encoder, projector, LoRA",
        ha="center", va="center", fontsize=8, fontstyle="italic",
        color=TRAIN_ORANGE, zorder=5)

# ── Generate K uses the OpenTSLM model ─────────────────────────────────
draw_arrow(ax, 8.5, 4.0, 9.8, 8.0, color=RL_GREEN, lw=1.8,
           style="-|>", connectionstyle="arc3,rad=0.3")
ax.text(9.15, 6.5, "policy\nsampling", ha="center", va="center",
        fontsize=8, fontweight="bold", fontstyle="italic",
        color=RL_GREEN, zorder=5, rotation=60)

# ══════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════
legend_y = 0.35
legend_x = 1.5

# Frozen
frozen_patch = FancyBboxPatch((legend_x, legend_y), 0.5, 0.35,
                               boxstyle="round,pad=0.05",
                               facecolor=FROZEN_BLUE_LIGHT,
                               edgecolor=FROZEN_BLUE, linewidth=1.5)
ax.add_patch(frozen_patch)
ax.text(legend_x + 0.65, legend_y + 0.175, "Frozen Component",
        ha="left", va="center", fontsize=9, color=TEXT_DARK)

# Trainable
train_patch = FancyBboxPatch((legend_x + 3.5, legend_y), 0.5, 0.35,
                              boxstyle="round,pad=0.05",
                              facecolor=TRAIN_ORANGE_LIGHT,
                              edgecolor=TRAIN_ORANGE, linewidth=1.5)
ax.add_patch(train_patch)
ax.text(legend_x + 4.15, legend_y + 0.175, "Trainable Component",
        ha="left", va="center", fontsize=9, color=TEXT_DARK)

# RL-specific
rl_patch = FancyBboxPatch((legend_x + 7.5, legend_y), 0.5, 0.35,
                           boxstyle="round,pad=0.05",
                           facecolor=RL_GREEN_LIGHT,
                           edgecolor=RL_GREEN, linewidth=1.5)
ax.add_patch(rl_patch)
ax.text(legend_x + 8.15, legend_y + 0.175, "RL-Specific Addition",
        ha="left", va="center", fontsize=9, color=TEXT_DARK)

# ── Save ───────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
out_path = "/home/wangni/notion-figures/llm-judge/fig_001.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Figure saved to {out_path}")
