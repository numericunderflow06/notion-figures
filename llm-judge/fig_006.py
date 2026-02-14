"""
fig_006: Trainable Components During RL vs SFT
Side-by-side comparison diagram showing which components are trainable/frozen
in each training stage for OpenTSLM.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
C_TRAIN    = "#4CAF50"   # green – trainable
C_TRAIN_BG = "#E8F5E9"   # light green fill
C_FROZEN   = "#9E9E9E"   # gray – frozen
C_FROZEN_BG= "#F5F5F5"   # light gray fill
C_REF      = "#7E57C2"   # purple – reference policy (frozen)
C_REF_BG   = "#EDE7F6"   # light purple fill
C_BORDER   = "#424242"
C_HEADER_SFT = "#1565C0" # dark blue
C_HEADER_RL  = "#C62828"  # dark red

# ── Figure setup ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.patch.set_facecolor("white")

# Title
fig.suptitle("Trainable Components: SFT vs GRPO-RL",
             fontsize=20, fontweight="bold", y=0.97, color="#212121")

# ── Component definitions ───────────────────────────────────────────
# Each component: (label, trainable_in_sft, trainable_in_rl, param_note, is_ref_policy)
# For SFT panel:
sft_components = [
    ("LLM Backbone\n(e.g. Llama-3.1-8B)",    False, "frozen",  "Frozen (billions of params)"),
    ("LoRA Adapters",                          True,  "train",   "Trainable"),
    ("Projector\n(MLPProjector)",              True,  "train",   "Trainable"),
    ("Time-Series Encoder\n(TransformerCNNEncoder)", True, "train", "Trainable"),
]

rl_components = [
    ("LLM Backbone\n(e.g. Llama-3.1-8B)",    False, "frozen",  "Frozen (billions of params)"),
    ("LoRA Adapters",                          True,  "train",   "Trainable (from SFT state)"),
    ("Projector\n(MLPProjector)",              True,  "train",   "Trainable (from SFT state)"),
    ("Time-Series Encoder\n(TransformerCNNEncoder)", True, "train", "Trainable (from SFT state)"),
    ("Reference Policy π_ref\n(frozen SFT copy)", False, "ref", "Frozen (~5–15M params)\nUsed for KL penalty only"),
]


def draw_panel(ax, title, title_color, components, show_ref=False):
    """Draw a single panel with stacked component boxes."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel header
    header = FancyBboxPatch(
        (0.3, 10.8), 9.4, 1.0,
        boxstyle="round,pad=0.15",
        facecolor=title_color, edgecolor=title_color,
        linewidth=2, alpha=0.9
    )
    ax.add_patch(header)
    ax.text(5, 11.3, title, ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")

    # Compute layout
    n = len(components)
    box_h = 1.5
    gap = 0.3
    total_h = n * box_h + (n - 1) * gap

    # Start y from top, leaving room for header
    start_y = 10.3 - 0.4  # below header

    for i, (label, trainable, mode, note) in enumerate(components):
        y = start_y - i * (box_h + gap) - box_h

        if mode == "ref":
            fc = C_REF_BG
            ec = C_REF
            status_text = "FROZEN (reference)"
            status_color = C_REF
            icon = "❄"
        elif mode == "train":
            fc = C_TRAIN_BG
            ec = C_TRAIN
            status_text = "TRAINABLE"
            status_color = C_TRAIN
            icon = "✓"
        else:
            fc = C_FROZEN_BG
            ec = C_FROZEN
            status_text = "FROZEN"
            status_color = C_FROZEN
            icon = "❄"

        # Main box
        box = FancyBboxPatch(
            (0.5, y), 9.0, box_h,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec,
            linewidth=2.5
        )
        ax.add_patch(box)

        # Status badge (right side)
        badge_w = 2.2
        if mode == "ref":
            badge_w = 2.8
        badge = FancyBboxPatch(
            (9.5 - badge_w - 0.2, y + box_h - 0.5), badge_w, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=ec, edgecolor=ec,
            linewidth=1, alpha=0.85
        )
        ax.add_patch(badge)
        ax.text(9.5 - badge_w / 2 - 0.2, y + box_h - 0.3,
                status_text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")

        # Component name (left-aligned)
        ax.text(1.0, y + box_h / 2 + 0.1, label,
                ha="left", va="center",
                fontsize=11.5, fontweight="bold", color="#212121")

        # Note text (smaller, below name if single line label, or offset)
        if note:
            ax.text(1.0, y + 0.25, note,
                    ha="left", va="center",
                    fontsize=9, color="#616161", style="italic")


# ── Draw SFT panel ──────────────────────────────────────────────────
draw_panel(axes[0],
           "Supervised Fine-Tuning (SFT)",
           C_HEADER_SFT,
           sft_components)

# Add "Teacher-forcing on CoT targets" annotation
axes[0].text(5.0, 1.0,
             "Training signal: cross-entropy loss\n(teacher-forcing on CoT targets)",
             ha="center", va="center",
             fontsize=10, color=C_HEADER_SFT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD",
                       edgecolor=C_HEADER_SFT, linewidth=1.5))

# ── Draw RL panel ───────────────────────────────────────────────────
draw_panel(axes[1],
           "GRPO Reinforcement Learning (RL)",
           C_HEADER_RL,
           rl_components)

# Add "GRPO signal" annotation
axes[1].text(5.0, 0.5,
             "Training signal: GRPO loss\n(LLM-judge rewards, KL penalty vs π_ref)",
             ha="center", va="center",
             fontsize=10, color=C_HEADER_RL, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFEBEE",
                       edgecolor=C_HEADER_RL, linewidth=1.5))

# ── Legend ──────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=C_TRAIN_BG, edgecolor=C_TRAIN, linewidth=2,
                   label="Trainable (weights updated)"),
    mpatches.Patch(facecolor=C_FROZEN_BG, edgecolor=C_FROZEN, linewidth=2,
                   label="Frozen (weights fixed)"),
    mpatches.Patch(facecolor=C_REF_BG, edgecolor=C_REF, linewidth=2,
                   label="Reference policy (frozen SFT copy, KL anchor)"),
]

fig.legend(handles=legend_elements,
           loc="lower center", ncol=3,
           fontsize=11, frameon=True,
           fancybox=True, shadow=False,
           edgecolor="#BDBDBD",
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.06, 1, 0.94])

# ── Save ────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/llm-judge/fig_006.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Figure saved to {out_path}")
