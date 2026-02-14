"""
fig_008: Trainable Components — SP vs. Flamingo during GRPO
Side-by-side architecture diagram showing which components are trainable
and which are frozen, plus reference model sharing strategy.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour palette ──────────────────────────────────────────────────
SP_TRAIN       = "#3B82F6"
SP_TRAIN_LIGHT = "#DBEAFE"
FL_TRAIN       = "#10B981"
FL_TRAIN_LIGHT = "#D1FAE5"
FROZEN_BG      = "#F3F4F6"
FROZEN_EDGE    = "#9CA3AF"
TEXT_DARK      = "#1F2937"
TEXT_MID       = "#6B7280"
REF_ICON_COLOR = "#F59E0B"
REF_EDGE       = "#D97706"
WHITE          = "#FFFFFF"

fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
fig.patch.set_facecolor(WHITE)
for ax in axes:
    ax.set_facecolor(WHITE)


def draw_box(ax, x, y, w, h, label, trained=True, color="#3B82F6",
             light_color="#DBEAFE", fontsize=11, ref_model=False,
             sublabel=None):
    if trained:
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012",
            facecolor=light_color, edgecolor=color,
            linewidth=2.2, zorder=3)
        txt_color = color
    else:
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012",
            facecolor=FROZEN_BG, edgecolor=FROZEN_EDGE,
            linewidth=1.8, linestyle="--", zorder=3)
        txt_color = TEXT_MID
    ax.add_patch(rect)

    cy = y + h / 2 + (0.02 if sublabel else 0)
    ax.text(x + w / 2, cy, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=txt_color, zorder=4)

    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.025, sublabel,
                ha="center", va="center", fontsize=7.5,
                fontstyle="italic", color=TEXT_MID, zorder=4)

    badge = "Trainable" if trained else "Frozen"
    badge_c = color if trained else FROZEN_EDGE
    ax.text(x + w - 0.02, y + h - 0.012, badge,
            ha="right", va="top", fontsize=7,
            color=badge_c, fontweight="bold", zorder=4)

    if ref_model:
        dx = x + 0.02
        dy = y + h - 0.018
        ax.plot(dx, dy, marker="D", markersize=6,
                color=REF_ICON_COLOR, markeredgecolor=REF_EDGE,
                markeredgewidth=0.8, zorder=5)
        ax.text(dx + 0.022, dy, "ref", ha="left", va="center",
                fontsize=6.5, color=REF_EDGE, fontweight="bold", zorder=5)


def draw_arrow(ax, x, y_from, y_to, color="#9CA3AF"):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14), zorder=2)


def draw_column(ax, title, color, light, components, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)
    ax.axis("off")

    # Title & subtitle
    ax.text(0.5, 0.97, title, ha="center", va="top",
            fontsize=16, fontweight="bold", color=color)
    ax.text(0.5, 0.92, subtitle, ha="center", va="top",
            fontsize=10, color=TEXT_MID)

    # Input label
    ax.text(0.5, 0.87, "Time-Series Input", ha="center", va="center",
            fontsize=9.5, color=TEXT_MID, fontstyle="italic")

    bx, bw = 0.08, 0.84
    box_h = 0.1
    gap = 0.04

    y_top = 0.82  # top of first box

    positions = []
    for i, comp in enumerate(components):
        by = y_top - i * (box_h + gap)
        draw_box(ax, bx, by, bw, box_h,
                 comp["name"],
                 trained=comp.get("trained", True),
                 color=color, light_color=light,
                 fontsize=comp.get("fontsize", 11),
                 ref_model=comp.get("ref_model", False),
                 sublabel=comp.get("sublabel"))
        positions.append((0.5, by + box_h, by))

    # Input → first box
    draw_arrow(ax, 0.5, 0.855, positions[0][1] + 0.005)
    # Between trainable boxes
    for i in range(len(positions) - 1):
        draw_arrow(ax, 0.5, positions[i][2] - 0.005,
                   positions[i + 1][1] + 0.005)

    # Frozen LLM backbone
    last_bot = positions[-1][2]
    llm_h = 0.16
    llm_y = last_bot - gap - llm_h
    draw_box(ax, bx, llm_y, bw, llm_h,
             "LLM Backbone (~1B params)", trained=False,
             fontsize=13)
    ax.text(bx + bw / 2, llm_y + llm_h * 0.30,
            "Shared between policy & reference",
            ha="center", va="center", fontsize=8.5,
            color=TEXT_MID, fontstyle="italic")

    draw_arrow(ax, 0.5, last_bot - 0.005, llm_y + llm_h + 0.005)

    # Output
    draw_arrow(ax, 0.5, llm_y - 0.005, llm_y - 0.045)
    ax.text(0.5, llm_y - 0.065, "CoT Rationale + Answer",
            ha="center", va="center", fontsize=9.5,
            color=TEXT_MID, fontstyle="italic")


# ═══════════════════════════════════════════════════════════════════
# LEFT — OpenTSLM-SP
# ═══════════════════════════════════════════════════════════════════
sp_components = [
    {"name": "TransformerCNNEncoder", "ref_model": True,
     "sublabel": "src/model/encoder/TransformerCNNEncoder.py"},
    {"name": "MLPProjector", "ref_model": True,
     "sublabel": "src/model/projector/MLPProjector.py"},
    {"name": "LoRA Adapters (q_proj, v_proj, ...)", "ref_model": True,
     "fontsize": 10.5},
]
draw_column(axes[0], "OpenTSLM-SP", SP_TRAIN, SP_TRAIN_LIGHT,
            sp_components, "Trainable during GRPO")

# ═══════════════════════════════════════════════════════════════════
# RIGHT — OpenTSLM-Flamingo
# ═══════════════════════════════════════════════════════════════════
fl_components = [
    {"name": "CNNTokenizer", "ref_model": True,
     "sublabel": "src/model/encoder/CNNTokenizer.py"},
    {"name": "Perceiver Resampler", "ref_model": True},
    {"name": "Gated Cross-Attention Layers", "ref_model": True,
     "sublabel": "Every 2 transformer blocks", "fontsize": 10.5},
]
draw_column(axes[1], "OpenTSLM-Flamingo", FL_TRAIN, FL_TRAIN_LIGHT,
            fl_components, "Trainable during GRPO")

# ── Divider ─────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.5, 0.5], [0.07, 0.93],
                          transform=fig.transFigure,
                          color="#D1D5DB", linewidth=1.2, zorder=1))

# ── Legend ──────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=SP_TRAIN_LIGHT, edgecolor=SP_TRAIN,
                   linewidth=2, label="SP trainable"),
    mpatches.Patch(facecolor=FL_TRAIN_LIGHT, edgecolor=FL_TRAIN,
                   linewidth=2, label="Flamingo trainable"),
    mpatches.Patch(facecolor=FROZEN_BG, edgecolor=FROZEN_EDGE,
                   linewidth=1.5, linestyle="--", label="Frozen (shared)"),
    plt.Line2D([0], [0], marker="D", color="w",
               markerfacecolor=REF_ICON_COLOR, markeredgecolor=REF_EDGE,
               markersize=8, label=r"Copied for reference model ($\pi_{\mathrm{ref}}$)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9.5, frameon=True, fancybox=True,
           edgecolor="#D1D5DB", facecolor=WHITE,
           handlelength=1.8, handletextpad=0.6, columnspacing=1.5,
           bbox_to_anchor=(0.5, 0.005))

# ── Title ───────────────────────────────────────────────────────
fig.suptitle("Trainable Components: SP vs. Flamingo during GRPO",
             fontsize=18, fontweight="bold", color=TEXT_DARK, y=0.995)

plt.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06,
                    wspace=0.1)

fig.savefig("/home/wangni/notion-figures/cot/fig_008.png",
            dpi=200, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)
print("Saved: /home/wangni/notion-figures/cot/fig_008.png")
