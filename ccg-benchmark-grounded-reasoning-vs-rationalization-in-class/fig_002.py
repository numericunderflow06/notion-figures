"""
fig_002: Factorized 8-Cell Rationale-Effect Taxonomy

Decision-tree visualization of the three orthogonal axes
(answer_changed, reasoning_changed, reasoning_direction)
decomposing into the 8 named cells from
source/poc_rationale_effect_judge.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/wangni/notion-figures/ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_002.png"

# Axis palette
COL_AX1 = "#4C78A8"   # answer_changed  (programmatic)
COL_AX2 = "#59A14F"   # reasoning_changed (LLM-judged)
COL_AX3 = "#E15759"   # reasoning_direction (LLM-judged)

# Cell category colors (tinted by their semantics)
LEAF_COLORS = {
    "no_effect":              "#D9D9D9",
    "label_forcing":          "#F4B183",
    "full_follow":            "#C00000",
    "knocked_off_away":       "#7F2BAA",
    "knocked_off_lateral":    "#B084CC",
    "reasoning_only_toward":  "#2E7D32",
    "reasoning_only_lateral": "#7CB342",
    "reasoning_only_away":    "#F9A825",
}

fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.set_axis_off()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ---------- Title ----------
ax.text(
    8, 9.55,
    "Factorized 8-Cell Rationale-Effect Taxonomy",
    ha="center", va="center",
    fontsize=18, fontweight="bold",
)
ax.text(
    8, 9.10,
    "Three orthogonal axes from poc_rationale_effect_judge.py "
    "decompose into 8 named cells\n"
    "(cross-tabbed with the 5-regime labels: grounded / rationalizing / "
    "ignoring / resistant / gullible)",
    ha="center", va="center",
    fontsize=10.5, style="italic", color="#444444",
)

# ---------- Helper to draw a node ----------
def node(x, y, w, h, label, fc, ec="black", fontsize=10.5, fontweight="normal", txtcolor="black"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.4,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color=txtcolor,
    )

def arrow(x1, y1, x2, y2, color="#555555", label=None, label_offset=(0, 0),
          label_color=None, label_bg=None, label_fontsize=9.5, label_weight="bold"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
    )
    ax.add_patch(a)
    if label is not None:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(
            mx, my, label,
            ha="center", va="center",
            fontsize=label_fontsize,
            fontweight=label_weight,
            color=label_color or color,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=label_bg or "white",
                ec="none",
                alpha=0.95,
            ),
        )

# ---------- Layout coords ----------
# Levels (y): root=8.1  L1=6.6  L2=5.0  leaves=3.0
Y_ROOT = 8.1
Y_L1   = 6.55
Y_L2   = 4.95
Y_LEAF = 2.85

# Root node
node(8, Y_ROOT, 4.8, 0.75,
     "Rationale effect of class signal\non a single generation",
     fc="#F2F2F2", ec="#222222", fontweight="bold", fontsize=11)

# L1 — axis 1 split: answer_changed
# Two branches:
node(4, Y_L1, 4.2, 0.78,
     "answer_changed = NO\n(programmatic match)",
     fc=COL_AX1, ec="black", fontweight="bold", fontsize=10.5, txtcolor="white")
node(12, Y_L1, 4.2, 0.78,
     "answer_changed = YES\n(programmatic match)",
     fc=COL_AX1, ec="black", fontweight="bold", fontsize=10.5, txtcolor="white")

arrow(8, Y_ROOT - 0.38, 4, Y_L1 + 0.40, color=COL_AX1)
arrow(8, Y_ROOT - 0.38, 12, Y_L1 + 0.40, color=COL_AX1)

# Axis-1 callout
ax.text(8, Y_L1 + 0.95, "Axis 1 · answer_changed  (programmatic)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=COL_AX1)

# L2 — axis 2 split: reasoning_changed under each L1 branch
# Left (answer_changed = NO)
node(2.1, Y_L2, 3.4, 0.72,
     "reasoning_changed = NO",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")
node(5.9, Y_L2, 3.4, 0.72,
     "reasoning_changed = YES",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")

arrow(4, Y_L1 - 0.40, 2.1, Y_L2 + 0.37, color=COL_AX2)
arrow(4, Y_L1 - 0.40, 5.9, Y_L2 + 0.37, color=COL_AX2)

# Right (answer_changed = YES)
node(10.1, Y_L2, 3.4, 0.72,
     "reasoning_changed = NO",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")
node(13.9, Y_L2, 3.4, 0.72,
     "reasoning_changed = YES",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")

arrow(12, Y_L1 - 0.40, 10.1, Y_L2 + 0.37, color=COL_AX2)
arrow(12, Y_L1 - 0.40, 13.9, Y_L2 + 0.37, color=COL_AX2)

# Axis-2 callout
ax.text(8, Y_L2 + 0.65, "Axis 2 · reasoning_changed  (LLM-judged)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=COL_AX2)

# ---------- Leaves ----------
# 1. answer NO,  reasoning NO  → no_effect            (single leaf)
# 2. answer NO,  reasoning YES → reasoning_only_{toward, lateral, away}
# 3. answer YES, reasoning NO  → label_forcing        (single leaf)
# 4. answer YES, reasoning YES → full_follow / knocked_off_lateral / knocked_off_away

leaf_w = 2.05
leaf_h = 0.95

def leaf(x, y, name, subtitle):
    node(x, y, leaf_w, leaf_h, f"{name}\n{subtitle}",
         fc=LEAF_COLORS[name], ec="black", fontweight="bold", fontsize=9.5)

# Branch A: no_effect (single leaf, sits straight under "reasoning_changed = NO")
leaf(2.1, Y_LEAF, "no_effect",
     "answer unchanged\nreasoning unchanged")
arrow(2.1, Y_L2 - 0.37, 2.1, Y_LEAF + 0.50, color="#555555")

# Branch B: reasoning_only_{toward, lateral, away}
# Place three leaves under x=5.9
x_b = [4.5, 6.0, 7.6]
names_b = ["reasoning_only_toward", "reasoning_only_lateral", "reasoning_only_away"]
subs_b  = ["answer unchanged\nreasoning ↗ toward signal",
           "answer unchanged\nreasoning ↔ lateral",
           "answer unchanged\nreasoning ↘ away from signal"]
dir_labels_b = ["toward", "lateral", "away"]
for xi, n, s, dl in zip(x_b, names_b, subs_b, dir_labels_b):
    leaf(xi, Y_LEAF, n, s)
    arrow(5.9, Y_L2 - 0.37, xi, Y_LEAF + 0.50,
          color=COL_AX3, label=dl,
          label_offset=(0.0, 0.05),
          label_bg="white", label_fontsize=8.5)

# Branch C: label_forcing (single leaf)
leaf(10.1, Y_LEAF, "label_forcing",
     "answer flipped\nreasoning unchanged")
arrow(10.1, Y_L2 - 0.37, 10.1, Y_LEAF + 0.50, color="#555555")

# Branch D: full_follow / knocked_off_lateral / knocked_off_away
x_d = [12.5, 14.0, 15.6]
names_d = ["full_follow", "knocked_off_lateral", "knocked_off_away"]
subs_d  = ["answer flipped\nreasoning ↗ toward signal",
           "answer flipped\nreasoning ↔ lateral",
           "answer flipped\nreasoning ↘ away from signal"]
dir_labels_d = ["toward", "lateral", "away"]
for xi, n, s, dl in zip(x_d, names_d, subs_d, dir_labels_d):
    leaf(xi, Y_LEAF, n, s)
    arrow(13.9, Y_L2 - 0.37, xi, Y_LEAF + 0.50,
          color=COL_AX3, label=dl,
          label_offset=(0.0, 0.05),
          label_bg="white", label_fontsize=8.5)

# Axis-3 callout (only applies on the YES-reasoning branches)
ax.text(8, Y_LEAF + 1.05,
        "Axis 3 · reasoning_direction  (LLM-judged · only when reasoning_changed = YES)",
        ha="center", va="center", fontsize=10, fontweight="bold", color=COL_AX3)

# ---------- Bottom: legend + caption ----------
# Axis legend
legend_handles = [
    mpatches.Patch(facecolor=COL_AX1, edgecolor="black",
                   label="Axis 1 · answer_changed  (programmatic)"),
    mpatches.Patch(facecolor=COL_AX2, edgecolor="black",
                   label="Axis 2 · reasoning_changed  (LLM-judged)"),
    mpatches.Patch(facecolor=COL_AX3, edgecolor="black",
                   label="Axis 3 · reasoning_direction  (LLM-judged: toward / lateral / away)"),
]
leg = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=False,
    fontsize=10,
)

# Caption
ax.text(
    8, 0.95,
    "The three axes are orthogonal: programmatic answer-flip detection "
    "is decoupled from the LLM-judged reasoning-change and direction calls.\n"
    "8 cells arise because reasoning_direction is only defined when "
    "reasoning_changed = YES (2 × [1 + 3] = 8).  Cross-tabs with the "
    "5-regime labels are how the page reads each cell.",
    ha="center", va="center",
    fontsize=9.5, color="#333333",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
