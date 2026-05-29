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

fig, ax = plt.subplots(figsize=(23, 11), dpi=200)
ax.set_xlim(0, 23)
ax.set_ylim(0, 11)
ax.set_axis_off()
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ---------- Title ----------
ax.text(
    11.5, 10.55,
    "Factorized 8-Cell Rationale-Effect Taxonomy",
    ha="center", va="center",
    fontsize=18, fontweight="bold",
)
ax.text(
    11.5, 10.05,
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
          label_color=None, label_bg=None, label_fontsize=9.0, label_weight="bold"):
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
Y_ROOT = 9.05
Y_L1   = 7.50
Y_L2   = 5.85
Y_LEAF = 3.55

# Root node (center of the tree area, x=12)
node(12, Y_ROOT, 5.0, 0.78,
     "Rationale effect of class signal\non a single generation",
     fc="#F2F2F2", ec="#222222", fontweight="bold", fontsize=11)

# L1 — axis 1 split: answer_changed
node(7, Y_L1, 4.4, 0.80,
     "answer_changed = NO\n(programmatic match)",
     fc=COL_AX1, ec="black", fontweight="bold", fontsize=10.5, txtcolor="white")
node(17, Y_L1, 4.4, 0.80,
     "answer_changed = YES\n(programmatic match)",
     fc=COL_AX1, ec="black", fontweight="bold", fontsize=10.5, txtcolor="white")

arrow(12, Y_ROOT - 0.39, 7, Y_L1 + 0.41, color=COL_AX1)
arrow(12, Y_ROOT - 0.39, 17, Y_L1 + 0.41, color=COL_AX1)

# L2 — axis 2 split: reasoning_changed under each L1 branch
node(4.5, Y_L2, 3.4, 0.74,
     "reasoning_changed = NO",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")
node(9.5, Y_L2, 3.4, 0.74,
     "reasoning_changed = YES",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")

arrow(7, Y_L1 - 0.41, 4.5, Y_L2 + 0.38, color=COL_AX2)
arrow(7, Y_L1 - 0.41, 9.5, Y_L2 + 0.38, color=COL_AX2)

node(14.5, Y_L2, 3.4, 0.74,
     "reasoning_changed = NO",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")
node(19.5, Y_L2, 3.4, 0.74,
     "reasoning_changed = YES",
     fc=COL_AX2, ec="black", fontweight="bold", fontsize=10, txtcolor="white")

arrow(17, Y_L1 - 0.41, 14.5, Y_L2 + 0.38, color=COL_AX2)
arrow(17, Y_L1 - 0.41, 19.5, Y_L2 + 0.38, color=COL_AX2)

# ---------- Axis labels (left-margin row labels) ----------
def axis_label(y, num, name, sub, color):
    ax.text(
        1.55, y,
        f"AXIS {num}\n{name}\n{sub}",
        ha="center", va="center",
        fontsize=10, fontweight="bold", color=color,
        bbox=dict(
            boxstyle="round,pad=0.45",
            fc="white", ec=color, lw=1.4,
        ),
    )

# Place each axis label vertically centered between the parent level
# and the child level it splits into — in the LEFT MARGIN so it does not
# collide with any connecting arrow.
axis_label((Y_ROOT + Y_L1) / 2, 1, "answer_changed", "(programmatic)",          COL_AX1)
axis_label((Y_L1   + Y_L2) / 2, 2, "reasoning_changed", "(LLM-judged)",         COL_AX2)
axis_label((Y_L2 + Y_LEAF) / 2, 3, "reasoning_direction",
           "(LLM-judged,\nonly if reasoning_changed = YES)",                    COL_AX3)

# ---------- Leaves ----------
leaf_w = 1.95
leaf_h = 1.05

def leaf(x, y, name, subtitle):
    node(x, y, leaf_w, leaf_h, f"{name}\n{subtitle}",
         fc=LEAF_COLORS[name], ec="black", fontweight="bold", fontsize=8.6)

# Branch A: no_effect (single leaf under left-NO L2)
leaf(4.5, Y_LEAF, "no_effect",
     "answer unchanged\nreasoning unchanged")
arrow(4.5, Y_L2 - 0.38, 4.5, Y_LEAF + 0.55, color="#555555")

# Branch B: reasoning_only_{toward, lateral, away}  (under x=9.5)
x_b = [7.3, 9.5, 11.7]
names_b = ["reasoning_only_toward", "reasoning_only_lateral", "reasoning_only_away"]
subs_b  = ["answer unchanged\nreasoning -> toward",
           "answer unchanged\nreasoning <-> lateral",
           "answer unchanged\nreasoning -> away"]
dir_labels_b = ["toward", "lateral", "away"]
for xi, n, s, dl in zip(x_b, names_b, subs_b, dir_labels_b):
    leaf(xi, Y_LEAF, n, s)
    arrow(9.5, Y_L2 - 0.38, xi, Y_LEAF + 0.55,
          color=COL_AX3, label=dl,
          label_offset=(0.0, 0.0),
          label_bg="white", label_fontsize=8.5)

# Branch C: label_forcing (single leaf under right-NO L2 at x=14.5)
leaf(14.5, Y_LEAF, "label_forcing",
     "answer flipped\nreasoning unchanged")
arrow(14.5, Y_L2 - 0.38, 14.5, Y_LEAF + 0.55, color="#555555")

# Branch D: full_follow / knocked_off_lateral / knocked_off_away  (under x=19.5)
x_d = [17.3, 19.5, 21.7]
names_d = ["full_follow", "knocked_off_lateral", "knocked_off_away"]
subs_d  = ["answer flipped\nreasoning -> toward",
           "answer flipped\nreasoning <-> lateral",
           "answer flipped\nreasoning -> away"]
dir_labels_d = ["toward", "lateral", "away"]
for xi, n, s, dl in zip(x_d, names_d, subs_d, dir_labels_d):
    leaf(xi, Y_LEAF, n, s)
    arrow(19.5, Y_L2 - 0.38, xi, Y_LEAF + 0.55,
          color=COL_AX3, label=dl,
          label_offset=(0.0, 0.0),
          label_bg="white", label_fontsize=8.5)

# ---------- Bottom: legend + caption ----------
legend_handles = [
    mpatches.Patch(facecolor=COL_AX1, edgecolor="black",
                   label="Axis 1 - answer_changed  (programmatic)"),
    mpatches.Patch(facecolor=COL_AX2, edgecolor="black",
                   label="Axis 2 - reasoning_changed  (LLM-judged)"),
    mpatches.Patch(facecolor=COL_AX3, edgecolor="black",
                   label="Axis 3 - reasoning_direction  (LLM-judged: toward / lateral / away)"),
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
    11.5, 1.30,
    "The three axes are orthogonal: programmatic answer-flip detection "
    "is decoupled from the LLM-judged reasoning-change and direction calls.\n"
    "8 cells arise because reasoning_direction is only defined when "
    "reasoning_changed = YES  (2 x [1 + 3] = 8).  "
    "Cross-tabs with the 5-regime labels are how the page reads each cell.",
    ha="center", va="center",
    fontsize=10, color="#333333",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
