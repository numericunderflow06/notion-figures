"""
fig_001: Reasoning-Regime 2x2 Taxonomy

Crosses class-signal correctness (gold vs. wrong) with final-answer correctness
(right vs. wrong). Quadrants labeled with regime names. Sidebar callout for
the degenerate-output regime.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------
# Palette
# ---------------------------------------------------------------
GOOD       = "#2E8B57"   # grounded, resistant
GOOD_LIGHT = "#D7EBDE"
AMBIG      = "#E67E22"   # rationalizing (good signal but post-hoc)
AMBIG_LIGHT = "#FAE3D0"
BAD        = "#C0392B"   # gullible, ignoring
BAD_LIGHT  = "#F4D4D0"
SIDEBAR    = "#7D6B91"
SIDEBAR_LT = "#E5DFEC"
AXIS_GREY  = "#2C3E50"

# ---------------------------------------------------------------
# Figure
# ---------------------------------------------------------------
fig = plt.figure(figsize=(13, 7.2))
fig.patch.set_facecolor("white")

# Main grid axis (left side) + sidebar axis (right side)
ax = fig.add_axes([0.08, 0.10, 0.62, 0.78])
sb = fig.add_axes([0.74, 0.10, 0.22, 0.78])

for a in (ax, sb):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)

# ---------------------------------------------------------------
# 2x2 grid coordinates
# ---------------------------------------------------------------
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Cells: (x, y, w, h)
# Columns:  left = answer right,  right = answer wrong
# Rows:     top  = gold signal,   bottom = wrong signal
cell_w, cell_h = 4.4, 3.8
x_left, x_right = 0.6, 5.2
y_top, y_bottom = 5.4, 1.4

cells = {
    "gold_right":  (x_left,  y_top,    AMBIG_LIGHT, AMBIG,
                    "Grounded  OR  Rationalizing",
                    "Gold signal + correct answer:\n"
                    "either genuine reasoning\n"
                    "or post-hoc justification."),
    "gold_wrong":  (x_right, y_top,    BAD_LIGHT,   BAD,
                    "Ignoring",
                    "Useful gold signal was\n"
                    "available but not used."),
    "wrong_right": (x_left,  y_bottom, GOOD_LIGHT,  GOOD,
                    "Resistant",
                    "Refused a misleading signal\n"
                    "and still answered correctly."),
    "wrong_wrong": (x_right, y_bottom, BAD_LIGHT,   BAD,
                    "Gullible",
                    "Followed a misleading signal\n"
                    "into a wrong answer."),
}

# Draw cells
for key, (cx, cy, fc, ec, title, body) in cells.items():
    box = FancyBboxPatch(
        (cx, cy), cell_w, cell_h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(cx + cell_w / 2, cy + cell_h - 0.75, title,
            ha="center", va="center",
            fontsize=14, fontweight="bold", color=ec, zorder=3)
    ax.text(cx + cell_w / 2, cy + cell_h / 2 - 0.55, body,
            ha="center", va="center",
            fontsize=10.5, color=AXIS_GREY, zorder=3)

# Good / Bad chips inside each cell
chip_meta = {
    "gold_right":  ("ambiguous",  AMBIG),
    "gold_wrong":  ("bad",        BAD),
    "wrong_right": ("good",       GOOD),
    "wrong_wrong": ("bad",        BAD),
}
for key, (label, color) in chip_meta.items():
    cx, cy, *_ = cells[key]
    chip = FancyBboxPatch(
        (cx + 0.18, cy + 0.22), 1.15, 0.55,
        boxstyle="round,pad=0.01,rounding_size=0.18",
        linewidth=0, facecolor=color, zorder=4,
    )
    ax.add_patch(chip)
    ax.text(cx + 0.18 + 1.15 / 2, cy + 0.22 + 0.55 / 2, label,
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5)

# ---------------------------------------------------------------
# Column headers (answer correctness)
# ---------------------------------------------------------------
ax.text(x_left  + cell_w / 2, 9.55, "Answer  RIGHT",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=AXIS_GREY)
ax.text(x_right + cell_w / 2, 9.55, "Answer  WRONG",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color=AXIS_GREY)
ax.text(5.0, 10.15, "Final-answer correctness",
        ha="center", va="center", fontsize=11.5,
        fontstyle="italic", color=AXIS_GREY)

# ---------------------------------------------------------------
# Row headers (signal correctness) -- left-side rotated labels
# ---------------------------------------------------------------
ax.text(0.05, y_top    + cell_h / 2, "GOLD\nclass signal",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=AXIS_GREY, rotation=90)
ax.text(0.05, y_bottom + cell_h / 2, "WRONG\nclass signal",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=AXIS_GREY, rotation=90)
ax.text(-0.55, 5.4, "Class-signal correctness",
        ha="center", va="center", fontsize=11.5,
        fontstyle="italic", color=AXIS_GREY, rotation=90)

# ---------------------------------------------------------------
# Title
# ---------------------------------------------------------------
fig.text(0.5, 0.955, "Reasoning-Regime 2x2 Taxonomy",
         ha="center", va="center", fontsize=17, fontweight="bold",
         color=AXIS_GREY)
fig.text(0.5, 0.918,
         "Class-signal correctness  x  Final-answer correctness  "
         "(CCG benchmark, 5-regime judge)",
         ha="center", va="center", fontsize=10.5,
         fontstyle="italic", color=AXIS_GREY)

# ---------------------------------------------------------------
# Sidebar: degenerate-output regime
# ---------------------------------------------------------------
sb.set_xlim(0, 10); sb.set_ylim(0, 10)

sb_box = FancyBboxPatch(
    (0.3, 0.6), 9.4, 8.8,
    boxstyle="round,pad=0.02,rounding_size=0.22",
    linewidth=2.0, edgecolor=SIDEBAR, facecolor=SIDEBAR_LT,
)
sb.add_patch(sb_box)

sb.text(5.0, 8.7, "Sidebar regime",
        ha="center", va="center", fontsize=10.5,
        fontstyle="italic", color=SIDEBAR)
sb.text(5.0, 7.95, "Degenerate output",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color=SIDEBAR)

# Divider line
sb.plot([1.0, 9.0], [7.35, 7.35], color=SIDEBAR, linewidth=1.0,
        alpha=0.6)

sb.text(5.0, 6.45,
        "Collapsed text:\nlow vocabulary,\nrepetition.",
        ha="center", va="center", fontsize=10.5, color=AXIS_GREY)

sb.text(5.0, 4.45,
        "Only producible by\n"
        "TRAINED injection\n"
        "surfaces with shared\n"
        "parameters when class\n"
        "gradients differ in\n"
        "magnitude.",
        ha="center", va="center", fontsize=10, color=AXIS_GREY)

sb.text(5.0, 1.65,
        "Inference-time\nprompt injection\ncannot produce it.",
        ha="center", va="center", fontsize=10,
        fontstyle="italic", color=SIDEBAR)

# ---------------------------------------------------------------
# Legend (bottom strip on main axis)
# ---------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=GOOD,  edgecolor="none",
                   label="good regime"),
    mpatches.Patch(facecolor=AMBIG, edgecolor="none",
                   label="ambiguous (grounded vs. rationalizing)"),
    mpatches.Patch(facecolor=BAD,   edgecolor="none",
                   label="bad regime"),
    mpatches.Patch(facecolor=SIDEBAR, edgecolor="none",
                   label="trained-only sidebar"),
]
leg = ax.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.10),
                ncol=4, frameon=False, fontsize=10)
for txt in leg.get_texts():
    txt.set_color(AXIS_GREY)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
out = ("/home/wangni/notion-figures/"
       "ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/"
       "fig_001.png")
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
