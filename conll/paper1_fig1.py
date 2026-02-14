"""
Paper 1, Figure 1: ZuCo Classification Accuracy by Initialization Strategy
Grouped bar chart comparing initialization strategies across ZuCo 1.0 and ZuCo 2.0.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------- verified data ----------
strategies = ["From-Scratch", "Sleep-Flamingo", "Full-Curriculum"]

# Values from verified facts (None = no reported value)
zuco1 = [62.25, 69.55, 70.51]   # ZuCo 1.0 accuracy (%)
zuco2 = [62.25, 68.29, None]    # ZuCo 2.0 accuracy (%)

# ---------- plot setup ----------
# Use wider spacing between groups by setting x positions manually
x = np.array([0.0, 1.1, 2.5])   # extra gap before Full-Curriculum for annotation room
n = len(strategies)

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_w = 0.30
gap = 0.04  # half-gap between paired bars

# Color palette
c1 = "#3A7CA5"   # ZuCo 1.0  (steel blue)
c2 = "#E07A5F"   # ZuCo 2.0  (terra cotta)
c_gain = "#2D6A4F"  # annotation green

# --- ZuCo 1.0 bars ---
ax.bar(
    x - bar_w / 2 - gap, zuco1, bar_w,
    label="ZuCo 1.0", color=c1, edgecolor="white", linewidth=0.6,
    zorder=3,
)

# --- ZuCo 2.0 bars (skip None) ---
for i, val in enumerate(zuco2):
    if val is not None:
        ax.bar(
            x[i] + bar_w / 2 + gap, val, bar_w,
            color=c2, edgecolor="white", linewidth=0.6,
            zorder=3,
            label="ZuCo 2.0" if i == 0 else "",
        )

# ---------- value labels on each bar ----------
def label_bar(xpos, yval, color):
    ax.text(
        xpos, yval + 0.35, f"{yval:.2f}%",
        ha="center", va="bottom", fontsize=9.5,
        color=color,
    )

for i, v in enumerate(zuco1):
    label_bar(x[i] - bar_w / 2 - gap, v, c1)
for i, v in enumerate(zuco2):
    if v is not None:
        label_bar(x[i] + bar_w / 2 + gap, v, c2)

# ---------- annotate the +6.04 pp transfer gain ----------
sf_bar_x = x[1] + bar_w / 2 + gap   # sleep-flamingo ZuCo 2.0 bar center x
fs_y = zuco2[0]                       # from-scratch  = 62.25
sf_y = zuco2[1]                       # sleep-flamingo = 68.29
mid_y = (fs_y + sf_y) / 2

# Vertical double-headed arrow just right of the sleep-flamingo ZuCo 2.0 bar
arrow_x = sf_bar_x + bar_w / 2 + 0.07
ax.annotate(
    "", xy=(arrow_x, sf_y - 0.25), xytext=(arrow_x, fs_y + 0.25),
    arrowprops=dict(arrowstyle="<->", lw=1.5, color=c_gain),
    zorder=5,
)

# Place annotation text in the gap between Sleep-Flamingo and Full-Curriculum groups
text_x = (x[1] + x[2]) / 2 + 0.1
ax.text(
    text_x, mid_y,
    "+6.04 pp\ncross-modality\ntransfer gain",
    fontsize=9, fontweight="bold", color=c_gain,
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f7f4", edgecolor=c_gain,
              alpha=0.95, linewidth=1.0),
    zorder=6,
)

# Thin horizontal connector from arrow to text box
ax.plot([arrow_x + 0.02, text_x - 0.38], [mid_y, mid_y],
        lw=1.0, color=c_gain, ls="-", zorder=4)

# ---------- axes & labels ----------
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=11)
ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
ax.set_title(
    "ZuCo Classification Accuracy by Initialization Strategy",
    fontsize=13, fontweight="bold", pad=14,
)
ax.set_ylim(55, 76)
ax.set_xlim(-0.55, x[-1] + 0.55)
ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

# Grid
ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

# Legend
ax.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc")

# Spine cleanup
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)

# Note for missing full-curriculum ZuCo 2.0 value
ax.text(
    x[2] + bar_w / 2 + gap, 56.0, "n/a",
    ha="center", va="bottom", fontsize=9, fontstyle="italic", color="#999999",
)

plt.tight_layout()
out = "/home/wangni/notion-figures/conll/paper1_fig1.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved to {out}")
