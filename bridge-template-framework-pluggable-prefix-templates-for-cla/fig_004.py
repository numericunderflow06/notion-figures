"""
fig_004 — Template Families and Counts
Horizontal bar chart of the 5 BTF template families + assertive fallback,
showing counts and the priority tiers each family spans.

Data source: bridge-template-framework verified facts (Section 2).
Current truth: 54 registered + 1 fallback = 55 total options
(vs spec's 51 — count drift noted via callout).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ---- Data ----------------------------------------------------------------
# Family, count, tiers spanned (per spec sections 4.2–4.6)
families = [
    ("confidence",   13, ["REJECT", "HIGH", "MEDIUM", "LOW"]),
    ("qtype",        11, ["HIGH", "MEDIUM", "LOW"]),
    ("evidence",     11, ["HIGH", "MEDIUM", "LOW"]),
    ("presentation", 10, ["MEDIUM", "LOW"]),
    ("hierarchy",     9, ["MEDIUM", "LOW"]),
    ("fallback (base)", 1, ["LOW"]),
]

# Order top-to-bottom: largest at top
families = families[::-1]  # so largest ends up at top with barh

# ---- Style ---------------------------------------------------------------
tier_colors = {
    "REJECT":  "#C0392B",  # red
    "HIGH":    "#E67E22",  # orange
    "MEDIUM":  "#2E86C1",  # blue
    "LOW":     "#7D3C98",  # purple
}
tier_order = ["REJECT", "HIGH", "MEDIUM", "LOW"]

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_positions = list(range(len(families)))
names  = [f[0] for f in families]
counts = [f[1] for f in families]
spans  = [f[2] for f in families]

# Main bar: total count, neutral fill
bar_color = "#D5DBDB"
bar_edge  = "#566573"
bars = ax.barh(
    y_positions, counts,
    color=bar_color, edgecolor=bar_edge, linewidth=1.2,
    height=0.62, zorder=2,
)

# Tier-span ribbon: a small stack of color blocks at the right end
# Each block = one tier the family spans. Placed *inside* the bar
# at the left, so total bar length still reads as count.
ribbon_block_w = 0.55  # in data units (count axis)
for y, span in zip(y_positions, spans):
    for i, tier in enumerate(tier_order):
        if tier in span:
            x0 = 0.15 + i * (ribbon_block_w + 0.08)
            rect = Rectangle(
                (x0, y - 0.18), ribbon_block_w, 0.36,
                facecolor=tier_colors[tier], edgecolor="white",
                linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)

# Count labels at end of each bar
for y, c in zip(y_positions, counts):
    ax.text(
        c + 0.25, y, str(c),
        va="center", ha="left",
        fontsize=12, fontweight="bold", color="#212F3C", zorder=4,
    )

# ---- Axes ----------------------------------------------------------------
ax.set_yticks(y_positions)
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel("Number of registered templates", fontsize=12)
ax.set_xlim(0, 16)
ax.set_xticks(range(0, 17, 2))
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=11)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#7F8C8D")
ax.spines["bottom"].set_color("#7F8C8D")
ax.grid(axis="x", linestyle=":", color="#BDC3C7", alpha=0.7, zorder=1)
ax.set_axisbelow(True)

# ---- Title ---------------------------------------------------------------
ax.set_title(
    "BTF Template Families: Counts and Priority-Tier Coverage",
    fontsize=14, fontweight="bold", pad=14, color="#1B2631",
)

# ---- Legend for tiers ----------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=tier_colors[t], edgecolor="white", label=t)
    for t in tier_order
]
leg = ax.legend(
    handles=legend_handles,
    title="Priority tier spanned",
    loc="lower right",
    fontsize=10, title_fontsize=10,
    frameon=True, framealpha=0.95,
    edgecolor="#BDC3C7",
)
leg.get_title().set_fontweight("bold")

# ---- Callout: count drift -----------------------------------------------
callout_text = (
    "Current truth: 54 registered + 1 fallback = 55\n"
    "Spec (about.txt) still reports 51 — count drift\n"
    "noted in docs/templates-catalog.md."
)
ax.text(
    0.985, 0.97, callout_text,
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=9.5, color="#1B2631",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#FEF9E7", edgecolor="#B7950B", linewidth=1.0,
    ),
    zorder=5,
)

# ---- Total annotation ----------------------------------------------------
total = sum(counts)
ax.text(
    0.015, -0.16,
    f"Total options shown: {total}  (≡ 54 registered families + 1 assertive fallback)",
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=10, color="#566573", style="italic",
)

plt.tight_layout()
out_path = (
    "/home/wangni/notion-figures/"
    "bridge-template-framework-pluggable-prefix-templates-for-cla/fig_004.png"
)
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
