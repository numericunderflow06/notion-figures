"""
fig_005: Betting PnL Backtester: OpenTSLM vs Baseline
Multi-panel comparison of betting performance metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Data from verified facts (betting_comparison.json, spec section 10)
# ---------------------------------------------------------------------------
models = ["OpenTSLM", "Baseline"]
pnl = [22.30, -2447.94]
bets_placed = [24, 100]
win_rate = [100.0, 2.0]
roi = [1.25, -97.9]

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
GREEN = "#2E8B57"      # sea-green for positive / OpenTSLM
RED = "#DC3545"         # red for negative / Baseline
LIGHT_GREEN = "#A8D5BA"
LIGHT_RED = "#F5A9A9"
BG_CARD = "#F7F9FC"
DARK_TEXT = "#1A1A2E"
GREY = "#6C757D"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 6), facecolor="white")

# Use gridspec: left panel (PnL bar) | right panel (metrics small multiples)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.35,
                      left=0.08, right=0.90, top=0.82, bottom=0.12)

# ========================== LEFT PANEL: PnL Bar Chart ==========================
ax_pnl = fig.add_subplot(gs[0, 0])

bar_colors = [GREEN, RED]
bars = ax_pnl.bar(models, pnl, width=0.55, color=bar_colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

# Value labels on bars
for bar, val in zip(bars, pnl):
    y = bar.get_height()
    sign = "+" if val > 0 else ""
    label = f"{sign}${val:,.2f}"
    if val >= 0:
        ax_pnl.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=14, fontweight="bold", color=GREEN)
    else:
        ax_pnl.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha="center", va="top",
                        fontsize=14, fontweight="bold", color=RED)

ax_pnl.axhline(0, color=GREY, linewidth=0.8, zorder=2)
ax_pnl.set_ylabel("Profit / Loss ($)", fontsize=12, color=DARK_TEXT)
ax_pnl.set_title("PnL Comparison", fontsize=14, fontweight="bold",
                  color=DARK_TEXT, pad=12)
ax_pnl.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x:,.0f}"))
ax_pnl.tick_params(axis="both", labelsize=11, colors=DARK_TEXT)
ax_pnl.set_ylim(-3200, 400)
ax_pnl.grid(axis="y", alpha=0.25, zorder=1)

# ========================== RIGHT PANEL: Key Metrics ==========================
# Three sub-panels for Bets Placed, Win Rate, ROI
gs_right = gs[0, 1].subgridspec(1, 3, wspace=0.50)

metrics = [
    {"title": "Bets Placed", "values": bets_placed, "fmt": "{:.0f}", "suffix": "",
     "ylim": (0, 120)},
    {"title": "Win Rate", "values": win_rate, "fmt": "{:.0f}", "suffix": "%",
     "ylim": (0, 120)},
    {"title": "ROI", "values": roi, "fmt": "{:+.1f}", "suffix": "%",
     "ylim": (-120, 20)},
]

for idx, m in enumerate(metrics):
    ax = fig.add_subplot(gs_right[0, idx])
    vals = m["values"]
    colors = [GREEN, RED]

    bars = ax.bar(models, vals, width=0.55, color=colors,
                  edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels
    for bar, val, clr in zip(bars, vals, colors):
        y = bar.get_height()
        label = m["fmt"].format(val) + m["suffix"]
        va = "bottom" if val >= 0 else "top"
        offset = 8 if val >= 0 else -8
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", va=va,
                    fontsize=13, fontweight="bold", color=clr,
                    annotation_clip=False)

    ax.axhline(0, color=GREY, linewidth=0.8, zorder=2)
    ax.set_title(m["title"], fontsize=12, fontweight="bold",
                 color=DARK_TEXT, pad=10)
    ax.set_ylim(m["ylim"])
    ax.tick_params(axis="x", labelsize=10, colors=DARK_TEXT)
    ax.tick_params(axis="y", labelsize=9, colors=GREY)
    ax.grid(axis="y", alpha=0.2, zorder=1)

    # Suffix on y-axis
    if m["suffix"] == "%":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _, s=m["suffix"]: f"{x:.0f}{s}"))

# ---------------------------------------------------------------------------
# Suptitle & subtitle
# ---------------------------------------------------------------------------
fig.suptitle("Betting PnL Backtester: OpenTSLM vs Baseline",
             fontsize=16, fontweight="bold", color=DARK_TEXT, y=0.98)
fig.text(0.5, 0.92,
         "Selective strategy (24 bets) vs indiscriminate strategy (100 bets)",
         ha="center", fontsize=11, color=GREY, style="italic")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/wangni/notion-figures/polymarkets/fig_005.png"
fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {out_path}")
