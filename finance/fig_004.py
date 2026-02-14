"""
fig_004: Company Coverage — Geographic Distribution
Grouped horizontal bar chart showing 14 companies across 6 European countries.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data (from STAGE6_FINANCIAL_REPORTS.md sections 3.2 and 4.2)
# ---------------------------------------------------------------------------
companies = {
    "Belgium": [
        ("Aedifica SA", 1497),
        ("Atenor", 1021),
        ("Montea NV", 919),
        ("Nextensa SA", 809),
        ("Retail Estates SA", 945),
        ("VastNed Belgium NV", 531),
    ],
    "Poland": [
        ("BBI Development SA", 2041),
        ("Echo Investment SA", 1575),
        ("Triton Development SA", 1113),
    ],
    "Germany": [
        ("Deutsche EuroShop AG", 1308),
        ("TAG Immobilien AG", 1746),
    ],
    "Austria": [
        ("Warimpex Finanz AG", 612),
    ],
    "Finland": [
        ("Citycon Oyj", 2254),
    ],
    "Norway": [
        ("Selvaag Bolig ASA", 1280),
    ],
}

# Sort countries by number of companies (descending), then alphabetically
countries_sorted = sorted(companies.keys(), key=lambda c: (-len(companies[c]), c))

# ---------------------------------------------------------------------------
# Colour palette — one colour per country
# ---------------------------------------------------------------------------
country_colors = {
    "Belgium":  "#2D2D2D",   # dark charcoal
    "Poland":   "#D64045",   # muted red
    "Germany":  "#E8A838",   # warm amber
    "Austria":  "#467599",   # steel blue
    "Finland":  "#5B9279",   # sage green
    "Norway":   "#7B6D8D",   # muted purple
}

# ---------------------------------------------------------------------------
# Build flat lists for plotting
# ---------------------------------------------------------------------------
bar_labels = []       # "Company Name"
bar_filings = []      # filing counts
bar_colors = []       # colour per bar
bar_country = []      # country label (for grouping)

group_boundaries = [] # y-positions where country groups start
group_labels = []     # country names for left labels
group_centers = []    # y-center for each group label

y = 0
gap = 0.6  # gap between country groups

for ci, country in enumerate(countries_sorted):
    if ci > 0:
        y += gap
    start_y = y
    for name, filings in companies[country]:
        bar_labels.append(name)
        bar_filings.append(filings)
        bar_colors.append(country_colors[country])
        bar_country.append(country)
        y += 1
    end_y = y - 1
    group_centers.append((start_y + end_y) / 2)
    group_labels.append(country)
    group_boundaries.append((start_y, end_y))

y_positions = []
idx = 0
y = 0
for ci, country in enumerate(countries_sorted):
    if ci > 0:
        y += gap
    for _ in companies[country]:
        y_positions.append(y)
        y += 1

y_positions = np.array(y_positions)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_height = 0.72

bars = ax.barh(
    y_positions, bar_filings, height=bar_height, color=bar_colors,
    edgecolor="white", linewidth=0.5, zorder=3,
)

# Company name + filing count labels
for yp, val, label, color in zip(y_positions, bar_filings, bar_labels, bar_colors):
    if val >= 800:
        # Company name inside the bar
        ax.text(
            30, yp, label,
            va="center", ha="left", fontsize=9.5, fontweight="600",
            color="white", zorder=5,
        )
        # Filing count just outside the bar
        ax.text(
            val + 40, yp, f"{val:,}",
            va="center", ha="left", fontsize=9.5, fontweight="500",
            color="#333333",
        )
    else:
        # Short bar: company name + count outside
        ax.text(
            val + 40, yp, f"{val:,}  —  {label}",
            va="center", ha="left", fontsize=9.5, fontweight="500",
            color="#333333",
        )

# Country group labels on the left + summary stats
for ci, country in enumerate(countries_sorted):
    n = len(companies[country])
    total = sum(f for _, f in companies[country])
    center = group_centers[ci]
    ax.text(
        -120, center,
        f"{country}\n({n} {'company' if n == 1 else 'companies'}, {total:,} filings)",
        va="center", ha="right", fontsize=10.5, fontweight="bold",
        color=country_colors[country],
    )

# Faint horizontal separator lines between groups
for ci in range(1, len(countries_sorted)):
    prev_end = group_boundaries[ci - 1][1]
    curr_start = group_boundaries[ci][0]
    # separator in the gap
    sep_y_pos = (y_positions[sum(len(companies[countries_sorted[j]])
                for j in range(ci)) - 1] +
                y_positions[sum(len(companies[countries_sorted[j]])
                for j in range(ci))]) / 2
    ax.axhline(sep_y_pos, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=1)

# Axis formatting
ax.set_xlim(0, max(bar_filings) + 700)
ax.set_ylim(min(y_positions) - 0.6, max(y_positions) + 0.6)
ax.invert_yaxis()

ax.set_xlabel("Number of Filings", fontsize=11, fontweight="600", labelpad=10)
ax.set_yticks([])
ax.tick_params(axis="x", labelsize=10)

# Light vertical gridlines
ax.xaxis.grid(True, color="#EEEEEE", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Remove spines except bottom
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#BBBBBB")

# Title and subtitle
fig.suptitle(
    "Company Coverage: Geographic Distribution",
    fontsize=15, fontweight="bold", color="#222222",
    x=0.5, y=0.97,
)
ax.set_title(
    "14 European real estate companies across 6 countries — filing counts per company",
    fontsize=10.5, color="#666666", pad=14,
)

# Legend — small colour swatches
legend_patches = [
    mpatches.Patch(facecolor=country_colors[c], label=c)
    for c in countries_sorted
]
ax.legend(
    handles=legend_patches, loc="lower right",
    fontsize=9, frameon=True, framealpha=0.9,
    edgecolor="#DDDDDD", ncol=3,
    title="Country", title_fontsize=9,
)

plt.tight_layout(rect=[0.18, 0.0, 1.0, 0.94])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = "/home/wangni/notion-figures/finance/fig_004.png"
fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved figure to {out_path}")
