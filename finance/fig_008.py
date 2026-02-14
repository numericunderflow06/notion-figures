"""
fig_008: Time-Based Dataset Split Visualization
Shows chronological train/val/test split with filing density histogram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import matplotlib.patheffects as pe

# --- Data from verified facts ---
# Train: 2001-01-09 to 2023-03-21, 14,104 samples (80%)
# Val:   2023-03-21 to 2024-07-08,  1,763 samples (10%)
# Test:  2024-07-09 to 2026-01-30,  1,764 samples (10%)

train_start = datetime(2001, 1, 9)
train_end = datetime(2023, 3, 21)
val_start = datetime(2023, 3, 21)
val_end = datetime(2024, 7, 8)
test_start = datetime(2024, 7, 9)
test_end = datetime(2026, 1, 30)

train_samples = 14_104
val_samples = 1_763
test_samples = 1_764
total_samples = train_samples + val_samples + test_samples

# Colors
c_train = "#3B82F6"       # Blue
c_val = "#F59E0B"          # Amber/Yellow
c_test = "#EF4444"         # Red
c_train_light = "#DBEAFE"
c_val_light = "#FEF3C7"
c_test_light = "#FEE2E2"
c_train_dark = "#1D4ED8"
c_val_dark = "#B45309"
c_test_dark = "#B91C1C"
c_bg = "#FFFFFF"
c_text = "#1F2937"
c_text_light = "#6B7280"
c_border = "#D1D5DB"
c_green = "#059669"
c_green_light = "#ECFDF5"

# --- Simulate filing density over time ---
np.random.seed(42)
all_start = datetime(2001, 1, 1)
all_end = datetime(2026, 2, 1)
total_days = (all_end - all_start).days

n_filings = total_samples
t = np.random.beta(2.5, 1.8, size=n_filings)
filing_days = np.sort(t * total_days)
filing_dates = [all_start + timedelta(days=int(d)) for d in filing_days]

# --- Date axis limits ---
xlim_left = datetime(2000, 1, 1)
xlim_right = datetime(2027, 3, 1)

# --- Create figure with 3 rows ---
fig = plt.figure(figsize=(14, 6.8))
fig.patch.set_facecolor(c_bg)

# GridSpec: band (top), gap for annotation, histogram (bottom)
gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 0.6, 1.6], hspace=0.02)
ax_band = fig.add_subplot(gs[0])
ax_gap = fig.add_subplot(gs[1])
ax_hist = fig.add_subplot(gs[2])

# ============================================================
# Top panel: Colored bands for train / val / test
# ============================================================
ax_band.set_xlim(xlim_left, xlim_right)
ax_band.set_ylim(0, 1)
ax_band.axis("off")

band_y0, band_h = 0.08, 0.70

# Draw bands
for start, end, color, light in [
    (train_start, train_end, c_train, c_train_light),
    (val_start, val_end, c_val, c_val_light),
    (test_start, test_end, c_test, c_test_light),
]:
    s_num = mdates.date2num(start)
    e_num = mdates.date2num(end)
    width = e_num - s_num
    rect = mpatches.FancyBboxPatch(
        (s_num, band_y0), width, band_h,
        boxstyle="round,pad=0.003",
        facecolor=light, edgecolor=color, linewidth=2.2,
        transform=ax_band.transData
    )
    ax_band.add_patch(rect)

ax_band.xaxis_date()

# Helper
def mid_date(d1, d2):
    return d1 + (d2 - d1) / 2

band_mid_y = band_y0 + band_h / 2

# Train label (large band, has room for 3 lines)
mid_t = mid_date(train_start, train_end)
ax_band.text(mid_t, band_mid_y + 0.12, "Train", fontsize=18, fontweight="bold",
             color=c_train_dark, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax_band.text(mid_t, band_mid_y - 0.03, f"{train_samples:,} samples  (80%)", fontsize=12,
             color=c_train, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax_band.text(mid_t, band_mid_y - 0.17, "2001-01-09  \u2192  2023-03-21", fontsize=9.5,
             color=c_text_light, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Val label (narrow band, compact)
mid_v = mid_date(val_start, val_end)
ax_band.text(mid_v, band_mid_y + 0.08, "Val", fontsize=13, fontweight="bold",
             color=c_val_dark, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax_band.text(mid_v, band_mid_y - 0.10, f"{val_samples:,}\n(10%)", fontsize=9,
             color=c_val_dark, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Test label (narrow band, compact)
mid_te = mid_date(test_start, test_end)
ax_band.text(mid_te, band_mid_y + 0.08, "Test", fontsize=13, fontweight="bold",
             color=c_test_dark, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
ax_band.text(mid_te, band_mid_y - 0.10, f"{test_samples:,}\n(10%)", fontsize=9,
             color=c_test_dark, ha="center", va="center",
             path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# Boundary date labels above the bands
# Stagger the middle two labels vertically to avoid overlap
boundary_info = [
    (train_start, "2001-01-09", "left",   0.90),
    (train_end,   "2023-03-21", "right",  0.90),
    (test_start,  "2024-07-09", "left",   0.82),
    (test_end,    "2026-01-30", "left",   0.90),
]
for bdate, label, ha, y_pos in boundary_info:
    # Vertical dashed line
    ax_band.axvline(bdate, ymin=0.0, ymax=1.0, color=c_border,
                    linestyle="--", linewidth=1.0, alpha=0.8)
    # Date label above band
    ax_band.text(bdate, y_pos, label, fontsize=8.5,
                 color=c_text_light, ha=ha, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=c_border, alpha=0.95, zorder=5))

# ============================================================
# Middle gap: "No Data Leakage" annotation
# ============================================================
ax_gap.set_xlim(xlim_left, xlim_right)
ax_gap.set_ylim(0, 1)
ax_gap.axis("off")
ax_gap.xaxis_date()

# Place the badge centered in the gap
badge_x = mid_date(datetime(2008, 1, 1), datetime(2020, 1, 1))
ax_gap.text(badge_x, 0.5,
            "  No Data Leakage  \u2014  strict chronological split  ",
            fontsize=11, fontweight="bold", color=c_green, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=c_green_light,
                      edgecolor=c_green, linewidth=1.5, alpha=0.95))

# Arrows from badge to split boundaries
for target_date, rad in [(train_end, -0.3), (test_start, -0.4)]:
    ax_gap.annotate(
        "", xy=(target_date, 0.85), xytext=(badge_x + timedelta(days=600), 0.55),
        arrowprops=dict(arrowstyle="-|>", color=c_green, lw=1.5,
                        connectionstyle=f"arc3,rad={rad}"),
    )

# ============================================================
# Bottom panel: Filing density histogram
# ============================================================
ax_hist.set_xlim(xlim_left, xlim_right)
ax_hist.xaxis_date()
ax_hist.xaxis.set_major_locator(mdates.YearLocator(2))
ax_hist.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Create histogram bins (yearly)
bin_edges = [datetime(y, 1, 1) for y in range(2001, 2027)]
bin_edges_num = [mdates.date2num(d) for d in bin_edges]
filing_dates_num = [mdates.date2num(d) for d in filing_dates]
counts, _ = np.histogram(filing_dates_num, bins=bin_edges_num)

# Color each bar by which split it belongs to
bar_colors = []
for i, year in enumerate(range(2001, 2026)):
    yr_mid = datetime(year, 7, 1)
    if yr_mid < val_start:
        bar_colors.append(c_train)
    elif yr_mid < test_start:
        bar_colors.append(c_val)
    else:
        bar_colors.append(c_test)

for i in range(len(counts)):
    left = bin_edges_num[i]
    width = bin_edges_num[i + 1] - bin_edges_num[i]
    ax_hist.bar(left, counts[i], width=width * 0.88, align="edge",
                color=bar_colors[i], alpha=0.6, edgecolor="white",
                linewidth=0.5)

# Y-axis label
ax_hist.set_ylabel("Filings\nper Year", fontsize=10, color=c_text, rotation=0,
                    labelpad=45, va="center")
ax_hist.set_xlabel("Filing Date", fontsize=10, color=c_text, labelpad=6)
ax_hist.tick_params(axis="both", labelsize=9, colors=c_text_light)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)
ax_hist.spines["left"].set_color(c_border)
ax_hist.spines["bottom"].set_color(c_border)
ax_hist.set_facecolor(c_bg)

# Boundary lines on histogram
for bdate in [train_end, test_start]:
    ax_hist.axvline(bdate, color=c_text_light, linestyle="--",
                    linewidth=1.0, alpha=0.6)

# ============================================================
# Title & Subtitle
# ============================================================
fig.suptitle("Time-Based Dataset Split \u2014 Chronological Train / Val / Test",
             fontsize=16, fontweight="bold", color=c_text, y=0.97)
fig.text(0.5, 0.93,
         f"Total: {total_samples:,} European real estate filings from 14 companies (6 countries)",
         fontsize=10.5, color=c_text_light, ha="center")

# ============================================================
# Save
# ============================================================
plt.savefig("/home/wangni/notion-figures/finance/fig_008.png",
            dpi=200, bbox_inches="tight", facecolor=c_bg, pad_inches=0.3)
plt.close()
print("Saved: /home/wangni/notion-figures/finance/fig_008.png")
