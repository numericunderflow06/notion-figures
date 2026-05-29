"""
fig_008: Channel Inventory — Space Heating vs DHW
Venn diagram showing the 13 TS_FEATURE_COLS split into CH (8) and DHW (8) subsets
with 3 shared overlap channels.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patheffects as path_effects

# Professional color palette
CH_COLOR = "#E07B39"      # warm orange for space heating
DHW_COLOR = "#3B82B5"     # cool blue for domestic hot water
SHARED_COLOR = "#6B5B95"  # purple for overlap
TEXT_COLOR = "#1F2937"
MUTED = "#6B7280"

# Channel lists
ch_only = [
    "outdoor_temp_min",
    "energy_consumption_ch",
    "CH_Energy_Out_Total_delta",
    "CH_E21_comp_energy_delta",
    "CH_EHeat_energy_delta",
]
shared = [
    "outdoor_temp_mean",
    "energy_consumption_total",
    "HP_Energy_Out_Total_delta",
]
dhw_only = [
    "energy_consumption_dhw",
    "DHW_Energy_Out_Total_delta",
    "DHW_E21_comp_energy_delta",
    "DHW_EHeat_energy_delta",
    "dhw_share",
]

fig, ax = plt.subplots(figsize=(14, 8.5), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(
    7.0, 8.05,
    "Channel Inventory: Space Heating vs DHW",
    ha="center", va="center",
    fontsize=17, fontweight="bold", color=TEXT_COLOR,
)
ax.text(
    7.0, 7.55,
    "13 TS_FEATURE_COLS partitioned into two 8-channel task subsets",
    ha="center", va="center",
    fontsize=11, color=MUTED, style="italic",
)

# --- Venn circles ---
ch_center = (4.5, 3.8)
dhw_center = (9.5, 3.8)
radius = 3.2

ch_circle = Circle(
    ch_center, radius,
    facecolor=CH_COLOR, edgecolor=CH_COLOR,
    alpha=0.22, linewidth=2.2,
)
dhw_circle = Circle(
    dhw_center, radius,
    facecolor=DHW_COLOR, edgecolor=DHW_COLOR,
    alpha=0.22, linewidth=2.2,
)
ax.add_patch(ch_circle)
ax.add_patch(dhw_circle)

# Outline circles on top so the borders stay crisp through the overlap tint
ax.add_patch(Circle(ch_center, radius, facecolor="none", edgecolor=CH_COLOR, linewidth=2.4))
ax.add_patch(Circle(dhw_center, radius, facecolor="none", edgecolor=DHW_COLOR, linewidth=2.4))

# --- Circle headers ---
ax.text(
    2.3, 6.55, "Space Heating (CH)",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color=CH_COLOR,
)
ax.text(
    2.3, 6.15, "8 channels",
    ha="center", va="center",
    fontsize=10, color=CH_COLOR, alpha=0.85,
)

ax.text(
    11.7, 6.55, "DHW (Hot Water)",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color=DHW_COLOR,
)
ax.text(
    11.7, 6.15, "8 channels",
    ha="center", va="center",
    fontsize=10, color=DHW_COLOR, alpha=0.85,
)

# Shared section header
ax.text(
    7.0, 6.05, "Shared (3)",
    ha="center", va="center",
    fontsize=11, fontweight="bold", color=SHARED_COLOR,
)

# --- Channel labels ---
mono = {"family": "monospace", "fontsize": 9.4}

# CH-only labels (left side)
ch_x = 2.55
ch_y_start = 5.35
ch_y_step = 0.55
for i, name in enumerate(ch_only):
    ax.text(
        ch_x, ch_y_start - i * ch_y_step, name,
        ha="center", va="center",
        color=TEXT_COLOR, **mono,
    )

# Shared labels (center overlap)
shared_x = 7.0
shared_y_start = 5.05
shared_y_step = 0.55
for i, name in enumerate(shared):
    ax.text(
        shared_x, shared_y_start - i * shared_y_step, name,
        ha="center", va="center",
        color=TEXT_COLOR, fontweight="bold", **mono,
    )

# DHW-only labels (right side)
dhw_x = 11.45
dhw_y_start = 5.35
dhw_y_step = 0.55
for i, name in enumerate(dhw_only):
    ax.text(
        dhw_x, dhw_y_start - i * dhw_y_step, name,
        ha="center", va="center",
        color=TEXT_COLOR, **mono,
    )

# --- Counts inside each region (bottom of each lobe) ---
ax.text(
    2.55, 1.55, "5 CH-only",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color=CH_COLOR, alpha=0.9,
)
ax.text(
    7.0, 1.55, "3 shared",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color=SHARED_COLOR, alpha=0.95,
)
ax.text(
    11.45, 1.55, "5 DHW-only",
    ha="center", va="center",
    fontsize=10, fontweight="bold", color=DHW_COLOR, alpha=0.9,
)

# --- Legend / summary box bottom-left ---
legend_x = 0.35
legend_y = 0.85
legend_w = 4.6
legend_h = 1.0
legend_box = FancyBboxPatch(
    (legend_x, legend_y - legend_h / 2),
    legend_w, legend_h,
    boxstyle="round,pad=0.06,rounding_size=0.12",
    facecolor="#F3F4F6", edgecolor="#D1D5DB", linewidth=1.0,
)
ax.add_patch(legend_box)
ax.text(
    legend_x + 0.18, legend_y + 0.22,
    "Union = 13 TS_FEATURE_COLS",
    ha="left", va="center",
    fontsize=10.5, fontweight="bold", color=TEXT_COLOR,
)
ax.text(
    legend_x + 0.18, legend_y - 0.18,
    "|CH| = 8,  |DHW| = 8,  |CH ∩ DHW| = 3",
    ha="left", va="center",
    fontsize=10, color=TEXT_COLOR, family="monospace",
)

# --- Footnote (bottom right) ---
ax.text(
    13.65, 0.85,
    "Note: outdoor_temp_min and dhw_share are\ntask-exclusive (CH and DHW respectively).",
    ha="right", va="center",
    fontsize=9, color=MUTED, style="italic",
)

# Save
out_path = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_008.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {out_path}")
