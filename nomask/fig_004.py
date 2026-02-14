"""
fig_004: NGAFID Sensor Channel Groups
Radial grouped visualization of the 23 NGAFID sensor channels organized by type,
showing cross-channel relationships that motivate the NoMask design.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

# ============================================================
# Data from verified facts
# ============================================================
groups = [
    ("Electrical",    ["volt1", "volt2", "amp1", "amp2"]),
    ("Fuel",          ["FQtyL", "FQtyR", "E1_FFlow"]),
    ("Engine",        ["E1_OilP", "E1_OilT", "E1_RPM"]),
    ("Cylinder",      ["CHT1", "CHT2", "CHT3", "CHT4",
                       "EGT1", "EGT2", "EGT3", "EGT4"]),
    ("Flight Params", ["IAS", "OAT", "AltB", "LatAc", "NormAc"]),
]

colors = {
    "Electrical":    "#3A86FF",
    "Fuel":          "#FF9F1C",
    "Engine":        "#E63946",
    "Cylinder":      "#2EC4B6",
    "Flight Params": "#7B2D8E",
}

light_colors = {
    "Electrical":    "#D6E6FF",
    "Fuel":          "#FFF0D4",
    "Engine":        "#FDDDE0",
    "Cylinder":      "#D4F5F1",
    "Flight Params": "#ECDAF3",
}

# Cross-channel correlations (from spec: faults manifest as correlated
# patterns across voltage, current, temperature, RPM, and other sensors)
correlations = [
    ("Engine",      "Cylinder"),
    ("Electrical",  "Engine"),
    ("Fuel",        "Engine"),
    ("Engine",      "Flight Params"),
    ("Cylinder",    "Flight Params"),
]

# ============================================================
# Helper functions
# ============================================================

def polar_xy(r, angle_deg):
    ar = np.radians(angle_deg)
    return r * np.cos(ar), r * np.sin(ar)


def draw_pill(ax, label, cx, cy, color, light_color, pw=0.28, ph=0.15):
    pill = mpatches.FancyBboxPatch(
        (cx - pw / 2, cy - ph / 2), pw, ph,
        boxstyle="round,pad=0.03",
        facecolor=light_color, edgecolor=color, linewidth=1.5, zorder=6,
    )
    ax.add_patch(pill)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=9.5, fontweight="semibold", color=color, zorder=7)


def draw_line(ax, x1, y1, x2, y2, color, lw=1.1, alpha=0.40, zorder=2):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
            alpha=alpha, zorder=zorder)


# ============================================================
# Layout — manually tuned angles to avoid overlap
# ============================================================
# Place Cylinder (8 channels) at bottom with plenty of room.
# Electrical at top. Other groups fill remaining space.
#
#   Electrical (top, 90°)
#   Flight Params (upper-left, 155°)
#   Cylinder (lower-left, 225°)   ← largest group, bottom area
#   Engine (lower-right, 315°)
#   Fuel (upper-right, 30°)

layout = {
    "Electrical":    90,
    "Fuel":          25,
    "Engine":        -40,
    "Cylinder":      -115,
    "Flight Params": 160,
}

group_label_r = 0.62
channel_r1 = 1.15       # single-ring channels
channel_r_inner = 1.08  # inner ring for Cylinder
channel_r_outer = 1.42  # outer ring for Cylinder

# ============================================================
# Draw
# ============================================================
fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

group_positions = {}

for gname, channels in groups:
    angle = layout[gname]
    color = colors[gname]
    light = light_colors[gname]
    n_ch = len(channels)

    gx, gy = polar_xy(group_label_r, angle)
    group_positions[gname] = (gx, gy)

    # --- Group label bubble ---
    bw, bh = 0.48, 0.22
    bubble = mpatches.FancyBboxPatch(
        (gx - bw / 2, gy - bh / 2), bw, bh,
        boxstyle="round,pad=0.045",
        facecolor=color, edgecolor="white", linewidth=2.5, zorder=10,
    )
    ax.add_patch(bubble)
    ax.text(gx, gy, gname, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white", zorder=11)

    # --- Channel pills ---
    if n_ch > 5:
        # Two concentric rings for Cylinder (8 sensors)
        half = n_ch // 2
        inner_chs = channels[:half]   # CHT1-4
        outer_chs = channels[half:]   # EGT1-4
        fan_span = 62

        inner_fan = np.linspace(angle - fan_span / 2,
                                angle + fan_span / 2, len(inner_chs))
        for ch, fa in zip(inner_chs, inner_fan):
            cx, cy = polar_xy(channel_r_inner, fa)
            draw_pill(ax, ch, cx, cy, color, light)
            # connector
            sx, sy = polar_xy(group_label_r + 0.16, fa)
            ex, ey = polar_xy(channel_r_inner - 0.16, fa)
            draw_line(ax, sx, sy, ex, ey, color)

        outer_fan = np.linspace(angle - fan_span / 2,
                                angle + fan_span / 2, len(outer_chs))
        for ch, fa in zip(outer_chs, outer_fan):
            cx, cy = polar_xy(channel_r_outer, fa)
            draw_pill(ax, ch, cx, cy, color, light)
            sx, sy = polar_xy(group_label_r + 0.16, fa)
            ex, ey = polar_xy(channel_r_outer - 0.16, fa)
            draw_line(ax, sx, sy, ex, ey, color)
    else:
        fan_span = max(28, 13 * n_ch)
        fan_angles = np.linspace(angle - fan_span / 2,
                                 angle + fan_span / 2, n_ch)
        for ch, fa in zip(channels, fan_angles):
            cx, cy = polar_xy(channel_r1, fa)
            draw_pill(ax, ch, cx, cy, color, light)
            sx, sy = polar_xy(group_label_r + 0.16, fa)
            ex, ey = polar_xy(channel_r1 - 0.16, fa)
            draw_line(ax, sx, sy, ex, ey, color)

# ============================================================
# Cross-channel correlation arcs (dashed)
# ============================================================
for g1, g2 in correlations:
    x1, y1 = group_positions[g1]
    x2, y2 = group_positions[g2]
    dx, dy = x2 - x1, y2 - y1
    dist = np.hypot(dx, dy)
    shrink = 0.27 / dist
    sx1, sy1 = x1 + dx * shrink, y1 + dy * shrink
    sx2, sy2 = x2 - dx * shrink, y2 - dy * shrink

    arc = FancyArrowPatch(
        (sx1, sy1), (sx2, sy2),
        connectionstyle="arc3,rad=0.15",
        arrowstyle="-",
        linewidth=1.6, linestyle=(0, (5, 4)),
        color="#AAAAAA", alpha=0.55, zorder=1,
    )
    ax.add_patch(arc)

# ============================================================
# Central aircraft schematic
# ============================================================
ac_color = "#444"
# Fuselage
ax.plot([0, 0], [-0.17, 0.21], color=ac_color, linewidth=4.5,
        solid_capstyle="round", zorder=8)
# Nose
ax.plot([0], [0.25], marker="^", color=ac_color, markersize=11, zorder=8)
# Wings
ax.plot([-0.22, 0.22], [0.06, 0.06], color=ac_color, linewidth=3.5,
        solid_capstyle="round", zorder=8)
# Tail
ax.plot([-0.11, 0.11], [-0.16, -0.16], color=ac_color, linewidth=2.8,
        solid_capstyle="round", zorder=8)

ax.text(0, -0.32, "Aircraft\nSubsystems", ha="center", va="center",
        fontsize=10, color="#555", fontstyle="italic", zorder=9,
        linespacing=1.1)

# ============================================================
# Legend
# ============================================================
ax.plot([], [], linestyle=(0, (5, 4)), color="#AAAAAA", linewidth=1.6,
        alpha=0.55, label="Cross-channel correlation")
ax.legend(loc="lower center", frameon=True, fontsize=10,
          fancybox=True, framealpha=0.92, edgecolor="#ccc",
          bbox_to_anchor=(0.5, 0.015))

# ============================================================
# Title + subtitle
# ============================================================
fig.suptitle("NGAFID Sensor Channel Groups",
             fontsize=18, fontweight="bold", y=0.955, color="#222")
ax.text(0, 1.82,
        "23 channels across 5 subsystem groups\n"
        "Cross-channel correlations motivate the NoMask attention design",
        ha="center", va="center", fontsize=11, color="#666",
        linespacing=1.4)

# ============================================================
# Save
# ============================================================
plt.tight_layout(rect=[0, 0.02, 1, 0.93])
plt.savefig("/home/wangni/notion-figures/nomask/fig_004.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: /home/wangni/notion-figures/nomask/fig_004.png")
