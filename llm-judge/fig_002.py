"""
fig_002: GRPO Training Loop — 7-step dataflow diagram.

Visualizes the cyclic GRPO training loop:
  1. SAMPLE  2. SCORE  3. NORMALIZE  4. COMPUTE LOG-PROBS
  5. COMPUTE LOSS  6. UPDATE  7. REFRESH

Key parameters annotated at relevant steps:
  K=4, T=0.7, epsilon=0.2, beta=0.04
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import numpy as np

# ---------------------------------------------------------------------------
# Layout configuration
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(18, 14), dpi=200, facecolor="white")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect("equal")
ax.axis("off")

# Title
fig.suptitle("GRPO Training Loop", fontsize=24, fontweight="bold",
             y=0.96, color="#1a1a2e")
ax.text(0, 1.28, "Group Relative Policy Optimization — 7-Step Training Cycle",
        ha="center", va="center", fontsize=14, color="#555555", style="italic")

# ---------------------------------------------------------------------------
# Step definitions (label, subtitle, annotation, color)
# ---------------------------------------------------------------------------
steps = [
    ("1. SAMPLE",            "Generate K responses\nper input prompt",
     "K = 4,  T = 0.7",     "#2196F3"),   # blue
    ("2. SCORE",             "LLM judge assigns\nscalar rewards",
     "r = 0.5·corr + 0.3·reas\n     + 0.2·cons",  "#4CAF50"),   # green
    ("3. NORMALIZE",         "Group-normalize rewards\n(mean-subtract, std-divide)",
     "â = (r − μ_g) / σ_g", "#FF9800"),   # orange
    ("4. LOG-PROBS",         "Per-token log-probs\nunder π_θ and π_ref",
     "Σ log π_θ(y_t | y_<t, x)", "#E91E63"),   # pink
    ("5. COMPUTE LOSS",      "Clipped surrogate\n+ KL penalty",
     "ε = 0.2,  β = 0.04",  "#9C27B0"),   # purple
    ("6. UPDATE",            "Gradient step on\nencoder + proj + LoRA",
     "θ ← θ − η ∇L",       "#00BCD4"),   # teal
    ("7. REFRESH",           "Sync reference\npolicy π_ref ← π_θ",
     "Periodic sync",       "#795548"),   # brown
]

N = len(steps)

# Place steps around an ellipse — larger radius to avoid overlap
cx, cy = 0.0, 0.0
rx, ry = 1.02, 0.98
# Start from top, go clockwise
angles = [np.pi / 2 - i * 2 * np.pi / N for i in range(N)]

box_w, box_h = 0.40, 0.28

positions = []
for angle in angles:
    x = cx + rx * np.cos(angle)
    y = cy + ry * np.sin(angle)
    positions.append((x, y))

# ---------------------------------------------------------------------------
# Helper: clipped box path for header masking
# ---------------------------------------------------------------------------
def rounded_rect_path(x0, y0, w, h, r):
    """Return a Path for a rounded rectangle."""
    verts = [
        (x0 + r, y0),
        (x0 + w - r, y0),
        (x0 + w, y0),
        (x0 + w, y0 + r),
        (x0 + w, y0 + h - r),
        (x0 + w, y0 + h),
        (x0 + w - r, y0 + h),
        (x0 + r, y0 + h),
        (x0, y0 + h),
        (x0, y0 + h - r),
        (x0, y0 + r),
        (x0, y0),
        (x0 + r, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
    ]
    return Path(verts, codes)

# ---------------------------------------------------------------------------
# Draw curved arrows between consecutive steps
# ---------------------------------------------------------------------------
def draw_arrow(ax, p1, p2):
    """Draw a curved arrow from p1 toward p2, stopping at box edge."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = np.hypot(dx, dy)
    # Shorten both ends
    shrink = 0.22
    frac = shrink / dist
    sx = x1 + dx * frac
    sy = y1 + dy * frac
    ex = x2 - dx * frac
    ey = y2 - dy * frac

    ax.annotate(
        "", xy=(ex, ey), xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.2",
            connectionstyle="arc3,rad=0.15",
            color="#555555", lw=2.2,
        ),
    )

for i in range(N):
    j = (i + 1) % N
    draw_arrow(ax, positions[i], positions[j])

# ---------------------------------------------------------------------------
# Draw step boxes
# ---------------------------------------------------------------------------
pad = 0.025
rounding = 0.035

for i, ((x, y), (label, subtitle, annotation, color)) in enumerate(
        zip(positions, steps)):

    bx = x - box_w / 2
    by = y - box_h / 2

    # Shadow
    shadow = mpatches.FancyBboxPatch(
        (bx + 0.01, by - 0.01), box_w, box_h,
        boxstyle=mpatches.BoxStyle.Round(pad=pad, rounding_size=rounding),
        facecolor="#bbbbbb", edgecolor="none", alpha=0.35, zorder=1,
    )
    ax.add_patch(shadow)

    # Main white box
    main_box = mpatches.FancyBboxPatch(
        (bx, by), box_w, box_h,
        boxstyle=mpatches.BoxStyle.Round(pad=pad, rounding_size=rounding),
        facecolor="white", edgecolor=color, linewidth=2.8, zorder=2,
    )
    ax.add_patch(main_box)

    # Colored header band — clipped to main box
    header_h = 0.075
    header = mpatches.FancyBboxPatch(
        (bx - pad, by + box_h - header_h + pad), box_w + 2 * pad, header_h,
        boxstyle=mpatches.BoxStyle.Round(pad=0.0, rounding_size=rounding),
        facecolor=color, edgecolor="none", alpha=0.95, zorder=3,
    )
    # Use clip path from the main box
    clip_path = rounded_rect_path(
        bx - pad, by - pad, box_w + 2 * pad, box_h + 2 * pad, rounding)
    clip_patch = mpatches.PathPatch(clip_path, transform=ax.transData)
    ax.add_patch(header)
    header.set_clip_path(clip_patch)

    # Step number + label in header
    ax.text(x, y + box_h / 2 - header_h / 2 + pad,
            label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=4)

    # Description
    ax.text(x, y - 0.015,
            subtitle, ha="center", va="center",
            fontsize=10, color="#333333", zorder=4, linespacing=1.3)

    # Parameter annotation
    ax.text(x, y - box_h / 2 + 0.04,
            annotation, ha="center", va="center",
            fontsize=8.5, color=color, fontweight="bold", zorder=4,
            fontstyle="italic", linespacing=1.2)

# ---------------------------------------------------------------------------
# Central circle with label
# ---------------------------------------------------------------------------
circle_bg = mpatches.Circle((cx, cy), 0.22, facecolor="#fafafa",
                             edgecolor="#aaaaaa", linewidth=2, zorder=5,
                             linestyle="--")
ax.add_patch(circle_bg)
ax.text(cx, cy + 0.05, "GRPO", ha="center", va="center",
        fontsize=20, fontweight="bold", color="#1a1a2e", zorder=6)
ax.text(cx, cy - 0.06, "Loop", ha="center", va="center",
        fontsize=14, color="#666666", zorder=6)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = "/home/wangni/notion-figures/llm-judge/fig_002.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_path}")
