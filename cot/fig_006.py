"""
fig_006: Reward Function Design
Visual breakdown of the two-component reward function for GRPO in OpenTSLM.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects
import numpy as np

# ── colours ──────────────────────────────────────────────────────────────
BG       = "#FFFFFF"
GREEN    = "#2E9E5A"   # pass / correct
GREEN_LT = "#E6F5EC"
RED      = "#D94040"   # fail / wrong
RED_LT   = "#FDE8E8"
BLUE     = "#3A7BD5"   # primary accent
BLUE_LT  = "#E8F0FE"
PURPLE   = "#7C4DFF"   # format accent
PURPLE_LT= "#F0E8FF"
ORANGE   = "#E67E22"
GRAY     = "#6B7280"
GRAY_LT  = "#F3F4F6"
DARK     = "#1F2937"
BORDER   = "#D1D5DB"

fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# ── Helper functions ─────────────────────────────────────────────────────
def rounded_box(x, y, w, h, fc, ec, lw=1.2, zorder=2, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(box)
    return box

def arrow(x1, y1, x2, y2, color=GRAY, lw=1.5, style="-|>", zorder=3):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

def txt(x, y, s, fs=10, ha="center", va="center", color=DARK, weight="normal",
        family="sans-serif", zorder=5, fontstyle="normal"):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color,
            fontweight=weight, fontfamily=family, fontstyle=fontstyle, zorder=zorder)

def mono(x, y, s, fs=8.5, ha="center", va="center", color=DARK, weight="normal", zorder=5):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=color,
            fontweight=weight, fontfamily="monospace", zorder=zorder)

# ════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════
txt(7, 9.55, "Reward Function Design", fs=16, weight="bold", color=DARK)
txt(7, 9.15, r"$r_i = r_{\mathrm{correct}} + r_{\mathrm{format}}$",
    fs=12, color=GRAY)

# ════════════════════════════════════════════════════════════════════════
# TOP: Generated Completion box (input to both branches)
# ════════════════════════════════════════════════════════════════════════
rounded_box(3.8, 8.2, 6.4, 0.7, GRAY_LT, BORDER)
txt(7, 8.65, "Generated Completion  y", fs=11, weight="bold")
mono(7, 8.38, '"<reasoning>... Answer: walking"', fs=9, color=GRAY)

# Arrows from completion down to two branches
arrow(5.2, 8.2, 3.6, 7.55, color=BLUE, lw=1.8)
arrow(8.8, 8.2, 10.4, 7.55, color=PURPLE, lw=1.8)

# ════════════════════════════════════════════════════════════════════════
# LEFT BRANCH: Correctness Reward
# ════════════════════════════════════════════════════════════════════════
# Branch header
rounded_box(0.4, 7.05, 6.3, 0.5, BLUE_LT, BLUE, lw=1.5)
txt(3.55, 7.3, "Correctness Reward  r_correct", fs=11, weight="bold", color=BLUE)

# Step 1: Regex extraction
arrow(3.55, 7.05, 3.55, 6.7, color=BLUE, lw=1.4)
rounded_box(0.6, 5.85, 5.9, 0.85, BG, BORDER)
txt(3.55, 6.42, "Step 1: Answer Extraction", fs=10, weight="bold")
mono(3.55, 6.08, r"re.search(r'answer:\s*(.+)', pred, re.IGNORECASE)", fs=7.5, color=BLUE)

# Step 2: Hierarchical matching
arrow(3.55, 5.85, 3.55, 5.5, color=BLUE, lw=1.4)
rounded_box(0.6, 4.4, 5.9, 1.1, BG, BORDER)
txt(3.55, 5.22, "Step 2: Three-Tier Matching Hierarchy", fs=10, weight="bold")

# Three tiers - each with label, center_x, half_width
tier_y = 4.8
tier_items = [
    ("1. Exact match", 1.55, 0.82),
    ("2. Prefix match", 3.55, 0.85),
    ("3. First-3-char", 5.55, 0.75),
]
for label, tx, hw in tier_items:
    rounded_box(tx - hw, tier_y - 0.2, hw * 2, 0.38, BLUE_LT, BLUE, lw=0.8)
    txt(tx, tier_y, label, fs=8, color=BLUE, weight="bold")

# Arrow between tiers (fallback) - draw below the boxes
for i in range(len(tier_items) - 1):
    x_from = tier_items[i][1] + tier_items[i][2] + 0.03
    x_to = tier_items[i+1][1] - tier_items[i+1][2] - 0.03
    arr_y = tier_y - 0.33
    ax.annotate("", xy=(x_to, arr_y),
                xytext=(x_from, arr_y),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0),
                zorder=3)
    mid_x = (x_from + x_to) / 2
    txt(mid_x, arr_y + 0.15, "fallback", fs=7, color=GRAY, fontstyle="italic")

# Result: binary reward
arrow(3.55, 4.4, 3.55, 4.05, color=BLUE, lw=1.4)
# Green pass / red fail boxes
rounded_box(0.8, 3.2, 2.4, 0.8, GREEN_LT, GREEN, lw=1.3)
txt(2.0, 3.75, "Match", fs=10, weight="bold", color=GREEN)
txt(2.0, 3.45, r"$r_{\mathrm{correct}} = 1.0$", fs=11, weight="bold", color=GREEN)

rounded_box(3.9, 3.2, 2.4, 0.8, RED_LT, RED, lw=1.3)
txt(5.1, 3.75, "No Match", fs=10, weight="bold", color=RED)
txt(5.1, 3.45, r"$r_{\mathrm{correct}} = 0.0$", fs=11, weight="bold", color=RED)

# ════════════════════════════════════════════════════════════════════════
# RIGHT BRANCH: Format Reward
# ════════════════════════════════════════════════════════════════════════
rounded_box(7.3, 7.05, 6.3, 0.5, PURPLE_LT, PURPLE, lw=1.5)
txt(10.45, 7.3, "Format Reward  r_format", fs=11, weight="bold", color=PURPLE)

# Three format conditions
arrow(10.45, 7.05, 10.45, 6.7, color=PURPLE, lw=1.4)
rounded_box(7.5, 4.4, 5.9, 2.3, BG, BORDER)
txt(10.45, 6.42, "Three Format Conditions (all must pass)", fs=10, weight="bold")

# Condition boxes
cond_data = [
    ('1', '"Answer: " appears exactly once', 5.93),
    ('2', 'Answer at end of generation\n(within last 30 tokens)', 5.35),
    ('3', 'Extracted label is a valid\ntask label (via get_labels())', 4.72),
]
for num, label, cy in cond_data:
    # Checkbox-style indicator
    rounded_box(7.75, cy - 0.22, 0.4, 0.4, GREEN_LT, GREEN, lw=0.8)
    txt(7.95, cy, "\u2713", fs=11, weight="bold", color=GREEN)
    txt(8.35, cy, label, fs=9, ha="left", color=DARK)

# Result: format reward
arrow(10.45, 4.4, 10.45, 4.05, color=PURPLE, lw=1.4)

rounded_box(7.7, 3.2, 2.4, 0.8, GREEN_LT, GREEN, lw=1.3)
txt(8.9, 3.75, "All Pass", fs=10, weight="bold", color=GREEN)
txt(8.9, 3.45, r"$r_{\mathrm{format}} = 0.2$", fs=11, weight="bold", color=GREEN)

rounded_box(10.8, 3.2, 2.4, 0.8, RED_LT, RED, lw=1.3)
txt(12.0, 3.75, "Any Fail", fs=10, weight="bold", color=RED)
txt(12.0, 3.45, r"$r_{\mathrm{format}} = 0.0$", fs=11, weight="bold", color=RED)

# ════════════════════════════════════════════════════════════════════════
# BOTTOM: Total Reward
# ════════════════════════════════════════════════════════════════════════
# Arrows merging down
arrow(3.55, 3.2, 5.5, 2.35, color=BLUE, lw=2.0)
arrow(10.45, 3.2, 8.5, 2.35, color=PURPLE, lw=2.0)

# Total reward box
rounded_box(4.2, 1.3, 5.6, 1.05, GRAY_LT, DARK, lw=2.0)
txt(7, 2.05, "Total Reward", fs=12, weight="bold", color=DARK)
txt(7, 1.65, r"$r_i = r_{\mathrm{correct}} + r_{\mathrm{format}}$", fs=12, color=DARK)

# Range bar below total reward
bar_y = 0.65
bar_x0 = 3.0
bar_x1 = 11.0
bar_w = bar_x1 - bar_x0

# Background bar
rounded_box(bar_x0, bar_y - 0.15, bar_w, 0.3, GRAY_LT, BORDER, lw=0.8)

# Gradient fill: red -> orange -> green
n_seg = 200
for i in range(n_seg):
    frac = i / n_seg
    sx = bar_x0 + 0.15 + frac * (bar_w - 0.3)
    sw = (bar_w - 0.3) / n_seg
    # Interpolate red -> yellow -> green
    if frac < 0.5:
        r = 0.85
        g = 0.35 + frac * 1.0
        b = 0.25
    else:
        r = 0.85 - (frac - 0.5) * 1.2
        g = 0.62 + (frac - 0.5) * 0.4
        b = 0.25 + (frac - 0.5) * 0.3
    r, g, b = max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b))
    ax.add_patch(plt.Rectangle((sx, bar_y - 0.1), sw, 0.2,
                                facecolor=(r, g, b), edgecolor="none", zorder=3))

# Tick marks at key reward values
ticks = [
    (0.0, "0.0"),
    (0.2, "0.2"),
    (1.0, "1.0"),
    (1.2, "1.2"),
]
for val, label in ticks:
    tx = bar_x0 + 0.15 + (val / 1.2) * (bar_w - 0.3)
    ax.plot([tx, tx], [bar_y - 0.18, bar_y + 0.18], color=DARK, lw=1.2, zorder=4)
    txt(tx, bar_y - 0.35, label, fs=9, weight="bold", color=DARK)

# Annotations for min/max
txt(bar_x0 - 0.3, bar_y, "min", fs=8, color=RED, weight="bold")
txt(bar_x1 + 0.35, bar_y, "max", fs=8, color=GREEN, weight="bold")

# Sub-labels
txt(bar_x0 + 0.15, bar_y + 0.35, "format only", fs=7, color=GRAY, ha="left",
    fontstyle="italic")
txt(bar_x0 + 0.15 + (1.0 / 1.2) * (bar_w - 0.3), bar_y + 0.35,
    "correct only", fs=7, color=GRAY, fontstyle="italic")
txt(bar_x0 + 0.15 + (bar_w - 0.3), bar_y + 0.35,
    "both", fs=7, color=GREEN, ha="right", weight="bold")

# Range label
txt(7, 0.15, "Reward Range: 0.0 \u2013 1.2", fs=10, weight="bold", color=DARK)

plt.tight_layout(pad=0.5)
plt.savefig("/home/wangni/notion-figures/cot/fig_006.png", dpi=200,
            facecolor=BG, bbox_inches="tight")
plt.close()
print("Saved: /home/wangni/notion-figures/cot/fig_006.png")
