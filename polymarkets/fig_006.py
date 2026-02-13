#!/usr/bin/env python3
"""
fig_006: Question Type Taxonomy and Ground Truth Methods
Tree/taxonomy diagram showing 5 question types organized into 3 categories.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ─── Colour palette ───────────────────────────────────────────────────
CAT_TREND = "#3B82F6"   # blue
CAT_STATE = "#10B981"   # green
CAT_EVOL  = "#8B5CF6"   # purple

FILL_TREND = "#DBEAFE"
FILL_STATE = "#D1FAE5"
FILL_EVOL  = "#EDE9FE"

CHIP_BG = {
    "increasing": "#93C5FD", "decreasing": "#FCA5A5", "stable": "#FDE68A",
    "volatile": "#F9A8D4", "yes": "#6EE7B7", "no": "#FCA5A5",
    "ambiguous": "#FDE68A",
}
CHIP_TEXT = "#1F2937"

BG         = "#FFFFFF"
TEXT_DARK  = "#111827"
TEXT_MID   = "#374151"
TEXT_LIGHT = "#6B7280"

# ─── Canvas ───────────────────────────────────────────────────────────
W, H = 22, 13
fig, ax = plt.subplots(figsize=(W, H), dpi=200)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ─── Helpers ──────────────────────────────────────────────────────────
def draw_box(x, y, w, h, fc, ec, lw=1.2, rad=0.15, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={rad}",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         zorder=zorder, clip_on=False)
    ax.add_patch(box)

def draw_chip(cx, cy, label, color, fs=8):
    tw = len(label) * 0.12 + 0.36
    chip = FancyBboxPatch((cx - tw/2, cy - 0.15), tw, 0.30,
                          boxstyle="round,pad=0.05", facecolor=color,
                          edgecolor="#9CA3AF", linewidth=0.6, zorder=5)
    ax.add_patch(chip)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
            color=CHIP_TEXT, fontweight="medium", zorder=6)

def conn(x1, y1, x2, y2, color="#9CA3AF", lw=1.5):
    """Straight line connection."""
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1,
            solid_capstyle="round")

def conn_elbow(x1, y1, x2, y2, color="#9CA3AF", lw=1.5):
    """Elbow connector: go down from (x1,y1), then across to x2, then down to (x2,y2)."""
    mid_y = (y1 + y2) / 2
    ax.plot([x1, x1, x2, x2], [y1, mid_y, mid_y, y2],
            color=color, lw=lw, zorder=1, solid_capstyle="round")

# ─── Title ────────────────────────────────────────────────────────────
ax.text(W/2, H - 0.55, "Question Type Taxonomy & Ground Truth Methods",
        ha="center", va="center", fontsize=18, fontweight="bold", color=TEXT_DARK)
ax.text(W/2, H - 1.05, "5 question types across 3 categories  ·  22,521 generated questions  ·  rule-based ground truth",
        ha="center", va="center", fontsize=11, color=TEXT_LIGHT)

# ─── Root node ────────────────────────────────────────────────────────
root_w, root_h = 3.8, 0.8
root_x = W/2 - root_w/2
root_y = H - 2.6
draw_box(root_x, root_y, root_w, root_h, "#F3F4F6", "#6B7280", lw=2, rad=0.18)
ax.text(root_x + root_w/2, root_y + root_h/2,
        "Question Generator", ha="center", va="center",
        fontsize=14, fontweight="bold", color=TEXT_DARK)

root_cx = root_x + root_w/2
root_bot = root_y

# ─── Category boxes ──────────────────────────────────────────────────
cat_w, cat_h = 4.2, 0.65
cat_y = H - 4.3

cats = [
    {"label": "Trend Analysis",  "color": CAT_TREND, "fill": "#EFF6FF",  "cx": 4.5},
    {"label": "Market State",    "color": CAT_STATE, "fill": "#ECFDF5",  "cx": 11.0},
    {"label": "Evolution",       "color": CAT_EVOL,  "fill": "#F5F3FF",  "cx": 17.5},
]

for c in cats:
    bx = c["cx"] - cat_w/2
    draw_box(bx, cat_y, cat_w, cat_h, c["fill"], c["color"], lw=2, rad=0.14)
    ax.text(c["cx"], cat_y + cat_h/2, c["label"],
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=c["color"])
    conn(root_cx, root_bot, c["cx"], cat_y + cat_h, color=c["color"], lw=1.8)

cat_bot = cat_y

# ─── Leaf definitions ────────────────────────────────────────────────
leaves = [
    {
        "type": "past_trend",
        "cat": "Trend Analysis", "color": CAT_TREND, "fill": FILL_TREND,
        "question": '"What was the trend in\nthe first period?"',
        "answers": ["increasing", "decreasing", "stable"],
        "method": "Linear regression slope\non first half of series",
        "detail": "slope > ε → increasing\nslope < −ε → decreasing\nelse → stable",
        "series_part": "First half",
    },
    {
        "type": "future_trend",
        "cat": "Trend Analysis", "color": CAT_TREND, "fill": FILL_TREND,
        "question": '"What is the trend in\nthe second period?"',
        "answers": ["increasing", "decreasing", "stable"],
        "method": "Linear regression slope\non second half (hidden)",
        "detail": "Same slope logic applied\nto unseen future half",
        "series_part": "Second half (forecasting)",
    },
    {
        "type": "volatility",
        "cat": "Market State", "color": CAT_STATE, "fill": FILL_STATE,
        "question": '"Was the prediction market\nstable or volatile?"',
        "answers": ["stable", "volatile"],
        "method": "Std dev of price changes",
        "detail": "std(Δp) > 0.001 → volatile\nelse → stable",
        "series_part": "First half",
    },
    {
        "type": "resolution",
        "cat": "Market State", "color": CAT_STATE, "fill": FILL_STATE,
        "question": '"Based on final probability,\ndid the event happen?"',
        "answers": ["yes", "no", "ambiguous"],
        "method": "Final probability threshold",
        "detail": "p > 0.95 → yes\np < 0.05 → no\nelse → ambiguous",
        "series_part": "Full series (final prob)",
    },
    {
        "type": "confidence_evolution",
        "cat": "Evolution", "color": CAT_EVOL, "fill": FILL_EVOL,
        "question": '"Did the market become\nmore confident?"',
        "answers": ["yes", "no"],
        "method": 'Mean |p − 0.5| comparison\nacross both halves',
        "detail": "conf = mean(|p − 0.5|)\nconf₂ > conf₁ × 1.05 → yes",
        "series_part": "Both halves compared",
    },
]

# ─── Leaf positions ──────────────────────────────────────────────────
# 5 leaves spread across width
leaf_w = 4.0
leaf_h = 4.6
gap = (W - 5 * leaf_w) / 6  # equal gaps

leaf_positions = []
for i in range(5):
    lx = gap + i * (leaf_w + gap)
    leaf_positions.append(lx)

leaf_y = 0.7  # bottom of leaf boxes

cat_cx_map = {c["label"]: c["cx"] for c in cats}

for i, leaf in enumerate(leaves):
    lx = leaf_positions[i]
    ly = leaf_y

    # Draw leaf box
    draw_box(lx, ly, leaf_w, leaf_h, leaf["fill"], leaf["color"], lw=1.6, rad=0.16)

    # Connection from category to leaf
    cat_cx = cat_cx_map[leaf["cat"]]
    leaf_cx = lx + leaf_w / 2
    conn_elbow(cat_cx, cat_bot, leaf_cx, ly + leaf_h, color=leaf["color"], lw=1.5)

    # ── Content inside the leaf box ──
    # Type name
    ty = ly + leaf_h - 0.38
    ax.text(leaf_cx, ty, leaf["type"],
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=leaf["color"], family="monospace")

    # Question
    qy = ty - 0.65
    ax.text(leaf_cx, qy, leaf["question"],
            ha="center", va="center", fontsize=9, color=TEXT_MID,
            style="italic", linespacing=1.3)

    # Answers label
    ay_label = qy - 0.58
    ax.text(lx + 0.3, ay_label, "Answers:", ha="left", va="center",
            fontsize=9, fontweight="bold", color=TEXT_DARK)

    # Answer chips — centered row
    answers = leaf["answers"]
    chip_spacing = 0.10
    total_cw = sum(len(a) * 0.12 + 0.36 for a in answers) + chip_spacing * (len(answers) - 1)
    chip_y = ay_label - 0.42
    cx_run = leaf_cx - total_cw / 2
    for ans in answers:
        cw = len(ans) * 0.12 + 0.36
        draw_chip(cx_run + cw / 2, chip_y, ans, CHIP_BG.get(ans, "#E5E7EB"), fs=8)
        cx_run += cw + chip_spacing

    # Ground truth label
    gt_label_y = chip_y - 0.5
    ax.text(lx + 0.3, gt_label_y, "Ground Truth:", ha="left", va="center",
            fontsize=9, fontweight="bold", color=TEXT_DARK)

    # Method text
    gt_y = gt_label_y - 0.48
    ax.text(leaf_cx, gt_y, leaf["method"],
            ha="center", va="center", fontsize=8.5, color=TEXT_MID, linespacing=1.3)

    # Detail / formula
    det_y = gt_y - 0.55
    ax.text(leaf_cx, det_y, leaf["detail"],
            ha="center", va="center", fontsize=7.5, color=TEXT_LIGHT,
            family="monospace", linespacing=1.4)

    # Series part badge
    badge_y = ly + 0.28
    ax.text(leaf_cx, badge_y, f"▸ {leaf['series_part']}",
            ha="center", va="center", fontsize=8, fontweight="medium",
            color=leaf["color"])

# ─── Legend ───────────────────────────────────────────────────────────
leg_y = 0.12
ax.text(0.6, leg_y, "Categories:", ha="left", va="center",
        fontsize=9.5, fontweight="bold", color=TEXT_DARK)

items = [
    ("Trend Analysis (2 types)", CAT_TREND),
    ("Market State (2 types)", CAT_STATE),
    ("Evolution (1 type)", CAT_EVOL),
]
lx_r = 3.0
for label, color in items:
    ax.plot(lx_r, leg_y, "s", color=color, markersize=9)
    ax.text(lx_r + 0.3, leg_y, label, ha="left", va="center",
            fontsize=9.5, color=TEXT_MID)
    lx_r += 4.5

ax.text(W - 0.5, leg_y, "Ground truth: rule-based computation on 720-point time series",
        ha="right", va="center", fontsize=8.5, color=TEXT_LIGHT, style="italic")

plt.tight_layout(pad=0.3)
plt.savefig("/home/wangni/notion-figures/polymarkets/fig_006.png",
            dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved: /home/wangni/notion-figures/polymarkets/fig_006.png")
