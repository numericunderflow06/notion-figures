"""
Figure 004: 9 Genomic LRB Tasks Overview
Visual taxonomy of the 9 supported genomic tasks organized into 4 categories.
Column-based layout: each category is a column with tasks stacked vertically.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Data (verified from GenomicsLRBDataset.py and GENOMICS_OPENTSLM_README.md) ──

categories = [
    {
        "name": "Variant Effect",
        "count": 3,
        "color": "#2166AC",
        "light": "#D1E5F0",
        "tasks": [
            ("Causal eQTL", "Binary Classification"),
            ("Pathogenic (ClinVar)", "Binary Classification"),
            ("Pathogenic (OMIM)", "Binary Classification"),
        ],
    },
    {
        "name": "Gene Expression",
        "count": 2,
        "color": "#B2182B",
        "light": "#FDDBC7",
        "tasks": [
            ("CAGE Prediction", "Regression"),
            ("Bulk RNA Expression", "Regression"),
        ],
    },
    {
        "name": "Chromatin Features",
        "count": 2,
        "color": "#D4760A",
        "light": "#FEE8C8",
        "tasks": [
            ("Histone Marks", "Multi-class Classification"),
            ("DNA Accessibility", "Binary Classification"),
        ],
    },
    {
        "name": "Regulatory Elements",
        "count": 2,
        "color": "#1B7837",
        "light": "#D9F0D3",
        "tasks": [
            ("Promoter", "Binary Classification"),
            ("Enhancer", "Binary Classification"),
        ],
    },
]

# Classification type badge colors
TYPE_COLORS = {
    "Binary Classification": "#5C6B73",
    "Regression": "#7B4EA3",
    "Multi-class Classification": "#C0392B",
}

# ── Figure ──
fig, ax = plt.subplots(figsize=(15, 9.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Layout constants ──
# 4 columns centered in the figure
col_w = 3.0          # column width
col_gap = 0.5        # gap between columns
total_cols_w = 4 * col_w + 3 * col_gap  # 13.5
col_x_start = (15 - total_cols_w) / 2 + col_w / 2  # center x of first column

# Root node
root_y = 8.7
root_w = 5.2
root_h = 0.65

# Category header
cat_y = 7.3
cat_h = 0.7

# Tasks start y and spacing
task_start_y = 6.0
task_h = 0.92
task_gap = 0.18
badge_h = 0.30

# Column x-centers
col_xs = [col_x_start + i * (col_w + col_gap) for i in range(4)]

# ── Helpers ──

def draw_box(cx, cy, w, h, fc, ec, text, fontsize=11, fw="bold", tc="white",
             lw=2.0, zorder=3, pad=0.12):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fw, color=tc,
            zorder=zorder + 1, linespacing=1.2)


def draw_badge(cx, cy, w, h, fc, text, fontsize=8.5):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=fc, edgecolor="none", alpha=0.92, zorder=5,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="medium", color="white", zorder=6)


# ── Root node ──
root_cx = 7.5
draw_box(root_cx, root_y, root_w, root_h,
         fc="#1B1B2F", ec="#1B1B2F",
         text="9 Genomic LRB Tasks", fontsize=15, fw="bold", tc="white",
         pad=0.15)

# ── Horizontal spine from root ──
spine_y = root_y - root_h / 2 - 0.25
ax.plot([root_cx, root_cx], [root_y - root_h / 2, spine_y],
        color="#AAAAAA", lw=2, zorder=1, solid_capstyle="round")
ax.plot([col_xs[0], col_xs[-1]], [spine_y, spine_y],
        color="#AAAAAA", lw=2, zorder=1, solid_capstyle="round")

# ── Draw each category column ──
for i, cat in enumerate(categories):
    cx = col_xs[i]
    color = cat["color"]
    light = cat["light"]

    # Vertical drop from spine to category header
    ax.plot([cx, cx], [spine_y, cat_y + cat_h / 2],
            color=color, lw=2.2, zorder=2, solid_capstyle="round")

    # Category header
    draw_box(cx, cat_y, col_w, cat_h,
             fc=color, ec=color,
             text=f"{cat['name']}\n({cat['count']} tasks)",
             fontsize=12, fw="bold", tc="white", lw=0)

    # Vertical line from category to first task
    first_task_top = task_start_y + task_h / 2
    ax.plot([cx, cx], [cat_y - cat_h / 2, first_task_top],
            color=color, lw=1.5, zorder=1, solid_capstyle="round",
            linestyle=(0, (4, 3)), alpha=0.5)

    # Task cards
    for j, (task_name, task_type) in enumerate(cat["tasks"]):
        ty = task_start_y - j * (task_h + task_gap)

        # Task card
        card = FancyBboxPatch(
            (cx - col_w / 2, ty - task_h / 2), col_w, task_h,
            boxstyle="round,pad=0.1",
            facecolor=light, edgecolor=color, linewidth=1.8, zorder=3,
        )
        ax.add_patch(card)

        # Task name
        ax.text(cx, ty + 0.15, task_name,
                ha="center", va="center",
                fontsize=11, fontweight="semibold", color="#1a1a1a", zorder=4)

        # Type badge
        badge_color = TYPE_COLORS[task_type]
        badge_w = col_w - 0.5
        draw_badge(cx, ty - 0.25, badge_w, badge_h, badge_color, task_type)

        # Connector dot at top of card
        ax.plot(cx, ty + task_h / 2, 'o', color=color, markersize=4, zorder=5)

# ── Legend (bottom center) ──
legend_y = 1.2
legend_items = [
    ("Binary Classification", TYPE_COLORS["Binary Classification"], "6 tasks"),
    ("Regression", TYPE_COLORS["Regression"], "2 tasks"),
    ("Multi-class Classification", TYPE_COLORS["Multi-class Classification"], "1 task"),
]

# Legend title
ax.text(2.0, legend_y, "Classification Type:", ha="left", va="center",
        fontsize=11, fontweight="bold", color="#444444")

lx = 5.2
for label, lcolor, count in legend_items:
    # Color swatch
    swatch = FancyBboxPatch(
        (lx, legend_y - 0.14), 0.45, 0.28,
        boxstyle="round,pad=0.04",
        facecolor=lcolor, edgecolor="none", alpha=0.92, zorder=3,
    )
    ax.add_patch(swatch)
    ax.text(lx + 0.65, legend_y, f"{label}  ({count})",
            ha="left", va="center",
            fontsize=10, color="#444444", zorder=4)
    lx += 3.6

# ── Title ──
ax.text(7.5, 9.5, "9 Genomic Long-Range Benchmark (LRB) Tasks",
        ha="center", va="bottom",
        fontsize=17, fontweight="bold", color="#1B1B2F",
        transform=ax.transData)

plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)
plt.savefig("/home/wangni/notion-figures/genomics/fig_004.png",
            dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.3)
plt.close()

print("Figure saved to /home/wangni/notion-figures/genomics/fig_004.png")
