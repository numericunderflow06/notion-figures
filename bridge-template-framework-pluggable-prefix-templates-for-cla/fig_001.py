"""
fig_001: BTF Pipeline — Context -> Config -> Cascade -> Render
Architecture diagram showing classifier output entering as TemplateContext,
combined with TemplateConfig, passing through the cascade selector, and
emitting a rendered prefix consumed by a downstream LLM.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT_PATH = "/home/wangni/notion-figures/bridge-template-framework-pluggable-prefix-templates-for-cla/fig_001.png"

# Cool blue palette + one accent (warm) for the cascade
COLOR_BG = "#FFFFFF"
COLOR_CONTEXT = "#E3F0FB"
COLOR_CONTEXT_EDGE = "#3B7BB0"
COLOR_CONFIG = "#D7E7F4"
COLOR_CONFIG_EDGE = "#2F6595"
COLOR_CASCADE = "#FCE3CB"          # accent
COLOR_CASCADE_EDGE = "#D46B1F"     # accent edge
COLOR_RENDER = "#CFE0EF"
COLOR_RENDER_EDGE = "#1F4E79"
COLOR_TEXT = "#1A1A1A"
COLOR_SIDE = "#6B6B6B"
COLOR_ARROW = "#33597A"

fig, ax = plt.subplots(figsize=(15, 6.6), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(COLOR_BG)

# --- Title ---
ax.text(
    50, 47.5,
    "BTF Pipeline: Context $\\rightarrow$ Config $\\rightarrow$ Cascade $\\rightarrow$ Render",
    ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_TEXT,
)

# Layout: four boxes horizontally arranged
# Each box: width 16, height 14, centered vertically around y=22
BOX_Y = 15
BOX_H = 16
BOX_W = 16
GAP = 5  # arrow gap

xs = [4, 4 + BOX_W + GAP, 4 + 2 * (BOX_W + GAP), 4 + 3 * (BOX_W + GAP)]

boxes = [
    {
        "x": xs[0], "title": "TemplateContext",
        "items": [
            "top-1 label",
            "top-k distribution",
            "confidence",
            "entropy",
            "question / dataset meta",
        ],
        "fc": COLOR_CONTEXT, "ec": COLOR_CONTEXT_EDGE,
    },
    {
        "x": xs[1], "title": "TemplateConfig",
        "items": [
            "gate thresholds",
            "confidence cutoffs",
            "entropy bounds",
            "family toggles",
            "priority weights",
        ],
        "fc": COLOR_CONFIG, "ec": COLOR_CONFIG_EDGE,
    },
    {
        "x": xs[2], "title": "Cascade selector",
        "items": [
            "priority-sorted union",
            "REGISTRY ∪ FALLBACK",
            "pure gate predicates",
            "first firing template",
            "select_template(...)",
        ],
        "fc": COLOR_CASCADE, "ec": COLOR_CASCADE_EDGE,
    },
    {
        "x": xs[3], "title": "Render",
        "items": [
            "pure render fn",
            "natural-language prefix",
            "opens chain of thought",
            "assertive fallback",
            "1 of 54 + 1 options",
        ],
        "fc": COLOR_RENDER, "ec": COLOR_RENDER_EDGE,
    },
]

def draw_box(ax, x, y, w, h, fc, ec, title, items, title_color):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        linewidth=1.8, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    # Title bar text
    ax.text(
        x + w / 2, y + h - 2.0, title,
        ha="center", va="center",
        fontsize=12.5, fontweight="bold", color=title_color,
    )
    # Separator
    ax.plot(
        [x + 1.2, x + w - 1.2],
        [y + h - 3.6, y + h - 3.6],
        color=ec, linewidth=1.0, alpha=0.5,
    )
    # Items
    item_top = y + h - 5.0
    line_h = 1.85
    for i, txt in enumerate(items):
        ax.text(
            x + w / 2, item_top - i * line_h, txt,
            ha="center", va="center",
            fontsize=10.2, color=COLOR_TEXT,
        )

for b in boxes:
    draw_box(
        ax, b["x"], BOX_Y, BOX_W, BOX_H,
        b["fc"], b["ec"], b["title"], b["items"], b["ec"],
    )

# --- Arrows between boxes ---
def draw_arrow(ax, x1, x2, y):
    arr = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="-|>", mutation_scale=20,
        linewidth=2.2, color=COLOR_ARROW,
    )
    ax.add_patch(arr)

arrow_y = BOX_Y + BOX_H / 2
for i in range(3):
    x1 = xs[i] + BOX_W + 0.3
    x2 = xs[i + 1] - 0.3
    draw_arrow(ax, x1, x2, arrow_y)

# --- Left edge: classifier source ---
clf_x = 1.0
ax.text(
    clf_x, arrow_y + 6.8,
    "Classifier",
    ha="left", va="center", fontsize=11, fontweight="bold", color=COLOR_SIDE,
)
clf_lines = [
    "top-1 prediction",
    "top-k distribution",
    "confidence",
    "entropy",
]
for i, t in enumerate(clf_lines):
    ax.text(
        clf_x, arrow_y + 4.7 - i * 1.7, t,
        ha="left", va="center", fontsize=9.5, color=COLOR_SIDE, style="italic",
    )
# Bracket-ish arrow from classifier region into Context box
in_arrow = FancyArrowPatch(
    (clf_x + 3.0, arrow_y - 2.5), (xs[0] - 0.2, arrow_y),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=COLOR_SIDE,
    connectionstyle="arc3,rad=-0.15",
)
ax.add_patch(in_arrow)

# --- Right edge: downstream LLM ---
llm_x = xs[3] + BOX_W + 0.2
ax.text(
    llm_x + 4.0, arrow_y + 6.8,
    "Downstream LLM",
    ha="center", va="center", fontsize=11, fontweight="bold", color=COLOR_SIDE,
)
llm_lines = [
    "receives prefix",
    "as opening of",
    "chain of thought",
]
for i, t in enumerate(llm_lines):
    ax.text(
        llm_x + 4.0, arrow_y + 4.7 - i * 1.7, t,
        ha="center", va="center", fontsize=9.5, color=COLOR_SIDE, style="italic",
    )
out_arrow = FancyArrowPatch(
    (llm_x, arrow_y), (llm_x + 7.5, arrow_y - 2.5),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=COLOR_SIDE,
    connectionstyle="arc3,rad=-0.15",
)
ax.add_patch(out_arrow)

# Caption under right arrow
ax.text(
    llm_x + 4.0, arrow_y - 5.2,
    "rendered prefix",
    ha="center", va="center", fontsize=10, color=COLOR_RENDER_EDGE, fontweight="bold",
)

# Caption under left arrow
ax.text(
    (clf_x + 3.0 + xs[0]) / 2, arrow_y - 5.2,
    "TemplateContext fields",
    ha="center", va="center", fontsize=10, color=COLOR_CONTEXT_EDGE, fontweight="bold",
)

# --- Bottom footer note ---
ax.text(
    50, 4.0,
    "Cascade walks the priority-sorted union of REGISTRY $\\cup$ ASSERTIVE_FALLBACK and returns the first firing template.",
    ha="center", va="center", fontsize=10, color=COLOR_TEXT, style="italic",
)

# --- Legend for accent color ---
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_CONTEXT,
           markeredgecolor=COLOR_CONTEXT_EDGE, markersize=12, label="Inputs"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_CASCADE,
           markeredgecolor=COLOR_CASCADE_EDGE, markersize=12, label="Selector (accent)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_RENDER,
           markeredgecolor=COLOR_RENDER_EDGE, markersize=12, label="Output"),
]
ax.legend(
    handles=legend_handles, loc="lower right",
    bbox_to_anchor=(0.99, 0.02), frameon=False, fontsize=9,
    ncol=3, handletextpad=0.4, columnspacing=1.2,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
