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
COLOR_SIDE_FC = "#F2F2F2"
COLOR_SIDE_EC = "#7A7A7A"
COLOR_TEXT = "#1A1A1A"
COLOR_SIDE = "#5A5A5A"
COLOR_ARROW = "#33597A"
COLOR_SIDE_ARROW = "#7A7A7A"

fig, ax = plt.subplots(figsize=(17, 7.6), dpi=200)
ax.set_xlim(0, 112)
ax.set_ylim(0, 52)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(COLOR_BG)

# --- Title ---
ax.text(
    56, 49.5,
    "BTF Pipeline: Context $\\rightarrow$ Config $\\rightarrow$ Cascade $\\rightarrow$ Render",
    ha="center", va="center", fontsize=15, fontweight="bold", color=COLOR_TEXT,
)

# Layout: four main pipeline boxes plus flanking side nodes
BOX_Y = 14
BOX_H = 16
BOX_W = 15
GAP = 4              # arrow gap between main boxes
SIDE_W = 12
SIDE_H = 14
SIDE_Y = BOX_Y + 1   # vertically centered relative to main boxes
GAP_SIDE = 6         # gap between side node and adjacent main box

CLF_X = 1
PIPE_START = CLF_X + SIDE_W + GAP_SIDE  # 18

xs = [
    PIPE_START,
    PIPE_START + BOX_W + GAP,
    PIPE_START + 2 * (BOX_W + GAP),
    PIPE_START + 3 * (BOX_W + GAP),
]
# = [18, 37, 56, 75]; box right edges at [33, 52, 71, 90]
LLM_X = xs[3] + BOX_W + GAP_SIDE  # 95; LLM box ends at 107

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


def draw_box(ax, x, y, w, h, fc, ec, title, items, title_color,
             title_fs=12.5, item_fs=10.2, item_top_offset=5.0, line_h=1.85):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.25,rounding_size=0.9",
        linewidth=1.8, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h - 2.0, title,
        ha="center", va="center",
        fontsize=title_fs, fontweight="bold", color=title_color,
    )
    ax.plot(
        [x + 1.2, x + w - 1.2],
        [y + h - 3.6, y + h - 3.6],
        color=ec, linewidth=1.0, alpha=0.5,
    )
    item_top = y + h - item_top_offset
    for i, txt in enumerate(items):
        ax.text(
            x + w / 2, item_top - i * line_h, txt,
            ha="center", va="center",
            fontsize=item_fs, color=COLOR_TEXT,
        )


for b in boxes:
    draw_box(
        ax, b["x"], BOX_Y, BOX_W, BOX_H,
        b["fc"], b["ec"], b["title"], b["items"], b["ec"],
    )

# --- Arrows between main boxes ---
def draw_arrow(ax, x1, x2, y, color=COLOR_ARROW, lw=2.2, mut=20):
    arr = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle="-|>", mutation_scale=mut,
        linewidth=lw, color=color,
    )
    ax.add_patch(arr)


arrow_y = BOX_Y + BOX_H / 2
for i in range(3):
    x1 = xs[i] + BOX_W + 0.3
    x2 = xs[i + 1] - 0.3
    draw_arrow(ax, x1, x2, arrow_y)

# --- Left side node: Classifier ---
clf_items = [
    "top-1 prediction",
    "top-k distribution",
    "confidence",
    "entropy",
    "question / dataset meta",
]
draw_box(
    ax, CLF_X, SIDE_Y, SIDE_W, SIDE_H,
    COLOR_SIDE_FC, COLOR_SIDE_EC,
    "Classifier", clf_items, COLOR_SIDE,
    title_fs=11.5, item_fs=8.8, item_top_offset=4.4, line_h=1.7,
)

# Arrow from Classifier into TemplateContext
draw_arrow(
    ax,
    CLF_X + SIDE_W + 0.3, xs[0] - 0.3, arrow_y,
    color=COLOR_SIDE_ARROW, lw=1.8, mut=18,
)

# --- Right side node: Downstream LLM ---
llm_items = [
    "receives prefix",
    "opens chain of thought",
    "generates answer",
]
LLM_H_SMALL = 11
LLM_Y = SIDE_Y + (SIDE_H - LLM_H_SMALL) / 2
draw_box(
    ax, LLM_X, LLM_Y, SIDE_W, LLM_H_SMALL,
    COLOR_SIDE_FC, COLOR_SIDE_EC,
    "Downstream LLM", llm_items, COLOR_SIDE,
    title_fs=11.5, item_fs=8.8, item_top_offset=4.0, line_h=1.7,
)

# Arrow from Render into Downstream LLM
draw_arrow(
    ax,
    xs[3] + BOX_W + 0.3, LLM_X - 0.3, arrow_y,
    color=COLOR_SIDE_ARROW, lw=1.8, mut=18,
)

# --- Bottom footer note ---
ax.text(
    56, 5.0,
    "Cascade walks the priority-sorted union of REGISTRY $\\cup$ ASSERTIVE_FALLBACK and returns the first firing template.",
    ha="center", va="center", fontsize=10, color=COLOR_TEXT, style="italic",
)

# --- Legend ---
legend_handles = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_SIDE_FC,
           markeredgecolor=COLOR_SIDE_EC, markersize=12, label="External"),
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
    ncol=4, handletextpad=0.4, columnspacing=1.2,
)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=COLOR_BG)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
