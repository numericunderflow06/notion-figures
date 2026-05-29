"""
fig_004: Core Study Matrix Coverage
Visual grid of models × injection methods, cells coloured by completion status.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# --- Matrix definition -------------------------------------------------------
columns = ["OpenTSLM-\nFlamingo", "GPT-4o\n(text TS)", "GPT-4o\n(plot TS)"]
rows = ["Prompt injection", "Soft-prompt", "Residual CCG", "Visual annotation"]

# Status codes:
#   "done"     -> green   (completed at n=50)
#   "partial"  -> yellow  (smoke / partial)
#   "pending"  -> grey    (not run yet)
#   "impossible" -> hatched (architecturally impossible — closed API surface)
# Layout: status[row_idx][col_idx]
status = [
    # OpenTSLM-Flamingo, GPT-4o text TS, GPT-4o plot TS
    ["done",       "done",       "done"],        # Prompt injection
    ["partial",    "impossible", "impossible"],  # Soft-prompt
    ["done",       "impossible", "impossible"],  # Residual CCG
    ["pending",    "impossible", "partial"],     # Visual annotation
]

# Optional annotation overlay text per cell
annotations = [
    ["", "", ""],
    ["", "no logits API", "no logits API"],
    ["", "API limitation\n= finding", "API limitation\n= finding"],
    ["N/A surface", "text-only model", ""],
]

# --- Colour palette ----------------------------------------------------------
COLOR = {
    "done":       "#4CAF50",  # green
    "partial":    "#FFC107",  # amber
    "pending":    "#BDBDBD",  # grey
    "impossible": "#ECEFF1",  # very light grey base (hatched on top)
}
EDGE = "#37474F"
HATCH_COLOR = "#546E7A"

# --- Figure -----------------------------------------------------------------
n_rows = len(rows)
n_cols = len(columns)

fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

cell_w = 1.0
cell_h = 1.0
left_pad = 0.0
top_pad = 0.0

# Draw cells (origin top-left; flip y so row 0 is on top)
for r in range(n_rows):
    for c in range(n_cols):
        s = status[r][c]
        x = left_pad + c * cell_w
        y = top_pad + (n_rows - 1 - r) * cell_h  # flip vertical

        rect = Rectangle(
            (x, y), cell_w, cell_h,
            facecolor=COLOR[s],
            edgecolor=EDGE,
            linewidth=1.5,
        )
        ax.add_patch(rect)

        # Hatching for architecturally impossible
        if s == "impossible":
            hatched = Rectangle(
                (x, y), cell_w, cell_h,
                facecolor="none",
                edgecolor=HATCH_COLOR,
                hatch="////",
                linewidth=0.0,
            )
            ax.add_patch(hatched)

        # Status label in cell
        status_label = {
            "done": "DONE\n(n=50)",
            "partial": "PARTIAL\n/ smoke",
            "pending": "PENDING",
            "impossible": "N/A",
        }[s]
        text_color = "white" if s in ("done", "pending") else "#212121"
        ax.text(
            x + cell_w / 2, y + cell_h / 2 + 0.18,
            status_label,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=text_color,
        )

        # Sub-annotation
        ann = annotations[r][c]
        if ann:
            ax.text(
                x + cell_w / 2, y + cell_h / 2 - 0.25,
                ann,
                ha="center", va="center",
                fontsize=9, style="italic",
                color="#263238" if s != "pending" else "white",
            )

# --- Axes / labels ----------------------------------------------------------
ax.set_xlim(left_pad - 0.05, left_pad + n_cols * cell_w + 0.05)
ax.set_ylim(top_pad - 0.05, top_pad + n_rows * cell_h + 0.05)
ax.set_aspect("equal")

# Column labels (top)
for c, col_name in enumerate(columns):
    ax.text(
        left_pad + c * cell_w + cell_w / 2,
        top_pad + n_rows * cell_h + 0.10,
        col_name,
        ha="center", va="bottom",
        fontsize=11, fontweight="bold",
        color="#212121",
    )

# Row labels (left)
for r, row_name in enumerate(rows):
    y = top_pad + (n_rows - 1 - r) * cell_h + cell_h / 2
    ax.text(
        left_pad - 0.08, y,
        row_name,
        ha="right", va="center",
        fontsize=12, fontweight="bold",
        color="#212121",
    )

# Group axis label (left side only — "Models" is stated in the title)
ax.text(
    left_pad - 1.25,
    top_pad + (n_rows * cell_h) / 2,
    "Injection method",
    ha="center", va="center",
    fontsize=13, fontweight="bold", color="#37474F",
    rotation=90,
)

# Title — placed via fig.suptitle for clean separation from column headers
fig.suptitle(
    "Core Study Matrix Coverage",
    fontsize=15, fontweight="bold", color="#212121",
    y=0.985,
)
fig.text(
    0.5, 0.945,
    "Models × injection methods — submission run inventory",
    ha="center", va="top",
    fontsize=11, style="italic", color="#455A64",
)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# --- Legend -----------------------------------------------------------------
legend_handles = [
    mpatches.Patch(facecolor=COLOR["done"],    edgecolor=EDGE, label="Done — n=50 evaluation"),
    mpatches.Patch(facecolor=COLOR["partial"], edgecolor=EDGE, label="Partial / smoke run"),
    mpatches.Patch(facecolor=COLOR["pending"], edgecolor=EDGE, label="Pending"),
    mpatches.Patch(facecolor=COLOR["impossible"], edgecolor=EDGE,
                   hatch="////", label="Architecturally impossible"),
]
leg = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=4,
    frameon=True,
    fontsize=10,
    title="Cell status",
    title_fontsize=11,
)
leg.get_frame().set_edgecolor("#CFD8DC")
leg.get_frame().set_facecolor("white")

# Footnote
fig.text(
    0.5, -0.005,
    "Hatched cells: closed-API models expose no parameter-level surface, so soft-prompt and residual CCG are\n"
    "structurally inaccessible. The GPT-4o × residual cell is reported as a finding (API limitation), not a gap.",
    ha="center", va="bottom",
    fontsize=9, style="italic", color="#455A64",
)

plt.subplots_adjust(top=0.86, bottom=0.16, left=0.13, right=0.97)
out_path = "/home/wangni/notion-figures/ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_004.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
