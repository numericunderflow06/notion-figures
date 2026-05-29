"""
fig_002: Shared Projector — One Linear, Three Paths
Zoomed view emphasizing that classifier, perceiver, and instruct tokens
go through the same nn.Linear(d_lat, d_llm) instance.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

# Colors
TEAL = "#1B9E9E"
ORANGE = "#E07A2B"
PURPLE = "#7B4FB0"
MATRIX_FILL = "#F0EAD6"
MATRIX_EDGE = "#3A3A3A"
INPUT_BG = "#F7F7F7"
OUTPUT_BG = "#F2F8FB"

fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(7.0, 8.55,
        "Shared Projector  —  One Linear, Three Paths",
        ha="center", va="center",
        fontsize=18, fontweight="bold", color="#202020")
ax.text(7.0, 8.10,
        r"A single $\mathrm{nn.Linear}(d_{\mathrm{lat}}, d_{\mathrm{llm}})$ instance serves classifier, perceiver, and instruct paths",
        ha="center", va="center",
        fontsize=11.5, style="italic", color="#505050")

# ---- Left column: latent space label ----
ax.text(1.7, 7.20, r"$\mathbb{R}^{d_{\mathrm{lat}}}$",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#303030")
ax.text(1.7, 6.75, "(latent space)",
        ha="center", va="center", fontsize=10, color="#707070")

# Three input tokens on the left
input_specs = [
    ("Classifier  (E_cls)",     "verdict embedding",        TEAL,   6.05),
    ("Perceiver  (Q_perc)",     "TS perceiver output",      ORANGE, 4.55),
    ("Instruct  (E_inst)",      "task-level tokens (opt.)", PURPLE, 3.05),
]

input_x = 1.7
input_w = 2.3
input_h = 0.95

for label, sub, color, y in input_specs:
    box = FancyBboxPatch(
        (input_x - input_w / 2, y - input_h / 2),
        input_w, input_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=INPUT_BG, zorder=3,
    )
    ax.add_patch(box)
    ax.text(input_x, y + 0.18, label,
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=color)
    ax.text(input_x, y - 0.22, sub,
            ha="center", va="center", fontsize=9.5, color="#555555")

# ---- Middle: shared projector matrix W ----
mat_cx, mat_cy = 7.0, 4.55
mat_w, mat_h = 2.4, 3.2

# Outer matrix box
mat_box = FancyBboxPatch(
    (mat_cx - mat_w / 2, mat_cy - mat_h / 2),
    mat_w, mat_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=2.4, edgecolor=MATRIX_EDGE, facecolor=MATRIX_FILL, zorder=4,
)
ax.add_patch(mat_box)

# Matrix grid (decorative cells to suggest a weight matrix)
n_rows, n_cols = 10, 6
cell_w = (mat_w - 0.5) / n_cols
cell_h = (mat_h - 0.9) / n_rows
grid_x0 = mat_cx - (mat_w - 0.5) / 2
grid_y0 = mat_cy - (mat_h - 1.1) / 2

rng = np.random.default_rng(7)
for i in range(n_rows):
    for j in range(n_cols):
        v = rng.uniform(0.55, 0.95)
        ax.add_patch(Rectangle(
            (grid_x0 + j * cell_w, grid_y0 + i * cell_h),
            cell_w * 0.92, cell_h * 0.88,
            facecolor=(0.55 + 0.35 * (1 - v), 0.55 + 0.30 * (1 - v), 0.50 + 0.25 * (1 - v)),
            edgecolor="none", alpha=0.55, zorder=5,
        ))

# Big W label
ax.text(mat_cx, mat_cy + mat_h / 2 - 0.35,
        r"$\mathbf{W}$",
        ha="center", va="center", fontsize=26, fontweight="bold",
        color="#202020", zorder=6)

# Shape annotation directly under the matrix (its own band)
ax.text(mat_cx, mat_cy - mat_h / 2 - 0.35,
        r"$\mathbf{W} \in \mathbb{R}^{d_{\mathrm{llm}} \times d_{\mathrm{lat}}}$",
        ha="center", va="center", fontsize=12.5, color="#202020")
ax.text(mat_cx, mat_cy - mat_h / 2 - 0.70,
        r"$\mathrm{nn.Linear}(d_{\mathrm{lat}},\ d_{\mathrm{llm}})$  —  single instance",
        ha="center", va="center", fontsize=10.5, color="#404040",
        fontfamily="monospace")

# Label across the top of the matrix
ax.text(mat_cx, mat_cy + mat_h / 2 + 0.30,
        "shared projector  P",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#202020")

# ---- Right column: d_llm space label ----
ax.text(12.3, 7.20, r"$\mathbb{R}^{d_{\mathrm{llm}}}$",
        ha="center", va="center", fontsize=18, fontweight="bold", color="#303030")
ax.text(12.3, 6.75, "(LLM token space)",
        ha="center", va="center", fontsize=10, color="#707070")

# Three output tokens on the right
output_specs = [
    (r"$P(E_{\mathrm{cls}})$",     "classifier token",  TEAL,   6.05),
    (r"$P(Q_{\mathrm{perc}})$",    "perceiver tokens",  ORANGE, 4.55),
    (r"$P(E_{\mathrm{inst}})$",    "instruct tokens",   PURPLE, 3.05),
]

output_x = 12.3
output_w = 2.3
output_h = 0.95

for label, sub, color, y in output_specs:
    box = FancyBboxPatch(
        (output_x - output_w / 2, y - output_h / 2),
        output_w, output_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.8, edgecolor=color, facecolor=OUTPUT_BG, zorder=3,
    )
    ax.add_patch(box)
    ax.text(output_x, y + 0.18, label,
            ha="center", va="center", fontsize=12.5, fontweight="bold", color=color)
    ax.text(output_x, y - 0.22, sub,
            ha="center", va="center", fontsize=9.5, color="#555555")

# ---- Arrows: inputs -> matrix ----
in_right_edge = input_x + input_w / 2
mat_left_edge = mat_cx - mat_w / 2

ys = [6.05, 4.55, 3.05]
colors = [TEAL, ORANGE, PURPLE]
for y, color in zip(ys, colors):
    arr = FancyArrowPatch(
        (in_right_edge + 0.05, y),
        (mat_left_edge - 0.05, y),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=3.0, color=color, zorder=2,
    )
    ax.add_patch(arr)

# ---- Arrows: matrix -> outputs ----
mat_right_edge = mat_cx + mat_w / 2
out_left_edge = output_x - output_w / 2

for y, color in zip(ys, colors):
    arr = FancyArrowPatch(
        (mat_right_edge + 0.05, y),
        (out_left_edge - 0.05, y),
        arrowstyle="-|>", mutation_scale=22,
        linewidth=3.0, color=color, zorder=2,
    )
    ax.add_patch(arr)

# Annotate the central operation
ax.text(mat_cx, 7.55,
        r"$\mathbf{x} \in \mathbb{R}^{d_{\mathrm{lat}}} \;\longmapsto\; \mathbf{W}\mathbf{x} \in \mathbb{R}^{d_{\mathrm{llm}}}$",
        ha="center", va="center", fontsize=12.5, color="#303030",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FAFAEC",
                  edgecolor="#BBB8A0", linewidth=1.0))

# ---- Legend row: clearly separated below the equation labels ----
legend_y = 1.85
# Thin separator line above the legend, sitting between the nn.Linear text and legend
ax.plot([1.6, 12.4], [2.30, 2.30], color="#D8D8D8", linewidth=0.8, zorder=1)

legend_items = [
    ("classifier path",  TEAL),
    ("perceiver path",   ORANGE),
    ("instruct path",    PURPLE),
]
legend_x_positions = [3.4, 6.7, 9.9]
for (label, color), lx in zip(legend_items, legend_x_positions):
    ax.add_patch(Rectangle((lx - 0.15, legend_y - 0.10), 0.45, 0.20,
                           facecolor=color, edgecolor="none", zorder=3))
    ax.text(lx + 0.40, legend_y, label,
            ha="left", va="center", fontsize=10.5, color="#303030")

# ---- Footer: defaults box (well below the legend) ----
footer_box = FancyBboxPatch(
    (1.2, 0.20), 11.6, 1.10,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.2, edgecolor="#888888", facecolor="#FCFCFC", zorder=1,
)
ax.add_patch(footer_box)

footer_top = 1.00
footer_bot = 0.50

ax.text(2.0, footer_top, "Defaults",
        ha="left", va="center", fontsize=11, fontweight="bold", color="#202020")
ax.text(2.0, footer_bot,
        r"$d_{\mathrm{lat}} = d_{\mathrm{llm}} = 3072$",
        ha="left", va="center", fontsize=11, color="#202020")

ax.text(6.2, footer_top, "Parameter count",
        ha="left", va="center", fontsize=11, fontweight="bold", color="#202020")
ax.text(6.2, footer_bot,
        r"$d_{\mathrm{llm}} \times d_{\mathrm{lat}} + d_{\mathrm{llm}} \approx 9.4\mathrm{M}$",
        ha="left", va="center", fontsize=11, color="#202020")

ax.text(10.6, footer_top, "Rationale",
        ha="left", va="center", fontsize=11, fontweight="bold", color="#202020")
ax.text(10.6, footer_bot,
        "shared latent geometry",
        ha="left", va="center", fontsize=10.5, color="#202020")

plt.tight_layout()
out_path = "/home/wangni/notion-figures/classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_002.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
