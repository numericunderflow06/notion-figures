"""
fig_008: Ablation Study Design Matrix
Visual matrix showing the six ablation studies (A1-A6) and which
architectural component or training choice each one isolates.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data from Section 5.3 of architecture_plan.md ──────────────────────

ablation_ids = ["A1", "A2", "A3", "A4", "A5", "A6"]

ablation_labels = [
    "Linear vs. MLP\nprojector",
    "No alignment\npre-training\n(no Stage 1)",
    "LoRA vs. full\nLLM fine-tuning",
    "No encoder\npre-training\n(no Stage 0)",
    "Token compression\nsize K",
    "Per-channel vs.\nshared pooling",
]

# Brief description of what each ablation tests
ablation_descriptions = [
    "Replace 2-layer MLP+GELU\nwith single linear layer",
    "Skip Stage 1; train projector\nand LLM jointly from start",
    "Replace full fine-tuning in\nStage 2 with LoRA (rank 16/64)",
    "Initialize encoder randomly;\ntrain during Stage 1",
    "Vary K ∈ {16, 32, 64, 128, 256}",
    "Channel-specific queries (K_c\nper channel) vs. shared queries",
]

# Columns: architectural components / training choices
component_labels = [
    "Projector\nType",
    "Alignment\nStage (S1)",
    "LLM\nFine-tuning",
    "Encoder\nPre-train (S0)",
    "Compression\nK",
    "Pooling\nStrategy",
]

# For each ablation (row) x component (col):
#   0 = default (unchanged)
#   1 = ablated (this is what changes)
# A1: changes projector type only
# A2: changes alignment stage only
# A3: changes LLM fine-tuning only
# A4: changes encoder pre-training only
# A5: changes compression K only
# A6: changes pooling strategy only
ablation_matrix = np.array([
    [1, 0, 0, 0, 0, 0],  # A1
    [0, 1, 0, 0, 0, 0],  # A2
    [0, 0, 1, 0, 0, 0],  # A3
    [0, 0, 0, 1, 0, 0],  # A4
    [0, 0, 0, 0, 1, 0],  # A5
    [0, 0, 0, 0, 0, 1],  # A6
])

# Default values for each component
default_values = [
    "2-layer MLP\n+ GELU",
    "Yes\n(Stage 1)",
    "Full\nfine-tuning",
    "Yes\n(Stage 0, MAE)",
    "K = 64",
    "Shared\npooling",
]

# Ablated values for each component (shown in the ablated cell)
ablated_values = [
    "Single\nlinear layer",
    "Skipped",
    "LoRA\n(rank 16/64)",
    "Random\ninit",
    "K ∈ {16, 32,\n128, 256}",
    "Per-channel\nqueries",
]

# ── Figure setup ────────────────────────────────────────────────────────

n_rows = len(ablation_ids)
n_cols = len(component_labels)

fig_width = 16
fig_height = 10

fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")

# Layout: main grid on the left, descriptions on the right
# Using gridspec for precise control
gs = fig.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3.2, 1.3],
    left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.02
)

ax_grid = fig.add_subplot(gs[0, 0])
ax_desc = fig.add_subplot(gs[0, 1])

# ── Colors ──────────────────────────────────────────────────────────────

color_default = "#D5E8D4"   # light green
color_ablated = "#F8CECC"   # light red/salmon
color_header_bg = "#DAE8FC" # light blue for headers
color_grid_line = "#666666"
color_text = "#333333"

# ── Draw the grid on ax_grid ────────────────────────────────────────────

cell_w = 1.0
cell_h = 1.0
header_h = 1.2  # taller header row
row_label_w = 0.7  # width for row labels (A1-A6)

ax_grid.set_xlim(-row_label_w - 0.05, n_cols * cell_w + 0.05)
ax_grid.set_ylim(-0.05, n_rows * cell_h + header_h + 0.05)
ax_grid.set_aspect("equal")
ax_grid.axis("off")

# Draw column headers
for j in range(n_cols):
    x = j * cell_w
    y = n_rows * cell_h
    rect = mpatches.FancyBboxPatch(
        (x, y), cell_w, header_h,
        boxstyle="round,pad=0.02",
        facecolor=color_header_bg,
        edgecolor=color_grid_line,
        linewidth=1.2
    )
    ax_grid.add_patch(rect)
    ax_grid.text(
        x + cell_w / 2, y + header_h / 2,
        component_labels[j],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color=color_text,
        linespacing=1.15
    )

# Draw data cells
for i in range(n_rows):
    y = (n_rows - 1 - i) * cell_h  # rows go top to bottom

    # Row label (A1, A2, ...)
    ax_grid.text(
        -row_label_w / 2, y + cell_h / 2,
        ablation_ids[i],
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color="#1a1a1a",
        fontfamily="monospace"
    )

    for j in range(n_cols):
        x = j * cell_w
        is_ablated = ablation_matrix[i, j] == 1
        bg_color = color_ablated if is_ablated else color_default

        rect = mpatches.FancyBboxPatch(
            (x, y), cell_w, cell_h,
            boxstyle="round,pad=0.02",
            facecolor=bg_color,
            edgecolor=color_grid_line,
            linewidth=0.8,
            alpha=0.95
        )
        ax_grid.add_patch(rect)

        # Cell text: show actual value
        if is_ablated:
            cell_text = ablated_values[j]
            text_weight = "bold"
            text_color = "#B85450"
            text_size = 8.5
        else:
            cell_text = default_values[j]
            text_weight = "normal"
            text_color = "#4A7A4C"
            text_size = 8.5

        ax_grid.text(
            x + cell_w / 2, y + cell_h / 2,
            cell_text,
            ha="center", va="center",
            fontsize=text_size, fontweight=text_weight,
            color=text_color,
            linespacing=1.1
        )

# ── Draw descriptions panel on ax_desc ──────────────────────────────────

ax_desc.set_xlim(0, 1)
ax_desc.set_ylim(-0.05, n_rows * cell_h + header_h + 0.05)
ax_desc.axis("off")

# Header for description column
desc_header_y = n_rows * cell_h
rect = mpatches.FancyBboxPatch(
    (0.02, desc_header_y), 0.96, header_h,
    boxstyle="round,pad=0.02",
    facecolor=color_header_bg,
    edgecolor=color_grid_line,
    linewidth=1.2
)
ax_desc.add_patch(rect)
ax_desc.text(
    0.5, desc_header_y + header_h / 2,
    "What Changes",
    ha="center", va="center",
    fontsize=10, fontweight="bold",
    color=color_text
)

# Description rows
for i in range(n_rows):
    y = (n_rows - 1 - i) * cell_h
    rect = mpatches.FancyBboxPatch(
        (0.02, y), 0.96, cell_h,
        boxstyle="round,pad=0.02",
        facecolor="#F5F5F5",
        edgecolor=color_grid_line,
        linewidth=0.6,
        alpha=0.8
    )
    ax_desc.add_patch(rect)
    ax_desc.text(
        0.5, y + cell_h / 2,
        ablation_descriptions[i],
        ha="center", va="center",
        fontsize=8, fontweight="normal",
        color="#555555",
        linespacing=1.15,
        style="italic"
    )

# ── Title ───────────────────────────────────────────────────────────────

fig.suptitle(
    "Ablation Study Design Matrix — LLaVA-TSM",
    fontsize=16, fontweight="bold", color="#1a1a1a",
    y=0.96
)

fig.text(
    0.5, 0.915,
    "Each row is one ablation study. Green cells = default setting retained; Red cells = ablated variant used.",
    ha="center", fontsize=10, color="#555555", style="italic"
)

# ── Legend ──────────────────────────────────────────────────────────────

legend_y = 0.015
legend_elements = [
    mpatches.Patch(facecolor=color_default, edgecolor=color_grid_line,
                   linewidth=0.8, label="Default (unchanged)"),
    mpatches.Patch(facecolor=color_ablated, edgecolor=color_grid_line,
                   linewidth=0.8, label="Ablated (modified)"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=True,
    fancybox=True,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, legend_y)
)

# ── Save ────────────────────────────────────────────────────────────────

output_path = "/home/wangni/notion-figures/llava/fig_008.png"
fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure saved to {output_path}")
