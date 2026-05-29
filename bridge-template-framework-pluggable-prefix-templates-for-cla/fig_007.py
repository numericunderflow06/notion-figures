"""fig_007: Demo Coverage Matrix — Contexts x Configs.

Builds a 4x7 grid showing which template (and which family) fires for
each (context, config) pair when running examples/demo.py. The cell
fills are derived from a real run of the demo cascade.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def wrap_template_name(name: str, width: int = 14) -> str:
    """Break a snake_case template name into <= 2 lines near the middle."""
    if len(name) <= width:
        return name
    parts = name.split("_")
    line1, line2 = "", ""
    for p in parts:
        if not line1 or len(line1) + 1 + len(p) <= width:
            line1 = p if not line1 else f"{line1}_{p}"
        else:
            line2 = p if not line2 else f"{line2}_{p}"
    return f"{line1}\n{line2}" if line2 else line1

OUT_PATH = Path(
    "/home/wangni/notion-figures/"
    "bridge-template-framework-pluggable-prefix-templates-for-cla/fig_007.png"
)

CONTEXTS = ["ECG_BINARY", "ECG_SEVEN_CLASS", "SLEEP_FIVE_CLASS", "HAR_EIGHT_CLASS"]
CONFIGS = [
    "baseline",
    "verify_aware",
    "markdown_roleplay",
    "clinical_reasoning",
    "low_confidence_aware",
    "hierarchical",
    "strict_ood",
]

# (template_name, family) keyed by (context_index, config_index), captured
# directly from `python examples/demo.py` against the current registry.
CELLS = {
    (0, 0): ("assertive_top1", "fallback"),
    (0, 1): ("verify_affirmative", "qtype"),
    (0, 2): ("markdown_heading", "presentation"),
    (0, 3): ("evidence_stub_library", "evidence"),
    (0, 4): ("assertive_top1", "fallback"),
    (0, 5): ("coarse_to_fine", "hierarchy"),
    (0, 6): ("assertive_top1", "fallback"),

    (1, 0): ("assertive_top1", "fallback"),
    (1, 1): ("assertive_top1", "fallback"),
    (1, 2): ("markdown_heading", "presentation"),
    (1, 3): ("evidence_stub_library", "evidence"),
    (1, 4): ("low_confidence_leading", "confidence"),
    (1, 5): ("coarse_to_fine", "hierarchy"),
    (1, 6): ("assertive_top1", "fallback"),

    (2, 0): ("assertive_top1", "fallback"),
    (2, 1): ("qtype_aware_fallback", "qtype"),
    (2, 2): ("markdown_heading", "presentation"),
    (2, 3): ("evidence_stub_library", "evidence"),
    (2, 4): ("low_confidence_leading", "confidence"),
    (2, 5): ("coarse_to_fine", "hierarchy"),
    (2, 6): ("assertive_top1", "fallback"),

    (3, 0): ("assertive_top1", "fallback"),
    (3, 1): ("assertive_top1", "fallback"),
    (3, 2): ("markdown_heading", "presentation"),
    (3, 3): ("evidence_stub_library", "evidence"),
    (3, 4): ("assertive_top1", "fallback"),
    (3, 5): ("coarse_to_fine", "hierarchy"),
    (3, 6): ("assertive_top1", "fallback"),
}

FAMILY_COLORS = {
    "confidence":   "#5B8FF9",  # blue
    "qtype":        "#F6BD16",  # gold
    "evidence":     "#5AD8A6",  # green
    "presentation": "#E86452",  # coral
    "hierarchy":    "#9270CA",  # purple
    "fallback":     "#CDD7E0",  # neutral gray
}
FAMILY_TEXT = {
    "confidence":   "#0B3FA2",
    "qtype":        "#7A5500",
    "evidence":     "#0B5A3A",
    "presentation": "#7A1F12",
    "hierarchy":    "#3F2877",
    "fallback":     "#414B57",
}

fig, ax = plt.subplots(figsize=(15.5, 7.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n_rows = len(CONTEXTS)
n_cols = len(CONFIGS)

cell_w = 1.0
cell_h = 1.0
pad = 0.06  # gap between cells

for r in range(n_rows):
    for c in range(n_cols):
        template, family = CELLS[(r, c)]
        x = c * cell_w
        y = (n_rows - 1 - r) * cell_h
        face = FAMILY_COLORS[family]
        text_color = FAMILY_TEXT[family]
        box = FancyBboxPatch(
            (x + pad, y + pad),
            cell_w - 2 * pad,
            cell_h - 2 * pad,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#3A4552",
            facecolor=face,
            alpha=0.92,
        )
        ax.add_patch(box)
        # Family label (bold, top)
        ax.text(
            x + cell_w / 2,
            y + cell_h / 2 + 0.22,
            family,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=text_color,
        )
        # Template name (smaller, italic, bottom; wrapped if long)
        ax.text(
            x + cell_w / 2,
            y + cell_h / 2 - 0.20,
            wrap_template_name(template, width=14),
            ha="center",
            va="center",
            fontsize=8.5,
            color=text_color,
            style="italic",
            linespacing=1.05,
        )

# Column headers (configs) — angled to prevent overlap
for c, cfg in enumerate(CONFIGS):
    ax.text(
        c * cell_w + cell_w / 2,
        n_rows * cell_h + 0.12,
        cfg,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#1F2933",
        rotation=25,
        rotation_mode="anchor",
    )

# Row headers (contexts)
for r, ctx in enumerate(CONTEXTS):
    y = (n_rows - 1 - r) * cell_h + cell_h / 2
    ax.text(
        -0.12,
        y,
        ctx,
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#1F2933",
    )

# Title and axis label hints
ax.text(
    n_cols * cell_w / 2,
    n_rows * cell_h + 1.55,
    "Demo Coverage Matrix: Contexts × Configs",
    ha="center",
    va="bottom",
    fontsize=15,
    fontweight="bold",
    color="#101820",
)
ax.text(
    n_cols * cell_w / 2,
    n_rows * cell_h + 1.22,
    "Family of template that fires in examples/demo.py for each (context, config) pair",
    ha="center",
    va="bottom",
    fontsize=10,
    color="#4A5568",
    style="italic",
)

# Axis-side labels
ax.text(
    n_cols * cell_w / 2,
    -0.55,
    "Configs (TemplateConfig flag combinations)",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#1F2933",
)
ax.text(
    -1.55,
    n_rows * cell_h / 2,
    "Contexts (TemplateContext fixtures)",
    ha="center",
    va="center",
    fontsize=10.5,
    color="#1F2933",
    rotation=90,
)

# Legend
legend_order = ["confidence", "qtype", "evidence", "presentation", "hierarchy", "fallback"]
handles = [
    mpatches.Patch(
        facecolor=FAMILY_COLORS[k],
        edgecolor="#3A4552",
        label=("assertive fallback (base)" if k == "fallback" else f"{k} family"),
    )
    for k in legend_order
]
ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=6,
    frameon=False,
    fontsize=9.5,
    handlelength=1.4,
    columnspacing=1.4,
)

# Frame
ax.set_xlim(-1.9, n_cols * cell_w + 0.3)
ax.set_ylim(-0.95, n_rows * cell_h + 2.0)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved figure to {OUT_PATH}")
