"""
fig_005: FiLM Conditioning Pathway
Pathway: classifier verdict v -> E_cls -> mean -> c -> gamma/beta MLPs
        -> modulation of perceiver queries Q_Pi
Equation: Q'_Pi = gamma(c) (.) Q_Pi + beta(c)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D


# Color palette (consistent, professional)
COL_INPUT      = "#E8F1FA"   # light blue
COL_INPUT_EDGE = "#3B6FB6"
COL_EMB        = "#E3EEDC"   # light green
COL_EMB_EDGE   = "#4F7A3A"
COL_CTX        = "#FFF1D6"   # warm gold
COL_CTX_EDGE   = "#B57A1A"
COL_MLP        = "#EADCF2"   # lavender
COL_MLP_EDGE   = "#6A3FA0"
COL_QUERY      = "#FBE2E2"   # rose
COL_QUERY_EDGE = "#A8423E"
COL_HIGHLIGHT  = "#FFE9A8"   # highlight band
COL_ARROW      = "#444444"
COL_TEXT       = "#222222"


def add_box(ax, x, y, w, h, label, face, edge,
            fontsize=10, fontweight="bold", round_pad=0.02):
    """Add a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={round_pad},rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=COL_TEXT)


def add_arrow(ax, x0, y0, x1, y1, color=COL_ARROW, lw=1.6, style="-|>"):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(13.5, 5.2), dpi=200)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 5.2)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title
ax.text(6.75, 4.95, "FiLM Conditioning Pathway",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=COL_TEXT)
ax.text(6.75, 4.62,
        r"$Q'_{\Pi} \;=\; \gamma(c)\,\odot\,Q_{\Pi} \;+\; \beta(c)$",
        ha="center", va="center",
        fontsize=12.5, color=COL_TEXT)

# --- Highlight band behind the Q_Pi -> Q'_Pi modulation region ---
band_y, band_h = 1.55, 1.55
band = FancyBboxPatch(
    (8.55, band_y), 4.65, band_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.2, edgecolor="#E0B84A", facecolor=COL_HIGHLIGHT, alpha=0.55,
)
ax.add_patch(band)
ax.text(10.875, band_y + band_h - 0.18,
        "Only the perceiver queries are modulated",
        ha="center", va="center",
        fontsize=9, fontstyle="italic", color="#6B4A0F")

# --- Row layout ---
row_y = 2.05   # bottom y of mid-row boxes
row_h = 0.85

# 1) classifier verdict v
add_box(ax, 0.20, row_y, 1.25, row_h, r"$v$", COL_INPUT, COL_INPUT_EDGE,
        fontsize=14)
ax.text(0.825, row_y - 0.30, "classifier\nverdict",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# 2) E_cls embedding lookup
add_box(ax, 1.85, row_y, 1.55, row_h, r"$E_{\mathrm{cls}}$",
        COL_EMB, COL_EMB_EDGE, fontsize=13)
ax.text(2.625, row_y - 0.30, "class embedding\ntable",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# 3) mean pooling (token sequence -> single vector)
add_box(ax, 3.80, row_y, 1.20, row_h, "mean", COL_EMB, COL_EMB_EDGE,
        fontsize=11)
ax.text(4.40, row_y - 0.30, "pool over\nverdict tokens",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# 4) c (conditioning vector)
add_box(ax, 5.35, row_y, 1.00, row_h, r"$c$", COL_CTX, COL_CTX_EDGE,
        fontsize=14)
ax.text(5.85, row_y - 0.30, "conditioning\nvector",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# 5) gamma and beta MLPs (stacked)
mlp_x, mlp_w = 6.85, 1.45
add_box(ax, mlp_x, 2.95, mlp_w, 0.80, r"MLP$_\gamma$",
        COL_MLP, COL_MLP_EDGE, fontsize=11)
add_box(ax, mlp_x, 1.85, mlp_w, 0.80, r"MLP$_\beta$",
        COL_MLP, COL_MLP_EDGE, fontsize=11)

# gamma(c) and beta(c) labels (outputs)
ax.text(mlp_x + mlp_w + 0.30, 3.35, r"$\gamma(c)$",
        ha="left", va="center", fontsize=12, color=COL_TEXT)
ax.text(mlp_x + mlp_w + 0.30, 2.25, r"$\beta(c)$",
        ha="left", va="center", fontsize=12, color=COL_TEXT)

# 6) elementwise product node (gamma . Q_Pi)
prod_cx, prod_cy = 9.55, 3.35
prod_r = 0.24
prod = Circle((prod_cx, prod_cy), prod_r,
              facecolor="white", edgecolor=COL_MLP_EDGE, linewidth=1.8)
ax.add_patch(prod)
ax.text(prod_cx, prod_cy, r"$\odot$",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COL_MLP_EDGE)
ax.text(prod_cx, prod_cy - 0.55, "element-wise",
        ha="center", va="center", fontsize=8.5,
        fontstyle="italic", color="#555555")

# 7) sum node (+ beta)
sum_cx, sum_cy = 11.05, 2.45
sum_r = 0.24
sum_node = Circle((sum_cx, sum_cy), sum_r,
                  facecolor="white", edgecolor=COL_MLP_EDGE, linewidth=1.8)
ax.add_patch(sum_node)
ax.text(sum_cx, sum_cy, r"$+$",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COL_MLP_EDGE)

# 8) Q_Pi (input queries) -- lower-left of highlight region
add_box(ax, 8.65, 1.80, 1.05, row_h, r"$Q_{\Pi}$",
        COL_QUERY, COL_QUERY_EDGE, fontsize=13)
ax.text(9.175, 1.80 - 0.30, "perceiver\nqueries",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# 9) Q'_Pi (modulated queries)
add_box(ax, 12.05, 2.05, 1.20, row_h, r"$Q'_{\Pi}$",
        COL_QUERY, COL_QUERY_EDGE, fontsize=13)
ax.text(12.65, 2.05 - 0.30, "modulated\nqueries",
        ha="center", va="center", fontsize=9, color=COL_TEXT)

# ----------- Arrows along the main pathway -----------
mid_y = row_y + row_h / 2  # = 2.475

# v -> E_cls
add_arrow(ax, 1.45, mid_y, 1.85, mid_y)
# E_cls -> mean
add_arrow(ax, 3.40, mid_y, 3.80, mid_y)
# mean -> c
add_arrow(ax, 5.00, mid_y, 5.35, mid_y)
# c -> MLP_gamma
add_arrow(ax, 6.35, mid_y + 0.10, mlp_x, 3.35)
# c -> MLP_beta
add_arrow(ax, 6.35, mid_y - 0.10, mlp_x, 2.25)

# MLP_gamma output -> product node (top branch)
add_arrow(ax, mlp_x + mlp_w, 3.35, prod_cx - prod_r, 3.35,
          color=COL_MLP_EDGE)

# Q_Pi -> product node (vertical up)
add_arrow(ax, 9.175, 1.80 + row_h, 9.175, prod_cy - prod_r,
          color=COL_QUERY_EDGE, lw=1.8)
# label on the Q_Pi branch
ax.text(9.30, (1.80 + row_h + prod_cy - prod_r) / 2,
        r"$Q_{\Pi}$",
        ha="left", va="center", fontsize=10, color=COL_QUERY_EDGE)

# product node -> sum node
add_arrow(ax, prod_cx + prod_r * 0.7, prod_cy - prod_r * 0.7,
          sum_cx - sum_r * 0.9, sum_cy + sum_r * 0.9,
          color=COL_MLP_EDGE)
# label gamma(c) . Q_Pi on the diagonal
ax.text((prod_cx + sum_cx) / 2 + 0.05,
        (prod_cy + sum_cy) / 2 + 0.18,
        r"$\gamma(c)\odot Q_{\Pi}$",
        ha="center", va="center", fontsize=9.5, color="#444",
        rotation=-30)

# MLP_beta output -> sum node
add_arrow(ax, mlp_x + mlp_w, 2.25, sum_cx - sum_r, sum_cy,
          color=COL_MLP_EDGE)

# sum node -> Q'_Pi
add_arrow(ax, sum_cx + sum_r, sum_cy, 12.05, mid_y,
          color=COL_QUERY_EDGE, lw=2.0)

# ----------- Legend strip at bottom -----------
legend_y = 0.45
legend_items = [
    (COL_INPUT,  COL_INPUT_EDGE,  "input"),
    (COL_EMB,    COL_EMB_EDGE,    "embedding / pool"),
    (COL_CTX,    COL_CTX_EDGE,    "conditioning"),
    (COL_MLP,    COL_MLP_EDGE,    "FiLM MLP"),
    (COL_QUERY,  COL_QUERY_EDGE,  "perceiver queries"),
]
x0 = 0.70
for face, edge, name in legend_items:
    swatch = FancyBboxPatch(
        (x0, legend_y), 0.35, 0.30,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.2, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(swatch)
    ax.text(x0 + 0.45, legend_y + 0.15, name,
            ha="left", va="center", fontsize=9, color=COL_TEXT)
    x0 += 0.45 + 0.05 + 0.18 + len(name) * 0.085

# Save
out_path = ("/home/wangni/notion-figures/"
            "classifier-aware-flamingo-caf-a-shared-projector-architectur/"
            "fig_005.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {out_path}")
