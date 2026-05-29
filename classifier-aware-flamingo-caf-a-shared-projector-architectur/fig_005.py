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


def add_line(ax, x0, y0, x1, y1, color=COL_ARROW, lw=1.6):
    """Plain segment, no arrow head."""
    seg = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(seg)


# ---------------- Figure ----------------
fig, ax = plt.subplots(figsize=(13.5, 5.6), dpi=200)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 5.6)
ax.axis("off")
fig.patch.set_facecolor("white")

# Title and equation
ax.text(6.75, 5.30, "FiLM Conditioning Pathway",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=COL_TEXT)
ax.text(6.75, 4.95,
        r"$Q'_{\Pi} \;=\; \gamma(c)\,\odot\,Q_{\Pi} \;+\; \beta(c)$",
        ha="center", va="center",
        fontsize=12.5, color=COL_TEXT)

# --- Highlight band behind the modulation region ---
band_y, band_h = 1.55, 2.10
band_x0, band_x1 = 8.55, 13.20
band = FancyBboxPatch(
    (band_x0, band_y), band_x1 - band_x0, band_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.2, edgecolor="#E0B84A", facecolor=COL_HIGHLIGHT, alpha=0.55,
)
ax.add_patch(band)

# Band annotation moved OUTSIDE the band (above it) so it never
# collides with the gamma(c) (.) Q_Pi label inside the band.
ax.text((band_x0 + band_x1) / 2, band_y + band_h + 0.25,
        "Only the perceiver queries are modulated element-wise",
        ha="center", va="center",
        fontsize=9.5, fontstyle="italic", color="#6B4A0F")

# --- Row layout (main horizontal flow) ---
row_y = 2.05
row_h = 0.85
mid_y = row_y + row_h / 2  # = 2.475

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

# 3) mean pooling
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

# 6) elementwise product node ⊙ — moved directly above Q_Pi box
prod_cx, prod_cy = 9.175, 3.35
prod_r = 0.24
prod = Circle((prod_cx, prod_cy), prod_r,
              facecolor="white", edgecolor=COL_MLP_EDGE, linewidth=1.8)
ax.add_patch(prod)
ax.text(prod_cx, prod_cy, r"$\odot$",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COL_MLP_EDGE)

# 7) sum node + — moved above Q_Pi box (clears box for routing)
sum_cx, sum_cy = 11.05, 2.95
sum_r = 0.24
sum_node = Circle((sum_cx, sum_cy), sum_r,
                  facecolor="white", edgecolor=COL_MLP_EDGE, linewidth=1.8)
ax.add_patch(sum_node)
ax.text(sum_cx, sum_cy, r"$+$",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COL_MLP_EDGE)

# 8) Q_Pi (perceiver queries)
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

# v -> E_cls -> mean -> c
add_arrow(ax, 1.45, mid_y, 1.85, mid_y)
add_arrow(ax, 3.40, mid_y, 3.80, mid_y)
add_arrow(ax, 5.00, mid_y, 5.35, mid_y)

# c -> MLP_gamma and c -> MLP_beta
add_arrow(ax, 6.35, mid_y + 0.10, mlp_x, 3.35)
add_arrow(ax, 6.35, mid_y - 0.10, mlp_x, 2.25)

# MLP_gamma output -> product node (horizontal)
add_arrow(ax, mlp_x + mlp_w, 3.35, prod_cx - prod_r, prod_cy,
          color=COL_MLP_EDGE)
ax.text(mlp_x + mlp_w + 0.18, 3.55, r"$\gamma(c)$",
        ha="left", va="center", fontsize=12, color=COL_TEXT)

# MLP_beta output -> sum node, routed UP and then RIGHT around the Q_Pi box.
# (Straight diagonal would pass through the Q_Pi box and collide with its label.)
beta_corner_y = sum_cy  # 2.95, above the Q_Pi box top (2.65)
# segment 1: up from MLP_beta output
add_line(ax, mlp_x + mlp_w, 2.25, mlp_x + mlp_w, beta_corner_y,
         color=COL_MLP_EDGE)
# segment 2: right to sum (with arrow head)
add_arrow(ax, mlp_x + mlp_w, beta_corner_y, sum_cx - sum_r, sum_cy,
          color=COL_MLP_EDGE)
# beta(c) label sits on the vertical segment, clearly outside Q_Pi box
ax.text(mlp_x + mlp_w + 0.18, 2.55, r"$\beta(c)$",
        ha="left", va="center", fontsize=12, color=COL_TEXT)

# Q_Pi -> product node (vertical up, no redundant inner Q_Pi label)
add_arrow(ax, 9.175, 1.80 + row_h, prod_cx, prod_cy - prod_r,
          color=COL_QUERY_EDGE, lw=1.8)

# product node -> sum node (diagonal, only mild slope so the label fits cleanly)
add_arrow(ax, prod_cx + prod_r * 0.7, prod_cy - prod_r * 0.7,
          sum_cx - sum_r * 0.9, sum_cy + sum_r * 0.9,
          color=COL_MLP_EDGE)
# gamma(c) ⊙ Q_Pi label, placed above the diagonal, well clear of
# both the band annotation (further up) and the beta horizontal segment (below).
ax.text((prod_cx + sum_cx) / 2, (prod_cy + sum_cy) / 2 + 0.20,
        r"$\gamma(c)\odot Q_{\Pi}$",
        ha="center", va="center", fontsize=9.5, color="#444",
        rotation=-12)

# sum node -> Q'_Pi (diagonal down-right)
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
