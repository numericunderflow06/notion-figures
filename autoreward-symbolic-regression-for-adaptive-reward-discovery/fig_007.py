"""
fig_007: Planned Module Architecture for AutoReward.

Component diagram showing the top-level autoreward/ package container with
five sub-packages (rubric/, reward_discovery/, grpo/, adaptive_search/,
instantiation/ecg/) and their external library dependencies (PySR,
Transformers, PEFT, chronos-forecasting, OpenTSLM Flamingo).

Arrows are routed as right-angled polylines through the gaps between rows
so they never cut through sub-package text. The entry-point bar
(run_algorithm_1 / run_algorithm_2) is the only accent-colored element.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Style constants — grayscale palette + single accent color
# ---------------------------------------------------------------------------
ACCENT          = "#C0392B"
CONTAINER_FACE  = "#FAFAFA"
CONTAINER_EDGE  = "#2C3E50"
SUBPKG_FACE     = "#ECEFF1"
SUBPKG_EDGE     = "#37474F"
EXTLIB_FACE     = "#FFFFFF"
EXTLIB_EDGE     = "#546E7A"
ARROW_COLOR     = "#37474F"
TEXT_DARK       = "#1A1A1A"
TEXT_MUTED      = "#37474F"

fig, ax = plt.subplots(figsize=(15, 10), dpi=200)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(7.5, 9.55, "Planned Module Architecture",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=TEXT_DARK)
ax.text(7.5, 9.15,
        "autoreward/ — five sub-packages and their external library dependencies",
        ha="center", va="center",
        fontsize=11, color=TEXT_MUTED, style="italic")

# ---------------------------------------------------------------------------
# External libraries column (left side)
# ---------------------------------------------------------------------------
ext_libs = [
    ("PySR",                 8.20),
    ("Transformers",         7.05),
    ("PEFT",                 5.90),
    ("chronos-forecasting",  4.75),
    ("OpenTSLM Flamingo",    3.60),
]
ext_box_x = 0.35
ext_box_w = 2.55
ext_box_h = 0.75

ax.text(ext_box_x + ext_box_w / 2, 8.85,
        "External Libraries", ha="center", va="center",
        fontsize=11, fontweight="bold", color=TEXT_MUTED)

ext_anchor = {}
for name, y in ext_libs:
    box = FancyBboxPatch(
        (ext_box_x, y - ext_box_h / 2), ext_box_w, ext_box_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=EXTLIB_EDGE, facecolor=EXTLIB_FACE,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(ext_box_x + ext_box_w / 2, y, name,
            ha="center", va="center",
            fontsize=10.5, color=TEXT_DARK, fontweight="medium",
            zorder=4)
    ext_anchor[name] = (ext_box_x + ext_box_w, y)

# ---------------------------------------------------------------------------
# autoreward/ outer container
# ---------------------------------------------------------------------------
cont_x, cont_y = 4.40, 0.80
cont_w, cont_h = 10.25, 8.05

container = FancyBboxPatch(
    (cont_x, cont_y), cont_w, cont_h,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    linewidth=2.0, edgecolor=CONTAINER_EDGE, facecolor=CONTAINER_FACE,
    zorder=1,
)
ax.add_patch(container)
ax.text(cont_x + 0.30, cont_y + cont_h - 0.35, "autoreward/",
        ha="left", va="center",
        fontsize=13, fontweight="bold", color=CONTAINER_EDGE,
        family="monospace", zorder=4)

# ---------------------------------------------------------------------------
# Entry-point bar — the single accent-colored element
# ---------------------------------------------------------------------------
ep_x = cont_x + 0.45
ep_y = cont_y + cont_h - 1.65        # 7.20
ep_w = cont_w - 0.90
ep_h = 0.85

entry = FancyBboxPatch(
    (ep_x, ep_y), ep_w, ep_h,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.8, edgecolor=ACCENT, facecolor="#FDECEA",
    zorder=3,
)
ax.add_patch(entry)
ax.text(ep_x + 0.20, ep_y + ep_h - 0.22, "entry point",
        ha="left", va="center",
        fontsize=9.5, color=ACCENT, fontweight="bold", style="italic",
        zorder=4)
ax.text(ep_x + ep_w / 2, ep_y + 0.32,
        "run_algorithm_1(·)   |   run_algorithm_2(·)",
        ha="center", va="center",
        fontsize=12, color=ACCENT, fontweight="bold", family="monospace",
        zorder=4)

# ---------------------------------------------------------------------------
# Sub-package boxes — top row of 3, bottom row of 2
# ---------------------------------------------------------------------------
sub_h = 1.85
gap_x = 0.30

top_y_bottom = cont_y + 3.65         # = 4.45  (y of top-row box bottom)
top_y_top    = top_y_bottom + sub_h  # = 6.30  (y of top-row box top)

bot_y_bottom = cont_y + 0.85         # = 1.65  (y of bot-row box bottom)
bot_y_top    = bot_y_bottom + sub_h  # = 3.50  (y of bot-row box top)

top_w = (cont_w - 0.90 - 2 * gap_x) / 3.0   # ≈ 2.917
bot_w = (cont_w - 0.90 - gap_x) / 2.0       # ≈ 4.525

subpkgs = [
    {
        "name": "rubric/",
        "desc": ("G-Eval LLM judge\n"
                 "+ heuristic checks\n"
                 "+ ICC / Spearman calibration"),
        "x": cont_x + 0.45,
        "y": top_y_bottom,
        "w": top_w,
        "row": "top",
    },
    {
        "name": "reward_discovery/",
        "desc": ("PySR-based symbolic regression\n"
                 "discovering reward formulas\n"
                 "over rubric dimensions"),
        "x": cont_x + 0.45 + top_w + gap_x,
        "y": top_y_bottom,
        "w": top_w,
        "row": "top",
    },
    {
        "name": "grpo/",
        "desc": ("GRPO training loop\n"
                 "with discovered formula\n"
                 "as scalar reward signal"),
        "x": cont_x + 0.45 + 2 * (top_w + gap_x),
        "y": top_y_bottom,
        "w": top_w,
        "row": "top",
    },
    {
        "name": "adaptive_search/",
        "desc": ("Pareto-front adaptive\n"
                 "formula search across\n"
                 "complexity / quality budget"),
        "x": cont_x + 0.45,
        "y": bot_y_bottom,
        "w": bot_w,
        "row": "bot",
    },
    {
        "name": "instantiation/ecg/",
        "desc": ("Clinical ECG instantiation\n"
                 "dataset + Chronos-2 oracle\n"
                 "+ OpenTSLM Flamingo policy"),
        "x": cont_x + 0.45 + bot_w + gap_x,
        "y": bot_y_bottom,
        "w": bot_w,
        "row": "bot",
    },
]
subpkg_lookup = {p["name"]: p for p in subpkgs}

for pkg in subpkgs:
    box = FancyBboxPatch(
        (pkg["x"], pkg["y"]), pkg["w"], sub_h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=SUBPKG_EDGE, facecolor=SUBPKG_FACE,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(pkg["x"] + pkg["w"] / 2, pkg["y"] + sub_h - 0.32,
            pkg["name"],
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=TEXT_DARK,
            family="monospace", zorder=4)
    ax.plot([pkg["x"] + 0.20, pkg["x"] + pkg["w"] - 0.20],
            [pkg["y"] + sub_h - 0.58] * 2,
            color=SUBPKG_EDGE, linewidth=0.6, alpha=0.5, zorder=4)
    ax.text(pkg["x"] + pkg["w"] / 2, pkg["y"] + sub_h / 2 - 0.40,
            pkg["desc"],
            ha="center", va="center",
            fontsize=9.5, color=TEXT_MUTED, linespacing=1.35, zorder=4)

# ---------------------------------------------------------------------------
# Dependency arrows — multi-segment routing through inter-row gaps
# ---------------------------------------------------------------------------
# (lib, target sub-package, is_top_row)
dependencies = [
    ("PySR",                 "reward_discovery/",  True),
    ("Transformers",         "grpo/",              True),
    ("Transformers",         "instantiation/ecg/", False),
    ("PEFT",                 "grpo/",              True),
    ("PEFT",                 "instantiation/ecg/", False),
    ("chronos-forecasting",  "instantiation/ecg/", False),
    ("OpenTSLM Flamingo",    "instantiation/ecg/", False),
]

# Target_x distribution along each sub-package's top edge
target_x_map = {}
target_counts = {}
for _, pkg, _ in dependencies:
    target_counts[pkg] = target_counts.get(pkg, 0) + 1
target_idx = {pkg: 0 for pkg in target_counts}
for lib, pkg, _ in dependencies:
    n = target_counts[pkg]
    i = target_idx[pkg]
    target_idx[pkg] += 1
    box = subpkg_lookup[pkg]
    if n == 1:
        x_pos = box["x"] + box["w"] / 2.0
    else:
        x_pos = box["x"] + box["w"] * (i + 1) / (n + 1)
    target_x_map[(lib, pkg)] = x_pos

# Unique lane_x for each arrow (so concurrent verticals never overlap)
lane_xs = [4.45, 4.51, 4.57, 4.63, 4.69, 4.75, 4.81]
lane_x_map = {
    (lib, pkg): lane_xs[i]
    for i, (lib, pkg, _) in enumerate(dependencies)
}

# Highway y — distinct value per arrow to keep horizontals separated
# Top highway gap: y in (6.30, 7.20). Bot highway gap: y in (3.50, 4.45).
top_highway = {
    ("PySR",         "reward_discovery/"): 7.05,
    ("Transformers", "grpo/"):              6.75,
    ("PEFT",         "grpo/"):              6.45,
}
bot_highway = {
    ("Transformers",        "instantiation/ecg/"): 4.30,
    ("PEFT",                "instantiation/ecg/"): 4.05,
    ("chronos-forecasting", "instantiation/ecg/"): 3.80,
    ("OpenTSLM Flamingo",   "instantiation/ecg/"): 3.65,
}
highway_y_map = {**top_highway, **bot_highway}


def draw_routed_arrow(ax, points, color, lw=1.0, alpha=0.7, zorder=2):
    """Draw a polyline through `points` with an arrowhead at the end."""
    if len(points) < 2:
        return
    # Draw all segments except the last as a PathPatch
    if len(points) > 2:
        verts = points[:-1]
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
        path = Path(verts, codes)
        patch = PathPatch(path, edgecolor=color, facecolor="none",
                          linewidth=lw, alpha=alpha,
                          joinstyle="round", capstyle="round",
                          zorder=zorder)
        ax.add_patch(patch)
    # Draw the final segment with an arrowhead
    p1 = points[-2]
    p2 = points[-1]
    arrow = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=8,
        color=color, alpha=alpha, linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(arrow)


for lib, pkg, is_top in dependencies:
    sx, sy = ext_anchor[lib]
    lane_x = lane_x_map[(lib, pkg)]
    highway_y = highway_y_map[(lib, pkg)]
    target_x = target_x_map[(lib, pkg)]
    target_top_y = top_y_top if is_top else bot_y_top  # 6.30 or 3.50

    points = [
        (sx, sy),                # right edge of lib box
        (lane_x, sy),            # enter left-margin lane horizontally
        (lane_x, highway_y),     # ascend/descend within the lane
        (target_x, highway_y),   # traverse highway above the target
        (target_x, target_top_y) # drop into the sub-package's top edge
    ]
    draw_routed_arrow(ax, points, color=ARROW_COLOR, lw=1.05, alpha=0.78)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_elems = [
    mpatches.Patch(facecolor="#FDECEA", edgecolor=ACCENT,
                   label="Entry point (accent)"),
    mpatches.Patch(facecolor=SUBPKG_FACE, edgecolor=SUBPKG_EDGE,
                   label="Sub-package"),
    mpatches.Patch(facecolor=EXTLIB_FACE, edgecolor=EXTLIB_EDGE,
                   label="External library"),
    Line2D([0], [0], color=ARROW_COLOR, linewidth=1.2,
           label="depends on  →"),
]
ax.legend(
    handles=legend_elems,
    loc="lower left",
    bbox_to_anchor=(0.005, 0.005),
    fontsize=9, frameon=True, framealpha=0.92,
    edgecolor=TEXT_MUTED, ncol=1,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = ("/home/wangni/notion-figures/"
            "autoreward-symbolic-regression-for-adaptive-reward-discovery/"
            "fig_007.png")

plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print(f"Saved: {out_path}")
