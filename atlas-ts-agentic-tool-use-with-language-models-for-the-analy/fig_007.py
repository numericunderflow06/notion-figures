"""
fig_007: BTF — Five Semantic Families of Renders
Hierarchical diagram of the Behavior-Template Framework showing
the root node, five family nodes with counts, example leaves,
and the assertive fallback r_star as a side node.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def draw_box(ax, x, y, w, h, text, facecolor, edgecolor="#333333",
             fontsize=11, fontweight="bold", textcolor="#1a1a1a",
             linestyle="-", linewidth=1.4, boxstyle="round,pad=0.02,rounding_size=0.06"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=boxstyle,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=3)


def draw_badge(ax, x, y, count, facecolor):
    # Circular count badge
    badge = mpatches.Circle((x, y), 0.18, facecolor=facecolor,
                            edgecolor="#1a1a1a", linewidth=1.2, zorder=4)
    ax.add_patch(badge)
    ax.text(x, y, str(count), ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1a1a1a", zorder=5)


def draw_connector(ax, x1, y1, x2, y2, linestyle="-", color="#444444", lw=1.4):
    line = Line2D([x1, x2], [y1, y2], color=color,
                  linestyle=linestyle, linewidth=lw, zorder=1)
    ax.add_line(line)


def main():
    fig, ax = plt.subplots(figsize=(15, 9), dpi=200)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(7.5, 8.55,
            "BTF — Five Semantic Families of Renders",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color="#1a1a1a")
    ax.text(7.5, 8.15,
            "Behavior-Template Framework: 54 deterministic renders + 1 assertive fallback",
            ha="center", va="center", fontsize=11, color="#555555", style="italic")

    # --- Root node ---
    root_x, root_y = 7.5, 7.1
    draw_box(ax, root_x, root_y, 3.4, 0.75,
             "BTF  (55 renders)",
             facecolor="#E8E2F5", edgecolor="#4B3F72",
             fontsize=14, fontweight="bold", linewidth=1.8)

    # --- Five family nodes ---
    families = [
        {
            "name": "Confidence-\ndistribution",
            "count": 13,
            "color": "#CFE6F8",     # pastel blue
            "edge": "#3E78A8",
            "examples": [
                "top-1 disclosure",
                "top-k disclosure",
                "entropy disclosure",
            ],
        },
        {
            "name": "Question-\ntype",
            "count": 11,
            "color": "#FBD9D6",     # pastel pink
            "edge": "#B0504A",
            "examples": [
                "classification cue",
                "regression cue",
                "forecast cue",
            ],
        },
        {
            "name": "Evidence-\nknowledge",
            "count": 11,
            "color": "#D4ECD2",     # pastel green
            "edge": "#3F7A3A",
            "examples": [
                "tool evidence",
                "model knowledge",
                "mixed disclosure",
            ],
        },
        {
            "name": "Presentation-\nformat",
            "count": 10,
            "color": "#FCE9C2",     # pastel yellow
            "edge": "#9A7726",
            "examples": [
                "numeric format",
                "verbal format",
                "structured format",
            ],
        },
        {
            "name": "Hierarchy /\nOOD",
            "count": 9,
            "color": "#E5D7F2",     # pastel purple
            "edge": "#6E4A9C",
            "examples": [
                "hierarchical label",
                "OOD flag",
                "abstention cue",
            ],
        },
    ]

    # Layout family boxes
    fam_y = 5.3
    box_w, box_h = 2.25, 1.05
    n = len(families)
    # x positions evenly spaced
    margin_x = 1.0
    span_x = 15 - 2 * margin_x
    fam_xs = [margin_x + span_x * (i + 0.5) / n for i in range(n)]

    # Connectors from root to families
    for fx in fam_xs:
        draw_connector(ax, root_x, root_y - 0.38, fx, fam_y + box_h / 2,
                       linestyle="-", color="#666666", lw=1.3)

    # Draw family boxes + badges + examples
    leaf_color = "#2a2a2a"
    for fam, fx in zip(families, fam_xs):
        draw_box(ax, fx, fam_y, box_w, box_h, fam["name"],
                 facecolor=fam["color"], edgecolor=fam["edge"],
                 fontsize=12, fontweight="bold", linewidth=1.6)
        # Count badge in upper-right of family box
        draw_badge(ax, fx + box_w / 2 - 0.05, fam_y + box_h / 2 - 0.05,
                   fam["count"], facecolor="white")

        # Connector from family box to examples group
        examples_top_y = fam_y - box_h / 2 - 0.15
        draw_connector(ax, fx, fam_y - box_h / 2, fx, examples_top_y,
                       linestyle="-", color=fam["edge"], lw=1.0)

        # Examples leaf-box (light)
        ex_lines = fam["examples"]
        ex_text = "\n".join(f"• {e}" for e in ex_lines)
        # Box around examples
        ex_h = 0.32 * len(ex_lines) + 0.25
        ex_w = box_w + 0.1
        ex_y = examples_top_y - ex_h / 2
        draw_box(ax, fx, ex_y, ex_w, ex_h, ex_text,
                 facecolor="#FAFAFA", edgecolor=fam["edge"],
                 fontsize=9.5, fontweight="normal", textcolor=leaf_color,
                 linestyle="--", linewidth=1.0,
                 boxstyle="round,pad=0.02,rounding_size=0.05")

    # --- Side node: r_star (assertive fallback) ---
    rstar_x, rstar_y = 13.7, 7.1
    draw_box(ax, rstar_x, rstar_y, 2.0, 0.75,
             r"$r_{\star}$  (1 render)",
             facecolor="#EDEDED", edgecolor="#555555",
             fontsize=12, fontweight="bold",
             linestyle="--", linewidth=1.6)
    ax.text(rstar_x, rstar_y - 0.55,
            "assertive fallback",
            ha="center", va="center", fontsize=9, color="#555555", style="italic")

    # Dashed connector from root to r_star
    draw_connector(ax, root_x + 1.7, root_y, rstar_x - 1.0, rstar_y,
                   linestyle="--", color="#777777", lw=1.4)

    # --- Legend / footer ---
    # Count summation note
    ax.text(7.5, 0.55,
            "54 family renders  (13 + 11 + 11 + 10 + 9)  +  1 fallback  =  55 total",
            ha="center", va="center", fontsize=10.5, color="#333333",
            fontweight="bold")
    ax.text(7.5, 0.22,
            "Each family encodes a distinct semantic axis along which ATLAS-A may surface evidence to the frozen LLM.",
            ha="center", va="center", fontsize=9.5, color="#666666", style="italic")

    out_path = ("/home/wangni/notion-figures/"
                "atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_007.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
