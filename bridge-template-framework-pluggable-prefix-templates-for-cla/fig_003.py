"""Figure 3: Priority Tier Hierarchy for the Bridge Template Framework.

Vertical band diagram of the 5 priority tiers used by the cascade in
`bridge_templates/base.py`. REJECT (10) sits on top — it is evaluated
first — and FALLBACK (1e9) sits at the bottom as the always-firing
unconditional assertive top-1 template. Counts and example names come
from the source modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap


OUTPUT = (
    "/home/wangni/notion-figures/"
    "bridge-template-framework-pluggable-prefix-templates-for-cla/fig_003.png"
)


TIERS = [
    {
        "name": "PRIORITY_REJECT",
        "value": 10,
        "count": 2,
        "summary": "rejection / OOD short-circuits",
        "examples": ["ood_reject", "hierarchy_plus_ood"],
        "color": "#C0392B",       # urgent red
        "text_color": "white",
    },
    {
        "name": "PRIORITY_HIGH",
        "value": 100,
        "count": 14,
        "summary": "confidence- and entropy-conditioned phrasings",
        "examples": [
            "high_confidence_calibrated",
            "entropy_ambiguous",
            "coarse_to_fine",
        ],
        "color": "#E67E22",       # orange
        "text_color": "white",
    },
    {
        "name": "PRIORITY_MEDIUM",
        "value": 200,
        "count": 12,
        "summary": "question-type-aware phrasings + hedged top-2",
        "examples": [
            "verify_affirmative",
            "choose_pairwise",
            "hedged_top2",
        ],
        "color": "#BDC3C7",       # neutral grey
        "text_color": "#2C3E50",
    },
    {
        "name": "PRIORITY_LOW",
        "value": 300,
        "count": 26,
        "summary": "evidence, presentation, dataset / hierarchy framing",
        "examples": [
            "feature_grounded",
            "json_structured",
            "dataset_prefix",
        ],
        "color": "#D4A24C",       # muted gold
        "text_color": "white",
    },
    {
        "name": "PRIORITY_FALLBACK",
        "value": 10**9,
        "count": 1,
        "summary": "unconditional assertive top-1 (cascade terminator)",
        "examples": ["assertive_top1"],
        "color": "#B8860B",       # warm gold
        "text_color": "white",
    },
]


def format_value(v: int) -> str:
    if v >= 10**6:
        return "10⁹"
    return str(v)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Title
    ax.text(
        6.0, 9.55,
        "Priority Tier Hierarchy",
        ha="center", va="center",
        fontsize=20, fontweight="bold", color="#2C3E50",
    )
    ax.text(
        6.0, 9.10,
        "Cascade walks REGISTRY ∪ {ASSERTIVE_FALLBACK} sorted by priority — "
        "lower values are inspected first",
        ha="center", va="center",
        fontsize=11, style="italic", color="#5D6D7E",
    )

    # Layout
    band_left = 0.6
    band_right = 11.4
    band_width = band_right - band_left
    top_y = 8.45
    bot_y = 0.50
    n = len(TIERS)
    band_height = (top_y - bot_y) / n - 0.10  # small gap

    # Draw the "evaluated first → last" arrow on the left
    arrow = FancyArrowPatch(
        (0.30, top_y - 0.05),
        (0.30, bot_y + 0.05),
        arrowstyle="-|>", mutation_scale=18,
        color="#34495E", lw=1.8,
    )
    ax.add_patch(arrow)
    ax.text(
        0.18, (top_y + bot_y) / 2,
        "cascade order",
        rotation=90, ha="center", va="center",
        fontsize=10, color="#34495E",
    )

    # Right-side annotation: top vs bottom
    ax.text(
        11.75, top_y - 0.15,
        "evaluated\nfirst",
        ha="center", va="top",
        fontsize=9, color="#7F8C8D",
    )
    ax.text(
        11.75, bot_y + 0.15,
        "always\nfires",
        ha="center", va="bottom",
        fontsize=9, color="#7F8C8D",
    )

    # Draw each band
    for i, tier in enumerate(TIERS):
        y_top = top_y - i * (band_height + 0.10)
        y_bot = y_top - band_height
        y_mid = (y_top + y_bot) / 2

        # Band background
        box = FancyBboxPatch(
            (band_left, y_bot),
            band_width, band_height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#2C3E50",
            facecolor=tier["color"],
            alpha=0.92,
        )
        ax.add_patch(box)

        # Left block: tier name + integer value
        ax.text(
            band_left + 0.30, y_mid + 0.22,
            tier["name"],
            ha="left", va="center",
            fontsize=14, fontweight="bold",
            color=tier["text_color"],
            family="monospace",
        )
        ax.text(
            band_left + 0.30, y_mid - 0.25,
            f"value = {format_value(tier['value'])}",
            ha="left", va="center",
            fontsize=11,
            color=tier["text_color"],
            family="monospace",
        )

        # Middle: count badge
        count_cx = band_left + 3.45
        ax.text(
            count_cx, y_mid + 0.20,
            f"{tier['count']}",
            ha="center", va="center",
            fontsize=22, fontweight="bold",
            color=tier["text_color"],
        )
        ax.text(
            count_cx, y_mid - 0.28,
            "template" + ("s" if tier["count"] != 1 else ""),
            ha="center", va="center",
            fontsize=9,
            color=tier["text_color"],
        )

        # Right side: summary + example template names
        text_left = band_left + 4.10
        ax.text(
            text_left, y_mid + 0.30,
            tier["summary"],
            ha="left", va="center",
            fontsize=10.5, style="italic",
            color=tier["text_color"],
        )
        examples_line = "  •  ".join(tier["examples"])
        ax.text(
            text_left, y_mid - 0.25,
            examples_line,
            ha="left", va="center",
            fontsize=10,
            color=tier["text_color"],
            family="monospace",
        )

    # Footer: source attribution
    ax.text(
        6.0, 0.12,
        "Source: bridge_templates/base.py (PRIORITY_* constants) and "
        "register(...) calls across confidence/qtype/evidence/presentation/hierarchy",
        ha="center", va="center",
        fontsize=8.5, color="#7F8C8D",
    )

    plt.savefig(OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
