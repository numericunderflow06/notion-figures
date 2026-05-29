"""
fig_007: Factorized Cell Counts — Flamingo, 8 Cells × 6 Conditions.

Heatmap of the 8 factorized rationale-effect cells across 3 datasets x {gold, wrong}.
Counts are taken verbatim from §8.6 verified summary:
  source/results/summaries/opentslm_flamingo_prompt_v1_2026-04-27_rationale_effects_summary.json
Each (dataset, condition) column sums to <=50 (50 samples, missing cells = 0).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT_PATH = Path(
    "/home/wangni/notion-figures/"
    "ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_007.png"
)

CELLS = [
    "no_effect",
    "label_forcing",
    "full_follow",
    "knocked_off_away",
    "knocked_off_lateral",
    "reasoning_only_toward",
    "reasoning_only_lateral",
    "reasoning_only_away",
]

# (dataset, condition) -> {cell: count}, taken verbatim from the verified JSON.
COUNTS = {
    ("ecgqa", "gold"): {
        "label_forcing": 11, "full_follow": 19, "no_effect": 18,
        "reasoning_only_toward": 1, "knocked_off_away": 1,
    },
    ("ecgqa", "wrong"): {
        "no_effect": 15, "knocked_off_away": 4, "full_follow": 19,
        "reasoning_only_toward": 3, "knocked_off_lateral": 2,
        "reasoning_only_lateral": 1, "label_forcing": 6,
    },
    ("har", "gold"): {
        "full_follow": 3, "knocked_off_away": 7, "no_effect": 26,
        "label_forcing": 10, "reasoning_only_toward": 3,
        "reasoning_only_lateral": 1,
    },
    ("har", "wrong"): {
        "knocked_off_away": 3, "full_follow": 7, "label_forcing": 17,
        "no_effect": 23,
    },
    ("sleep", "gold"): {
        "no_effect": 20, "reasoning_only_away": 4, "reasoning_only_toward": 2,
        "full_follow": 6, "label_forcing": 5, "knocked_off_away": 12,
        "reasoning_only_lateral": 1,
    },
    ("sleep", "wrong"): {
        "label_forcing": 6, "full_follow": 20, "knocked_off_away": 12,
        "no_effect": 12,
    },
}

DATASETS = ["ecgqa", "har", "sleep"]
CONDITIONS = ["gold", "wrong"]
COLUMNS = [(d, c) for d in DATASETS for c in CONDITIONS]  # 6 columns


def build_matrix() -> np.ndarray:
    matrix = np.zeros((len(CELLS), len(COLUMNS)), dtype=int)
    for j, (dataset, condition) in enumerate(COLUMNS):
        for i, cell in enumerate(CELLS):
            matrix[i, j] = COUNTS[(dataset, condition)].get(cell, 0)
    return matrix


def main() -> None:
    matrix = build_matrix()
    vmax = 50  # max per spec — 50 samples per (dataset, condition)

    fig, ax = plt.subplots(figsize=(11.0, 7.2), dpi=200)
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(COLUMNS)))
    ax.set_xticklabels([c.upper() for _, c in COLUMNS], fontsize=11)
    ax.set_yticks(np.arange(len(CELLS)))
    ax.set_yticklabels(CELLS, fontsize=11)

    # Dataset group labels above the x-axis (one per pair of columns).
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([0.5, 2.5, 4.5])
    sec.set_xticklabels(
        [d.upper() for d in DATASETS], fontsize=13, fontweight="bold"
    )
    sec.tick_params(axis="x", which="both", length=0, pad=10)
    for spine in sec.spines.values():
        spine.set_visible(False)

    # Thin dataset separators on the heatmap (between pair-columns).
    for sep in (1.5, 3.5):
        ax.axvline(sep, color="white", linewidth=2.0)
        ax.axvline(sep, color="0.25", linewidth=0.6, linestyle="--")

    # Annotate cell values; pick a text color that contrasts with the colormap.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            text_color = "white" if v < vmax * 0.55 else "black"
            ax.text(
                j, i, str(v),
                ha="center", va="center",
                fontsize=11, color=text_color,
                fontweight="bold" if v > 0 else "normal",
            )

    ax.set_xticks(np.arange(-0.5, len(COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CELLS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Count (max = 50 per column)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Injected class signal", fontsize=12, labelpad=8)
    ax.set_ylabel("Factorized rationale-effect cell", fontsize=12, labelpad=8)

    ax.set_title(
        "Factorized Cell Counts — OpenTSLM-Flamingo (prompt v1, 2026-04-27)\n"
        "8 cells × (3 datasets × {gold, wrong}); 50 samples per column",
        fontsize=13, fontweight="bold", pad=46,
    )

    caption = (
        "Caveat: 'full_follow' = (answer flipped) ∧ (reasoning shifted toward injected); "
        "for wrong-class signals it strongly implies gullible-rationalization, but does not "
        "by itself distinguish genuine reconsideration from post-hoc label-driven rewriting "
        "(see §8.6, judge prompt v3). Empty cells indicate zero observed samples."
    )
    fig.text(
        0.5, 0.012, caption,
        ha="center", va="bottom",
        fontsize=8.5, color="0.25", style="italic", wrap=True,
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
