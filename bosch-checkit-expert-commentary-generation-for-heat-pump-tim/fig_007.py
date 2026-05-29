"""fig_007: Generated vs Gold Length (Brevity Asymmetry).

Scatter of per-sample generated text length vs gold reference length, colored
by task (space heating / DHW), with y=x parity line and per-task brevity-
penalty annotations near the cluster centroids. Marginal histograms show the
length distribution on each axis.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Load the 76 test predictions
# ---------------------------------------------------------------------------
PRED_PATH = Path("/home/wangni/bosch-checkit/results/test_predictions.jsonl")

records = [json.loads(line) for line in PRED_PATH.read_text().splitlines() if line.strip()]
assert len(records) == 76, f"Expected 76 test entries, found {len(records)}"


def task_of(rec: dict) -> str:
    """Classify a record as CH (space heating) or DHW.

    The post_prompt names the commentary task explicitly. We look for the
    German keyword 'Trinkwasser' (DHW) vs 'Heizung'/'Heizkreis' (space heating).
    Fallback heuristic: gold text mentioning Trinkwasser → DHW.
    """
    text = " ".join(
        str(rec.get(k, "")) for k in ("pre_prompt", "post_prompt", "gold")
    ).lower()
    if "trinkwasser" in text or "warmwasser" in text or "dhw" in text:
        return "DHW"
    return "CH"


gold_len = np.array([len(r["gold"]) for r in records], dtype=float)
gen_len = np.array([len(r["generated"]) for r in records], dtype=float)
tasks = np.array([task_of(r) for r in records])

ch_mask = tasks == "CH"
dhw_mask = tasks == "DHW"

# Sanity-check against the spec's aggregate stats (gold CH mean ≈396, DHW ≈312)
print(
    f"CH samples: {ch_mask.sum():d}  gold mean={gold_len[ch_mask].mean():.0f}  "
    f"gen mean={gen_len[ch_mask].mean():.0f}"
)
print(
    f"DHW samples: {dhw_mask.sum():d}  gold mean={gold_len[dhw_mask].mean():.0f}  "
    f"gen mean={gen_len[dhw_mask].mean():.0f}"
)

# Brevity penalty values from the spec (computed offline by evaluate_bosch.py)
BP_CH = 0.8378
BP_DHW = 0.5145

# Colors consistent with fig_005 / fig_006 (CH = warm orange, DHW = cool blue)
COLOR_CH = "#E07A2A"
COLOR_DHW = "#1F77B4"

# ---------------------------------------------------------------------------
# Figure layout: scatter + marginal histograms
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8), facecolor="white")
gs = GridSpec(
    2,
    2,
    width_ratios=[4.2, 1.0],
    height_ratios=[1.0, 4.2],
    hspace=0.04,
    wspace=0.04,
    left=0.10,
    right=0.97,
    top=0.93,
    bottom=0.10,
)

ax_main = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

# Axis limits with padding so annotations are not clipped
lo = 0.0
hi = float(max(gold_len.max(), gen_len.max())) * 1.08
ax_main.set_xlim(lo, hi)
ax_main.set_ylim(lo, hi)

# ----- main scatter -----
ax_main.plot(
    [lo, hi],
    [lo, hi],
    linestyle="--",
    color="#777777",
    linewidth=1.2,
    zorder=1,
    label="y = x (parity)",
)

ax_main.scatter(
    gold_len[ch_mask],
    gen_len[ch_mask],
    s=70,
    color=COLOR_CH,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.85,
    label=f"Space heating (CH), n={ch_mask.sum()}",
    zorder=3,
)
ax_main.scatter(
    gold_len[dhw_mask],
    gen_len[dhw_mask],
    s=70,
    color=COLOR_DHW,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.85,
    label=f"DHW (hot water), n={dhw_mask.sum()}",
    zorder=3,
)

# ----- centroid markers + brevity-penalty annotations -----
ch_cx, ch_cy = gold_len[ch_mask].mean(), gen_len[ch_mask].mean()
dhw_cx, dhw_cy = gold_len[dhw_mask].mean(), gen_len[dhw_mask].mean()

for cx, cy, color in [(ch_cx, ch_cy, COLOR_CH), (dhw_cx, dhw_cy, COLOR_DHW)]:
    ax_main.scatter(
        [cx], [cy],
        s=260,
        facecolor="white",
        edgecolor=color,
        linewidth=2.4,
        marker="X",
        zorder=4,
    )

ax_main.annotate(
    f"CH centroid\n({ch_cx:.0f}, {ch_cy:.0f})\nBP = {BP_CH:.4f}",
    xy=(ch_cx, ch_cy),
    xytext=(ch_cx + 90, ch_cy - 110),
    fontsize=10,
    color="#5a3210",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="#fff3e6",
        edgecolor=COLOR_CH,
        linewidth=1.0,
    ),
    arrowprops=dict(arrowstyle="-", color=COLOR_CH, lw=1.0),
    zorder=5,
)
ax_main.annotate(
    f"DHW centroid\n({dhw_cx:.0f}, {dhw_cy:.0f})\nBP = {BP_DHW:.4f}",
    xy=(dhw_cx, dhw_cy),
    xytext=(dhw_cx - 220, dhw_cy + 130),
    fontsize=10,
    color="#0f3b66",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="#e6f0fa",
        edgecolor=COLOR_DHW,
        linewidth=1.0,
    ),
    arrowprops=dict(arrowstyle="-", color=COLOR_DHW, lw=1.0),
    zorder=5,
)

# Quadrant shading: below the diagonal = under-generation
under_x = np.array([lo, hi, hi])
under_y = np.array([lo, hi, lo])
ax_main.fill(under_x, under_y, color="#cccccc", alpha=0.08, zorder=0)
ax_main.text(
    hi * 0.78,
    hi * 0.10,
    "under-generation\n(generated < gold)",
    fontsize=9,
    color="#555555",
    style="italic",
    ha="center",
    va="center",
    zorder=2,
)

ax_main.set_xlabel("Gold reference length (characters)", fontsize=12)
ax_main.set_ylabel("Generated length (characters)", fontsize=12)
ax_main.grid(True, linestyle=":", linewidth=0.6, color="#bbbbbb", alpha=0.7)
ax_main.set_axisbelow(True)

legend = ax_main.legend(
    loc="upper left",
    fontsize=10,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
)

# ----- marginal histograms -----
bins = np.linspace(lo, hi, 26)

ax_top.hist(
    gold_len[ch_mask], bins=bins, color=COLOR_CH, alpha=0.55,
    edgecolor="white", linewidth=0.5, label="CH gold",
)
ax_top.hist(
    gold_len[dhw_mask], bins=bins, color=COLOR_DHW, alpha=0.55,
    edgecolor="white", linewidth=0.5, label="DHW gold",
)
ax_top.set_ylabel("count", fontsize=9)
ax_top.tick_params(axis="x", labelbottom=False, length=0)
ax_top.tick_params(axis="y", labelsize=8)
ax_top.spines["top"].set_visible(False)
ax_top.spines["right"].set_visible(False)
ax_top.set_title(
    "Generated vs Gold Length per Test Sample (n = 76) — Brevity Asymmetry",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

ax_right.hist(
    gen_len[ch_mask], bins=bins, color=COLOR_CH, alpha=0.55,
    edgecolor="white", linewidth=0.5, orientation="horizontal",
)
ax_right.hist(
    gen_len[dhw_mask], bins=bins, color=COLOR_DHW, alpha=0.55,
    edgecolor="white", linewidth=0.5, orientation="horizontal",
)
ax_right.set_xlabel("count", fontsize=9)
ax_right.tick_params(axis="y", labelleft=False, length=0)
ax_right.tick_params(axis="x", labelsize=8)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# ----- subtitle / interpretation strip -----
fig.text(
    0.10,
    0.025,
    "BP = BLEU brevity penalty (exp(1 − r/c) when c < r).  "
    "CH stays near parity (BP ≈ 0.84); DHW outputs are ~half the reference length (BP ≈ 0.51).",
    fontsize=9,
    color="#444444",
    ha="left",
)

# ---------------------------------------------------------------------------
out_path = Path(
    "/home/wangni/notion-figures/"
    "bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_007.png"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"saved {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
