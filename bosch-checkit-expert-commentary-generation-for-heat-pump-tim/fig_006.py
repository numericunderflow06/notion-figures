"""
fig_006: STS Cosine Similarity Distribution by Task

Per-sample STS cosine on the 76 test predictions, split by task
(space heating vs DHW). Computed from test_predictions.jsonl using
paraphrase-multilingual-MiniLM-L12-v2. The recomputed aggregates match
eval_metrics.json exactly:
  CH  (n=38): mean 92.29 / std 6.86  / min 63.90 / max 99.32
  DHW (n=38): mean 79.84 / std 12.23 / min 43.03 / max 97.22

Layout: split violin + jittered strip per task, with mean lines and a
stats annotation box. Colors match fig_005 / fig_007 (orange = Space
Heating, green = DHW).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DATA_PATH = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/sts_per_sample.json"
OUT_PATH = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_006.png"

# Bosch palette (matches fig_005)
CH_COLOR = "#E67E22"   # orange — Space Heating
DHW_COLOR = "#27AE60"  # green  — DHW
EDGE = "#222222"

with open(DATA_PATH) as f:
    data = json.load(f)

ch = np.array(data["space_heating"], dtype=float)
dhw = np.array(data["dhw"], dtype=float)


def stats(x):
    return dict(n=len(x), mean=x.mean(), std=x.std(ddof=0), mn=x.min(), mx=x.max())


s_ch = stats(ch)
s_dhw = stats(dhw)

fig, ax = plt.subplots(figsize=(11.0, 6.6), facecolor="white")
ax.set_facecolor("white")

positions = [1.0, 2.0]
parts = ax.violinplot(
    [ch, dhw],
    positions=positions,
    widths=0.78,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)
for body, color in zip(parts["bodies"], [CH_COLOR, DHW_COLOR]):
    body.set_facecolor(color)
    body.set_edgecolor(EDGE)
    body.set_alpha(0.55)
    body.set_linewidth(1.0)

# Box plot inside violin (median + IQR) for robust summary
bp = ax.boxplot(
    [ch, dhw],
    positions=positions,
    widths=0.12,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.0),
    boxprops=dict(facecolor="#333333", edgecolor="#333333"),
    whiskerprops=dict(color="#333333", linewidth=1.2),
    capprops=dict(color="#333333", linewidth=1.2),
    zorder=3,
)

# Jittered strip
rng = np.random.default_rng(42)
for pos, vals, color in [(positions[0], ch, CH_COLOR), (positions[1], dhw, DHW_COLOR)]:
    jitter = rng.uniform(-0.10, 0.10, size=len(vals))
    ax.scatter(
        pos + jitter,
        vals,
        s=26,
        color=color,
        edgecolor=EDGE,
        linewidth=0.5,
        alpha=0.85,
        zorder=4,
    )

# Mean lines
for pos, s, color in [(positions[0], s_ch, CH_COLOR), (positions[1], s_dhw, DHW_COLOR)]:
    ax.hlines(
        s["mean"],
        pos - 0.42,
        pos + 0.42,
        colors=color,
        linestyles="dashed",
        linewidth=2.0,
        zorder=5,
    )
    ax.annotate(
        f"mean = {s['mean']:.2f}",
        xy=(pos + 0.44, s["mean"]),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=10,
        color=color,
        fontweight="bold",
        va="center",
        ha="left",
    )

# Axis cosmetics
ax.set_xticks(positions)
ax.set_xticklabels(
    [
        f"Space Heating (Heizbetrieb)\nn = {s_ch['n']}",
        f"Domestic Hot Water (Warmwasser)\nn = {s_dhw['n']}",
    ],
    fontsize=11,
)
ax.set_xlim(0.45, 3.05)
ax.set_ylabel("STS cosine similarity (×100)", fontsize=12)
ax.set_ylim(35, 105)
ax.set_yticks(np.arange(40, 101, 10))
ax.yaxis.grid(True, linestyle=":", color="#BBBBBB", alpha=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=10)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")

# Reference bands (qualitative interpretation)
ax.axhspan(90, 105, color="#27AE60", alpha=0.06, zorder=0)
ax.axhspan(75, 90, color="#F1C40F", alpha=0.06, zorder=0)
ax.axhspan(35, 75, color="#C0392B", alpha=0.05, zorder=0)
ax.text(2.62, 96.0, "≥ 90\nstrong",     fontsize=8.5, color="#1E7E34", ha="left", va="center")
ax.text(2.62, 86.0, "75–90\nmoderate",  fontsize=8.5, color="#8A6D00", ha="left", va="center")
ax.text(2.62, 65.0, "< 75\nweak",       fontsize=8.5, color="#922B21", ha="left", va="center")

# Stats annotation boxes
def fmt(s):
    return (
        f"n     = {s['n']}\n"
        f"mean = {s['mean']:.2f}\n"
        f"std    = {s['std']:.2f}\n"
        f"min   = {s['mn']:.2f}\n"
        f"max  = {s['mx']:.2f}"
    )


bbox_ch = dict(boxstyle="round,pad=0.45", fc="white", ec=CH_COLOR, lw=1.4)
bbox_dhw = dict(boxstyle="round,pad=0.45", fc="white", ec=DHW_COLOR, lw=1.4)

ax.text(
    0.04, 0.04, "Space Heating\n" + fmt(s_ch),
    transform=ax.transAxes, fontsize=9.5, family="monospace",
    va="bottom", ha="left", color=EDGE, bbox=bbox_ch,
)
ax.text(
    0.28, 0.04, "DHW\n" + fmt(s_dhw),
    transform=ax.transAxes, fontsize=9.5, family="monospace",
    va="bottom", ha="left", color=EDGE, bbox=bbox_dhw,
)

# Annotate the DHW left tail / outlier (longest tail noted in spec)
dhw_min = float(np.min(dhw))
ax.annotate(
    f"DHW worst case ({dhw_min:.1f})",
    xy=(positions[1] + 0.02, dhw_min),
    xytext=(positions[1] + 0.32, dhw_min + 8),
    fontsize=9, color=DHW_COLOR,
    arrowprops=dict(arrowstyle="->", color=DHW_COLOR, lw=1.2),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=DHW_COLOR, lw=1.0),
)

# Highlight ΔMean gap
gap = s_ch["mean"] - s_dhw["mean"]
ax.annotate(
    "",
    xy=(1.50, s_ch["mean"]),
    xytext=(1.50, s_dhw["mean"]),
    arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4),
)
ax.text(
    1.52, (s_ch["mean"] + s_dhw["mean"]) / 2,
    f"Δ mean = {gap:.2f}",
    fontsize=10, color="#222222", fontweight="bold",
    va="center", ha="left",
    bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFFCC", ec="#888888", lw=0.8),
)

# Legend
legend_elems = [
    Patch(facecolor=CH_COLOR, edgecolor=EDGE, alpha=0.55, label="Space Heating"),
    Patch(facecolor=DHW_COLOR, edgecolor=EDGE, alpha=0.55, label="DHW"),
    plt.Line2D([0], [0], color="#333333", lw=2.0, label="median (box) / IQR"),
    plt.Line2D([0], [0], color=EDGE, lw=0, marker="o",
               markerfacecolor="#888888", markeredgecolor=EDGE,
               markersize=6, label="per-sample STS"),
    plt.Line2D([0], [0], color="#444444", lw=1.6, linestyle="--", label="task mean"),
]
ax.legend(
    handles=legend_elems, loc="upper right",
    bbox_to_anchor=(0.99, 0.99),
    fontsize=9, ncol=1,
    frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
)

# Titles
fig.suptitle(
    "STS Cosine Similarity Distribution by Task — Bosch CheckIt Run #1",
    fontsize=14, fontweight="bold", y=0.985,
)
ax.set_title(
    "Per-sample STS on the 76-system test split  ·  "
    "paraphrase-multilingual-MiniLM-L12-v2  ·  "
    "Space Heating tight near 90+, DHW wider with long left tail",
    fontsize=10.5, color="#555555", style="italic", pad=10,
)

fig.text(
    0.5, 0.005,
    "Computed from results/test_predictions.jsonl; aggregates match eval_metrics.json "
    "(CH 92.29 ± 6.86 / DHW 79.84 ± 12.23).",
    ha="center", va="bottom", fontsize=8.5, color="#666666", style="italic",
)

fig.tight_layout(rect=[0, 0.025, 1, 0.96])
fig.savefig(OUT_PATH, dpi=200, facecolor="white", bbox_inches="tight")
print(f"Saved figure to {OUT_PATH}")
