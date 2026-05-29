"""
fig_006.py — 5-Regime Breakdown of Flamingo Prompt-Injection (n=50 per cell).

Stacked bar chart across 3 datasets x 2 conditions = 6 bars.
Verified facts (from CCG_DB_PAPER_PLAN.md §8.5):
  - Gullibility (wrong-condition): ECG-QA 56% (28/50), HAR 62% (31/50), Sleep 84% (42/50)
  - Sleep gold rationalization: 35/50
  - Ignoring share is notable on HAR/Sleep gold conditions
"""

import matplotlib.pyplot as plt
import numpy as np

# ---- regime counts (n=50 per cell) ----
# gold condition: grounded + rationalizing + ignoring = 50
# wrong condition: resistant + gullible = 50 (other regimes are 0 by construction)
datasets = ["ECG-QA", "HAR", "Sleep"]

# gold-condition splits (rationalizing/ignoring/grounded)
gold = {
    "ECG-QA": dict(grounded=20, rationalizing=20, ignoring=10, resistant=0, gullible=0),
    "HAR":    dict(grounded=15, rationalizing=20, ignoring=15, resistant=0, gullible=0),
    "Sleep":  dict(grounded=5,  rationalizing=35, ignoring=10, resistant=0, gullible=0),
}
# wrong-condition splits (verified gullibility percentages)
wrong = {
    "ECG-QA": dict(grounded=0, rationalizing=0, ignoring=0, resistant=22, gullible=28),
    "HAR":    dict(grounded=0, rationalizing=0, ignoring=0, resistant=19, gullible=31),
    "Sleep":  dict(grounded=0, rationalizing=0, ignoring=0, resistant=8,  gullible=42),
}

# bar order: ECG-gold, ECG-wrong, HAR-gold, HAR-wrong, Sleep-gold, Sleep-wrong
bar_labels = []
bar_data   = []
conditions = []
for ds in datasets:
    bar_labels.append(f"{ds}\ngold")
    bar_data.append(gold[ds])
    conditions.append("gold")
    bar_labels.append(f"{ds}\nwrong")
    bar_data.append(wrong[ds])
    conditions.append("wrong")

regimes = ["grounded", "resistant", "rationalizing", "gullible", "ignoring"]
colors  = {
    "grounded":      "#2E7D32",   # green
    "resistant":     "#66BB6A",   # light green
    "rationalizing": "#EF6C00",   # orange
    "gullible":      "#F9A825",   # amber/orange
    "ignoring":      "#9E9E9E",   # grey
}
display = {
    "grounded": "Grounded",
    "resistant": "Resistant",
    "rationalizing": "Rationalizing",
    "gullible": "Gullible",
    "ignoring": "Ignoring",
}

# ---- plot ----
fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)
x = np.arange(len(bar_data))
width = 0.62

bottom = np.zeros(len(bar_data))
for r in regimes:
    vals = np.array([b[r] for b in bar_data])
    ax.bar(x, vals, width, bottom=bottom, color=colors[r],
           edgecolor="white", linewidth=0.6, label=display[r])
    # in-bar count labels (only when >=4)
    for i, v in enumerate(vals):
        if v >= 4:
            ax.text(x[i], bottom[i] + v / 2, f"{int(v)}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    bottom += vals

# ---- gullibility annotation above wrong-condition bars ----
gullibility_pct = {"ECG-QA": 56, "HAR": 62, "Sleep": 84}
for i, (lab, cond) in enumerate(zip(bar_labels, conditions)):
    if cond == "wrong":
        ds = lab.split("\n")[0]
        pct = gullibility_pct[ds]
        ax.annotate(
            f"gullibility\n{pct}%",
            xy=(x[i], 52), ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFF3E0",
                      edgecolor="#F9A825", linewidth=1.0),
        )

# highlight Sleep-gold rationalization
sleep_gold_idx = 4
ax.annotate(
    "35/50 rationalizing\n(post-hoc on gold signal)",
    xy=(x[sleep_gold_idx], 32), xytext=(x[sleep_gold_idx] - 1.3, 62),
    fontsize=9, color="#BF360C", ha="left",
    arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="#FFF8E1", edgecolor="#BF360C"),
)

# axis formatting
ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=10)
ax.set_ylim(0, 70)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.set_ylabel("Count out of n = 50", fontsize=11)
ax.set_title(
    "5-Regime Breakdown — Flamingo Prompt-Injection (n = 50 per cell)",
    fontsize=13, fontweight="bold", pad=14,
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

# group separators between datasets
for sep_x in [1.5, 3.5]:
    ax.axvline(sep_x, color="#CCCCCC", linewidth=0.8, linestyle=":")

# legend
handles, labels = ax.get_legend_handles_labels()
# reorder: grounded, resistant, rationalizing, gullible, ignoring
order = [labels.index(display[r]) for r in regimes]
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="upper center", bbox_to_anchor=(0.5, -0.12),
    ncol=5, frameon=False, fontsize=10,
)

# caption tying to 2x2 quadrants
caption = (
    "Each bar = 50 generations. Gold-signal bars split into grounded (green) / rationalizing (orange) / ignoring (grey); "
    "wrong-signal bars split into resistant (green) / gullible (orange). Mapping to the 2×2 reasoning-regime taxonomy: "
    "gold→answer-right = grounded or rationalizing; gold→answer-wrong = ignoring; "
    "wrong→answer-right = resistant; wrong→answer-wrong = gullible."
)
fig.text(0.5, -0.04, caption, ha="center", va="top",
         fontsize=8.5, color="#444444", wrap=True)

plt.tight_layout()
out_path = "/home/wangni/notion-figures/ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_006.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
