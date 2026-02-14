"""
fig_006: Parameter Budget Breakdown
Two-level visualization:
  (1) Main bar chart comparing Encoder (~1.5M), TPA Total (~4.3M), LLM (~1B)
      with broken y-axis to handle the ~230x scale difference.
  (2) Inset donut chart breaking TPA into ATPE (~16K), Anchor Injector (~260K),
      Cross-Attention (~4M) with manually placed labels to avoid overlap.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
main_labels = ["Encoder", "TPA (Total)", "Frozen LLM\n(Llama-3.2-1B)"]
main_values = [1.5, 4.3, 1000.0]  # in millions

tpa_labels = ["ATPE", "Anchor Injector", "Cross-Attention"]
tpa_values_k = [16, 260, 4000]  # in thousands
tpa_pcts = [v / sum(tpa_values_k) * 100 for v in tpa_values_k]

# ── Colors ────────────────────────────────────────────────────────────────────
color_encoder = "#4A90C4"      # steel blue
color_tpa_total = "#E07B54"    # warm orange
color_llm = "#6BAF7B"          # sage green

color_atpe = "#FFC857"         # golden yellow
color_anchor = "#E8945A"       # medium orange
color_crossattn = "#C44E52"    # warm red

tpa_colors = [color_atpe, color_anchor, color_crossattn]
main_colors = [color_encoder, color_tpa_total, color_llm]

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7), facecolor="white")

# Gridspec: left = broken bar chart (2 rows), right = donut chart (spanning both)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[2.5, 1],
                      hspace=0.06, wspace=0.32,
                      left=0.07, right=0.96, top=0.89, bottom=0.08)

# ── Left: broken y-axis bar chart ────────────────────────────────────────────
ax_top = fig.add_subplot(gs[0, 0])
ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

bar_x = np.arange(len(main_labels))
bar_width = 0.50

for ax in [ax_top, ax_bot]:
    bars = ax.bar(bar_x, main_values, bar_width, color=main_colors,
                  edgecolor="white", linewidth=1.8, zorder=3)

# Top portion: only shows the LLM bar
ax_top.set_ylim(850, 1080)
ax_top.set_yticks([900, 1000])
ax_top.set_yticklabels(["900M", "1,000M"], fontsize=10)

# Bottom portion: shows Encoder and TPA bars
ax_bot.set_ylim(0, 10)
ax_bot.set_yticks([0, 2, 4, 6, 8, 10])
ax_bot.set_yticklabels(["0", "2M", "4M", "6M", "8M", "10M"], fontsize=10)

# Hide spines between the two axes
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.tick_params(bottom=False, labelbottom=False)

# Diagonal break marks
d = 0.015
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

# Annotate bars with exact parameter counts
annotations = ["~1.5M", "~4.3M", "~1B"]
ann_colors = ["#2E5C82", "#B85A33", "#3E7A4E"]  # darker versions

# Encoder and TPA labels on bottom axes
for i in range(2):
    ax_bot.annotate(annotations[i],
                    xy=(bar_x[i], main_values[i]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color=ann_colors[i])

# LLM label on top axis
ax_top.annotate(annotations[2],
                xy=(bar_x[2], main_values[2]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=ann_colors[2])

# X-axis labels
ax_bot.set_xticks(bar_x)
ax_bot.set_xticklabels(main_labels, fontsize=11, fontweight="medium")

# Y-axis label
fig.text(0.02, 0.50, "Number of Parameters", rotation=90,
         va="center", ha="center", fontsize=12, fontweight="medium")

# Grid
for ax in [ax_top, ax_bot]:
    ax.yaxis.grid(True, alpha=0.25, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

ax_top.set_title("Overall Parameter Budget", fontsize=14, fontweight="bold", pad=14)

# ── Right: TPA donut chart with manual label placement ───────────────────────
ax_donut = fig.add_subplot(gs[:, 1])

# Create donut chart (pie with a white center)
wedges, _ = ax_donut.pie(
    tpa_values_k,
    startangle=90,
    colors=tpa_colors,
    wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2.5),
    radius=1.0,
)

# Center text (inside the donut hole)
ax_donut.text(0, 0.08, "Total", ha="center", va="center",
              fontsize=12, color="#666666", fontweight="medium", zorder=10)
ax_donut.text(0, -0.12, "~4.3M", ha="center", va="center",
              fontsize=17, color="#333333", fontweight="bold", zorder=10)

# Manual label positions to avoid overlap
# ATPE is tiny (0.4%), Anchor is small (6.0%), Cross-Attn is dominant (93.0%)
label_info = [
    # (label_text, xytext position, color)
    ("ATPE\n~16K (0.4%)",        (1.45, 1.05),  color_atpe),
    ("Anchor Injector\n~260K (6.0%)", (1.45, 0.45), color_anchor),
    ("Cross-Attention\n~4M (93.0%)",  (1.35, -0.85), color_crossattn),
]

for i, (wedge, (label, xytext, color)) in enumerate(zip(wedges, label_info)):
    # Compute the midpoint angle of each wedge for the arrow source
    ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
    x_src = 0.78 * np.cos(ang)
    y_src = 0.78 * np.sin(ang)

    ax_donut.annotate(
        label,
        xy=(x_src, y_src),
        xytext=xytext,
        fontsize=10.5, fontweight="semibold",
        ha="left", va="center",
        color="#333333",
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=1.8,
            connectionstyle="arc3,rad=-0.1" if i < 2 else "arc3,rad=0.1",
        ),
        bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.15, ec="none"),
    )

ax_donut.set_title("TPA Component Breakdown", fontsize=14, fontweight="bold", pad=18)

# ── Connection arrow from TPA bar to donut chart ─────────────────────────────
con = mpatches.FancyArrowPatch(
    posA=(0.49, 0.48), posB=(0.55, 0.48),
    arrowstyle="->,head_width=5,head_length=4",
    color=color_tpa_total, lw=2.0, alpha=0.6,
    transform=fig.transFigure, clip_on=False,
)
fig.patches.append(con)
fig.text(0.52, 0.51, "breakdown", ha="center", va="bottom",
         fontsize=9, fontstyle="italic", color=color_tpa_total, alpha=0.7)

# ── Styling ──────────────────────────────────────────────────────────────────
for ax in [ax_top, ax_bot]:
    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

# Main figure title
fig.suptitle("Parameter Budget Breakdown", fontsize=16, fontweight="bold",
             y=0.96, color="#222222")

plt.savefig("/home/wangni/notion-figures/alignment/fig_006.png",
            dpi=200, facecolor="white", bbox_inches="tight")
plt.close()
print("Saved: /home/wangni/notion-figures/alignment/fig_006.png")
