import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Data from verified facts (section 5.3) ---
strategies = [
    "Full curriculum\n(through Stage 6)",
    "HAR-flamingo",
    "Sleep-flamingo",
    "From scratch",
]
zuco1_acc = [70.51, 69.65, 69.25, 68.63]
zuco2_acc = [59.95, 59.59, 68.29, 62.25]

# --- Layout ---
fig, ax = plt.subplots(figsize=(9, 5.5))

x = np.arange(len(strategies))
bar_width = 0.32
gap = 0.04  # half-gap between paired bars

bars1 = ax.bar(
    x - bar_width / 2 - gap,
    zuco1_acc,
    bar_width,
    label="ZuCo 1.0",
    color="#3A7CA5",
    edgecolor="white",
    linewidth=0.6,
    zorder=3,
)
bars2 = ax.bar(
    x + bar_width / 2 + gap,
    zuco2_acc,
    bar_width,
    label="ZuCo 2.0",
    color="#E8833A",
    edgecolor="white",
    linewidth=0.6,
    zorder=3,
)

# --- Value labels on bars ---
def add_labels(bars, bold_idx=None):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        weight = "bold" if i == bold_idx else "normal"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.35,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight=weight,
            color="#222222",
        )

add_labels(bars1, bold_idx=0)  # bold the best ZuCo 1.0 result
add_labels(bars2)

# --- Horizontal reference line at 70% ---
ax.axhline(y=70.0, color="#888888", linestyle="--", linewidth=0.9, zorder=2)
ax.text(
    len(strategies) - 0.5 + 0.05,
    70.15,
    "70%",
    va="bottom",
    ha="left",
    fontsize=9,
    color="#666666",
)

# --- Annotation for best result ---
best_bar = bars1[0]
ax.annotate(
    "Best ZuCo 1.0\naccuracy",
    xy=(best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
    xytext=(best_bar.get_x() + best_bar.get_width() / 2 + 0.55, 73.2),
    fontsize=9,
    fontweight="bold",
    color="#3A7CA5",
    ha="center",
    arrowprops=dict(
        arrowstyle="-|>",
        color="#3A7CA5",
        lw=1.2,
        connectionstyle="arc3,rad=-0.15",
    ),
)

# --- Axes formatting ---
ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=10.5)
ax.set_ylabel("Classification Accuracy (%)", fontsize=11.5, labelpad=8)
ax.set_title(
    "Downstream Transfer Learning: ZuCo Classification Accuracy",
    fontsize=13,
    fontweight="bold",
    pad=14,
)

# y-axis range and grid
ax.set_ylim(55, 76)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
ax.grid(axis="y", which="major", linestyle="-", linewidth=0.4, color="#d0d0d0", zorder=0)
ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.3, color="#e4e4e4", zorder=0)

# Clean spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#aaaaaa")
ax.spines["bottom"].set_color("#aaaaaa")
ax.tick_params(axis="both", colors="#444444", labelsize=10)

# Legend
ax.legend(
    fontsize=10.5,
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
)

# Subtitle / caption
fig.text(
    0.5,
    0.01,
    "Full curriculum (through Stage 6, financial data) achieves the highest ZuCo 1.0 accuracy,\n"
    "demonstrating cross-domain transfer from financial report training.",
    ha="center",
    fontsize=9.5,
    color="#555555",
    style="italic",
)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(
    "/home/wangni/notion-figures/finance/fig_007.png",
    dpi=200,
    facecolor="white",
    bbox_inches="tight",
)
plt.close()
print("Saved: /home/wangni/notion-figures/finance/fig_007.png")
