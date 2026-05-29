"""
fig_004: Training & Validation Loss (Run #1)
Per-epoch train and validation cross-entropy over the 9 epochs of Run #1,
with markers at the best epoch (4) and the early-stop trigger (9).
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data from code/results/loss_history.txt
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
train_loss = [1.357889, 0.547931, 0.388702, 0.294428,
              0.231066, 0.180978, 0.157094, 0.124758, 0.107078]
val_loss = [0.767319, 0.666477, 0.650798, 0.638826,
            0.678126, 0.674294, 0.690632, 0.710141, 0.721832]

best_epoch = 4
best_val = 0.638826
early_stop_epoch = 9
patience = 5

# Style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# High-contrast colors
TRAIN_COLOR = "#1f77b4"   # blue
VAL_COLOR   = "#d62728"   # red
BEST_COLOR  = "#2ca02c"   # green
STOP_COLOR  = "#7f7f7f"   # gray

# Lines
ax.plot(epochs, train_loss, marker="o", linewidth=2.2, color=TRAIN_COLOR,
        label="Train loss", markersize=6, zorder=3)
ax.plot(epochs, val_loss, marker="s", linewidth=2.2, color=VAL_COLOR,
        label="Validation loss", markersize=6, zorder=3)

# Star marker for best epoch
ax.plot(best_epoch, best_val, marker="*", markersize=22, color=BEST_COLOR,
        markeredgecolor="black", markeredgewidth=0.8,
        linestyle="None", label=f"Best (saved): epoch {best_epoch}, val={best_val:.3f}",
        zorder=5)

# Vertical dashed line for early stop
ax.axvline(x=early_stop_epoch, color=STOP_COLOR, linestyle="--",
           linewidth=1.8, alpha=0.85, zorder=2,
           label=f"Early stop (patience={patience})")

# Annotations
ax.annotate(
    f"Best (saved)\nepoch {best_epoch}, val={best_val:.3f}",
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 0.45, best_val - 0.18),
    fontsize=10, color="black",
    arrowprops=dict(arrowstyle="->", color=BEST_COLOR, lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", fc="#eaf7ea", ec=BEST_COLOR, lw=1.0),
)

ax.annotate(
    f"Early stop\n(patience={patience})",
    xy=(early_stop_epoch, val_loss[-1]),
    xytext=(early_stop_epoch - 1.6, val_loss[-1] + 0.18),
    fontsize=10, color="black",
    arrowprops=dict(arrowstyle="->", color=STOP_COLOR, lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec=STOP_COLOR, lw=1.0),
)

# Axes
ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
ax.set_ylabel("Cross-entropy loss", fontsize=12, fontweight="bold")
ax.set_title("Training & Validation Loss — Run #1\n"
             "OpenTSLM-Flamingo (Llama-3.2-1B) on Bosch Check-It (180 train / 38 val systems)",
             fontsize=12.5, fontweight="bold", pad=12)

ax.set_xlim(0.5, 9.5)
ax.set_ylim(0, 1.5)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xticks(epochs)

# Grid
ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
ax.set_axisbelow(True)

# Legend (top-right)
leg = ax.legend(loc="upper right", frameon=True, fontsize=10,
                framealpha=0.95, edgecolor="#cccccc")
leg.get_frame().set_facecolor("white")

# Tidy spines
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#444444")

plt.tight_layout()

out_path = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_004.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved: {out_path}")
