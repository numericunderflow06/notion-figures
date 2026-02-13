"""
fig_002: Parameter Distribution — Trainable vs. Frozen
Donut chart showing parameter budget breakdown for OpenTSLM-Aviation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data from spec ---
components = [
    ("Qwen2.5-0.5B\n(Frozen)", 500, "frozen"),
    ("Gated Cross-Attention\n(Trainable)", 240, "trainable"),
    ("Perceiver Resampler\n(Trainable)", 2, "trainable"),
    ("CNN Encoder\n(Trainable)", 1, "trainable"),
]

labels_raw = [c[0] for c in components]
sizes = [c[1] for c in components]
categories = [c[2] for c in components]
total = sum(sizes)

# --- Colors ---
frozen_color = "#5B7FA5"
trainable_colors = ["#E8793A", "#F5A623", "#FFCB47"]

colors = []
t_idx = 0
for cat in categories:
    if cat == "frozen":
        colors.append(frozen_color)
    else:
        colors.append(trainable_colors[t_idx])
        t_idx += 1

# --- Figure ---
fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor="white")
ax.set_facecolor("white")

# Shift donut to the left to make room for a right-side legend/table
center_x, center_y = -0.15, 0.0

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
    startangle=140,
    colors=colors,
    pctdistance=0.78,
    wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2.5),
    textprops=dict(fontsize=10.5, fontweight="bold", color="white"),
    center=(center_x, center_y),
    radius=1.0,
)

for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
    at.set_color("white")

# --- Center text ---
ax.text(center_x, center_y + 0.08, f"~{total}M", ha="center", va="center",
        fontsize=24, fontweight="bold", color="#222222", family="sans-serif")
ax.text(center_x, center_y - 0.14, "Total Params", ha="center", va="center",
        fontsize=10, color="#777777", family="sans-serif")

# --- Right-side breakdown table ---
table_x = 1.15
table_y_start = 0.75
row_height = 0.36

component_info = [
    ("Qwen2.5-0.5B", "Frozen", "~500M", frozen_color),
    ("Gated Cross-Attention", "Trainable", "~240M", trainable_colors[0]),
    ("Perceiver Resampler", "Trainable", "~2M", trainable_colors[1]),
    ("CNN Encoder", "Trainable", "~1M", trainable_colors[2]),
]

for i, (name, tag, param_str, color) in enumerate(component_info):
    y = table_y_start - i * row_height

    # Color dot
    ax.plot(table_x, y, 'o', color=color, markersize=10, markeredgecolor="white",
            markeredgewidth=1.5, transform=ax.transData, clip_on=False)

    # Component name
    ax.text(table_x + 0.12, y + 0.04, name, fontsize=10.5, fontweight="bold",
            color="#333333", va="center", ha="left", transform=ax.transData, clip_on=False)

    # Tag + param count
    pct = sizes[i] / total * 100
    ax.text(table_x + 0.12, y - 0.12, f"{tag}  ·  {param_str}  ({pct:.1f}%)",
            fontsize=9, color="#777777", va="center", ha="left",
            transform=ax.transData, clip_on=False)

# --- Summary bar at bottom ---
total_trainable = sum(s for _, s, c in components if c == "trainable")
total_frozen = sum(s for _, s, c in components if c == "frozen")

# Draw a small stacked horizontal bar as summary
bar_y = -1.15
bar_left = -1.1
bar_width = 3.5
frozen_w = bar_width * (total_frozen / total)
trainable_w = bar_width * (total_trainable / total)

# Frozen portion
ax.barh(bar_y, frozen_w, left=bar_left, height=0.18, color=frozen_color,
        edgecolor="white", linewidth=1)
# Trainable portion
ax.barh(bar_y, trainable_w, left=bar_left + frozen_w, height=0.18,
        color=trainable_colors[0], edgecolor="white", linewidth=1)

# Labels on the bar
ax.text(bar_left + frozen_w / 2, bar_y, f"Frozen: {total_frozen}M ({total_frozen/total*100:.0f}%)",
        ha="center", va="center", fontsize=9, fontweight="bold", color="white")
ax.text(bar_left + frozen_w + trainable_w / 2, bar_y,
        f"Trainable: {total_trainable}M ({total_trainable/total*100:.0f}%)",
        ha="center", va="center", fontsize=9, fontweight="bold", color="white")

# --- Title ---
fig.suptitle(
    "Parameter Distribution: Trainable vs. Frozen",
    fontsize=15,
    fontweight="bold",
    color="#222222",
    y=0.97,
    family="sans-serif",
)

ax.set_xlim(-1.5, 3.0)
ax.set_ylim(-1.45, 1.35)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(rect=[0, 0.0, 1, 0.93])

# --- Save ---
out_path = "/home/wangni/notion-figures/maintenance/fig_002.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.3)
plt.close(fig)
print(f"Saved: {out_path}")
