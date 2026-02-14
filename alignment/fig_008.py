"""
fig_008: Related Work Positioning
Comparison chart positioning TPA against prior work along two axes:
  X-axis: Cross-modal integration (TS only → Text+TS)
  Y-axis: Temporal alignment capability (None → Ordinal → Explicit)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Data from verified facts (Section 12) ---
# Positions: (cross_modal, temporal_alignment)
# Cross-modal scale: 0 = TS only, 1 = Text+TS
# Temporal alignment scale: 0 = None, ~1 = Ordinal, ~2 = Explicit

methods = {
    "Chronos": {
        "x": 0.12,   # TS only (tokenization, no text)
        "y": 0.15,   # No alignment (tokenization, no temporal alignment)
        "marker": "D",
        "color": "#7B8794",
    },
    "TFT": {
        "x": 0.18,   # TS only (forecasting only)
        "y": 1.05,   # Temporal features (uses temporal covariates, ordinal-level)
        "marker": "s",
        "color": "#E8A838",
    },
    "Time-LLM": {
        "x": 0.72,   # Text+TS (reprograms TS as text tokens)
        "y": 0.15,   # No alignment
        "marker": "^",
        "color": "#5B9BD5",
    },
    "Flamingo\n(OpenTSLM)": {
        "x": 0.78,   # Multimodal (text + TS interleaved)
        "y": 0.95,   # Ordinal alignment (learnable ordinal positional embeddings)
        "marker": "o",
        "color": "#ED7D31",
    },
    "TPA (Ours)": {
        "x": 0.88,   # Full cross-modal integration
        "y": 1.90,   # Explicit temporal alignment
        "marker": "*",
        "color": "#C0392B",
    },
}

# --- Figure setup ---
fig, axes = plt.subplots(figsize=(8.5, 6.5), dpi=200, facecolor="white")
axes.set_facecolor("white")

# Quadrant shading to emphasize the upper-right region
axes.axhspan(1.4, 2.15, xmin=0.55, xmax=1.0, color="#C0392B", alpha=0.04, zorder=0)

# Light quadrant dividers
axes.axhline(y=0.65, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
axes.axhline(y=1.4, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
axes.axvline(x=0.5, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)

# Y-axis region labels (left margin)
axes.text(-0.18, 0.30, "None", fontsize=9, color="#888888", ha="center", va="center",
          fontstyle="italic")
axes.text(-0.18, 1.02, "Ordinal", fontsize=9, color="#888888", ha="center", va="center",
          fontstyle="italic")
axes.text(-0.18, 1.75, "Explicit", fontsize=9, color="#888888", ha="center", va="center",
          fontstyle="italic")

# X-axis region labels (bottom)
axes.text(0.22, -0.22, "TS Only", fontsize=9, color="#888888", ha="center", va="center",
          fontstyle="italic")
axes.text(0.78, -0.22, "Text + TS", fontsize=9, color="#888888", ha="center", va="center",
          fontstyle="italic")

# --- Plot each method ---
for name, info in methods.items():
    ms = 18 if name == "TPA (Ours)" else 12
    edge_w = 2.0 if name == "TPA (Ours)" else 1.2
    edge_c = "#8B0000" if name == "TPA (Ours)" else "#333333"
    z = 10 if name == "TPA (Ours)" else 5

    axes.scatter(
        info["x"], info["y"],
        marker=info["marker"],
        s=ms**2,
        c=info["color"],
        edgecolors=edge_c,
        linewidths=edge_w,
        zorder=z,
    )

    # Method name label positioning
    fontweight = "bold" if name == "TPA (Ours)" else "semibold"
    fsize = 12 if name == "TPA (Ours)" else 10.5

    if name == "Chronos":
        lx, ly = info["x"] + 0.06, info["y"] + 0.12
        ha_name, va_name = "left", "bottom"
    elif name == "TFT":
        lx, ly = info["x"] + 0.06, info["y"] + 0.10
        ha_name, va_name = "left", "bottom"
    elif name == "Time-LLM":
        lx, ly = info["x"] + 0.06, info["y"] + 0.12
        ha_name, va_name = "left", "bottom"
    elif "Flamingo" in name:
        lx, ly = info["x"] - 0.28, info["y"] + 0.10
        ha_name, va_name = "center", "bottom"
    else:  # TPA
        lx, ly = info["x"] + 0.02, info["y"] + 0.12
        ha_name, va_name = "center", "bottom"

    arrow_props = dict(arrowstyle="-", color="#AAAAAA", lw=0.8) if name == "TPA (Ours)" else None
    axes.annotate(
        name, xy=(info["x"], info["y"]), xytext=(lx, ly),
        fontsize=fsize, fontweight=fontweight, color=info["color"],
        ha=ha_name, va=va_name,
        arrowprops=arrow_props,
        zorder=15,
    )

# --- Brief annotations (descriptions) ---
annot_style = dict(fontsize=8, color="#666666", ha="left", va="top",
                   fontstyle="italic", linespacing=1.3)

# Chronos annotation
axes.annotate("Tokenization\nNo alignment",
              xy=(0.12, 0.15), xytext=(0.22, 0.50),
              fontsize=8, color="#666666", fontstyle="italic", linespacing=1.3,
              ha="left", va="top",
              arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
# TFT annotation
axes.annotate("Temporal features\nForecasting only",
              xy=(0.18, 1.05), xytext=(0.28, 1.38),
              fontsize=8, color="#666666", fontstyle="italic", linespacing=1.3,
              ha="left", va="top",
              arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
# Time-LLM annotation
axes.annotate("TS reprogrammed\nas text tokens",
              xy=(0.72, 0.15), xytext=(0.82, 0.50),
              fontsize=8, color="#666666", fontstyle="italic", linespacing=1.3,
              ha="left", va="top",
              arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
# Flamingo annotation
axes.annotate("Ordinal positional\nembeddings",
              xy=(0.78, 0.95), xytext=(0.88, 0.82),
              fontsize=8, color="#666666", fontstyle="italic", linespacing=1.3,
              ha="left", va="top",
              arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))

# TPA annotation — more prominent, placed below-left of TPA marker
axes.annotate(
    "ATPE + Anchor Injection\n+ Temporal Cross-Attention",
    xy=(0.88, 1.90), xytext=(0.52, 1.55),
    fontsize=8.5, color="#8B0000", ha="center", va="top",
    fontstyle="italic", linespacing=1.3,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#C0392B", alpha=0.07,
              edgecolor="#C0392B", linewidth=0.6),
    arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=0.8, alpha=0.5),
)

# --- Axes configuration ---
axes.set_xlim(-0.08, 1.08)
axes.set_ylim(-0.35, 2.15)

axes.set_xlabel("Cross-Modal Integration", fontsize=12, fontweight="semibold",
                labelpad=10)
axes.set_ylabel("Temporal Alignment Capability", fontsize=12, fontweight="semibold",
                labelpad=10)

# Hide numeric ticks (regions labeled textually instead)
axes.set_xticks([0.0, 0.5, 1.0])
axes.set_xticklabels(["", "", ""], fontsize=9)
axes.set_yticks([0.0, 0.65, 1.4, 2.0])
axes.set_yticklabels(["", "", "", ""], fontsize=9)

# Directional arrows on axes
axes.annotate("", xy=(1.06, -0.35), xytext=(0.0, -0.35),
              arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
              annotation_clip=False)
axes.annotate("", xy=(-0.08, 2.13), xytext=(-0.08, 0.0),
              arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5),
              annotation_clip=False)

# Remove spines (clean academic look)
for spine in axes.spines.values():
    spine.set_visible(False)
axes.tick_params(left=False, bottom=False)

# --- Title ---
axes.set_title("Related Work Positioning: Temporal Alignment vs. Cross-Modal Integration",
               fontsize=13, fontweight="bold", pad=18, color="#222222")

# --- Legend ---
legend_elements = []
for name, info in methods.items():
    ms = 10 if name == "TPA (Ours)" else 7
    legend_elements.append(
        plt.Line2D([0], [0], marker=info["marker"], color="w",
                   markerfacecolor=info["color"], markeredgecolor="#333333",
                   markersize=ms, label=name, markeredgewidth=0.8)
    )

leg = axes.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  frameon=True, fancybox=True, framealpha=0.9,
                  edgecolor="#CCCCCC", borderpad=0.8)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()
plt.savefig("/home/wangni/notion-figures/alignment/fig_008.png",
            dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()

print("Saved: /home/wangni/notion-figures/alignment/fig_008.png")
