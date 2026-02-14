"""
fig_003: ATPE — From Ordinal to Temporal Encoding
Detailed diagram of the Absolute Temporal Positional Encoding module.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
BG       = "#FFFFFF"
DARK     = "#1B2A4A"
ACCENT1  = "#2E86C1"   # blue – main flow
ACCENT2  = "#E74C3C"   # red – original (contrast)
ACCENT3  = "#27AE60"   # green – output
ACCENT4  = "#8E44AD"   # purple – projection
LIGHT1   = "#D6EAF8"   # light blue
LIGHT2   = "#FADBD8"   # light red
LIGHT3   = "#D5F5E3"   # light green
LIGHT4   = "#E8DAEF"   # light purple
GRAY     = "#BDC3C7"
LGRAY    = "#F2F3F4"
TXTCOL   = "#2C3E50"

fig = plt.figure(figsize=(11, 16), facecolor=BG, dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 11)
ax.set_ylim(0, 16)
ax.set_aspect("equal")
ax.axis("off")

# ── helper: rounded box ────────────────────────────────────────────
def draw_box(x, y, w, h, facecolor, edgecolor, label, fontsize=11,
             fontweight="bold", textcolor=DARK, alpha=1.0, lw=1.5,
             sublabel=None, sublabel_size=9):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 0.2, sublabel, ha="center", va="center",
                fontsize=sublabel_size, color=textcolor, style="italic", zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=4)

def draw_arrow(x1, y1, x2, y2, color=DARK, lw=1.8, style="->"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=14, zorder=5)
    ax.add_patch(arrow)

# =====================================================================
#  TITLE
# =====================================================================
ax.text(5.5, 15.55, "ATPE: From Ordinal to Temporal Encoding",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=DARK, zorder=10)
ax.text(5.5, 15.2, "Absolute Temporal Positional Encoding Module",
        ha="center", va="center", fontsize=11, color=TXTCOL, style="italic")

# =====================================================================
#  LEFT COLUMN — ORIGINAL APPROACH (contrast)
# =====================================================================
left_x = 0.6
col_w  = 3.0

ax.text(left_x + col_w / 2, 14.65, "Original Approach",
        ha="center", fontsize=12, fontweight="bold", color=ACCENT2)

# Ordinal indices box
draw_box(left_x, 13.55, col_w, 0.85, LIGHT2, ACCENT2,
         "Ordinal Indices", fontsize=11)
# small example indices
indices = ["0", "1", "2", "…", "N-1"]
idx_y = 13.05
for i, idx in enumerate(indices):
    ix = left_x + 0.35 + i * 0.52
    ax.text(ix, idx_y, idx, ha="center", va="center", fontsize=9,
            fontfamily="monospace", color=ACCENT2, fontweight="bold")

draw_arrow(left_x + col_w / 2, 13.55, left_x + col_w / 2, 12.55, color=ACCENT2)

# Learnable embedding
draw_box(left_x, 11.7, col_w, 0.85, LIGHT2, ACCENT2,
         "nn.Parameter", fontsize=11,
         sublabel="torch.randn(1, 2600, 128)")

draw_arrow(left_x + col_w / 2, 11.7, left_x + col_w / 2, 10.75, color=ACCENT2)

# Output
draw_box(left_x, 9.9, col_w, 0.85, LIGHT2, ACCENT2,
         "Positional Embeddings", fontsize=10,
         sublabel="No temporal meaning")

# Limitations annotation
lim_y = 9.2
ax.text(left_x + col_w / 2, lim_y, "Limitations", ha="center",
        fontsize=10, fontweight="bold", color=ACCENT2)
limitations = [
    "• Encodes order, not time",
    "• No cross-series alignment",
    "• Position 0 ≠ same real date",
]
for i, lim in enumerate(limitations):
    ax.text(left_x + 0.15, lim_y - 0.35 - i * 0.3, lim, ha="left",
            fontsize=8.5, color=ACCENT2, va="center")

# =====================================================================
#  RIGHT COLUMN — ATPE (main flow)
# =====================================================================
right_x = 4.6
rcol_w  = 3.4

ax.text(right_x + rcol_w / 2, 14.65, "ATPE (Proposed)",
        ha="center", fontsize=12, fontweight="bold", color=ACCENT1)

# ── Stage 1: Raw timestamps ──
stage1_y = 13.55
draw_box(right_x, stage1_y, rcol_w, 0.85, LIGHT1, ACCENT1,
         "Raw Timestamps", fontsize=11,
         sublabel="Patch midpoints")

# Example timestamps
ts_examples = ["Day 0", "Day 4", "Day 8", "Day 12", "…", "Day T"]
ts_y = stage1_y - 0.07
for i, t in enumerate(ts_examples):
    tx = right_x + 0.3 + i * 0.53
    ax.text(tx, ts_y, t, ha="center", va="center", fontsize=7,
            fontfamily="monospace", color=ACCENT1, fontweight="bold")

draw_arrow(right_x + rcol_w / 2, stage1_y, right_x + rcol_w / 2, stage1_y - 0.8,
           color=ACCENT1)

# ── Stage 2: Normalization ──
stage2_y = 11.9
draw_box(right_x, stage2_y, rcol_w, 0.85, LIGHT1, ACCENT1,
         "Normalize to [0, 1]", fontsize=11,
         sublabel="t̂ = (t − t_min) / (t_max − t_min)")

# Normalized examples
norm_ex = ["0.00", "0.15", "0.31", "0.46", "…", "1.00"]
ne_y = stage2_y - 0.07
for i, n in enumerate(norm_ex):
    nx = right_x + 0.3 + i * 0.53
    ax.text(nx, ne_y, n, ha="center", va="center", fontsize=7,
            fontfamily="monospace", color=ACCENT1, fontweight="bold")

draw_arrow(right_x + rcol_w / 2, stage2_y, right_x + rcol_w / 2, stage2_y - 0.8,
           color=ACCENT1)

# ── Stage 3: Multi-scale sinusoidal encoding ──
stage3_y = 10.05
draw_box(right_x, stage3_y, rcol_w, 1.05, LIGHT1, ACCENT1,
         "Multi-Scale Sinusoidal", fontsize=11,
         sublabel="sin/cos at different frequencies")

# Formula annotation — placed inside box below sublabel
formula_y = stage3_y + 0.12
ax.text(right_x + rcol_w / 2, formula_y,
        r"$PE(t, 2i) = \sin(t \; / \; 10000^{2i/d})$",
        ha="center", va="center", fontsize=8.5, color=DARK, zorder=4)

draw_arrow(right_x + rcol_w / 2, stage3_y, right_x + rcol_w / 2, stage3_y - 0.85,
           color=ACCENT1)

# ── Stage 3b: Sinusoidal heatmap ──
heatmap_x = 8.5
heatmap_y = 10.6
hm_w, hm_h = 2.05, 2.0

# Generate sinusoidal encoding matrix for heatmap
n_pos = 20
d_model = 64  # show 64 dims for visualization
positions = np.linspace(0, 1, n_pos)
encoding = np.zeros((n_pos, d_model))
for i in range(d_model // 2):
    freq = 1.0 / (10000 ** (2 * i / 128))
    encoding[:, 2 * i]     = np.sin(positions * freq * 2 * np.pi * (i + 1))
    encoding[:, 2 * i + 1] = np.cos(positions * freq * 2 * np.pi * (i + 1))

# Create inset axes for heatmap
hm_ax = fig.add_axes([heatmap_x / 11, heatmap_y / 16,
                       hm_w / 11, hm_h / 16])
im = hm_ax.imshow(encoding.T, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-1, vmax=1)
hm_ax.set_xlabel("Norm. Time", fontsize=6.5, color=DARK, labelpad=1)
hm_ax.set_ylabel("Encoding Dim", fontsize=6.5, color=DARK, labelpad=1)
hm_ax.set_title("Sinusoidal Encoding Matrix", fontsize=7.5,
                 fontweight="bold", color=ACCENT1, pad=3)
hm_ax.tick_params(axis="both", labelsize=5.5, colors=DARK, pad=1)
hm_ax.set_xticks([0, n_pos - 1])
hm_ax.set_xticklabels(["0", "1"])
hm_ax.set_yticks([0, d_model - 1])
hm_ax.set_yticklabels(["0", f"{d_model-1}"])

# Colorbar
cbar = fig.colorbar(im, ax=hm_ax, fraction=0.06, pad=0.04)
cbar.ax.tick_params(labelsize=5.5)
cbar.set_ticks([-1, 0, 1])


# ── Stage 4: Learnable linear projection ──
stage4_y = 8.5
draw_box(right_x, stage4_y, rcol_w, 0.85, LIGHT4, ACCENT4,
         "Learnable Linear Projection", fontsize=10.5,
         sublabel="time_proj: Linear(d_model, d_model)")

draw_arrow(right_x + rcol_w / 2, stage4_y, right_x + rcol_w / 2, stage4_y - 0.85,
           color=ACCENT4)

# ── Stage 5: Output ──
stage5_y = 6.8
draw_box(right_x, stage5_y, rcol_w, 0.85, LIGHT3, ACCENT3,
         "Temporal Embeddings", fontsize=11,
         sublabel="d_model = 128")

# =====================================================================
#  PARAMETER ANNOTATION
# =====================================================================
param_x = right_x + rcol_w + 0.15
param_y = 8.9
ax.annotate("~16K params\n(128 × 128)",
            xy=(right_x + rcol_w, stage4_y + 0.42),
            xytext=(param_x + 0.05, param_y),
            fontsize=9, fontweight="bold", color=ACCENT4,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=ACCENT4, lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT4,
                      edgecolor=ACCENT4, lw=1))

# =====================================================================
#  BACKWARD COMPATIBILITY NOTE
# =====================================================================
compat_y = 5.9
compat_w = 6.2
draw_box(right_x - 0.8, compat_y, compat_w, 0.7, "#FEF9E7", "#F39C12",
         "", fontsize=9)
ax.text(right_x - 0.8 + compat_w / 2, compat_y + 0.45,
        "Backward Compatibility", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#D68910")
ax.text(right_x - 0.8 + compat_w / 2, compat_y + 0.15,
        "With uniform timestamps (no real-world time data),\n"
        "ATPE reduces to standard sinusoidal positional encoding.",
        ha="center", va="center", fontsize=8.5, color="#7D6608")

# =====================================================================
#  CROSS-SERIES ALIGNMENT ILLUSTRATION
# =====================================================================
cs_y = 4.2
ax.text(5.5, cs_y + 0.85, "Cross-Series Temporal Alignment",
        ha="center", fontsize=12, fontweight="bold", color=DARK)
ax.text(5.5, cs_y + 0.55,
        "All series in a sample share the same normalized [0, 1] time axis",
        ha="center", fontsize=9, color=TXTCOL, style="italic")

# Draw two series on the same axis
axis_left = 1.2
axis_right = 9.8
axis_y_s1 = cs_y
axis_y_s2 = cs_y - 0.7

# Time axis line
for ay in [axis_y_s1, axis_y_s2]:
    ax.plot([axis_left, axis_right], [ay, ay], color=GRAY, lw=1.5, zorder=2)
    # Tick marks
    for tv in np.linspace(0, 1, 6):
        tx = axis_left + tv * (axis_right - axis_left)
        ax.plot([tx, tx], [ay - 0.05, ay + 0.05], color=GRAY, lw=1, zorder=2)

# Series 1 patches
s1_colors = [ACCENT1] * 5
s1_starts = [0.0, 0.15, 0.3, 0.6, 0.85]
s1_widths = [0.12, 0.12, 0.12, 0.12, 0.12]
ax.text(axis_left - 0.2, axis_y_s1, "Series A", ha="right", va="center",
        fontsize=9, fontweight="bold", color=ACCENT1)
for i, (s, w) in enumerate(zip(s1_starts, s1_widths)):
    px = axis_left + s * (axis_right - axis_left)
    pw = w * (axis_right - axis_left)
    rect = FancyBboxPatch((px, axis_y_s1 - 0.1), pw, 0.2,
                           boxstyle="round,pad=0.02",
                           facecolor=ACCENT1, edgecolor="white",
                           alpha=0.7, lw=0.8, zorder=3)
    ax.add_patch(rect)

# Series 2 patches (different distribution but same axis)
ax.text(axis_left - 0.2, axis_y_s2, "Series B", ha="right", va="center",
        fontsize=9, fontweight="bold", color=ACCENT3)
s2_starts = [0.0, 0.2, 0.4, 0.55, 0.75]
for i, (s, w) in enumerate(zip(s2_starts, s1_widths)):
    px = axis_left + s * (axis_right - axis_left)
    pw = w * (axis_right - axis_left)
    rect = FancyBboxPatch((px, axis_y_s2 - 0.1), pw, 0.2,
                           boxstyle="round,pad=0.02",
                           facecolor=ACCENT3, edgecolor="white",
                           alpha=0.7, lw=0.8, zorder=3)
    ax.add_patch(rect)

# Normalized time labels
for tv, label in [(0, "0.0"), (0.2, "0.2"), (0.4, "0.4"),
                   (0.6, "0.6"), (0.8, "0.8"), (1.0, "1.0")]:
    tx = axis_left + tv * (axis_right - axis_left)
    ax.text(tx, axis_y_s2 - 0.28, label, ha="center", va="top",
            fontsize=8, color=TXTCOL, fontfamily="monospace")

ax.text((axis_left + axis_right) / 2, axis_y_s2 - 0.55,
        "Normalized Time Axis",
        ha="center", va="top", fontsize=9, fontweight="bold", color=DARK)

# Dashed alignment lines between matching temporal regions
for t_align in [0.2, 0.6]:
    tx = axis_left + t_align * (axis_right - axis_left)
    ax.plot([tx, tx], [axis_y_s2 + 0.12, axis_y_s1 - 0.12],
            color="#F39C12", lw=1.2, ls="--", alpha=0.7, zorder=2)
    ax.text(tx, (axis_y_s1 + axis_y_s2) / 2, "⬌",
            ha="center", va="center", fontsize=8, color="#F39C12")

# =====================================================================
#  CONTRAST ARROW between left and right columns
# =====================================================================
contrast_y = 11.0
ax.annotate("",
            xy=(right_x - 0.15, contrast_y),
            xytext=(left_x + col_w + 0.15, contrast_y),
            arrowprops=dict(arrowstyle="->", color=DARK, lw=2,
                            connectionstyle="arc3,rad=0.0"))
ax.text((left_x + col_w + right_x) / 2, contrast_y + 0.2,
        "Replace with", ha="center", va="center", fontsize=9,
        fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                  edgecolor=DARK, lw=1))

# =====================================================================
#  KEY PROPERTIES BOX
# =====================================================================
kp_y = 1.4
kp_h = 1.2
kp_w = 9.2
kp_x = 1.0
draw_box(kp_x, kp_y, kp_w, kp_h, LGRAY, DARK, "", lw=1.2)
ax.text(kp_x + kp_w / 2, kp_y + kp_h - 0.15, "Key Properties of ATPE",
        ha="center", va="center", fontsize=11, fontweight="bold", color=DARK)

props = [
    ("Continuous", "Handles irregular & variable-length series"),
    ("Multi-scale", "sin/cos capture patterns at all frequencies"),
    ("Learnable", "time_proj adapts encoding to task"),
    ("Compatible", "Reduces to standard PE with uniform time"),
]
for i, (title, desc) in enumerate(props):
    col = i % 2
    row = i // 2
    px = kp_x + 0.35 + col * 4.6
    py = kp_y + kp_h - 0.55 - row * 0.4
    ax.text(px, py, f"▸ {title}:", ha="left", va="center",
            fontsize=9, fontweight="bold", color=ACCENT1)
    ax.text(px + 1.55, py, desc, ha="left", va="center",
            fontsize=8.5, color=TXTCOL)

# =====================================================================
#  SAVE
# =====================================================================
fig.savefig("/home/wangni/notion-figures/alignment/fig_003.png",
            dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
plt.close(fig)
print("✓ Saved fig_003.png")
