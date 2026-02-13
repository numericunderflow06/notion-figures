"""
OpenTSLM-Aviation Architecture Pipeline Diagram
fig_001: End-to-end architecture from sensor input to maintenance classification.

Data coordinates = inches (xlim/ylim match figsize).
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Colours ─────────────────────────────────────────────────────────
FROZEN_BG   = "#4A90D9"
FROZEN_EDGE = "#2C5F8A"
TRAIN_BG    = "#E8873A"
TRAIN_EDGE  = "#B0612A"
INPUT_BG    = "#6BBF6B"
INPUT_EDGE  = "#3D7A3D"
OUTPUT_BG   = "#C25D7B"
OUTPUT_EDGE = "#8A3A54"
ARROW_CLR   = "#555555"
SHAPE_CLR   = "#333333"
BG_COLOR    = "white"
TEXT_WHITE   = "white"
TEXT_DARK    = "#222222"

# ── Figure ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 24, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor=BG_COLOR)
ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor(BG_COLOR)


# ── Helpers ─────────────────────────────────────────────────────────
def draw_box(x, y, w, h, label, sublabel, bg, edge,
             text_color=TEXT_WHITE, fontsize=12, sublabel_size=9):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=bg, edgecolor=edge, linewidth=2.0, zorder=3)
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        n = sublabel.count('\n') + 1
        gap = 0.12 + 0.05 * n
        ax.text(cx, cy + gap, label, ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(cx, cy - gap, sublabel, ha="center", va="top",
                fontsize=sublabel_size, color=text_color, zorder=4,
                style="italic", linespacing=1.15)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)


def draw_arrow(x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_CLR,
                                lw=1.8, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.32
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7.5, fontfamily="monospace", color=SHAPE_CLR,
                bbox=dict(boxstyle="round,pad=0.12", fc="#F5F5F0",
                          ec="#CCCCCC", lw=0.6, alpha=0.95),
                zorder=5)


# ── Layout ──────────────────────────────────────────────────────────
BOX_H = 1.8
BOX_Y = 2.0

# Widths and gaps in inches
box_ws = [2.9, 2.4, 3.2, 3.2, 2.9, 2.8]
gap_ws = [0.90, 1.05, 1.00, 1.30, 1.30]

total_w = sum(box_ws) + sum(gap_ws)   # ~22.85
margin = (FIG_W - total_w) / 2
xs = []
cx = margin
for i, bw in enumerate(box_ws):
    xs.append(cx)
    if i < len(gap_ws):
        cx += bw + gap_ws[i]

CY = BOX_Y + BOX_H / 2

# ── Boxes ───────────────────────────────────────────────────────────
specs = [
    ("23 Sensor Channels",    "Variable-length time series",                           INPUT_BG,  INPUT_EDGE,  11,   8.5),
    ("CNN Encoder",           "patch=4, dim=128",                                      TRAIN_BG,  TRAIN_EDGE,  11,   8.5),
    ("Perceiver Resampler",   "2 layers, 8 heads, dim=896\n64 learned latent queries", TRAIN_BG,  TRAIN_EDGE,  10.5, 7.5),
    ("Gated Cross-Attention", "24 layers (one per LM layer)\n\u03b1 gates init=0",     TRAIN_BG,  TRAIN_EDGE,  10.5, 7.5),
    ("Qwen2.5-0.5B",         "24 layers, hidden=896\n500M params (frozen)",            FROZEN_BG, FROZEN_EDGE, 12,   7.5),
    ("Maintenance Issue",     "Classification (19 classes)",                            OUTPUT_BG, OUTPUT_EDGE, 11,   8.5),
]

for i, (label, sublabel, bg, edge, fs, sfs) in enumerate(specs):
    draw_box(xs[i], BOX_Y, box_ws[i], BOX_H, label, sublabel, bg, edge,
             fontsize=fs, sublabel_size=sfs)

# ── Arrows ──────────────────────────────────────────────────────────
arrow_labels = [
    "[B, 23, T]",
    "[B, 23, T/4, 128]",
    "[B, 23, 64, 896]",
    "interleaved\ninjection",
    "[B, seq, 151936]",
]
for i in range(5):
    x0 = xs[i] + box_ws[i] + 0.03
    x1 = xs[i + 1] - 0.03
    draw_arrow(x0, CY, x1, CY, label=arrow_labels[i])

# ── Integration bracket: GCA ↔ LLM ────────────────────────────────
gca_cx = xs[3] + box_ws[3] / 2
llm_cx = xs[4] + box_ws[4] / 2
mid_x  = (gca_cx + llm_cx) / 2
bkt_base = BOX_Y + BOX_H + 0.12
bkt_top  = bkt_base + 0.50

ax.plot([gca_cx, llm_cx], [bkt_base, bkt_base], color="#777", lw=1.2, zorder=3)
for tx in [gca_cx, llm_cx]:
    ax.plot([tx, tx], [bkt_base - 0.07, bkt_base], color="#777", lw=1.2, zorder=3)
ax.plot([mid_x, mid_x], [bkt_base, bkt_top], color="#777", lw=1.2, zorder=3)
ax.text(mid_x, bkt_top + 0.06,
        "Cross-attention layers interleaved inside LM transformer blocks",
        ha="center", va="bottom", fontsize=8, color="#555",
        style="italic", zorder=4)

# ── Sensor groups below input box ──────────────────────────────────
groups = ["Electrical (4)", "Fuel (3)", "Engine (3)",
          "CHT (4)", "EGT (4)", "Flight (5)"]
for i, g in enumerate(groups):
    col, row = i % 3, i // 3
    gx = xs[0] + 0.08 + col * 0.98
    gy = BOX_Y - 0.12 - row * 0.22
    ax.text(gx, gy, g, fontsize=6, color="#555", va="top", zorder=4)

# ── Parameter counts ───────────────────────────────────────────────
for idx, txt, clr in [
    (1, "~1M params",   TRAIN_EDGE),
    (2, "~2M params",   TRAIN_EDGE),
    (3, "~240M params", TRAIN_EDGE),
    (4, "~500M params", FROZEN_EDGE),
]:
    ax.text(xs[idx] + box_ws[idx] / 2, BOX_Y - 0.15, txt,
            ha="center", va="top", fontsize=7.5, color=clr,
            fontweight="bold", zorder=4)

# ── Summary bar ────────────────────────────────────────────────────
centre_x = FIG_W / 2
ax.text(centre_x, 0.50,
        "Trainable: ~250M parameters   |   Frozen: ~500M parameters",
        ha="center", va="center", fontsize=10, color=TEXT_DARK,
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.22", fc="#F0F0F0", ec="#CCCCCC", lw=1))

# ── Legend ──────────────────────────────────────────────────────────
legend_y = 5.55
for i, (bg, ec, lbl) in enumerate([
    (TRAIN_BG, TRAIN_EDGE, "Trainable"),
    (FROZEN_BG, FROZEN_EDGE, "Frozen"),
    (INPUT_BG, INPUT_EDGE, "Input Data"),
    (OUTPUT_BG, OUTPUT_EDGE, "Output"),
]):
    lx = centre_x - 5.0 + i * 2.6
    patch = FancyBboxPatch(
        (lx, legend_y), 0.38, 0.26,
        boxstyle="round,pad=0.03",
        facecolor=bg, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(patch)
    ax.text(lx + 0.50, legend_y + 0.13, lbl,
            fontsize=9.5, va="center", color=TEXT_DARK, zorder=4)

# ── Title ───────────────────────────────────────────────────────────
ax.text(centre_x, 6.35,
        "OpenTSLM-Aviation: Architecture Pipeline",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=TEXT_DARK, zorder=4)

# ── Save ────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/maintenance/fig_001.png"
fig.savefig(out_path, dpi=200, facecolor=BG_COLOR,
            bbox_inches="tight", pad_inches=0.2)
plt.close(fig)
print(f"Figure saved to {out_path}")
