"""
OpenTSLM Flamingo Architecture for Eye-Tracking Classification
Figure fig_001: End-to-end architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ─── Color Palette ───────────────────────────────────────────────────────────
TRAINABLE_FILL = "#D6EAF8"
TRAINABLE_EDGE = "#2980B9"
FROZEN_FILL = "#E5E7E9"
FROZEN_EDGE = "#7F8C8D"
INPUT_FILL = "#D5F5E3"
INPUT_EDGE = "#27AE60"
OUTPUT_FILL = "#FADBD8"
OUTPUT_EDGE = "#C0392B"
ARROW_COLOR = "#2C3E50"
ANNOT_COLOR = "#6C3483"
SUBLABEL_COLOR = "#555555"
BG_COLOR = "white"

fig, ax = plt.subplots(1, 1, figsize=(24, 9))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-1, 24)
ax.set_ylim(-2.5, 8.0)
ax.set_aspect("equal")
ax.axis("off")

# ─── Helper Functions ────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, fill, edge, lw=2.0, rounding=0.3):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={rounding}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=2.0,
               style="->", connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=18, connectionstyle=connectionstyle, zorder=1
    )
    ax.add_patch(arrow)
    return arrow


def dim_annotation(ax, x, y, text, fontsize=8, color=ANNOT_COLOR):
    """Add a dimension annotation with a small box."""
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold", zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.8))


def step_circle(ax, x, y, num):
    """Draw a numbered step circle."""
    circle = plt.Circle((x, y), 0.22, facecolor="#2C3E50",
                         edgecolor="white", linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=6)


# ─── Layout ──────────────────────────────────────────────────────────────────
cy = 3.0          # vertical center
box_h = 2.6       # main box height
small_h = 1.8     # input/output box height
gap = 0.6         # horizontal gap between boxes

# Box positions: (x, width)
boxes = {
    "input":     (0.0,   2.6),
    "cnn":       (3.2,   3.8),
    "perceiver": (7.6,   3.6),
    "gca":       (11.8,  3.6),
    "llama":     (16.2,  3.6),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INPUT
# ═══════════════════════════════════════════════════════════════════════════════
ix, iw = boxes["input"]
iy = cy - small_h / 2
draw_box(ax, ix, iy, iw, small_h, INPUT_FILL, INPUT_EDGE)
ax.text(ix + iw/2, cy + 0.25, "Eye-Tracking", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1B2631", zorder=3)
ax.text(ix + iw/2, cy - 0.15, "Time Series", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1B2631", zorder=3)
ax.text(ix + iw/2, cy - 0.6, "(B, L)", ha="center", va="center",
        fontsize=9, fontstyle="italic", color=SUBLABEL_COLOR, zorder=3)
step_circle(ax, ix + iw/2, iy + small_h + 0.35, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CNN ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
cx, cw = boxes["cnn"]
cy_b = cy - box_h / 2
draw_box(ax, cx, cy_b, cw, box_h, TRAINABLE_FILL, TRAINABLE_EDGE)

ax.text(cx + cw/2, cy + 0.85, "CNN Encoder", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)

details = [
    ("Conv1d (k=4, s=4, 1→128)", "#2471A3", "bold", 9),
    ("Positional Embedding (2600, 128)", SUBLABEL_COLOR, "normal", 8.5),
    ("LayerNorm(128) + Dropout", SUBLABEL_COLOR, "normal", 8.5),
]
for i, (text, color, weight, fs) in enumerate(details):
    ax.text(cx + cw/2, cy + 0.35 - i * 0.4, text,
            ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=3)

ax.text(cx + cw/2, cy_b - 0.28, "TRAINABLE",
        ha="center", va="center", fontsize=7.5, color=TRAINABLE_EDGE,
        fontweight="bold", zorder=4)
step_circle(ax, cx + cw/2, cy_b + box_h + 0.35, 2)

# Arrow: Input → CNN
draw_arrow(ax, ix + iw + 0.08, cy, cx - 0.08, cy)
dim_annotation(ax, (ix + iw + cx) / 2, cy + 0.6, "(B, L, 1)")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERCEIVER RESAMPLER
# ═══════════════════════════════════════════════════════════════════════════════
px, pw = boxes["perceiver"]
py_b = cy - box_h / 2
draw_box(ax, px, py_b, pw, box_h, TRAINABLE_FILL, TRAINABLE_EDGE)

ax.text(px + pw/2, cy + 0.85, "Perceiver", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)
ax.text(px + pw/2, cy + 0.45, "Resampler", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)

p_details = [
    ("6 Attention Layers", "#2471A3", "bold", 9),
    ("64 Learnable Latents", "#2471A3", "bold", 9),
    ("8 heads, dim=128", SUBLABEL_COLOR, "normal", 8.5),
    ("FFN: 128→512→128", SUBLABEL_COLOR, "normal", 8.5),
]
for i, (text, color, weight, fs) in enumerate(p_details):
    ax.text(px + pw/2, cy + 0.05 - i * 0.35, text,
            ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=3)

ax.text(px + pw/2, py_b - 0.28, "TRAINABLE",
        ha="center", va="center", fontsize=7.5, color=TRAINABLE_EDGE,
        fontweight="bold", zorder=4)
step_circle(ax, px + pw/2, py_b + box_h + 0.35, 3)

# Arrow: CNN → Perceiver
draw_arrow(ax, cx + cw + 0.08, cy, px - 0.08, cy)
dim_annotation(ax, (cx + cw + px) / 2, cy + 0.6, "(B, N_patches, 128)")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. GATED CROSS-ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════
gx, gw = boxes["gca"]
gy_b = cy - box_h / 2
draw_box(ax, gx, gy_b, gw, box_h, TRAINABLE_FILL, TRAINABLE_EDGE)

ax.text(gx + gw/2, cy + 0.85, "Gated Cross-", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)
ax.text(gx + gw/2, cy + 0.45, "Attention", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)

g_details = [
    ("× 28 layers (one per decoder)", "#2471A3", "bold", 8.5),
    ("Q: 3072→512  KV: 128→1024", SUBLABEL_COLOR, "normal", 8),
    ("Gates init=0 (tanh gating)", SUBLABEL_COLOR, "normal", 8),
    ("FFN: 3072→12288→3072", SUBLABEL_COLOR, "normal", 8),
]
for i, (text, color, weight, fs) in enumerate(g_details):
    ax.text(gx + gw/2, cy + 0.05 - i * 0.35, text,
            ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=3)

ax.text(gx + gw/2, gy_b - 0.28, "TRAINABLE",
        ha="center", va="center", fontsize=7.5, color=TRAINABLE_EDGE,
        fontweight="bold", zorder=4)
step_circle(ax, gx + gw/2, gy_b + box_h + 0.35, 4)

# Arrow: Perceiver → GCA
draw_arrow(ax, px + pw + 0.08, cy, gx - 0.08, cy)
dim_annotation(ax, (px + pw + gx) / 2, cy + 0.6, "(B, 64, 128)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. LLAMA 3.2 3B
# ═══════════════════════════════════════════════════════════════════════════════
lx, lw = boxes["llama"]
ly_b = cy - box_h / 2
draw_box(ax, lx, ly_b, lw, box_h, FROZEN_FILL, FROZEN_EDGE)

ax.text(lx + lw/2, cy + 0.85, "Llama 3.2 3B", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1B2631", zorder=3)

l_details = [
    ("28 Decoder Layers", "#566573", "bold", 9),
    ("Hidden: 3072, Heads: 24", SUBLABEL_COLOR, "normal", 8.5),
    ("GQA (8 KV heads)", SUBLABEL_COLOR, "normal", 8.5),
    ("FFN: 3072→8192→3072", SUBLABEL_COLOR, "normal", 8.5),
]
for i, (text, color, weight, fs) in enumerate(l_details):
    ax.text(lx + lw/2, cy + 0.35 - i * 0.35, text,
            ha="center", va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=3)

ax.text(lx + lw/2, ly_b - 0.28, "FROZEN",
        ha="center", va="center", fontsize=7.5, color=FROZEN_EDGE,
        fontweight="bold", zorder=4)
step_circle(ax, lx + lw/2, ly_b + box_h + 0.35, 5)

# Arrow: GCA → Llama (thicker injection arrow)
draw_arrow(ax, gx + gw + 0.08, cy, lx - 0.08, cy, lw=2.8, style="-|>")

# "Inject" label between GCA and Llama
mid_gl = (gx + gw + lx) / 2
ax.text(mid_gl, cy + 0.85, "Inject at\neach layer",
        ha="center", va="center", fontsize=8.5, color="#7D6608",
        fontweight="bold", zorder=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9E7",
                  edgecolor="#F39C12", alpha=0.95, linewidth=1.0))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OUTPUT (below Llama)
# ═══════════════════════════════════════════════════════════════════════════════
ox = lx + 0.3
oy = cy - box_h/2 - 2.3
ow, oh = 3.0, 1.6
draw_box(ax, ox, oy, ow, oh, OUTPUT_FILL, OUTPUT_EDGE)

ax.text(ox + ow/2, oy + oh/2 + 0.25, "Classification Output",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#1B2631", zorder=3)
ax.text(ox + ow/2, oy + oh/2 - 0.25, "NR / TSR",
        ha="center", va="center", fontsize=10, fontstyle="italic",
        color=SUBLABEL_COLOR, zorder=3)

# Arrow: Llama → Output
draw_arrow(ax, lx + lw/2, ly_b - 0.08, ox + ow/2, oy + oh + 0.08)

# Generation annotation next to the downward arrow
dim_annotation(ax, lx + lw/2 + 2.5, (ly_b + oy + oh) / 2,
               "Generates:\n\"Normal Reading\"\nor \"Task-Specific Reading\"",
               fontsize=7.5, color="#943126")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TEXT PROMPT (above, feeding into Llama)
# ═══════════════════════════════════════════════════════════════════════════════
tx = 13.0
ty = cy + box_h/2 + 1.3
tw, th = 5.2, 1.2
draw_box(ax, tx, ty, tw, th, "#FDF2E9", "#E67E22", lw=1.5)

ax.text(tx + tw/2, ty + th/2 + 0.15, "Text Prompt",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#1B2631", zorder=3)
ax.text(tx + tw/2, ty + th/2 - 0.25,
        '"<image> {ts_text} <|endofchunk|> Classify:"',
        ha="center", va="center", fontsize=8, fontstyle="italic",
        color=SUBLABEL_COLOR, zorder=3)

# Arrow from text prompt to Llama
draw_arrow(ax, tx + tw/2 + 1.3, ty - 0.08, lx + lw/2, ly_b + box_h + 0.08,
           color="#E67E22", lw=1.8, connectionstyle="arc3,rad=-0.15")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(11.5, 7.5, "OpenTSLM Flamingo Architecture for Eye-Tracking Classification",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="#1B2631", zorder=5)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. LEGEND
# ═══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (TRAINABLE_FILL, TRAINABLE_EDGE, "Trainable"),
    (FROZEN_FILL, FROZEN_EDGE, "Frozen"),
    (INPUT_FILL, INPUT_EDGE, "Input"),
    (OUTPUT_FILL, OUTPUT_EDGE, "Output"),
]
legend_y = -2.0
for i, (fill, edge, label) in enumerate(legend_items):
    lx_i = 0.0 + i * 3.2
    box = FancyBboxPatch(
        (lx_i, legend_y - 0.2), 0.5, 0.4,
        boxstyle="round,pad=0.08",
        facecolor=fill, edgecolor=edge, linewidth=1.5, zorder=2
    )
    ax.add_patch(box)
    ax.text(lx_i + 0.7, legend_y, label,
            ha="left", va="center", fontsize=9.5, color="#2C3E50",
            fontweight="bold", zorder=3)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
plt.savefig("/home/wangni/notion-figures/zuco/fig_001.png",
            dpi=200, bbox_inches="tight", facecolor="white",
            edgecolor="none", pad_inches=0.3)
plt.close()

print("Figure saved to /home/wangni/notion-figures/zuco/fig_001.png")
