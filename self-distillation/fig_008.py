"""
fig_008: OpenTSLM Architecture Variants with SDFT Addition
Side-by-side comparison of OpenTSLMSP and OpenTSLMFlamingo,
showing the new compute_logits() method added by SDFT.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Color palette ──────────────────────────────────────────────
C_LLM      = "#3B82F6"   # blue – LLM backbone
C_ENCODER  = "#8B5CF6"   # purple – encoder/tokenizer
C_PROJ     = "#06B6D4"   # cyan – projector / cross-attn
C_METHOD   = "#E5E7EB"   # light gray – existing methods
C_NEW      = "#F59E0B"   # amber – new method (compute_logits)
C_NEW_BG   = "#FEF3C7"   # pale amber background
C_BORDER   = "#374151"   # dark gray borders
C_TEXT     = "#1F2937"    # near-black text
C_ARROW    = "#6B7280"   # medium gray arrows
C_TITLE_BG = "#F3F4F6"   # very light gray for column headers
C_NEW_BADGE = "#DC2626"  # red for "NEW" badge

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper functions ───────────────────────────────────────────

def draw_box(ax, x, y, w, h, color, label, fontsize=12, fontweight="bold",
             textcolor="white", border_color=None, border_width=1.5,
             alpha=1.0, style="round,pad=0.1"):
    """Draw a rounded rectangle with centered label."""
    bc = border_color or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        facecolor=color, edgecolor=bc,
        linewidth=border_width, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor, zorder=3)
    return box


def draw_arrow_down(ax, x, y_top, y_bot, color=C_ARROW):
    """Draw a downward arrow between two y-coordinates."""
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.8, mutation_scale=14),
                zorder=2)


def draw_method_box(ax, x, y, w, h, label, is_new=False):
    """Draw a method box, highlighted if new."""
    if is_new:
        bg = C_NEW_BG
        ec = C_NEW
        lw = 2.5
    else:
        bg = C_METHOD
        ec = "#9CA3AF"
        lw = 1.2
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08",
        facecolor=bg, edgecolor=ec,
        linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=10,
            fontweight="bold" if is_new else "normal",
            fontfamily="monospace",
            color=C_NEW if is_new else C_TEXT, zorder=3)
    # "NEW" badge — positioned above the box, centered
    if is_new:
        badge_cx = x + w / 2
        badge_y = y + h + 0.05
        badge = FancyBboxPatch(
            (badge_cx - 0.28, badge_y), 0.56, 0.22,
            boxstyle="round,pad=0.04",
            facecolor=C_NEW_BADGE, edgecolor=C_NEW_BADGE,
            linewidth=0, zorder=4
        )
        ax.add_patch(badge)
        ax.text(badge_cx, badge_y + 0.11, "NEW",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=5)


# ── Layout constants ──────────────────────────────────────────
# Two columns: SP on left, Flamingo on right
col_left_x  = 0.8       # left column center-ish
col_right_x = 8.0       # right column center-ish
box_w       = 4.8        # width of architecture boxes
box_h       = 0.7        # height of architecture boxes
gap         = 0.35       # vertical gap between boxes

# ── Main title ────────────────────────────────────────────────
ax.text(7.5, 8.55, "OpenTSLM Architecture Variants with SDFT Addition",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color=C_TEXT)
ax.text(7.5, 8.15, "SDFT adds only compute_logits() — no architectural changes",
        ha="center", va="center", fontsize=11, fontstyle="italic",
        color="#6B7280")

# ── Column headers ────────────────────────────────────────────
for cx, title in [(col_left_x, "OpenTSLM_SP"), (col_right_x, "OpenTSLM_Flamingo")]:
    header = FancyBboxPatch(
        (cx - 0.1, 7.35), box_w + 0.2, 0.55,
        boxstyle="round,pad=0.08",
        facecolor=C_TITLE_BG, edgecolor="#D1D5DB",
        linewidth=1.2, zorder=1
    )
    ax.add_patch(header)
    ax.text(cx + box_w / 2, 7.625, title,
            ha="center", va="center", fontsize=13, fontweight="bold",
            fontfamily="monospace", color=C_TEXT)

# ═══════════════════════════════════════════════════════════════
#  LEFT COLUMN: OpenTSLMSP
# ═══════════════════════════════════════════════════════════════
lx = col_left_x
ly_start = 6.8

# Input
ly = ly_start
draw_box(ax, lx, ly, box_w, 0.5, "#E5E7EB", "Time-Series Input",
         fontsize=11, fontweight="normal", textcolor=C_TEXT,
         border_color="#9CA3AF")

# Arrow
ly_arrow_top = ly - 0.05
ly_next = ly - 0.5 - gap
draw_arrow_down(ax, lx + box_w / 2, ly_arrow_top, ly_next + box_h)

# TransformerCNNEncoder
ly = ly_next
draw_box(ax, lx, ly, box_w, box_h, C_ENCODER, "TransformerCNNEncoder",
         fontsize=12, textcolor="white")

# Arrow
draw_arrow_down(ax, lx + box_w / 2, ly - 0.05, ly - gap - box_h + box_h - 0.05)

# MLP Projector
ly2 = ly - gap - box_h
draw_box(ax, lx, ly2, box_w, box_h, C_PROJ, "MLP Projector",
         fontsize=12, textcolor="white")

# Arrow
draw_arrow_down(ax, lx + box_w / 2, ly2 - 0.05, ly2 - gap - box_h + box_h - 0.05)

# Llama-3.2-1B
ly3 = ly2 - gap - box_h
draw_box(ax, lx, ly3, box_w, box_h, C_LLM, "Llama-3.2-1B",
         fontsize=13, textcolor="white")

# ── Methods section (left) ────────────────────────────────────
method_y_top = ly3 - 0.5
method_label_y = method_y_top + 0.05
ax.text(lx + box_w / 2, method_label_y, "Methods",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#6B7280")

# Dashed enclosure for methods
methods_rect = FancyBboxPatch(
    (lx - 0.15, method_y_top - 1.35), box_w + 0.3, 1.3,
    boxstyle="round,pad=0.08",
    facecolor="white", edgecolor="#D1D5DB",
    linewidth=1.0, linestyle="--", zorder=1
)
ax.add_patch(methods_rect)

mw = 1.28   # method box width
mh = 0.42  # method box height
mgap = 0.1
m_start_x = lx + (box_w - 3 * mw - 2 * mgap) / 2
my = method_y_top - 1.1

draw_method_box(ax, m_start_x, my, mw, mh, "compute_loss()")
draw_method_box(ax, m_start_x + mw + mgap, my, mw, mh, "generate()")
draw_method_box(ax, m_start_x + 2 * (mw + mgap), my, mw, mh,
                "compute_logits()", is_new=True)

# ═══════════════════════════════════════════════════════════════
#  RIGHT COLUMN: OpenTSLMFlamingo
# ═══════════════════════════════════════════════════════════════
rx = col_right_x
ry_start = 6.8

# Input
ry = ry_start
draw_box(ax, rx, ry, box_w, 0.5, "#E5E7EB", "Time-Series Input",
         fontsize=11, fontweight="normal", textcolor=C_TEXT,
         border_color="#9CA3AF")

# Arrow
draw_arrow_down(ax, rx + box_w / 2, ry - 0.05, ry - 0.5 - gap + box_h)

# CNNTokenizer
ry_enc = ry - 0.5 - gap
draw_box(ax, rx, ry_enc, box_w, box_h, C_ENCODER, "CNNTokenizer",
         fontsize=12, textcolor="white")

# Arrow
draw_arrow_down(ax, rx + box_w / 2, ry_enc - 0.05,
                ry_enc - gap - box_h + box_h - 0.05)

# Gated Cross-Attention
ry2 = ry_enc - gap - box_h
draw_box(ax, rx, ry2, box_w, box_h, C_PROJ, "Gated Cross-Attention",
         fontsize=12, textcolor="white")

# Arrow
draw_arrow_down(ax, rx + box_w / 2, ry2 - 0.05,
                ry2 - gap - box_h + box_h - 0.05)

# Llama-3.2-1B
ry3 = ry2 - gap - box_h
draw_box(ax, rx, ry3, box_w, box_h, C_LLM, "Llama-3.2-1B",
         fontsize=13, textcolor="white")

# ── Methods section (right) ───────────────────────────────────
method_y_top_r = ry3 - 0.5
method_label_y_r = method_y_top_r + 0.05
ax.text(rx + box_w / 2, method_label_y_r, "Methods",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color="#6B7280")

methods_rect_r = FancyBboxPatch(
    (rx - 0.15, method_y_top_r - 1.35), box_w + 0.3, 1.3,
    boxstyle="round,pad=0.08",
    facecolor="white", edgecolor="#D1D5DB",
    linewidth=1.0, linestyle="--", zorder=1
)
ax.add_patch(methods_rect_r)

m_start_x_r = rx + (box_w - 3 * mw - 2 * mgap) / 2
my_r = method_y_top_r - 1.1

draw_method_box(ax, m_start_x_r, my_r, mw, mh, "compute_loss()")
draw_method_box(ax, m_start_x_r + mw + mgap, my_r, mw, mh, "generate()")
draw_method_box(ax, m_start_x_r + 2 * (mw + mgap), my_r, mw, mh,
                "compute_logits()", is_new=True)

# ── Legend / annotation at bottom ─────────────────────────────
legend_y = 0.45

# Existing method swatch
swatch_existing = FancyBboxPatch(
    (3.8, legend_y - 0.12), 0.35, 0.24,
    boxstyle="round,pad=0.04",
    facecolor=C_METHOD, edgecolor="#9CA3AF", linewidth=1.0, zorder=2
)
ax.add_patch(swatch_existing)
ax.text(4.3, legend_y, "Existing method", ha="left", va="center",
        fontsize=10, color=C_TEXT)

# New method swatch
swatch_new = FancyBboxPatch(
    (6.8, legend_y - 0.12), 0.35, 0.24,
    boxstyle="round,pad=0.04",
    facecolor=C_NEW_BG, edgecolor=C_NEW, linewidth=2.0, zorder=2
)
ax.add_patch(swatch_new)
ax.text(7.3, legend_y, "Added by SDFT", ha="left", va="center",
        fontsize=10, color=C_TEXT)

# Badge swatch
badge_sw = FancyBboxPatch(
    (9.7, legend_y - 0.08), 0.5, 0.22,
    boxstyle="round,pad=0.04",
    facecolor=C_NEW_BADGE, edgecolor=C_NEW_BADGE,
    linewidth=0, zorder=2
)
ax.add_patch(badge_sw)
ax.text(9.95, legend_y + 0.02, "NEW", ha="center", va="center",
        fontsize=7, fontweight="bold", color="white", zorder=3)
ax.text(10.35, legend_y, "= SDFT addition (method only)", ha="left",
        va="center", fontsize=10, color=C_TEXT)

# ── Divider line between columns ──────────────────────────────
ax.plot([7.4, 7.4], [1.0, 7.9], color="#E5E7EB", linewidth=1.5,
        linestyle="--", zorder=0)

# ── Save ──────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
fig.savefig("/home/wangni/notion-figures/self-distillation/fig_008.png",
            dpi=200, facecolor="white", bbox_inches="tight")
plt.close()
print("Saved: /home/wangni/notion-figures/self-distillation/fig_008.png")
