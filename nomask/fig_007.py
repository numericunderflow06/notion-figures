"""
fig_007: Cross-Channel Diagnostic Reasoning Example
Split diagram showing how 'cylinder compression issue' diagnosis benefits
from cross-channel attention (unmasked) vs. limited single-channel view (masked).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────
BG_WHITE = "#FFFFFF"
PANEL_BG_LEFT = "#F5F5F8"
PANEL_BG_RIGHT = "#EDF6F0"

CHT_COLOR = "#E05A33"
EGT_COLOR = "#E8913A"
RPM_COLOR = "#D94560"
FADED = "#D0D0D6"
POST_PROMPT_COLOR = "#3D6BA5"
DIAG_COLOR = "#2E7D52"
ARROW_ACTIVE = "#3D6BA5"
ARROW_MASKED = "#999999"
BLOCK_BORDER = "#888888"

# ── Helper: draw a tiny schematic waveform ──────────────────────────
def draw_waveform(ax, cx, cy, w=0.38, h=0.12, color="#333333",
                  alpha=1.0, style="normal"):
    n = 40
    x = np.linspace(cx - w / 2, cx + w / 2, n)
    if style == "diverge":
        y1 = cy + h * 0.5 * np.sin(np.linspace(0, 3 * np.pi, n)) * np.linspace(0.3, 1.0, n)
        y2 = cy - h * 0.5 * np.sin(np.linspace(0, 3 * np.pi, n)) * np.linspace(0.3, 1.0, n)
        ax.plot(x, y1, color=color, lw=1.2, alpha=alpha, zorder=5)
        ax.plot(x, y2, color=color, lw=1.2, alpha=alpha, zorder=5)
    elif style == "unstable":
        rng = np.random.RandomState(42)
        y = cy + h * 0.6 * np.sin(np.linspace(0, 6 * np.pi, n)) * rng.uniform(0.4, 1.0, n)
        ax.plot(x, y, color=color, lw=1.2, alpha=alpha, zorder=5)
    else:
        y = cy + h * 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
        ax.plot(x, y, color=color, lw=1.2, alpha=alpha, zorder=5)


def draw_channel_box(ax, x, y, w, h, label, color, alpha=1.0,
                     waveform_style="normal", show_waveform=True):
    fc = color if alpha >= 0.9 else FADED
    ec = BLOCK_BORDER if alpha >= 0.9 else "#CCCCCC"
    real_alpha = max(alpha, 0.4)
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec,
        linewidth=1.0, alpha=real_alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.70, label,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="#FFFFFF" if alpha >= 0.9 else "#AAAAAA",
            alpha=real_alpha, zorder=6)
    if show_waveform:
        draw_waveform(ax, x + w / 2, y + h * 0.28, w=w * 0.65, h=h * 0.22,
                      color="#FFFFFF" if alpha >= 0.9 else "#CCCCCC",
                      alpha=real_alpha, style=waveform_style)


def draw_arrow_line(ax, x0, y0, x1, y1, color, lw=1.5, alpha=1.0,
                    rad=0.0):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=12,
        color=color, lw=lw, alpha=alpha, zorder=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


# ── Figure setup ────────────────────────────────────────────────────
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 8.5),
                                  gridspec_kw={"wspace": 0.06})
fig.patch.set_facecolor(BG_WHITE)

for ax in (ax_l, ax_r):
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.3, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")

# ── Panel backgrounds ──────────────────────────────────────────────
ax_l.add_patch(FancyBboxPatch((0.0, 0.0), 10.0, 10.2,
               boxstyle="round,pad=0.2", facecolor=PANEL_BG_LEFT,
               edgecolor="#BBBBBB", lw=1.5, zorder=0))
ax_r.add_patch(FancyBboxPatch((0.0, 0.0), 10.0, 10.2,
               boxstyle="round,pad=0.2", facecolor=PANEL_BG_RIGHT,
               edgecolor="#88B8A0", lw=1.5, zorder=0))

# ── Panel titles ────────────────────────────────────────────────────
ax_l.text(5.0, 9.7, "Masked: Limited View", ha="center", va="center",
          fontsize=14, fontweight="bold", color="#555555", zorder=6)
ax_l.text(5.0, 9.25, "Post-prompt attends to last channel only (torch.eq)",
          ha="center", va="center", fontsize=9, color="#888888", zorder=6)

ax_r.text(5.0, 9.7, "Unmasked: Full Cross-Channel View", ha="center",
          va="center", fontsize=14, fontweight="bold", color="#2E7D52", zorder=6)
ax_r.text(5.0, 9.25, "Post-prompt attends to all channels (torch.ge)",
          ha="center", va="center", fontsize=9, color="#5A9E78", zorder=6)

# ═══════════════════════════════════════════════════════════════════
# Channel layout parameters
# ═══════════════════════════════════════════════════════════════════
box_w = 1.5
box_h = 0.85

groups = [
    ("CHT1-4", ["CHT1", "CHT2", "CHT3", "CHT4"], CHT_COLOR, "diverge"),
    ("EGT1-4", ["EGT1", "EGT2", "EGT3", "EGT4"], EGT_COLOR, "normal"),
    ("E1_RPM", ["E1_RPM"], RPM_COLOR, "unstable"),
]

# Y positions for the group labels
group_label_y = [8.6, 6.55, 5.0]
# Y positions for the box tops
group_box_y = [8.45, 6.40, 4.85]

# ═══════════════════════════════════════════════════════════════════
# LEFT PANEL – Masked
# ═══════════════════════════════════════════════════════════════════

left_channel_boxes = []  # (cx, cy_bottom, cy_top, is_last_overall)
total_channels = sum(len(g[1]) for g in groups)
channel_idx = 0

for gi, (grp_name, channels, color, wf_style) in enumerate(groups):
    gy_label = group_label_y[gi]
    gy_box_top = group_box_y[gi]

    ax_l.text(5.0, gy_label, grp_name, ha="center", va="bottom",
              fontsize=10, fontweight="bold", color="#AAAAAA", zorder=6)

    for ci, ch in enumerate(channels):
        cx = 5.0 - (len(channels) - 1) * (box_w + 0.3) / 2 + ci * (box_w + 0.3)
        cy_bot = gy_box_top - box_h
        is_last = (channel_idx == total_channels - 1)
        a = 1.0 if is_last else 0.3
        draw_channel_box(ax_l, cx - box_w / 2, cy_bot, box_w, box_h, ch,
                         color, alpha=a, waveform_style=wf_style)
        left_channel_boxes.append((cx, cy_bot, cy_bot + box_h, is_last))
        channel_idx += 1

# X marks on faded channels
for cx, cy_bot, cy_top, is_last in left_channel_boxes:
    if not is_last:
        sz = 0.22
        ax_l.plot([cx - sz, cx + sz], [cy_bot + (cy_top - cy_bot) / 2 - sz,
                  cy_bot + (cy_top - cy_bot) / 2 + sz],
                  color="#CC3333", lw=2.0, alpha=0.45, zorder=7)
        ax_l.plot([cx - sz, cx + sz], [cy_bot + (cy_top - cy_bot) / 2 + sz,
                  cy_bot + (cy_top - cy_bot) / 2 - sz],
                  color="#CC3333", lw=2.0, alpha=0.45, zorder=7)

# Post-prompt box (left)
pp_x, pp_y = 2.5, 2.3
pp_w, pp_h = 2.5, 0.85
pp_rect_l = FancyBboxPatch(
    (pp_x, pp_y), pp_w, pp_h,
    boxstyle="round,pad=0.05",
    facecolor=POST_PROMPT_COLOR, edgecolor="#2B5280",
    linewidth=1.5, zorder=3,
)
ax_l.add_patch(pp_rect_l)
ax_l.text(pp_x + pp_w / 2, pp_y + pp_h / 2, "Post-Prompt Region",
          ha="center", va="center", fontsize=11, fontweight="bold",
          color="#FFFFFF", zorder=6)

# Arrow from post-prompt to last channel (E1_RPM)
last_cx, last_cy_bot, last_cy_top, _ = left_channel_boxes[-1]
draw_arrow_line(ax_l,
                pp_x + pp_w / 2 + 0.2, pp_y + pp_h,
                last_cx, last_cy_bot,
                ARROW_MASKED, lw=2.8, alpha=0.85, rad=0.0)

# Annotation
ax_l.text(pp_x + pp_w + 0.3, pp_y + pp_h + 0.6,
          "Only last channel\nvisible",
          ha="left", va="center", fontsize=9.5, fontstyle="italic",
          color="#999999", zorder=6,
          bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF",
                    edgecolor="#CCCCCC", alpha=0.85))

# Diagnosis box (left)
diag_x, diag_y = 1.5, 0.5
diag_w, diag_h = 7.0, 0.8
diag_rect_l = FancyBboxPatch(
    (diag_x, diag_y), diag_w, diag_h,
    boxstyle="round,pad=0.05",
    facecolor="#E0E0E0", edgecolor="#AAAAAA",
    linewidth=1.5, zorder=3,
)
ax_l.add_patch(diag_rect_l)
ax_l.text(diag_x + diag_w / 2, diag_y + diag_h / 2,
          "Diagnosis: ???  (insufficient cross-channel information)",
          ha="center", va="center", fontsize=10.5, fontweight="bold",
          color="#888888", zorder=6)

draw_arrow_line(ax_l, pp_x + pp_w / 2, pp_y,
                diag_x + diag_w / 2, diag_y + diag_h,
                ARROW_MASKED, lw=2.0, alpha=0.55, rad=0.0)

# ═══════════════════════════════════════════════════════════════════
# RIGHT PANEL – Unmasked
# ═══════════════════════════════════════════════════════════════════

right_channel_centres = []
right_channel_colors = []

for gi, (grp_name, channels, color, wf_style) in enumerate(groups):
    gy_label = group_label_y[gi]
    gy_box_top = group_box_y[gi]

    ax_r.text(5.0, gy_label, grp_name, ha="center", va="bottom",
              fontsize=10, fontweight="bold", color=color, zorder=6)

    for ci, ch in enumerate(channels):
        cx = 5.0 - (len(channels) - 1) * (box_w + 0.3) / 2 + ci * (box_w + 0.3)
        cy_bot = gy_box_top - box_h
        draw_channel_box(ax_r, cx - box_w / 2, cy_bot, box_w, box_h, ch,
                         color, alpha=1.0, waveform_style=wf_style)
        right_channel_centres.append((cx, cy_bot))
        right_channel_colors.append(color)

    # Group annotations on right edge
    rightmost_cx = 5.0 + (len(channels) - 1) * (box_w + 0.3) / 2
    anno_x = rightmost_cx + box_w / 2 + 0.25
    anno_y = gy_box_top - box_h / 2
    if grp_name == "CHT1-4":
        ax_r.text(anno_x, anno_y, "Temperature\ndivergence",
                  ha="left", va="center", fontsize=8.5, fontstyle="italic",
                  color=CHT_COLOR, zorder=6)
    elif grp_name == "EGT1-4":
        ax_r.text(anno_x, anno_y, "Exhaust\npatterns",
                  ha="left", va="center", fontsize=8.5, fontstyle="italic",
                  color=EGT_COLOR, zorder=6)
    elif grp_name == "E1_RPM":
        ax_r.text(5.0 + box_w / 2 + 0.25, anno_y, "RPM\ninstability",
                  ha="left", va="center", fontsize=8.5, fontstyle="italic",
                  color=RPM_COLOR, zorder=6)

# Post-prompt box (right) — centred at x=5
pp_r_x = 3.75
pp_r_w = 2.5
pp_rect_r = FancyBboxPatch(
    (pp_r_x, pp_y), pp_r_w, pp_h,
    boxstyle="round,pad=0.05",
    facecolor=POST_PROMPT_COLOR, edgecolor="#2B5280",
    linewidth=1.5, zorder=5,
)
ax_r.add_patch(pp_rect_r)
ax_r.text(pp_r_x + pp_r_w / 2, pp_y + pp_h / 2, "Post-Prompt\nRegion",
          ha="center", va="center", fontsize=10.5, fontweight="bold",
          color="#FFFFFF", zorder=6)

# Attention beams from post-prompt to ALL channels
pp_top_x_r = pp_r_x + pp_r_w / 2
pp_top_y = pp_y + pp_h

n_arrows = len(right_channel_centres)
for idx, ((cx, cy_bot), acolor) in enumerate(zip(right_channel_centres, right_channel_colors)):
    spread = (idx / (n_arrows - 1) - 0.5) * 1.6 if n_arrows > 1 else 0
    ox = pp_top_x_r + spread * 0.25
    rad = (idx / (n_arrows - 1) - 0.5) * 0.20 if n_arrows > 1 else 0
    draw_arrow_line(ax_r, ox, pp_top_y, cx, cy_bot,
                    acolor, lw=1.6, alpha=0.7, rad=rad)

# Annotation — to the left, well clear of the box
ax_r.text(0.6, pp_y + pp_h * 0.45,
          "All channels\nvisible simultaneously",
          ha="center", va="center", fontsize=9, fontstyle="italic",
          color="#2E7D52", zorder=6,
          bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF",
                    edgecolor="#88B8A0", alpha=0.90))
# Small connector line from annotation to post-prompt box
ax_r.annotate("", xy=(pp_r_x, pp_y + pp_h * 0.45),
              xytext=(1.55, pp_y + pp_h * 0.45),
              arrowprops=dict(arrowstyle="-", color="#88B8A0",
                              lw=1.0, ls="--"), zorder=4)

# Diagnosis box (right)
diag_rect_r = FancyBboxPatch(
    (diag_x, diag_y), diag_w, diag_h,
    boxstyle="round,pad=0.05",
    facecolor=DIAG_COLOR, edgecolor="#1B5C3B",
    linewidth=1.5, zorder=3,
)
ax_r.add_patch(diag_rect_r)
ax_r.text(diag_x + diag_w / 2, diag_y + diag_h / 2,
          'Diagnosis: "cylinder compression issue"',
          ha="center", va="center", fontsize=11, fontweight="bold",
          color="#FFFFFF", zorder=6)

draw_arrow_line(ax_r, pp_r_x + pp_r_w / 2, pp_y,
                diag_x + diag_w / 2, diag_y + diag_h,
                DIAG_COLOR, lw=2.5, alpha=0.85, rad=0.0)

# ── Overall title ───────────────────────────────────────────────────
fig.suptitle("Cross-Channel Diagnostic Reasoning: Cylinder Compression Issue",
             fontsize=16, fontweight="bold", y=0.98, color="#333333")

# ── Save ────────────────────────────────────────────────────────────
out = "/home/wangni/notion-figures/nomask/fig_007.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_WHITE)
plt.close(fig)
print(f"Saved figure to {out}")
