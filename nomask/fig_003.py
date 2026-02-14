"""
fig_003: Cross-Attention Masking Code Flow
Vertical flowchart showing the 5-step masking logic with a fork at step 4.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
BG           = "#FFFFFF"
BOX_FILL     = "#F0F4FA"
BOX_EDGE     = "#4A6FA5"
FORK_EQ_FILL = "#FFF0F0"
FORK_EQ_EDGE = "#C0392B"
FORK_GE_FILL = "#EAFAF1"
FORK_GE_EDGE = "#27AE60"
ARROW_CLR    = "#3B5998"
HIGHLIGHT_FILL = "#FFF3E0"
HIGHLIGHT_EDGE = "#C0392B"
TEXT_DARK    = "#1A1A2E"
TEXT_MED     = "#3D3D5C"
CODE_CLR     = "#B71C1C"
STEP_NUM_BG  = "#4A6FA5"

# ── figure setup ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 16.5), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 11)
ax.set_ylim(0, 16.5)
ax.axis("off")

# ── title ───────────────────────────────────────────────────────────
ax.text(5.5, 16.15, "Cross-Attention Masking Code Flow",
        fontsize=18, fontweight="bold", ha="center", va="center",
        color=TEXT_DARK, fontfamily="sans-serif")
ax.text(5.5, 15.82, "MaskedCrossAttention  ·  open_flamingo/src/helpers.py",
        fontsize=10, ha="center", va="center", color=TEXT_MED,
        fontfamily="monospace", style="italic")

# ── helpers ─────────────────────────────────────────────────────────
def draw_step_box(ax, cx, cy, w, h, step_num, title, desc, code,
                  fill=BOX_FILL, edge=BOX_EDGE, code_color=CODE_CLR,
                  edge_lw=2.0):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=fill, edgecolor=edge, linewidth=edge_lw,
                         zorder=2)
    ax.add_patch(box)
    badge = plt.Circle((cx - w/2 + 0.35, cy + h/2 - 0.05), 0.22,
                        color=STEP_NUM_BG, zorder=3)
    ax.add_patch(badge)
    ax.text(cx - w/2 + 0.35, cy + h/2 - 0.05, str(step_num),
            fontsize=11, fontweight="bold", color="white",
            ha="center", va="center", zorder=4)
    ax.text(cx, cy + h/2 - 0.35, title,
            fontsize=12, fontweight="bold", ha="center", va="center",
            color=TEXT_DARK, fontfamily="sans-serif", zorder=3)
    ax.text(cx, cy - 0.05, desc,
            fontsize=9.5, ha="center", va="center",
            color=TEXT_MED, fontfamily="sans-serif", zorder=3,
            linespacing=1.3)
    ax.text(cx, cy - h/2 + 0.35, code,
            fontsize=9.5, fontweight="bold", ha="center", va="center",
            color=code_color, fontfamily="monospace", zorder=3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FAFAFA",
                      edgecolor="#CCCCCC", linewidth=0.8))

def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_CLR, lw=2.0,
               style="->", connectionstyle="arc3,rad=0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=18,
                            connectionstyle=connectionstyle,
                            zorder=1)
    ax.add_patch(arrow)

# ── layout positions ────────────────────────────────────────────────
cx = 5.5
box_w = 7.0
box_h = 1.35
gap = 0.40

# Top: input_ids label → step 1
input_y = 15.40       # well below subtitle at 15.82
y1 = input_y - 0.65   # step 1 center
y2 = y1 - box_h - gap
y3 = y2 - box_h - gap
y4_header = y3 - box_h - gap - 0.10
hdr_h = 1.25
y4_branch = y4_header - 1.75
branch_h = 1.6
y5 = y4_branch - branch_h/2 - gap - box_h/2 - 0.15

# ── Input label + arrow ────────────────────────────────────────────
ax.text(cx, input_y + 0.12, "input_ids",
        fontsize=12, fontweight="bold", ha="center", va="center",
        color=TEXT_DARK, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#E3F2FD",
                  edgecolor="#1976D2", linewidth=1.5))
draw_arrow(ax, cx, input_y - 0.08, cx, y1 + box_h/2)

# ── Step 1 ──────────────────────────────────────────────────────────
draw_step_box(ax, cx, y1, box_w, box_h,
              step_num=1,
              title="Media Location Detection",
              desc="Identify <image> token positions in input sequence",
              code="media_locations = (input_ids == media_token_id)")
draw_arrow(ax, cx, y1 - box_h/2, cx, y2 + box_h/2)

# ── Step 2 ──────────────────────────────────────────────────────────
draw_step_box(ax, cx, y2, box_w, box_h,
              step_num=2,
              title="Cumulative Position Counter",
              desc="Assign block index to each text token (0 = before any media)",
              code="text_time = media_locations.cumsum(dim=-1)")
draw_arrow(ax, cx, y2 - box_h/2, cx, y3 + box_h/2)

# ── Step 3 ──────────────────────────────────────────────────────────
draw_step_box(ax, cx, y3, box_w, box_h,
              step_num=3,
              title="Media Time Indices",
              desc="Create sequential index for each media (time series) item",
              code="media_time = torch.arange(T_img) + 1")
draw_arrow(ax, cx, y3 - box_h/2, cx, y4_header + hdr_h/2)

# ── Step 4 header (highlighted) ────────────────────────────────────
hdr_w = 7.0
draw_step_box(ax, cx, y4_header, hdr_w, hdr_h,
              step_num=4,
              title="Mask Operation Selection",
              desc="mask_op chosen by only_attend_immediate_media flag",
              code="mask = mask_op(text_time, media_time)",
              fill=HIGHLIGHT_FILL, edge=HIGHLIGHT_EDGE, edge_lw=3.0)

# "MODIFICATION POINT" badge to the right
ax.text(cx + hdr_w/2 + 0.15, y4_header,
        "MODIFICATION\nPOINT",
        fontsize=7.5, fontweight="bold", color="#C0392B",
        ha="left", va="center", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE",
                  edgecolor="#C0392B", linewidth=1.2))

# ── Flag annotation between header and branches ─────────────────────
flag_y = y4_header - hdr_h/2 - 0.30
ax.text(cx, flag_y,
        "only_attend_immediate_media",
        fontsize=8.5, fontfamily="monospace", color="#555555",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF9C4",
                  edgecolor="#F9A825", linewidth=1.0),
        zorder=5)
ax.text(cx - 1.95, flag_y,
        "True  \u2192", fontsize=8.5, color="#C0392B", fontweight="bold",
        ha="right", va="center", fontfamily="monospace", zorder=5)
ax.text(cx + 1.95, flag_y,
        "\u2190  False", fontsize=8.5, color="#27AE60", fontweight="bold",
        ha="left", va="center", fontfamily="monospace", zorder=5)

# ── Fork branches ───────────────────────────────────────────────────
branch_left_x = 2.85
branch_right_x = 8.15
branch_w = 3.9

# Fork arrows
draw_arrow(ax, cx - 1.2, y4_header - hdr_h/2,
           branch_left_x, y4_branch + branch_h/2,
           connectionstyle="arc3,rad=0.12", lw=2.2, color="#C0392B")
draw_arrow(ax, cx + 1.2, y4_header - hdr_h/2,
           branch_right_x, y4_branch + branch_h/2,
           connectionstyle="arc3,rad=-0.12", lw=2.2, color="#27AE60")

# Left branch: torch.eq (Masked — original)
box_eq = FancyBboxPatch((branch_left_x - branch_w/2, y4_branch - branch_h/2),
                         branch_w, branch_h,
                         boxstyle="round,pad=0.15",
                         facecolor=FORK_EQ_FILL, edgecolor=FORK_EQ_EDGE,
                         linewidth=2.5, linestyle="--", zorder=2)
ax.add_patch(box_eq)
ax.text(branch_left_x, y4_branch + branch_h/2 - 0.32,
        "Masked (Original)", fontsize=11, fontweight="bold",
        ha="center", va="center", color="#C0392B")
ax.text(branch_left_x, y4_branch + 0.05,
        "Strict equality: each text block\nattends only to its own channel",
        fontsize=9, ha="center", va="center", color=TEXT_MED,
        linespacing=1.3)
ax.text(branch_left_x, y4_branch - branch_h/2 + 0.35,
        "mask_op = torch.eq",
        fontsize=9.5, fontweight="bold", ha="center", va="center",
        color="#C0392B", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FAFAFA",
                  edgecolor="#E0E0E0", linewidth=0.8))

# Right branch: torch.ge (Causal Unmasked — proposed)
box_ge = FancyBboxPatch((branch_right_x - branch_w/2, y4_branch - branch_h/2),
                         branch_w, branch_h,
                         boxstyle="round,pad=0.15",
                         facecolor=FORK_GE_FILL, edgecolor=FORK_GE_EDGE,
                         linewidth=2.5, zorder=2)
ax.add_patch(box_ge)
ax.text(branch_right_x, y4_branch + branch_h/2 - 0.32,
        "Causal Unmasked (Proposed)", fontsize=11, fontweight="bold",
        ha="center", va="center", color="#27AE60")
ax.text(branch_right_x, y4_branch + 0.05,
        "Causal inequality: each text block\nattends to current + prior channels",
        fontsize=9, ha="center", va="center", color=TEXT_MED,
        linespacing=1.3)
ax.text(branch_right_x, y4_branch - branch_h/2 + 0.35,
        "mask_op = torch.ge",
        fontsize=9.5, fontweight="bold", ha="center", va="center",
        color="#27AE60", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FAFAFA",
                  edgecolor="#E0E0E0", linewidth=0.8))

# Converging arrows → step 5
draw_arrow(ax, branch_left_x, y4_branch - branch_h/2,
           cx, y5 + box_h/2,
           connectionstyle="arc3,rad=-0.12", lw=2.0)
draw_arrow(ax, branch_right_x, y4_branch - branch_h/2,
           cx, y5 + box_h/2,
           connectionstyle="arc3,rad=0.12", lw=2.0)

# ── Step 5 ──────────────────────────────────────────────────────────
draw_step_box(ax, cx, y5, box_w, box_h,
              step_num=5,
              title="Pre-Media Zeroing",
              desc="Zero cross-attention output for text before first <image> token",
              code="text_without_media_mask = (text_time == 0)")

# ── Output label ────────────────────────────────────────────────────
out_y = y5 - box_h/2 - 0.55
draw_arrow(ax, cx, y5 - box_h/2, cx, out_y + 0.15)
ax.text(cx, out_y - 0.05, "Masked Cross-Attention Output",
        fontsize=11, fontweight="bold", ha="center", va="center",
        color=TEXT_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EAF6",
                  edgecolor="#3F51B5", linewidth=1.5))

# ── Legend (positioned right below output) ──────────────────────────
legend_y = out_y - 0.75
legend_x = 1.5
ax.text(legend_x, legend_y, "Legend:", fontsize=9, fontweight="bold",
        color=TEXT_DARK, va="center")

line_eq = plt.Line2D([legend_x + 1.0, legend_x + 2.0], [legend_y, legend_y],
                     color="#C0392B", linewidth=2.2, linestyle="--")
ax.add_line(line_eq)
ax.text(legend_x + 2.2, legend_y, "Original (torch.eq)",
        fontsize=8.5, color="#C0392B", va="center", fontfamily="monospace")

line_ge = plt.Line2D([legend_x + 5.2, legend_x + 6.2], [legend_y, legend_y],
                     color="#27AE60", linewidth=2.2, linestyle="-")
ax.add_line(line_ge)
ax.text(legend_x + 6.4, legend_y, "Proposed (torch.ge)",
        fontsize=8.5, color="#27AE60", va="center", fontfamily="monospace")

plt.savefig("/home/wangni/notion-figures/nomask/fig_003.png",
            dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
plt.close()

print("Saved: /home/wangni/notion-figures/nomask/fig_003.png")
