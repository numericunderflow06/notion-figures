"""
fig_006: Experimental Conditions Comparison
Visual summary comparing the three experimental conditions (Masked Baseline,
Causal Unmasked, Full Unmasked) showing key parameter differences, mask behavior,
and expected trade-offs side by side.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------- colour palette ----------
COL_MASKED = "#2B6CB0"
COL_CAUSAL = "#DD8C1A"
COL_FULL   = "#C53030"

COL_MASKED_LIGHT  = "#BEE3F8"
COL_CAUSAL_LIGHT  = "#FEEBC8"
COL_FULL_LIGHT    = "#FED7D7"

COL_MASKED_MED    = "#63B3ED"
COL_CAUSAL_MED    = "#F6AD55"
COL_FULL_MED      = "#FC8181"

BG_WHITE = "#FFFFFF"
TEXT_DARK = "#1A202C"
TEXT_MED  = "#4A5568"
GRID_LINE = "#E2E8F0"

# ---------- figure ----------
fig = plt.figure(figsize=(18, 14.5), facecolor=BG_WHITE, dpi=200)

# Title
fig.text(0.50, 0.975, "Experimental Conditions Comparison",
         fontsize=22, fontweight="bold", color=TEXT_DARK, ha="center", va="top",
         fontfamily="sans-serif")
fig.text(0.50, 0.952, "Three cross-attention mask configurations for OpenTSLM-NoMask",
         fontsize=13, color=TEXT_MED, ha="center", va="top", fontfamily="sans-serif")

# ── Column definitions ──
cols = [
    {
        "name": "Masked Baseline",
        "color": COL_MASKED, "light": COL_MASKED_LIGHT, "med": COL_MASKED_MED,
        "params": [
            ("only_attend_immediate_media", "True"),
            ("fully_unmasked", "False"),
            ("mask_op", "torch.eq"),
        ],
        "bullets": [
            "Each text block attends only\nto its own channel's media",
            "Post-prompt attends to last\nchannel only",
            "Strong grounding between\ndescriptions and time series",
            "Information bottleneck at\ndiagnosis generation",
            "Original OpenTSLM design",
        ],
        "matrix": np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]),
    },
    {
        "name": "Causal Unmasked",
        "color": COL_CAUSAL, "light": COL_CAUSAL_LIGHT, "med": COL_CAUSAL_MED,
        "params": [
            ("only_attend_immediate_media", "False"),
            ("fully_unmasked", "False"),
            ("mask_op", "torch.ge"),
        ],
        "bullets": [
            "Each text block attends to its\nown and all prior channels",
            "Post-prompt attends to all\nchannels simultaneously",
            "Partial grounding preserved\n(block 1 still sees only ch 1)",
            "Direct cross-channel fusion\nin post-prompt region",
            "Primary proposal (minimal\nsingle-flag change)",
        ],
        "matrix": np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]),
    },
    {
        "name": "Full Unmasked",
        "color": COL_FULL, "light": COL_FULL_LIGHT, "med": COL_FULL_MED,
        "params": [
            ("only_attend_immediate_media", "N/A"),
            ("fully_unmasked", "True"),
            ("mask_op", "None (mask skipped)"),
        ],
        "bullets": [
            "All text tokens attend to all\nmedia tokens regardless of position",
            "Post-prompt attends to all\nchannels simultaneously",
            "No positional grounding;\nmodel must learn alignment",
            "Maximum cross-channel\ninformation flow everywhere",
            "Ablation study variant\n(most aggressive change)",
        ],
        "matrix": np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]),
    },
]

# Layout constants
x_starts = [0.035, 0.355, 0.675]
col_w = 0.285
top_y = 0.920


def draw_grid_fig(col_info, x0, grid_top, grid_h):
    """Draw the 5x4 attention grid using figure-level patches and text.

    Draws all cell patches first, then all text on top to prevent overlap.
    """
    c = col_info["color"]
    mat = col_info["matrix"]

    row_labels = ["Blk 1", "Blk 2", "Blk 3", "Blk 4", "Post"]
    col_labels = ["M1", "M2", "M3", "M4"]

    # Usable area inside the box (with padding)
    pad_l = 0.050   # left padding for row labels
    pad_r = 0.008
    pad_t = 0.032   # top padding for col labels
    pad_b = 0.008

    inner_x = x0 + pad_l
    inner_y_top = grid_top - pad_t
    inner_w = col_w - pad_l - pad_r
    inner_h = grid_h - pad_t - pad_b

    cell_w = inner_w / 4
    cell_h = inner_h / 5

    # --- Pass 1: Draw all cell patches ---
    for i in range(5):
        for j in range(4):
            cx = inner_x + j * cell_w
            cy = inner_y_top - (i + 1) * cell_h

            cell_pad = 0.003
            if mat[i, j] == 1:
                fc = c
                ec = c
            else:
                fc = "#F7FAFC"
                ec = "#CBD5E0"

            rect = FancyBboxPatch(
                (cx + cell_pad, cy + cell_pad),
                cell_w - 2 * cell_pad, cell_h - 2 * cell_pad,
                boxstyle="round,pad=0.003", linewidth=0.8,
                edgecolor=ec, facecolor=fc,
                transform=fig.transFigure, clip_on=False, zorder=2)
            fig.patches.append(rect)

    # Dashed highlight for post-prompt row (last row, index 4)
    hl_y_bottom = inner_y_top - 5 * cell_h
    highlight = FancyBboxPatch(
        (inner_x - 0.003, hl_y_bottom - 0.001),
        inner_w + 0.006, cell_h + 0.002,
        boxstyle="round,pad=0.004", linewidth=2,
        edgecolor=c, facecolor="none", linestyle="--", alpha=0.7,
        transform=fig.transFigure, clip_on=False, zorder=5)
    fig.patches.append(highlight)

    # --- Pass 2: Draw all text on top of patches ---
    # Column headers
    for j, lab in enumerate(col_labels):
        cx = inner_x + (j + 0.5) * cell_w
        cy = grid_top - pad_t * 0.45
        fig.text(cx, cy, lab, fontsize=8.5, fontweight="bold",
                 ha="center", va="center", color=TEXT_MED,
                 transform=fig.transFigure, fontfamily="sans-serif",
                 zorder=10)

    # Row labels (with white background to ensure visibility)
    for i in range(5):
        ry = inner_y_top - (i + 0.5) * cell_h
        fig.text(inner_x - 0.012, ry, row_labels[i], fontsize=7.5,
                 ha="right", va="center", color=TEXT_MED,
                 transform=fig.transFigure, fontfamily="sans-serif",
                 linespacing=1.1, zorder=15,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor="none", alpha=0.95))

    # Cell symbols (checkmarks / crosses)
    for i in range(5):
        for j in range(4):
            cx = inner_x + j * cell_w
            cy = inner_y_top - (i + 1) * cell_h

            if mat[i, j] == 1:
                symb = "\u2713"
                symb_c = "white"
            else:
                symb = "\u2717"
                symb_c = "#CBD5E0"

            fig.text(cx + cell_w / 2, cy + cell_h / 2, symb,
                     fontsize=13, fontweight="bold", color=symb_c,
                     ha="center", va="center",
                     transform=fig.transFigure, zorder=10)


def draw_column(col_info, x0):
    """Draw one complete condition column."""
    c = col_info["color"]
    cl = col_info["light"]
    cm = col_info["med"]

    # ── Header banner ──
    header_h = 0.040
    header = FancyBboxPatch(
        (x0, top_y - header_h), col_w, header_h,
        boxstyle="round,pad=0.006", linewidth=2,
        edgecolor=c, facecolor=c,
        transform=fig.transFigure, clip_on=False)
    fig.patches.append(header)
    fig.text(x0 + col_w / 2, top_y - header_h / 2, col_info["name"],
             fontsize=15, fontweight="bold", color="white",
             ha="center", va="center", fontfamily="sans-serif",
             transform=fig.transFigure)

    # ── Parameters box ──
    param_top = top_y - header_h - 0.010
    param_h = 0.095
    param_box = FancyBboxPatch(
        (x0, param_top - param_h), col_w, param_h,
        boxstyle="round,pad=0.006", linewidth=1.5,
        edgecolor=cm, facecolor=cl,
        transform=fig.transFigure, clip_on=False)
    fig.patches.append(param_box)

    fig.text(x0 + col_w / 2, param_top - 0.008, "Parameters",
             fontsize=11, fontweight="bold", color=c,
             ha="center", va="top", fontfamily="sans-serif",
             transform=fig.transFigure)

    py = param_top - 0.030
    for pname, pval in col_info["params"]:
        fig.text(x0 + 0.012, py, f"{pname}:", fontsize=9.5, color=TEXT_MED,
                 ha="left", va="top", fontfamily="monospace",
                 transform=fig.transFigure)
        fig.text(x0 + col_w - 0.012, py, pval, fontsize=9.5, fontweight="bold",
                 color=c, ha="right", va="top", fontfamily="monospace",
                 transform=fig.transFigure)
        py -= 0.022

    # ── Attention pattern diagram ──
    diag_top = param_top - param_h - 0.010
    diag_h = 0.240
    diag_box = FancyBboxPatch(
        (x0, diag_top - diag_h), col_w, diag_h,
        boxstyle="round,pad=0.006", linewidth=1.5,
        edgecolor=cm, facecolor=BG_WHITE,
        transform=fig.transFigure, clip_on=False)
    fig.patches.append(diag_box)

    fig.text(x0 + col_w / 2, diag_top - 0.006, "Attention Pattern (4-channel)",
             fontsize=11, fontweight="bold", color=c,
             ha="center", va="top", fontfamily="sans-serif",
             transform=fig.transFigure)
    fig.text(x0 + col_w / 2, diag_top - 0.022,
             "dashed = diagnosis region (post-prompt)",
             fontsize=7.5, color=TEXT_MED, fontstyle="italic",
             ha="center", va="top", fontfamily="sans-serif",
             transform=fig.transFigure)

    draw_grid_fig(col_info, x0, diag_top - 0.032, diag_h - 0.040)

    # ── Key characteristics ──
    char_top = diag_top - diag_h - 0.010
    char_h = 0.215
    char_box = FancyBboxPatch(
        (x0, char_top - char_h), col_w, char_h,
        boxstyle="round,pad=0.006", linewidth=1.5,
        edgecolor=cm, facecolor=cl,
        transform=fig.transFigure, clip_on=False)
    fig.patches.append(char_box)

    fig.text(x0 + col_w / 2, char_top - 0.006, "Key Characteristics",
             fontsize=11, fontweight="bold", color=c,
             ha="center", va="top", fontfamily="sans-serif",
             transform=fig.transFigure)

    by = char_top - 0.030
    for bullet in col_info["bullets"]:
        fig.text(x0 + 0.014, by, "\u2022", fontsize=12, color=c,
                 ha="left", va="top", fontfamily="sans-serif",
                 transform=fig.transFigure)
        fig.text(x0 + 0.026, by, bullet, fontsize=9.5, color=TEXT_DARK,
                 ha="left", va="top", fontfamily="sans-serif",
                 linespacing=1.25,
                 transform=fig.transFigure)
        n_lines = bullet.count("\n") + 1
        by -= 0.020 * n_lines + 0.010


# Draw all three columns
for col_info, x0 in zip(cols, x_starts):
    draw_column(col_info, x0)


# ══════════════════════════════════════════════════════════════════
# Bottom arrow: Expressiveness vs Grounding trade-off spectrum
# ══════════════════════════════════════════════════════════════════
arrow_y_center = 0.050
arrow_x_left = 0.06
arrow_x_right = 0.94

# Background strip
strip = FancyBboxPatch(
    (0.025, 0.010), 0.95, 0.082,
    boxstyle="round,pad=0.008", linewidth=1.5,
    edgecolor=GRID_LINE, facecolor="#F7FAFC",
    transform=fig.transFigure, clip_on=False)
fig.patches.append(strip)

# Arrow label
fig.text(0.50, 0.086, "Trade-off Spectrum",
         fontsize=12, fontweight="bold", color=TEXT_DARK,
         ha="center", va="top", fontfamily="sans-serif",
         transform=fig.transFigure)

# Draw gradient bar
bar_h = 0.014
n_seg = 300
xs = np.linspace(arrow_x_left, arrow_x_right, n_seg + 1)
for i in range(n_seg):
    t = i / n_seg
    if t < 0.5:
        s = t / 0.5
        r = int((1 - s) * 0x2B + s * 0xDD) / 255
        g = int((1 - s) * 0x6C + s * 0x8C) / 255
        b = int((1 - s) * 0xB0 + s * 0x1A) / 255
    else:
        s = (t - 0.5) / 0.5
        r = int((1 - s) * 0xDD + s * 0xC5) / 255
        g = int((1 - s) * 0x8C + s * 0x30) / 255
        b = int((1 - s) * 0x1A + s * 0x30) / 255
    seg = mpatches.Rectangle(
        (xs[i], arrow_y_center - bar_h / 2), xs[i + 1] - xs[i], bar_h,
        facecolor=(r, g, b), edgecolor="none",
        transform=fig.transFigure, clip_on=False)
    fig.patches.append(seg)

# Arrow heads
fig.text(arrow_x_left - 0.012, arrow_y_center, "\u25C0", fontsize=15,
         color=COL_MASKED, ha="right", va="center", transform=fig.transFigure)
fig.text(arrow_x_right + 0.012, arrow_y_center, "\u25B6", fontsize=15,
         color=COL_FULL, ha="left", va="center", transform=fig.transFigure)

# Left label
fig.text(arrow_x_left + 0.005, arrow_y_center + 0.016, "Conservative",
         fontsize=10, fontweight="bold", color=COL_MASKED,
         ha="left", va="bottom", transform=fig.transFigure)
fig.text(arrow_x_left + 0.005, arrow_y_center - 0.018, "Strong grounding",
         fontsize=9, color=COL_MASKED,
         ha="left", va="top", transform=fig.transFigure)

# Right label
fig.text(arrow_x_right - 0.005, arrow_y_center + 0.016, "Aggressive",
         fontsize=10, fontweight="bold", color=COL_FULL,
         ha="right", va="bottom", transform=fig.transFigure)
fig.text(arrow_x_right - 0.005, arrow_y_center - 0.018, "Maximum expressiveness",
         fontsize=9, color=COL_FULL,
         ha="right", va="top", transform=fig.transFigure)

# Center label
fig.text(0.50, arrow_y_center + 0.016, "Balanced",
         fontsize=10, fontweight="bold", color=COL_CAUSAL,
         ha="center", va="bottom", transform=fig.transFigure)
fig.text(0.50, arrow_y_center - 0.018, "Causal cross-channel fusion",
         fontsize=9, color=COL_CAUSAL,
         ha="center", va="top", transform=fig.transFigure)

# Condition dots on the gradient bar
for mx, mc in [
    (x_starts[0] + col_w / 2, COL_MASKED),
    (x_starts[1] + col_w / 2, COL_CAUSAL),
    (x_starts[2] + col_w / 2, COL_FULL),
]:
    dot = mpatches.Circle(
        (mx, arrow_y_center), radius=0.008,
        facecolor="white", edgecolor=mc, linewidth=2.5,
        transform=fig.transFigure, clip_on=False, zorder=10)
    fig.patches.append(dot)
    inner_dot = mpatches.Circle(
        (mx, arrow_y_center), radius=0.004,
        facecolor=mc, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=11)
    fig.patches.append(inner_dot)


# Save
out_path = "/home/wangni/notion-figures/nomask/fig_006.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_WHITE, pad_inches=0.12)
plt.close(fig)
print(f"Saved: {out_path}")
