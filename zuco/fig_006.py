"""
fig_006: Eye-Tracking Metrics Overview
Illustrative diagram of the 5 eye-tracking metrics used as input channels:
FFD, GD, GPT, TRT, nFixations — showing how they relate to eye movement during reading.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Color palette ──────────────────────────────────────────────────────
COLORS = {
    "FFD":        "#E63946",   # red
    "GD":         "#457B9D",   # steel blue
    "GPT":        "#2A9D8F",   # teal
    "TRT":        "#E9C46A",   # golden/amber
    "nFixations": "#6A4C93",   # purple
}
# Darker TRT for text readability
TRT_TEXT = "#C4A030"

BG_COLOR   = "white"
WORD_BG    = "#EAEEF3"
WORD_EDGE  = "#B0BEC5"
SCANPATH   = "#555555"

# ── Sentence words ────────────────────────────────────────────────────
words = ["The", "researcher", "analyzed", "the", "complex", "data", "carefully", "."]
n_words = len(words)

# ── Figure layout ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14), facecolor=BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.set_aspect("equal")
ax.axis("off")

# ════════════════════════════════════════════════════════════════════════
# SECTION 1: Title (y ~ 13.0–13.6)
# ════════════════════════════════════════════════════════════════════════
ax.text(8, 13.55, "Eye-Tracking Metrics Overview",
        fontsize=22, fontweight="bold", ha="center", va="center", color="#1A1A2E")
ax.text(8, 13.1, "5 word-level metrics captured during reading, used as input channels for classification",
        fontsize=11.5, ha="center", va="center", color="#555555", style="italic")

# ════════════════════════════════════════════════════════════════════════
# SECTION 2: Scanpath + Sentence (y ~ 10.5–12.5)
# ════════════════════════════════════════════════════════════════════════

# Draw word boxes
word_y = 10.8
box_h  = 0.6
gap    = 0.15
word_widths = [0.7, 1.7, 1.4, 0.7, 1.2, 0.8, 1.5, 0.4]
total_w = sum(word_widths) + gap * (n_words - 1)
start_x = 8 - total_w / 2

word_centers_x = []
word_lefts = []
word_rights = []
x = start_x
for i, (w, ww) in enumerate(zip(words, word_widths)):
    rect = FancyBboxPatch((x, word_y - box_h / 2), ww, box_h,
                          boxstyle="round,pad=0.05",
                          facecolor=WORD_BG, edgecolor=WORD_EDGE, linewidth=1.2)
    ax.add_patch(rect)
    cx = x + ww / 2
    word_centers_x.append(cx)
    word_lefts.append(x)
    word_rights.append(x + ww)
    ax.text(cx, word_y, w, fontsize=13, ha="center", va="center",
            fontfamily="monospace", fontweight="bold", color="#1A1A2E")
    x += ww + gap

ax.text(start_x - 0.15, word_y, "Sentence:",
        fontsize=10, ha="right", va="center", color="#888888", fontweight="bold")

# Scanpath
fixation_seq = [
    (0, 1), (1, 1), (1, 2), (2, 1), (3, 1),
    (4, 1), (5, 1), (4, 2), (5, 2), (6, 1), (7, 1),
]

scan_y = 11.8
fix_radius = 0.13

# Saccade arrows
for k in range(len(fixation_seq) - 1):
    wi1, wi2 = fixation_seq[k][0], fixation_seq[k + 1][0]
    x1, x2 = word_centers_x[wi1], word_centers_x[wi2]
    if wi2 < wi1:  # regression
        mid_x = (x1 + x2) / 2
        mid_y = scan_y + 0.55
        t = np.linspace(0, 1, 40)
        sx = (1 - t)**2 * x1 + 2 * (1 - t) * t * mid_x + t**2 * x2
        sy = (1 - t)**2 * scan_y + 2 * (1 - t) * t * mid_y + t**2 * scan_y
        ax.plot(sx, sy, color="#CC4444", linewidth=1.2, alpha=0.6, linestyle="--")
        ax.annotate("", xy=(x2, scan_y), xytext=(sx[-3], sy[-3]),
                    arrowprops=dict(arrowstyle="->", color="#CC4444", lw=1.2))
    else:
        ax.annotate("", xy=(x2, scan_y), xytext=(x1, scan_y),
                    arrowprops=dict(arrowstyle="->", color=SCANPATH, lw=0.9, alpha=0.5))

# Fixation dots
for k, (wi, fn) in enumerate(fixation_seq):
    r = fix_radius * (0.85 + 0.15 * fn)
    circle = plt.Circle((word_centers_x[wi], scan_y), r,
                         facecolor=SCANPATH, edgecolor="white", linewidth=0.8, alpha=0.7, zorder=5)
    ax.add_patch(circle)

ax.text(start_x - 0.15, scan_y, "Scanpath:",
        fontsize=10, ha="right", va="center", color="#888888", fontweight="bold")

# Scanpath legend
ax.plot([11.5, 12.2], [12.5, 12.5], color=SCANPATH, linewidth=1, alpha=0.6)
ax.annotate("", xy=(12.2, 12.5), xytext=(11.9, 12.5),
            arrowprops=dict(arrowstyle="->", color=SCANPATH, lw=0.9))
ax.text(12.35, 12.5, "saccade", fontsize=9, va="center", color="#777")
ax.plot([13.3, 14.0], [12.5, 12.5], color="#CC4444", linewidth=1, alpha=0.6, linestyle="--")
ax.annotate("", xy=(14.0, 12.5), xytext=(13.7, 12.5),
            arrowprops=dict(arrowstyle="->", color="#CC4444", lw=0.9))
ax.text(14.15, 12.5, "regression", fontsize=9, va="center", color="#777")

# Highlight "complex" as example word
focus_word = 4
fw_left = word_lefts[focus_word]
fw_right = word_rights[focus_word]
fw_cx = word_centers_x[focus_word]
highlight = FancyBboxPatch((fw_left - 0.06, word_y - box_h / 2 - 0.06),
                           (fw_right - fw_left) + 0.12, box_h + 0.12,
                           boxstyle="round,pad=0.05",
                           facecolor="none", edgecolor="#E63946",
                           linewidth=2.2, linestyle="--")
ax.add_patch(highlight)
ax.text(fw_cx, word_y - box_h / 2 - 0.2, "example word ▼",
        fontsize=8.5, ha="center", va="top", color="#E63946", fontweight="bold")

# ════════════════════════════════════════════════════════════════════════
# SECTION 3: Fixation Timeline + Metric Spans (y ~ 6.8–10.0)
# ════════════════════════════════════════════════════════════════════════

# Background panel for timeline section
tl_panel = FancyBboxPatch((1.0, 6.55), 14.0, 3.5,
                           boxstyle="round,pad=0.15",
                           facecolor="#FAFBFC", edgecolor="#E0E4E8", linewidth=1)
ax.add_patch(tl_panel)

ax.text(8, 9.8, 'Fixation timeline for "complex" — what each metric captures',
        fontsize=11, ha="center", va="center", color="#333", fontweight="bold")

# Timeline axis
tl_y = 9.25
tl_x_left = 2.5
tl_x_right = 13.5
ax.annotate("", xy=(tl_x_right + 0.3, tl_y), xytext=(tl_x_left - 0.3, tl_y),
            arrowprops=dict(arrowstyle="->", color="#BBB", lw=1.3))
ax.text(tl_x_right + 0.45, tl_y, "time", fontsize=9, va="center", color="#999", style="italic")

# Fixation blocks
fix1_left, fix1_right = 3.5, 5.0
fix1_cx = (fix1_left + fix1_right) / 2
sacc1_right = 7.2
fix2_left, fix2_right = 10.5, 11.8
fix2_cx = (fix2_left + fix2_right) / 2
block_h = 0.35

# Fix 1
fix1_rect = FancyBboxPatch((fix1_left, tl_y - block_h / 2), fix1_right - fix1_left, block_h,
                            boxstyle="round,pad=0.04", facecolor="#4A4A4A", edgecolor="white", linewidth=1)
ax.add_patch(fix1_rect)
ax.text(fix1_cx, tl_y, "Fixation 1", fontsize=8.5, ha="center", va="center", color="white", fontweight="bold")
ax.text(fix1_cx, tl_y + 0.35, "1st pass", fontsize=8, ha="center", va="bottom", color="#888")

# Saccade away
ax.annotate("", xy=(sacc1_right, tl_y), xytext=(fix1_right + 0.15, tl_y),
            arrowprops=dict(arrowstyle="->", color="#BBB", lw=1.2, linestyle="--"))
ax.text((fix1_right + 0.15 + sacc1_right) / 2, tl_y + 0.28, "eyes leave word",
        fontsize=7.5, ha="center", va="bottom", color="#AAA", style="italic")

# Regression back
ax.annotate("", xy=(fix2_left - 0.15, tl_y), xytext=(sacc1_right + 0.15, tl_y),
            arrowprops=dict(arrowstyle="->", color="#CC4444", lw=1.2, linestyle="--"))
ax.text((sacc1_right + fix2_left) / 2, tl_y + 0.28, "regression back",
        fontsize=7.5, ha="center", va="bottom", color="#CC4444", style="italic")

# Fix 2
fix2_rect = FancyBboxPatch((fix2_left, tl_y - block_h / 2), fix2_right - fix2_left, block_h,
                            boxstyle="round,pad=0.04", facecolor="#4A4A4A", edgecolor="white", linewidth=1)
ax.add_patch(fix2_rect)
ax.text(fix2_cx, tl_y, "Fixation 2", fontsize=8.5, ha="center", va="center", color="white", fontweight="bold")
ax.text(fix2_cx, tl_y + 0.35, "2nd pass", fontsize=8, ha="center", va="bottom", color="#888")

# ── Metric span brackets under timeline ────────────────────────────────
span_y_start = tl_y - 0.45
span_gap = 0.42

def draw_bracket(ax, xl, xr, y, color, label, label_x=None):
    """Draw U-bracket with label."""
    ax.plot([xl, xl], [y + 0.08, y - 0.08], color=color, lw=2.2, solid_capstyle="round")
    ax.plot([xl, xr], [y - 0.08, y - 0.08], color=color, lw=2.5, solid_capstyle="round")
    ax.plot([xr, xr], [y + 0.08, y - 0.08], color=color, lw=2.2, solid_capstyle="round")
    lx = label_x if label_x else (xl + xr) / 2
    tcol = TRT_TEXT if label == "TRT" else color
    ax.text(lx, y - 0.23, label, fontsize=10.5, fontweight="bold",
            ha="center", va="top", color=tcol)

# FFD — just Fix 1
y = span_y_start
draw_bracket(ax, fix1_left, fix1_right, y, COLORS["FFD"], "FFD")
ax.text(fix1_right + 0.2, y - 0.05, "first fixation only",
        fontsize=8, va="center", ha="left", color="#999", style="italic")

# GD — all first-pass fixations (same as FFD here since only 1 first-pass fixation on this word)
y = span_y_start - span_gap
draw_bracket(ax, fix1_left, fix1_right, y, COLORS["GD"], "GD")
ax.text(fix1_right + 0.2, y - 0.05, "all first-pass fixations",
        fontsize=8, va="center", ha="left", color="#999", style="italic")

# GPT — first entry to moving past
y = span_y_start - 2 * span_gap
draw_bracket(ax, fix1_left, sacc1_right, y, COLORS["GPT"], "GPT")
ax.text(sacc1_right + 0.2, y - 0.05, "includes time until eyes move past",
        fontsize=8, va="center", ha="left", color="#999", style="italic")

# TRT — Fix 1 + Fix 2
y = span_y_start - 3 * span_gap
draw_bracket(ax, fix1_left, fix1_right, y, COLORS["TRT"], "")
ax.text((fix1_right + fix2_left) / 2, y - 0.02, "+", fontsize=13, ha="center", va="center",
        color=TRT_TEXT, fontweight="bold")
draw_bracket(ax, fix2_left, fix2_right, y, COLORS["TRT"], "")
ax.text((fix1_left + fix2_right) / 2, y - 0.28, "TRT", fontsize=10.5, fontweight="bold",
        ha="center", va="top", color=TRT_TEXT)
ax.text(fix2_right + 0.2, y - 0.05, "all fixations, all passes",
        fontsize=8, va="center", ha="left", color="#999", style="italic")

# nFixations — count
y = span_y_start - 4 * span_gap
c = COLORS["nFixations"]
for j, (fl, fr) in enumerate([(fix1_left, fix1_right), (fix2_left, fix2_right)]):
    cmx = (fl + fr) / 2
    circle = plt.Circle((cmx, y), 0.2, facecolor=c, edgecolor="white",
                         linewidth=1.5, alpha=0.2, zorder=4)
    ax.add_patch(circle)
    ax.text(cmx, y, str(j + 1), fontsize=10, fontweight="bold",
            ha="center", va="center", color=c, zorder=5)
ax.text((fix1_cx + fix2_cx) / 2, y - 0.3, "nFixations = 2",
        fontsize=10.5, fontweight="bold", ha="center", va="top", color=c)

# ════════════════════════════════════════════════════════════════════════
# SECTION 4: Metric Definition Cards (y ~ 0.8–6.0)
# ════════════════════════════════════════════════════════════════════════

# Background panel
cards_panel = FancyBboxPatch((0.5, 0.6), 15.0, 5.5,
                              boxstyle="round,pad=0.15",
                              facecolor="#F5F7FA", edgecolor="#D0D8E0", linewidth=1)
ax.add_patch(cards_panel)

ax.text(8, 5.7, "Metric Definitions", fontsize=15, fontweight="bold",
        ha="center", va="center", color="#1A1A2E")

metrics_info = [
    ("FFD", "First Fixation\nDuration",
     "Duration (ms) of the very first\nfixation on a word.\nCaptures initial recognition.",
     "1st fixation only"),
    ("GD", "Gaze\nDuration",
     "Sum of all fixation durations\non a word before the eyes\nleave it (first pass only).",
     "All 1st-pass fixations"),
    ("GPT", "Go-Past\nTime",
     "Time from first entering a\nword region until moving past\nit — includes regressions.",
     "1st entry → past word"),
    ("TRT", "Total Reading\nTime",
     "Total time spent fixating on\na word across all passes\nthrough the text.",
     "All fixations, all passes"),
    ("nFixations", "Number of\nFixations",
     "Count of individual fixation\nevents on a word across\nall passes combined.",
     "Count of fixation events"),
]

card_w = 2.65
card_h = 3.6
card_gap = 0.3
total_cards_w = 5 * card_w + 4 * card_gap
cards_start_x = 8 - total_cards_w / 2
card_base_y = 1.0

for i, (abbr, full_name, desc, scope) in enumerate(metrics_info):
    cx = cards_start_x + i * (card_w + card_gap)
    cy = card_base_y
    color = COLORS[abbr]
    text_color = TRT_TEXT if abbr == "TRT" else color

    # Card background
    card_bg = FancyBboxPatch((cx, cy), card_w, card_h,
                              boxstyle="round,pad=0.1",
                              facecolor="white", edgecolor=color,
                              linewidth=2.2)
    ax.add_patch(card_bg)

    # Colored header band
    header_h = 0.85
    stripe = FancyBboxPatch((cx + 0.06, cy + card_h - header_h - 0.06),
                             card_w - 0.12, header_h,
                             boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor="none", alpha=0.13)
    ax.add_patch(stripe)

    # Abbreviation
    ax.text(cx + card_w / 2, cy + card_h - 0.38, abbr,
            fontsize=15, fontweight="bold", ha="center", va="center",
            color=text_color, fontfamily="monospace")

    # Full name
    ax.text(cx + card_w / 2, cy + card_h - 0.75, full_name,
            fontsize=8, ha="center", va="top", color="#333",
            fontweight="bold", linespacing=1.2)

    # Description
    ax.text(cx + card_w / 2, cy + card_h - 1.55, desc,
            fontsize=8, ha="center", va="top", color="#555",
            linespacing=1.45)

    # Scope tag at bottom
    tag_w = len(scope) * 0.085 + 0.35
    tag_x = cx + card_w / 2 - tag_w / 2
    tag_bg = FancyBboxPatch((tag_x, cy + 0.2), tag_w, 0.38,
                             boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor="none", alpha=0.12)
    ax.add_patch(tag_bg)
    ax.text(cx + card_w / 2, cy + 0.39, scope,
            fontsize=7.8, ha="center", va="center", color=text_color,
            fontweight="bold")

# ════════════════════════════════════════════════════════════════════════
# Footer: z-score note
# ════════════════════════════════════════════════════════════════════════
ax.text(8, 0.3, "Each metric is independently z-score normalized per sentence:  z = (x − μ) / σ",
        fontsize=9.5, ha="center", va="center", color="#777",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", edgecolor="#CCC", linewidth=0.8))

# ── Save ───────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/zuco/fig_006.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR, pad_inches=0.2)
plt.close(fig)
print(f"Saved: {out_path}")
