"""
fig_003: Prompt Format Structure
Visual representation of the prompt template showing how text and time series tokens interleave.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colors ──────────────────────────────────────────────────────────────────
GREEN      = "#2E8B57"   # text tokens
GREEN_LT   = "#E8F5EC"   # text token fill
BLUE       = "#2B6CB0"   # <image> special token
BLUE_LT    = "#DBEAFE"   # <image> fill
RED        = "#C53030"    # <|endofchunk|> special token
RED_LT     = "#FEE2E2"   # <|endofchunk|> fill
GRAY       = "#6B7280"   # annotations / arrows
GRAY_LT    = "#F3F4F6"   # background sections
ORANGE     = "#D97706"   # post-prompt highlight
ORANGE_LT  = "#FEF3C7"
BG_SECTION = "#F9FAFB"

fig, ax = plt.subplots(figsize=(11, 18))
ax.set_xlim(0, 11)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helper: draw a rounded box with text ────────────────────────────────────
def draw_box(x, y, w, h, facecolor, edgecolor, text, fontsize=10,
             fontweight="normal", text_color="black", alpha=1.0, ha="center",
             zorder=2, linestyle="-", linewidth=1.2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                         zorder=zorder)
    ax.add_patch(box)
    tx = x + w / 2 if ha == "center" else x + 0.25
    ax.text(tx, y + h / 2, text, fontsize=fontsize,
            fontweight=fontweight, color=text_color,
            ha=ha, va="center", zorder=zorder + 1)

# ── Helper: annotation arrow + label ────────────────────────────────────────
def annotate(x_start, y_start, x_end, y_end, text, fontsize=9):
    ax.annotate(text,
                xy=(x_end, y_end), xytext=(x_start, y_start),
                fontsize=fontsize, color=GRAY,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRAY,
                                connectionstyle="arc3,rad=0.15",
                                linewidth=1.0))

# ── Title ───────────────────────────────────────────────────────────────────
ax.text(5.5, 17.5, "Prompt Format Structure", fontsize=18, fontweight="bold",
        ha="center", va="center", color="#1a1a2e")
ax.text(5.5, 17.1, "Text and time-series token interleaving in OpenTSLM-Aviation",
        fontsize=11, ha="center", va="center", color=GRAY)

# ── Flow connector parameters ──────────────────────────────────────────────
cx = 5.5          # center x for main flow
bw = 7.0          # box width
bx = cx - bw / 2  # box left x

cur_y = 16.4      # current y (top-down)

# ── DOWN-ARROW helper ──────────────────────────────────────────────────────
def down_arrow(y_top, length=0.25):
    ax.annotate("", xy=(cx, y_top - length), xytext=(cx, y_top),
                arrowprops=dict(arrowstyle="-|>", color="#9CA3AF", lw=1.5))

# ═══════════════════════════════════════════════════════════════════════════
# 1  PRE-PROMPT
# ═══════════════════════════════════════════════════════════════════════════
h = 1.55
cur_y -= h + 0.15
# Background
bg = FancyBboxPatch((bx - 0.15, cur_y - 0.1), bw + 0.3, h + 0.2,
                     boxstyle="round,pad=0.15", facecolor=GREEN_LT,
                     edgecolor=GREEN, linewidth=1.5, alpha=0.35, zorder=1)
ax.add_patch(bg)

ax.text(cx, cur_y + h - 0.15, "PRE-PROMPT  (text tokens)", fontsize=12,
        fontweight="bold", ha="center", va="center", color=GREEN)

pre_lines = [
    '"You are analyzing flight data from a general',
    'aviation aircraft. This data comes from the NGAFID..."',
    '',
    'Describes sensor categories: electrical, fuel,',
    'engine, cylinder, flight parameters.',
]
for i, line in enumerate(pre_lines):
    ax.text(cx, cur_y + h - 0.50 - i * 0.22, line, fontsize=8.5,
            ha="center", va="center", color="#374151",
            style="italic" if i < 2 else "normal")

# Annotation
annotate(bx + bw + 0.6, cur_y + h / 2 + 0.25, bx + bw + 0.15, cur_y + h / 2 + 0.25,
         "System context\n& instructions", fontsize=9)

down_arrow(cur_y - 0.05)

# ═══════════════════════════════════════════════════════════════════════════
# 2  CHANNEL BLOCKS  (show 3 examples + ellipsis)
# ═══════════════════════════════════════════════════════════════════════════
channels_to_show = [
    ("volt1", "Electrical system voltage reading from bus 1 ..."),
    ("E1 CHT1", "Cylinder head temperature for cylinder #1 ..."),
    ("AltMSL", "Altitude above mean sea level ..."),
]

# Section label
cur_y -= 0.55
ax.text(cx, cur_y, "REPEATED FOR EACH SENSOR CHANNEL  (×23 channels)",
        fontsize=11, fontweight="bold", ha="center", va="center", color="#1E40AF")

# Left brace approximation: vertical line + ticks
brace_x = bx - 0.55
brace_top = cur_y - 0.25
brace_bot = None  # set after drawing channels

block_h = 1.25
gap = 0.35

for idx, (ch_name, ch_desc) in enumerate(channels_to_show):
    cur_y -= block_h + gap

    # Channel block background
    cbg = FancyBboxPatch((bx - 0.05, cur_y - 0.05), bw + 0.1, block_h + 0.1,
                          boxstyle="round,pad=0.1", facecolor="#F0F4FF",
                          edgecolor="#93A8D4", linewidth=1.0, linestyle="--",
                          alpha=0.5, zorder=1)
    ax.add_patch(cbg)

    # Row 1: <image> token
    iw = 1.8
    draw_box(bx + 0.15, cur_y + block_h - 0.45, iw, 0.35,
             BLUE_LT, BLUE, "<image>", fontsize=10, fontweight="bold",
             text_color=BLUE)

    # Token ID annotation (only first)
    if idx == 0:
        ax.text(bx + 0.15 + iw + 0.2, cur_y + block_h - 0.275,
                "token ID 151666", fontsize=8, color=BLUE, va="center",
                style="italic")
        annotate(bx + bw + 0.6, cur_y + block_h - 0.275,
                 bx + bw + 0.15, cur_y + block_h - 0.275,
                 "Cross-attention\ninjection point", fontsize=9)

    # Row 2: channel_name: description (text)
    draw_box(bx + 0.15, cur_y + block_h - 0.90, bw - 0.65, 0.35,
             GREEN_LT, GREEN,
             f"  {ch_name}: {ch_desc}",
             fontsize=9, text_color="#1a1a2e", ha="left")

    # Row 3: <|endofchunk|>
    ew = 2.6
    draw_box(bx + 0.15, cur_y + block_h - 1.28, ew, 0.30,
             RED_LT, RED, "<|endofchunk|>", fontsize=10, fontweight="bold",
             text_color=RED)

    if idx == 0:
        ax.text(bx + 0.15 + ew + 0.2, cur_y + block_h - 1.13,
                "token ID 151665", fontsize=8, color=RED, va="center",
                style="italic")

    # Channel label
    ax.text(bx - 0.15, cur_y + block_h / 2,
            f"Ch {['1','12','23'][idx]}", fontsize=9, fontweight="bold",
            ha="right", va="center", color="#6B7280")

    if idx < len(channels_to_show) - 1:
        down_arrow(cur_y - 0.05)

    # After first channel, add ellipsis
    if idx == 0:
        cur_y -= 0.65
        ax.text(cx, cur_y + 0.15, "⋮", fontsize=28, ha="center", va="center",
                color="#9CA3AF")
        ax.text(cx + 0.6, cur_y + 0.15, "(channels 2–11)", fontsize=9,
                ha="left", va="center", color="#9CA3AF")
        cur_y -= 0.15

    if idx == 1:
        cur_y -= 0.65
        ax.text(cx, cur_y + 0.15, "⋮", fontsize=28, ha="center", va="center",
                color="#9CA3AF")
        ax.text(cx + 0.6, cur_y + 0.15, "(channels 13–22)", fontsize=9,
                ha="left", va="center", color="#9CA3AF")
        cur_y -= 0.15

brace_bot = cur_y + 0.1

# Draw left bracket for repeated section
ax.plot([brace_x, brace_x], [brace_bot, brace_top], color="#1E40AF",
        linewidth=2.0, solid_capstyle="round")
ax.plot([brace_x, brace_x + 0.15], [brace_top, brace_top], color="#1E40AF",
        linewidth=2.0, solid_capstyle="round")
ax.plot([brace_x, brace_x + 0.15], [brace_bot, brace_bot], color="#1E40AF",
        linewidth=2.0, solid_capstyle="round")

ax.text(brace_x - 0.15, (brace_top + brace_bot) / 2,
        "×23", fontsize=13, fontweight="bold", rotation=90,
        ha="center", va="center", color="#1E40AF")

down_arrow(cur_y - 0.05)

# ═══════════════════════════════════════════════════════════════════════════
# 3  POST-PROMPT
# ═══════════════════════════════════════════════════════════════════════════
h_post = 1.5
cur_y -= h_post + 0.45

bg2 = FancyBboxPatch((bx - 0.15, cur_y - 0.1), bw + 0.3, h_post + 0.2,
                      boxstyle="round,pad=0.15", facecolor=ORANGE_LT,
                      edgecolor=ORANGE, linewidth=1.5, alpha=0.45, zorder=1)
ax.add_patch(bg2)

ax.text(cx, cur_y + h_post - 0.15, "POST-PROMPT  (text tokens)", fontsize=12,
        fontweight="bold", ha="center", va="center", color=ORANGE)

post_lines = [
    '"Based on the sensor data patterns shown above,',
    'identify which maintenance issue was discovered..."',
    '',
    'Lists 19 possible maintenance issue categories.',
    'Ends with:  "The maintenance issue identified',
    'for this aircraft was:"',
]
for i, line in enumerate(post_lines):
    ax.text(cx, cur_y + h_post - 0.48 - i * 0.20, line, fontsize=8.5,
            ha="center", va="center", color="#374151",
            style="italic" if (i < 2 or i >= 4) else "normal")

annotate(bx + bw + 0.6, cur_y + h_post / 2, bx + bw + 0.15, cur_y + h_post / 2,
         "Classification\ninstruction", fontsize=9)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
leg_y = cur_y - 0.85
leg_x = 1.2
spacing = 3.2

items = [
    (GREEN_LT, GREEN, "Text tokens"),
    (BLUE_LT, BLUE, "<image> token"),
    (RED_LT, RED, "<|endofchunk|> token"),
]

for i, (fc, ec, label) in enumerate(items):
    lx = leg_x + i * spacing
    box = FancyBboxPatch((lx, leg_y), 0.35, 0.25,
                         boxstyle="round,pad=0.05", facecolor=fc,
                         edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    ax.text(lx + 0.5, leg_y + 0.125, label, fontsize=10, va="center",
            color=ec, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/wangni/notion-figures/maintenance/fig_003.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: /home/wangni/notion-figures/maintenance/fig_003.png")
