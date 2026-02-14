"""
fig_003: LLM Judge Scoring System
Diagram showing three scoring dimensions, their weights, the weighted reward formula,
and format compliance bonuses/penalties.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# --- Color palette ---
BG_WHITE = "#FFFFFF"
CARD_BG = "#F7F9FC"
HEADER_CORRECTNESS = "#2563EB"   # blue
HEADER_REASONING = "#7C3AED"     # purple
HEADER_CONSISTENCY = "#0891B2"   # teal
FORMULA_BG = "#1E293B"          # dark slate
FORMULA_TEXT = "#FFFFFF"
BONUS_GREEN = "#16A34A"
PENALTY_RED = "#DC2626"
SCALE_TRACK = "#E2E8F0"
SCALE_FILL_CORR = "#93C5FD"
SCALE_FILL_REAS = "#C4B5FD"
SCALE_FILL_CONS = "#67E8F9"
WEIGHT_BADGE = "#FCD34D"
ARROW_COLOR = "#64748B"
TEXT_DARK = "#1E293B"
TEXT_MED = "#475569"
TEXT_LIGHT = "#94A3B8"
BORDER_COLOR = "#CBD5E1"

fig, ax = plt.subplots(figsize=(15, 10), facecolor=BG_WHITE)
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")

# === Title ===
ax.text(7.5, 9.55, "LLM Judge Scoring System", fontsize=18, fontweight="bold",
        ha="center", va="center", color=TEXT_DARK, fontfamily="sans-serif")
ax.text(7.5, 9.15, "Three-dimensional evaluation with weighted reward aggregation",
        fontsize=10, ha="center", va="center", color=TEXT_MED, fontfamily="sans-serif")

# ------------------------------------------------------------------
# Helper: draw a scoring dimension card
# ------------------------------------------------------------------
def draw_dimension_card(ax, cx, top_y, header_color, scale_fill,
                        title, weight_str, weight_val, descriptions):
    """Draw one scoring dimension card centred at cx."""
    card_w, card_h = 3.6, 4.1
    left = cx - card_w / 2
    bottom = top_y - card_h

    # Card background with rounded corners
    card = FancyBboxPatch((left, bottom), card_w, card_h,
                          boxstyle="round,pad=0.12", linewidth=1.2,
                          edgecolor=BORDER_COLOR, facecolor=CARD_BG, zorder=2)
    ax.add_patch(card)

    # Header bar
    hdr_h = 0.55
    hdr = FancyBboxPatch((left, top_y - hdr_h), card_w, hdr_h,
                         boxstyle="round,pad=0.08", linewidth=0,
                         facecolor=header_color, zorder=3)
    ax.add_patch(hdr)
    ax.text(cx, top_y - hdr_h / 2, title, fontsize=11, fontweight="bold",
            ha="center", va="center", color="#FFFFFF", zorder=4, fontfamily="sans-serif")

    # Weight badge
    badge_y = top_y - hdr_h - 0.35
    badge = FancyBboxPatch((cx - 0.55, badge_y - 0.17), 1.1, 0.34,
                           boxstyle="round,pad=0.06", linewidth=0,
                           facecolor=WEIGHT_BADGE, zorder=3)
    ax.add_patch(badge)
    ax.text(cx, badge_y, f"Weight: {weight_val}", fontsize=9, fontweight="bold",
            ha="center", va="center", color=TEXT_DARK, zorder=4, fontfamily="sans-serif")

    # Scale bar (0-5)
    scale_y = badge_y - 0.55
    bar_w = 2.8
    bar_h = 0.28
    bar_left = cx - bar_w / 2

    # Background track
    track = FancyBboxPatch((bar_left, scale_y - bar_h / 2), bar_w, bar_h,
                           boxstyle="round,pad=0.04", linewidth=0,
                           facecolor=SCALE_TRACK, zorder=3)
    ax.add_patch(track)

    # Filled portion (show ~4 out of 5 as illustrative)
    fill_frac = 0.8
    fill = FancyBboxPatch((bar_left, scale_y - bar_h / 2), bar_w * fill_frac, bar_h,
                          boxstyle="round,pad=0.04", linewidth=0,
                          facecolor=scale_fill, zorder=4)
    ax.add_patch(fill)

    # Scale ticks
    for i in range(6):
        tx = bar_left + bar_w * i / 5
        ax.plot([tx, tx], [scale_y - bar_h / 2 - 0.04, scale_y - bar_h / 2],
                color=TEXT_LIGHT, linewidth=0.8, zorder=5)
        ax.text(tx, scale_y - bar_h / 2 - 0.14, str(i), fontsize=7,
                ha="center", va="top", color=TEXT_MED, zorder=5, fontfamily="sans-serif")

    ax.text(cx, scale_y + bar_h / 2 + 0.12, "Score Range 0 – 5", fontsize=7.5,
            ha="center", va="bottom", color=TEXT_MED, zorder=5, fontfamily="sans-serif",
            fontstyle="italic")

    # Description lines
    desc_y = scale_y - 0.55
    for i, (label, text) in enumerate(descriptions):
        y = desc_y - i * 0.38
        ax.text(left + 0.22, y, f"{label}:", fontsize=7.5, fontweight="bold",
                ha="left", va="center", color=header_color, zorder=4, fontfamily="sans-serif")
        ax.text(left + 0.22 + len(label) * 0.065 + 0.18, y, text, fontsize=7.5,
                ha="left", va="center", color=TEXT_MED, zorder=4, fontfamily="sans-serif")

    return cx, bottom  # return bottom-centre for arrow

# --- Three dimension cards ---
card_top = 8.6

_, bot1 = draw_dimension_card(
    ax, 2.3, card_top, HEADER_CORRECTNESS, SCALE_FILL_CORR,
    "Answer Correctness", "w\u2081", "0.5",
    [("Exact match", "score 5"),
     ("Semantically equiv.", "score 4"),
     ("Partial", "score 2"),
     ("Wrong", "score 0")]
)

_, bot2 = draw_dimension_card(
    ax, 7.0, card_top, HEADER_REASONING, SCALE_FILL_REAS,
    "Reasoning Quality", "w\u2082", "0.3",
    [("Feature ID", "relevant TS features"),
     ("Logic", "sound logical steps"),
     ("Depth", "analytical rigor"),
     ("Clarity", "coherent explanation")]
)

_, bot3 = draw_dimension_card(
    ax, 12.2, card_top, HEADER_CONSISTENCY, SCALE_FILL_CONS,
    "Reasoning-Answer Consistency", "w\u2083", "0.2",
    [("Alignment", "reasoning \u2192 answer"),
     ("Support", "evidence backs claim"),
     ("No contradict.", "internally consistent"),
     ("Logical flow", "step-by-step valid")]
)

# === Converging arrows to formula box ===
arrow_kw = dict(arrowstyle="-|>", color=ARROW_COLOR, linewidth=1.5,
                mutation_scale=14, connectionstyle="arc3,rad=0", zorder=6)

formula_top = 3.25
for cx in [2.3, 7.0, 12.2]:
    ax.annotate("", xy=(7.5, formula_top + 0.05), xytext=(cx, bot1 + 0.02),
                arrowprops=arrow_kw)

# === Formula box ===
fw, fh = 8.4, 1.1
fleft = 7.5 - fw / 2
fbottom = formula_top - fh

formula_box = FancyBboxPatch((fleft, fbottom), fw, fh,
                             boxstyle="round,pad=0.15", linewidth=0,
                             facecolor=FORMULA_BG, zorder=5)
ax.add_patch(formula_box)

ax.text(7.5, formula_top - 0.28, "Weighted Reward Formula", fontsize=9,
        ha="center", va="center", color=TEXT_LIGHT, zorder=6, fontfamily="sans-serif",
        fontstyle="italic")

ax.text(7.5, formula_top - 0.68,
        r"$r \;=\; 0.5 \times \mathrm{correctness} \;+\; 0.3 \times \mathrm{reasoning} \;+\; 0.2 \times \mathrm{consistency}$",
        fontsize=13, ha="center", va="center", color=FORMULA_TEXT, zorder=6,
        fontfamily="sans-serif")

# === Format compliance callouts ===
# Bonus callout (left)
bonus_cx, bonus_cy = 3.2, 1.3
bw, bh = 4.2, 1.1
bonus_box = FancyBboxPatch((bonus_cx - bw / 2, bonus_cy - bh / 2), bw, bh,
                           boxstyle="round,pad=0.12", linewidth=1.5,
                           edgecolor=BONUS_GREEN, facecolor="#F0FDF4", zorder=5)
ax.add_patch(bonus_box)

ax.text(bonus_cx, bonus_cy + 0.22, "Format Compliance Bonus",
        fontsize=9.5, fontweight="bold", ha="center", va="center",
        color=BONUS_GREEN, zorder=6, fontfamily="sans-serif")
ax.text(bonus_cx, bonus_cy - 0.12, "+0.5  to reward",
        fontsize=14, fontweight="bold", ha="center", va="center",
        color=BONUS_GREEN, zorder=6, fontfamily="sans-serif")
ax.text(bonus_cx, bonus_cy - 0.42,
        'Follows  <reasoning> Answer: <label>  format',
        fontsize=7.5, ha="center", va="center",
        color=TEXT_MED, zorder=6, fontfamily="sans-serif", fontstyle="italic")

# Penalty callout (right)
penalty_cx, penalty_cy = 11.8, 1.3
pw, ph = 4.2, 1.1
penalty_box = FancyBboxPatch((penalty_cx - pw / 2, penalty_cy - ph / 2), pw, ph,
                             boxstyle="round,pad=0.12", linewidth=1.5,
                             edgecolor=PENALTY_RED, facecolor="#FEF2F2", zorder=5)
ax.add_patch(penalty_box)

ax.text(penalty_cx, penalty_cy + 0.22, "Format Non-Compliance Penalty",
        fontsize=9.5, fontweight="bold", ha="center", va="center",
        color=PENALTY_RED, zorder=6, fontfamily="sans-serif")
ax.text(penalty_cx, penalty_cy - 0.12, "\u22121.0  from reward",
        fontsize=14, fontweight="bold", ha="center", va="center",
        color=PENALTY_RED, zorder=6, fontfamily="sans-serif")
ax.text(penalty_cx, penalty_cy - 0.42,
        "Missing required format structure",
        fontsize=7.5, ha="center", va="center",
        color=TEXT_MED, zorder=6, fontfamily="sans-serif", fontstyle="italic")

# Arrows from formula to callouts
arrow_down_kw = dict(arrowstyle="-|>", color=ARROW_COLOR, linewidth=1.3,
                     mutation_scale=12, zorder=6)
ax.annotate("", xy=(bonus_cx, bonus_cy + bh / 2 + 0.02),
            xytext=(5.5, fbottom - 0.02), arrowprops=arrow_down_kw)
ax.annotate("", xy=(penalty_cx, penalty_cy + ph / 2 + 0.02),
            xytext=(9.5, fbottom - 0.02), arrowprops=arrow_down_kw)

# Small +/- labels on arrows
ax.text(4.2, 1.98, "+", fontsize=14, fontweight="bold", ha="center", va="center",
        color=BONUS_GREEN, zorder=7, fontfamily="sans-serif")
ax.text(10.8, 1.98, "\u2013", fontsize=14, fontweight="bold", ha="center", va="center",
        color=PENALTY_RED, zorder=7, fontfamily="sans-serif")

# === Final reward summary at bottom centre ===
fr_cx, fr_cy = 7.5, 0.35
ax.text(fr_cx, fr_cy,
        "Final Reward  =  r  +  format bonus/penalty        (used as GRPO reward signal)",
        fontsize=9, ha="center", va="center", color=TEXT_MED, zorder=6,
        fontfamily="sans-serif", fontstyle="italic")

plt.tight_layout(pad=0.3)
fig.savefig("/home/wangni/notion-figures/llm-judge/fig_003.png", dpi=200,
            facecolor=BG_WHITE, bbox_inches="tight")
plt.close()

print("Saved: /home/wangni/notion-figures/llm-judge/fig_003.png")
