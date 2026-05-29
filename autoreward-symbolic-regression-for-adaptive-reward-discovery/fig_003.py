"""
fig_003: Two-Tier Judge Contract and Calibration Loop
Shows the calibration relationship between the full G-Eval LLM judge
and the lightweight per-dimension heuristics used during RL.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Palette (muted blue / muted orange + neutrals)
# ------------------------------------------------------------
BLUE_FILL = "#cfe0f3"
BLUE_EDGE = "#3b6ea5"
BLUE_DARK = "#1f3f6b"

ORANGE_FILL = "#fbe2c8"
ORANGE_EDGE = "#c97a2c"
ORANGE_DARK = "#8a4a13"

GATE_FILL = "#fff4c2"
GATE_EDGE = "#a4892c"

FAIL_RED = "#b14a4a"
OK_GREEN = "#3f8a4a"
NEUTRAL = "#444444"

# ------------------------------------------------------------
# Figure / axes
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 11), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------
ax.text(50, 97,
        "Two-Tier Judge Contract and Calibration Loop",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="#222222")
ax.text(50, 93.5,
        "Expensive LLM judge calibrates cheap per-dimension heuristics that score every RL rollout",
        ha="center", va="center",
        fontsize=11, color="#555555", style="italic")

# ------------------------------------------------------------
# Column headers
# ------------------------------------------------------------
# Left column: Calibration tier
ax.add_patch(FancyBboxPatch((4, 86), 42, 4.6,
                            boxstyle="round,pad=0.2,rounding_size=0.6",
                            linewidth=1.6, edgecolor=BLUE_EDGE,
                            facecolor=BLUE_EDGE))
ax.text(25, 88.3, "TIER 1  —  Calibration (offline, expensive)",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")

# Right column: RL tier
ax.add_patch(FancyBboxPatch((54, 86), 42, 4.6,
                            boxstyle="round,pad=0.2,rounding_size=0.6",
                            linewidth=1.6, edgecolor=ORANGE_EDGE,
                            facecolor=ORANGE_EDGE))
ax.text(75, 88.3, "TIER 2  —  RL Scoring (online, cheap)",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")

# Column divider
ax.add_line(Line2D([50, 50], [3, 85], linestyle=(0, (3, 3)),
                   color="#bbbbbb", linewidth=1.0))

# ------------------------------------------------------------
# Helper: rounded labelled box
# ------------------------------------------------------------
def box(x, y, w, h, title, body, fill, edge, title_color=None,
        title_size=11, body_size=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                linewidth=1.5, edgecolor=edge,
                                facecolor=fill))
    ax.text(x + w / 2, y + h - 1.3, title,
            ha="center", va="top",
            fontsize=title_size, fontweight="bold",
            color=title_color or edge)
    ax.text(x + w / 2, y + h - 3.2, body,
            ha="center", va="top",
            fontsize=body_size, color="#222222")

def gate(x, y, w, h, title, body, fill=GATE_FILL, edge=GATE_EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.25,rounding_size=0.6",
                                linewidth=1.8, edgecolor=edge,
                                facecolor=fill))
    ax.text(x + w / 2, y + h - 1.2, title,
            ha="center", va="top",
            fontsize=11, fontweight="bold", color="#7a5e10")
    ax.text(x + w / 2, y + h - 3.0, body,
            ha="center", va="top",
            fontsize=9.5, color="#222222")

def arrow(x1, y1, x2, y2, color=NEUTRAL, lw=1.7, style="-",
          mut=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>",
                                 mutation_scale=mut,
                                 linewidth=lw,
                                 color=color,
                                 linestyle=style,
                                 shrinkA=2, shrinkB=2))

# ============================================================
# LEFT COLUMN — Calibration path
# ============================================================

# 1. G-Eval LLM judge
box(6, 75, 38, 8,
    "G-Eval LLM Judge  (full rubric, 8 dimensions)",
    "Temperature  τ = 0     •     3 independent passes per (x, ŷ) pair",
    BLUE_FILL, BLUE_EDGE)

# 2. ICC(2,1) reliability gate
gate(8, 64.5, 34, 7.5,
     "Gate A — Inter-pass reliability",
     "ICC(2,1)  ≥  0.7  on judge passes")

# 3. Per-dimension heuristic candidates
box(6, 53, 38, 8,
    "Per-dimension Heuristic Candidates  h_d(x, ŷ)",
    "Cheap, programmatic proxies for each rubric dimension d",
    BLUE_FILL, BLUE_EDGE)

# 4. 200-pair held-out evaluation
box(6, 41.5, 38, 8,
    "Held-out Calibration Set",
    "200 (x, ŷ) pairs  •  judge score  vs  heuristic score",
    BLUE_FILL, BLUE_EDGE)

# 5. Spearman acceptance gate
gate(8, 31, 34, 7.5,
     "Gate B — Per-dimension agreement",
     "Spearman  ρ  ≥  0.8   ( judge  vs  h_d )")

# 6. Acceptance / rejection split
box(6, 20.5, 17, 7.5,
    "Accepted h_d",
    "ship to Tier 2",
    "#d8efd5", OK_GREEN, title_color=OK_GREEN)

box(27, 20.5, 17, 7.5,
    "Rejected h_d",
    "redesign proxy",
    "#f1d4d4", FAIL_RED, title_color=FAIL_RED)

# ============================================================
# RIGHT COLUMN — RL scoring path
# ============================================================

# A. GRPO rollout source
box(56, 75, 38, 8,
    "Policy  π_θ   ( OpenTSLM-Flamingo + LoRA )",
    "Generates K rollouts per medical time-series prompt x",
    ORANGE_FILL, ORANGE_EDGE)

# B. Accepted heuristic bank
box(56, 64.5, 38, 7.5,
    "Accepted Heuristic Bank   { h_d }  d ∈ accepted",
    "imported from Tier 1 calibration",
    ORANGE_FILL, ORANGE_EDGE)

# C. Per-dimension scoring on every rollout
box(56, 53, 38, 8,
    "Per-rollout Scoring  (cheap, parallel)",
    "score vector  s(ŷ)  =  [ h_1(x,ŷ), …, h_D(x,ŷ) ]",
    ORANGE_FILL, ORANGE_EDGE)

# D. Symbolic reward formula
box(56, 41.5, 38, 8,
    "Symbolic Reward  R(s)  ( from PySR )",
    "compact formula over accepted dimensions",
    ORANGE_FILL, ORANGE_EDGE)

# E. GRPO update
box(56, 31, 38, 7.5,
    "GRPO Policy Update",
    "advantage from R(s) over the K-rollout group",
    ORANGE_FILL, ORANGE_EDGE)

# F. Periodic drift audit
box(56, 20.5, 38, 7.5,
    "Periodic Drift Audit",
    "re-sample 200 pairs  →  re-check ICC and ρ on new policy",
    "#f6e7d3", ORANGE_EDGE)

# ============================================================
# ARROWS — Calibration column (top → bottom)
# ============================================================
arrow(25, 75, 25, 72.2, color=BLUE_DARK)         # judge → ICC gate
arrow(25, 64.5, 25, 61.2, color=BLUE_DARK)       # ICC pass → heuristics
arrow(25, 53, 25, 49.7, color=BLUE_DARK)         # heuristics → held-out eval
arrow(25, 41.5, 25, 38.7, color=BLUE_DARK)       # held-out → Spearman gate

# Spearman → accept / reject split
arrow(20, 31, 14.5, 28.2, color=OK_GREEN)        # accept branch
arrow(30, 31, 35.5, 28.2, color=FAIL_RED)        # reject branch

# Gate annotations (PASS / FAIL labels)
ax.text(43.6, 68.4, "pass", fontsize=8.5, color=OK_GREEN,
        fontweight="bold", ha="left", va="center")
ax.text(43.6, 35,   "pass", fontsize=8.5, color=OK_GREEN,
        fontweight="bold", ha="left", va="center")

# ICC fail → back to judge (re-score with new pass set)
arrow(8, 68.2, 4.5, 68.2, color=FAIL_RED, style=(0, (4, 2)))
arrow(4.5, 68.2, 4.5, 79, color=FAIL_RED, style=(0, (4, 2)))
arrow(4.5, 79, 6, 79, color=FAIL_RED, style=(0, (4, 2)))
ax.text(3.8, 73.5, "ICC < 0.7\nrerun judge",
        fontsize=8, color=FAIL_RED, ha="right", va="center",
        fontweight="bold")

# ============================================================
# ARROWS — RL column (top → bottom)
# ============================================================
arrow(75, 75, 75, 72.0, color=ORANGE_DARK)       # policy → heuristic bank (apply)
arrow(75, 64.5, 75, 61.0, color=ORANGE_DARK)     # bank → scoring
arrow(75, 53, 75, 49.5, color=ORANGE_DARK)       # scoring → symbolic reward
arrow(75, 41.5, 75, 38.5, color=ORANGE_DARK)     # reward → GRPO
arrow(75, 31, 75, 28.0, color=ORANGE_DARK)       # GRPO → drift audit

# ============================================================
# CROSS-COLUMN ARROWS  (Tier 1 ↔ Tier 2)
# ============================================================
# Accepted heuristics ship from left to right.
# Route under all boxes, then up the OUTSIDE RIGHT edge so the line
# never crosses a column box.
arrow(14.5, 20.5, 14.5, 17.8, color=OK_GREEN, lw=2.0, mut=14)
ax.add_patch(FancyArrowPatch((14.5, 17.8), (95, 17.8),
                             arrowstyle="-",
                             linewidth=2.0,
                             color=OK_GREEN,
                             shrinkA=0, shrinkB=0))
ax.text(45, 18.7, "ship accepted  h_d  to RL bank",
        fontsize=9.5, color=OK_GREEN, fontweight="bold",
        ha="center", va="bottom")
ax.add_patch(FancyArrowPatch((95, 17.8), (95, 68),
                             arrowstyle="-",
                             linewidth=2.0,
                             color=OK_GREEN,
                             shrinkA=0, shrinkB=0))
arrow(95, 68, 94.2, 68, color=OK_GREEN, lw=2.0, mut=14)

# Drift audit triggers re-calibration → back into Tier 1
# Route along the far right edge so it does not collide with any column box
arrow(75, 20.5, 98.5, 20.5, color=FAIL_RED, lw=1.7,
      style=(0, (5, 2)))
arrow(98.5, 20.5, 98.5, 92, color=FAIL_RED, lw=1.7,
      style=(0, (5, 2)))
# Drop into the calibration column at the top, just above the LLM judge
arrow(98.5, 92, 25, 92, color=FAIL_RED, lw=1.7,
      style=(0, (5, 2)))
arrow(25, 92, 25, 83.4, color=FAIL_RED, lw=1.7,
      style=(0, (5, 2)))
ax.text(97.5, 55, "ICC or ρ drift\n→ re-calibrate",
        fontsize=9, color=FAIL_RED, fontweight="bold",
        ha="right", va="center", rotation=90)

# ============================================================
# Bottom note: thresholds summary
# ============================================================
ax.add_patch(FancyBboxPatch((6, 6), 88, 9,
                            boxstyle="round,pad=0.3,rounding_size=0.6",
                            linewidth=1.2, edgecolor="#999999",
                            facecolor="#f7f7f7"))
ax.text(50, 12.6, "Two-tier contract",
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="#222222")
ax.text(50, 9.6,
        "Tier 1 (LLM judge) defines ground-truth rubric scores; Tier 2 (heuristics) "
        "scores every RL rollout at negligible cost.",
        ha="center", va="center", fontsize=9.5, color="#333333")
ax.text(50, 7.4,
        "Heuristic  h_d  is admitted to the RL loop only if  ICC(2,1) ≥ 0.7  (Gate A)  "
        "and  Spearman ρ ≥ 0.8  vs. judge on the 200-pair held-out set  (Gate B).",
        ha="center", va="center", fontsize=9.5, color="#333333")

# ============================================================
# Legend
# ============================================================
legend_handles = [
    Line2D([0], [0], color=BLUE_DARK, lw=2.0, label="Tier 1 calibration flow"),
    Line2D([0], [0], color=ORANGE_DARK, lw=2.0, label="Tier 2 RL flow"),
    Line2D([0], [0], color=OK_GREEN, lw=2.0,
           label="Gate pass / accepted heuristic"),
    Line2D([0], [0], color=FAIL_RED, lw=2.0, linestyle=(0, (4, 2)),
           label="Gate fail / re-calibration loop"),
]
leg = ax.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.015),
                ncol=4, frameon=False, fontsize=9.5)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
out_path = ("/home/wangni/notion-figures/"
            "autoreward-symbolic-regression-for-adaptive-reward-discovery/"
            "fig_003.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="white", pad_inches=0.2)
plt.close(fig)
print(f"Saved: {out_path}")
