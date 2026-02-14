#!/usr/bin/env python3
"""
fig_004 — Extended Curriculum Pipeline
Timeline/pipeline visualization showing all curriculum stages from stage1 through stage5,
with the new RL stages (stage3_rl_grpo, stage4_rl_grpo, stage5_rl_grpo) highlighted.
Shows the SFT→RL pairing for each CoT domain.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Data from plan.md Section 5.1 ──────────────────────────────────────────
stages = [
    # (stage_key, display_label, stage_type, dataset_label, domain_group)
    ("stage1_mcq",       "stage1_mcq",       "sft",  "TSQA / M4",            None),
    ("stage2_captioning","stage2_captioning", "sft",  "TSQA / M4",            None),
    ("stage3_cot",       "stage3_cot",        "sft",  "HAR-CoT\n(68,542)",    "HAR"),
    ("stage3_rl_grpo",   "stage3_rl_grpo",    "rl",   "HAR-CoT\n(68,542)",    "HAR"),
    ("stage4_sleep_cot", "stage4_sleep_cot",  "sft",  "Sleep-CoT\n(7,434)",   "Sleep"),
    ("stage4_rl_grpo",   "stage4_rl_grpo",    "rl",   "Sleep-CoT\n(7,434)",   "Sleep"),
    ("stage5_ecg_cot",   "stage5_ecg_cot",    "sft",  "ECG-QA-CoT\n(159,313)","ECG"),
    ("stage5_rl_grpo",   "stage5_rl_grpo",    "rl",   "ECG-QA-CoT\n(159,313)","ECG"),
]

# ── Colors ──────────────────────────────────────────────────────────────────
SFT_COLOR       = "#3B82F6"   # blue
SFT_EDGE        = "#1E40AF"
SFT_TEXT        = "#FFFFFF"
RL_COLOR        = "#F97316"   # orange
RL_EDGE         = "#C2410C"
RL_TEXT         = "#FFFFFF"
NEW_BADGE_BG    = "#DC2626"   # red badge
NEW_BADGE_TEXT  = "#FFFFFF"
ARROW_COLOR     = "#6B7280"
GROUP_BG_COLORS = {"HAR": "#DBEAFE", "Sleep": "#E0E7FF", "ECG": "#FEF3C7"}
GROUP_EDGE      = {"HAR": "#93C5FD", "Sleep": "#A5B4FC", "ECG": "#FCD34D"}

# ── Layout constants ────────────────────────────────────────────────────────
BOX_W   = 1.55
BOX_H   = 0.72
GAP     = 0.35          # gap between boxes inside a group
GROUP_GAP = 0.60        # gap between groups / standalone stages
Y_CENTER = 2.0          # vertical center for boxes
DATASET_Y = Y_CENTER - BOX_H/2 - 0.55   # dataset labels below boxes

fig, ax = plt.subplots(figsize=(18, 5.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── Compute x positions and group spans ─────────────────────────────────────
x_positions = []
x = 0.5  # starting x

# Track groups for background rectangles
# Groups: stages within the same domain_group get grouped together
group_spans = {}  # domain -> (x_start, x_end)

for i, (key, label, stype, dataset, group) in enumerate(stages):
    # Determine gap before this stage
    if i == 0:
        pass  # no gap before the first stage
    else:
        prev_group = stages[i-1][4]
        if group is not None and prev_group == group:
            x += GAP   # same group → small gap
        else:
            x += GROUP_GAP  # different group → larger gap

    x_positions.append(x)

    if group is not None:
        if group not in group_spans:
            group_spans[group] = [x, x + BOX_W]
        else:
            group_spans[group][1] = x + BOX_W

    x += BOX_W

total_width = x + 0.5

# ── Draw group background rectangles ───────────────────────────────────────
PAD = 0.18
for group, (gx0, gx1) in group_spans.items():
    rect = FancyBboxPatch(
        (gx0 - PAD, Y_CENTER - BOX_H/2 - 0.22),
        gx1 - gx0 + 2*PAD,
        BOX_H + 0.44,
        boxstyle="round,pad=0.12",
        facecolor=GROUP_BG_COLORS[group],
        edgecolor=GROUP_EDGE[group],
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(rect)
    # Group label above
    ax.text(
        (gx0 + gx1)/2, Y_CENTER + BOX_H/2 + 0.38,
        f"{group} Domain  (SFT → RL)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
        color="#374151",
        zorder=5,
    )

# ── Draw arrows between consecutive stages ─────────────────────────────────
for i in range(len(stages) - 1):
    x0 = x_positions[i] + BOX_W
    x1 = x_positions[i + 1]
    mid_y = Y_CENTER

    ax.annotate(
        "",
        xy=(x1 - 0.02, mid_y),
        xytext=(x0 + 0.02, mid_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=2,
    )

# ── Draw stage boxes ───────────────────────────────────────────────────────
for i, (key, label, stype, dataset, group) in enumerate(stages):
    xb = x_positions[i]
    yb = Y_CENTER - BOX_H / 2

    bg = SFT_COLOR if stype == "sft" else RL_COLOR
    ec = SFT_EDGE  if stype == "sft" else RL_EDGE
    tc = SFT_TEXT   if stype == "sft" else RL_TEXT

    # Main box
    box = FancyBboxPatch(
        (xb, yb), BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        facecolor=bg, edgecolor=ec, linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(box)

    # Stage name inside box
    ax.text(
        xb + BOX_W/2, Y_CENTER + 0.03,
        label,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=tc,
        fontfamily="monospace",
        zorder=4,
    )

    # Type tag below the name inside the box
    type_label = "SFT" if stype == "sft" else "GRPO-RL"
    ax.text(
        xb + BOX_W/2, Y_CENTER - 0.22,
        type_label,
        ha="center", va="center",
        fontsize=8.5, color=tc, fontstyle="italic",
        alpha=0.85,
        zorder=4,
    )

    # Dataset label below box
    ax.text(
        xb + BOX_W/2, DATASET_Y,
        dataset,
        ha="center", va="top",
        fontsize=8.5, color="#4B5563",
        linespacing=1.15,
        zorder=4,
    )

    # "NEW" badge for RL stages
    if stype == "rl":
        badge_x = xb + BOX_W - 0.12
        badge_y = yb + BOX_H - 0.06
        badge = FancyBboxPatch(
            (badge_x - 0.22, badge_y - 0.08), 0.44, 0.22,
            boxstyle="round,pad=0.04",
            facecolor=NEW_BADGE_BG, edgecolor="white", linewidth=1.2,
            zorder=5,
        )
        ax.add_patch(badge)
        ax.text(
            badge_x, badge_y + 0.03,
            "NEW",
            ha="center", va="center",
            fontsize=7, fontweight="bold", color=NEW_BADGE_TEXT,
            zorder=6,
        )

# ── Title ───────────────────────────────────────────────────────────────────
ax.text(
    (x_positions[0] + x_positions[-1] + BOX_W) / 2,
    Y_CENTER + BOX_H/2 + 0.95,
    "Extended Curriculum Pipeline — OpenTSLM with GRPO-RL Stages",
    ha="center", va="bottom",
    fontsize=14, fontweight="bold", color="#111827",
    zorder=5,
)

# ── Subtitle ────────────────────────────────────────────────────────────────
ax.text(
    (x_positions[0] + x_positions[-1] + BOX_W) / 2,
    Y_CENTER + BOX_H/2 + 0.75,
    "Each CoT domain pairs a supervised fine-tuning (SFT) stage with a new reinforcement learning (GRPO-RL) stage",
    ha="center", va="bottom",
    fontsize=10, color="#6B7280",
    zorder=5,
)

# ── Legend ───────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=SFT_COLOR, edgecolor=SFT_EDGE, linewidth=1.5,
                   label="Supervised Fine-Tuning (SFT)"),
    mpatches.Patch(facecolor=RL_COLOR, edgecolor=RL_EDGE, linewidth=1.5,
                   label="GRPO Reinforcement Learning (NEW)"),
]
legend = ax.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=2,
    fontsize=10,
    frameon=True,
    fancybox=True,
    shadow=False,
    edgecolor="#D1D5DB",
)

# ── Axis cleanup ────────────────────────────────────────────────────────────
ax.set_xlim(x_positions[0] - 0.4, x_positions[-1] + BOX_W + 0.4)
ax.set_ylim(DATASET_Y - 0.8, Y_CENTER + BOX_H/2 + 1.25)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout(pad=0.5)

# ── Save ────────────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/llm-judge/fig_004.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
