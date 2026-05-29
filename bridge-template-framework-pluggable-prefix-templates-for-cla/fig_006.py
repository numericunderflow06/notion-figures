"""
fig_006: Gate Composition: From Hyperparameters to Selected Template

Three parallel mini-flows showing how TemplateConfig flags + TemplateContext
route through the cascade (priority-ordered REGISTRY + ASSERTIVE_FALLBACK)
to a specific selected template.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_PATH = "/home/wangni/notion-figures/bridge-template-framework-pluggable-prefix-templates-for-cla/fig_006.png"

# ---- palette ----
BG_A = "#F5F7FB"   # path A row band
BG_B = "#F4F9F4"   # path B row band
BG_C = "#FBF6F2"   # path C row band

COL_CONFIG = "#3B5BA5"
COL_CONTEXT = "#5B7FBF"
COL_FIRE = "#2E7D5B"
COL_SKIP = "#B0B0B0"
COL_SHORT = "#C04848"
COL_TEMPLATE = "#4A3F8A"

FAMILY_COLORS = {
    "confidence":   "#4F86C6",
    "qtype":        "#7AA76A",
    "evidence":     "#D29A4A",
    "presentation": "#A975B0",
    "hierarchy":    "#C26666",
    "fallback":     "#7F7F7F",
}

# ---- figure ----
fig, ax = plt.subplots(figsize=(15.5, 10.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title
ax.text(50, 97.2, "Gate Composition: From Hyperparameters to Selected Template",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1F2A44")
ax.text(50, 94.0,
        "Three example paths through the priority-ordered cascade "
        "(REGISTRY ∪ ASSERTIVE_FALLBACK). Cascade returns first firing template.",
        ha="center", va="center", fontsize=10, color="#475063", style="italic")

# Column headers
HEADER_Y = 89.5
ax.text(11.5, HEADER_Y, "TemplateConfig flags\n+  TemplateContext",
        ha="center", va="center", fontsize=11, fontweight="bold", color=COL_CONFIG)
ax.text(50.0, HEADER_Y, "Cascade evaluation (priority-sorted, short-circuit)",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#333333")
ax.text(89.0, HEADER_Y, "Selected template",
        ha="center", va="center", fontsize=11, fontweight="bold", color=COL_TEMPLATE)

# header underline
ax.plot([3, 97], [86.8, 86.8], color="#CCCCCC", linewidth=0.8)

# ---- path-band geometry ----
ROW_TOP = [82.0, 56.5, 31.0]
ROW_BOT = [60.0, 34.5, 9.0]
BANDS   = [BG_A, BG_B, BG_C]

for top, bot, bg in zip(ROW_TOP, ROW_BOT, BANDS):
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.0, bot), 96, top - bot,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=0.6, edgecolor="#D8DEE8", facecolor=bg, zorder=0))

# Path labels (left margin tag)
PATH_LABELS = [
    ("Path A", "high-confidence  +  verify qtype"),
    ("Path B", "low-confidence  +  OOD signal"),
    ("Path C", "evidence  +  markdown roleplay combo"),
]
for (tag, sub), top, bot in zip(PATH_LABELS, ROW_TOP, ROW_BOT):
    mid_y = (top + bot) / 2
    ax.text(3.6, mid_y + 1.3, tag, ha="left", va="center",
            fontsize=10, fontweight="bold", color="#222222")
    ax.text(3.6, mid_y - 1.0, sub, ha="left", va="center",
            fontsize=8.5, color="#4A4A4A", style="italic")

# ---- helpers ----
def draw_box(x, y, w, h, text, fc, ec, fontsize=8.5, fontweight="normal",
             text_color="white", boxstyle="round,pad=0.25,rounding_size=0.6",
             lw=0.8, alpha=1.0, zorder=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                alpha=alpha, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=zorder + 1)

def draw_arrow(x1, y1, x2, y2, color="#666666", lw=1.0, style="-|>", zorder=4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=style, mutation_scale=10,
                                 color=color, linewidth=lw, zorder=zorder))

def cascade_node(cx, cy, label, status, fontsize=8.0):
    """status in {'skip','fire','short'}"""
    if status == "fire":
        fc, ec, tc, ls = "#E6F2EA", COL_FIRE, "#1F4F38", "-"
        marker = "[FIRES]"
    elif status == "short":
        fc, ec, tc, ls = "#F4F4F4", "#BBBBBB", "#888888", ":"
        marker = "[skipped — short-circuit]"
    else:
        fc, ec, tc, ls = "#FAFAFA", "#CCCCCC", "#888888", ":"
        marker = "[skip]"
    w, h = 18.0, 4.4
    rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle="round,pad=0.18,rounding_size=0.5",
                          facecolor=fc, edgecolor=ec, linewidth=0.9,
                          linestyle=ls, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.65, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=tc, zorder=4)
    ax.text(cx, cy - 1.05, marker, ha="center", va="center",
            fontsize=7.0, color=tc, style="italic", zorder=4)
    return cx - w / 2, cx + w / 2

# ---- configuration block (left) ----
def draw_config_block(cx, cy, lines, width=18.5, height=14.0):
    x = cx - width / 2
    y = cy - height / 2
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor="white", edgecolor=COL_CONFIG,
                                linewidth=1.0, zorder=3))
    ax.text(x + width / 2, y + height - 1.5, "TemplateConfig + Context",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=COL_CONFIG, zorder=4)
    line_y = y + height - 3.4
    for ln in lines:
        ax.text(x + 0.8, line_y, ln, ha="left", va="top",
                fontsize=7.4, color="#1F2A44", zorder=4,
                family="monospace")
        line_y -= 1.55

# ---- selected-template block (right) ----
def draw_template_block(cx, cy, family, name, render_preview, width=20.0, height=14.0):
    x = cx - width / 2
    y = cy - height / 2
    color = FAMILY_COLORS[family]
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                                boxstyle="round,pad=0.25,rounding_size=0.7",
                                facecolor="white", edgecolor=color,
                                linewidth=1.4, zorder=3))
    # family tag
    ax.add_patch(FancyBboxPatch((x + 0.6, y + height - 2.6), width - 1.2, 1.9,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor=color, edgecolor=color, linewidth=0,
                                alpha=0.85, zorder=4))
    ax.text(x + width / 2, y + height - 1.65, f"family: {family}",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color="white", zorder=5)
    # template name
    ax.text(x + width / 2, y + height - 4.5, name,
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            color="#222222", zorder=5, family="monospace")
    # render preview
    ax.text(x + width / 2, y + 2.6, render_preview,
            ha="center", va="center", fontsize=7.0, color="#444444",
            style="italic", zorder=5, wrap=True)
    ax.text(x + width / 2, y + 0.9, "→ rendered prefix → LLM",
            ha="center", va="center", fontsize=6.8, color="#666666", zorder=5)

# ====================================================================
# PATH A — high-confidence + verify qtype
# ====================================================================
yA = (ROW_TOP[0] + ROW_BOT[0]) / 2

draw_config_block(
    11.5, yA,
    [
        "top1_prob = 0.93",
        "entropy   = 0.21",
        "question  = 'Verify ...'",
        "qtype     = verify",
        "evidence  = None",
        "high_conf_thresh = 0.85",
    ]
)

# cascade nodes for path A
A_nodes = [
    (28, "high_confidence_assertion", "fire"),
    (48, "verify_qtype_template",     "short"),
    (68, "presentation.markdown",     "short"),
]
prev_right = 21.0
for cx, label, status in A_nodes:
    lx, rx = cascade_node(cx, yA, label, status)
    draw_arrow(prev_right, yA, lx, yA,
               color="#7A7A7A" if status != "skip" else "#BFBFBF",
               lw=1.1)
    prev_right = rx

# Path A — selected
draw_template_block(89, yA, "confidence", "high_confidence_assertion",
                    "\"With high confidence, the\nlabel is X. Reasoning ...\"")
draw_arrow(prev_right, yA, 79.0, yA, color=FAMILY_COLORS["confidence"], lw=1.4)

# ====================================================================
# PATH B — low-confidence + OOD signal
# ====================================================================
yB = (ROW_TOP[1] + ROW_BOT[1]) / 2

draw_config_block(
    11.5, yB,
    [
        "top1_prob   = 0.34",
        "entropy     = 1.78",
        "ood_score   = 0.91",
        "qtype       = (none)",
        "low_conf_thresh = 0.50",
        "ood_thresh      = 0.80",
    ]
)

B_nodes = [
    (28, "high_confidence_assertion",   "skip"),
    (48, "ood_low_confidence_template", "fire"),
    (68, "evidence.dataset_hierarchy",  "short"),
]
prev_right = 21.0
for cx, label, status in B_nodes:
    lx, rx = cascade_node(cx, yB, label, status)
    draw_arrow(prev_right, yB, lx, yB,
               color="#7A7A7A" if status != "skip" else "#BFBFBF",
               lw=1.1)
    prev_right = rx

draw_template_block(89, yB, "confidence", "ood_low_confidence_template",
                    "\"Confidence is low and the\ninput looks out-of-distribution;\nproceed cautiously ...\"")
draw_arrow(prev_right, yB, 79.0, yB, color=FAMILY_COLORS["confidence"], lw=1.4)

# ====================================================================
# PATH C — evidence + markdown roleplay combo
# ====================================================================
yC = (ROW_TOP[2] + ROW_BOT[2]) / 2

draw_config_block(
    11.5, yC,
    [
        "top1_prob = 0.71",
        "evidence  = ['ECG seg #3']",
        "qtype     = open",
        "format    = markdown",
        "roleplay  = clinician",
        "combo_enabled = True",
    ]
)

C_nodes = [
    (28, "high_confidence_assertion", "skip"),
    (48, "evidence_present_template", "skip"),
    (68, "markdown_plus_roleplay",    "fire"),
]
prev_right = 21.0
for cx, label, status in C_nodes:
    lx, rx = cascade_node(cx, yC, label, status)
    draw_arrow(prev_right, yC, lx, yC,
               color="#7A7A7A" if status != "skip" else "#BFBFBF",
               lw=1.1)
    prev_right = rx

draw_template_block(89, yC, "presentation", "markdown_plus_roleplay",
                    "\"As a clinician, I will reason\nin markdown using the cited\nevidence ...\"")
draw_arrow(prev_right, yC, 79.0, yC, color=FAMILY_COLORS["presentation"], lw=1.4)

# ====================================================================
# Legend (bottom)
# ====================================================================
LEG_Y = 4.2
ax.add_patch(FancyBboxPatch((2.0, 1.2), 96, 5.0,
                            boxstyle="round,pad=0.3,rounding_size=0.8",
                            facecolor="#FAFBFD", edgecolor="#D0D5DD",
                            linewidth=0.6, zorder=0))

def legend_chip(x, y, fc, ec, ls, label, marker):
    ax.add_patch(FancyBboxPatch((x, y - 1.3), 4.0, 2.6,
                                boxstyle="round,pad=0.1,rounding_size=0.35",
                                facecolor=fc, edgecolor=ec, linewidth=0.9,
                                linestyle=ls, zorder=2))
    ax.text(x + 2.0, y, marker, ha="center", va="center",
            fontsize=6.5, color=ec, fontweight="bold", zorder=3)
    ax.text(x + 4.8, y, label, ha="left", va="center",
            fontsize=8.2, color="#333333", zorder=3)

legend_chip(4.0,  LEG_Y, "#E6F2EA", COL_FIRE, "-",  "gate fires — render & return",  "[FIRES]")
legend_chip(28.0, LEG_Y, "#F4F4F4", "#BBBBBB", ":", "short-circuit (never evaluated)", "[short]")
legend_chip(58.0, LEG_Y, "#FAFAFA", "#CCCCCC", ":", "predicate evaluated False (skip)", "[skip]")

ax.text(82.0, LEG_Y + 0.8,
        "Cascade order: REGISTRY entries sorted by priority,",
        ha="left", va="center", fontsize=7.8, color="#444444")
ax.text(82.0, LEG_Y - 0.8,
        "then ASSERTIVE_FALLBACK guarantees a render.",
        ha="left", va="center", fontsize=7.8, color="#444444")

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PATH}")
