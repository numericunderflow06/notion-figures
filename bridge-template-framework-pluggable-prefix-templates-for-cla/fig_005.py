"""Figure 005: TemplateContext and TemplateConfig anatomy.

Side-by-side schematic of the two main dataclasses showing all required
and optional fields, grouped by purpose, with color-coded mapping to
template families that read each field.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

PALETTE = {
    "confidence":   "#1f6feb",   # blue
    "qtype":        "#a371f7",   # purple
    "evidence":     "#3fb950",   # green
    "presentation": "#db6d28",   # orange
    "hierarchy":    "#f85149",   # red
    "fallback":     "#6e7681",   # gray
}

CTX_REQ_BG   = "#fff8c5"  # highlighted yellow for required
CTX_OPT_BG   = "#f6f8fa"  # light gray for optional
CTX_DERIV_BG = "#dbeafe"  # light blue for derived props
BOX_EDGE     = "#24292f"
TEXT_DARK    = "#0f1115"
TEXT_MUTED   = "#57606a"


# ---------------------------------------------------------------------------
# Data: exact fields from bridge_templates/base.py
# ---------------------------------------------------------------------------

CTX_REQUIRED = [
    ("class_idx: int",                "top-1 argmax index"),
    ("class_names: List[str]",        "all class names"),
    ("top_k_indices: List[int]",      "top-k indices, sorted desc"),
    ("top_k_probs: List[float]",      "top-k probabilities"),
    ("confidence: float",             "p(top-1) = top_k_probs[0]"),
    ("entropy: float",                "Shannon entropy, in nats"),
    ("qtype: str",                    "verify / choose / query / unknown"),
]

CTX_OPTIONAL = [
    ("question_text: Optional[str]",          "raw question, for echo/rephrase"),
    ("dataset_name: Optional[str]",           "for dataset prefix / analogy"),
    ("hierarchy_map: Optional[Dict[str,str]]","class -> parent, for hierarchy"),
    ("extra: Dict[str, Any]",                  "free-form bag (e.g. features)"),
]

CTX_DERIVED = [
    ("top1_name -> class_names[class_idx]",           "name of the top-1 class"),
    ("top2_gap  -> p(top1) - p(top2)",                "hedge gate input"),
]

# TemplateConfig fields, grouped by the comment headers in base.py
CFG_GROUPS = [
    ("Confidence / distribution gates", "confidence", [
        "hedge_epsilon: float = 0.0",
        "low_conf_tau: float = 0.0",
        "very_low_conf_tau: float = 0.0",
        "med_conf_low: float = 1.0",
        "med_conf_high: float = 1.0",
        "entropy_tau: float = inf",
        "list_top2: bool = False",
        "list_top3: bool = False",
        "show_numeric_confidence: bool = False",
    ]),
    ("Question-type gates", "qtype", [
        "use_verify: bool = False",
        "use_query: bool = False",
        "use_choose_pairwise: bool = False",
        "use_choose_list: bool = False",
        "use_unknown_explicit: bool = False",
        "use_negation: bool = False",
        "echo_question: bool = False",
        "rephrase_question: bool = False",
    ]),
    ("Evidence / knowledge gates", "evidence", [
        'evidence_mode: str = "none"',
        "use_procedural: bool = False",
        "use_two_sentence: bool = False",
        "use_counterfactual: bool = False",
        "use_differential: bool = False",
        "use_citation: bool = False",
        "use_reinforced: bool = False",
        'reinforcement_phrase: str = "signal patterns"',
        "show_entropy_value: bool = False",
        "show_top2_probs: bool = False",
        "show_full_snapshot: bool = False",
    ]),
    ("Presentation / structure gates", "presentation", [
        'format_mode: str = "plain"',
        "use_roleplay: bool = False",
        "use_third_person: bool = False",
    ]),
    ("Hierarchy / cross-dataset / OOD gates", "hierarchy", [
        "use_hierarchy: bool = False",
        "coarse_only: bool = False",
        "ood_tau: float = inf",
        "show_dataset_prefix: bool = False",
        "use_analogy: bool = False",
        "multi_label: bool = False",
    ]),
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(17, 14.5), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(-2, 100)
ax.set_axis_off()
fig.patch.set_facecolor("white")

# Title
ax.text(
    50, 98.2,
    "TemplateContext and TemplateConfig — Anatomy",
    ha="center", va="top",
    fontsize=18, fontweight="bold", color=TEXT_DARK,
)
ax.text(
    50, 95.5,
    "Inputs the cascade reads (left) and hyperparameters that gate every template (right). "
    "Color bars on the right map each config group to its template family.",
    ha="center", va="top",
    fontsize=10.5, color=TEXT_MUTED, style="italic",
)


def header_box(x, y, w, h, title, subtitle, edge):
    """Section header strip."""
    ax.add_patch(FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        linewidth=1.6, edgecolor=edge, facecolor="white",
    ))
    ax.text(x + 0.8, y - h / 2 + 0.45, title,
            ha="left", va="center", fontsize=12.5, fontweight="bold", color=TEXT_DARK)
    ax.text(x + w - 0.8, y - h / 2 + 0.45, subtitle,
            ha="right", va="center", fontsize=9, color=TEXT_MUTED, style="italic")


def field_row(x, y, w, h, name, desc, bg, name_color=TEXT_DARK,
              left_bar_color=None, mono_size=9.0, desc_size=8.3):
    """One field row with optional left color bar."""
    ax.add_patch(Rectangle(
        (x, y - h), w, h,
        linewidth=0.5, edgecolor="#d0d7de", facecolor=bg,
    ))
    if left_bar_color is not None:
        ax.add_patch(Rectangle(
            (x, y - h), 0.45, h,
            linewidth=0, facecolor=left_bar_color,
        ))
    pad_left = 1.1 if left_bar_color is not None else 0.7
    ax.text(x + pad_left, y - h / 2 + 0.05, name,
            ha="left", va="center", fontsize=mono_size,
            family="monospace", color=name_color)
    if desc:
        ax.text(x + w - 0.6, y - h / 2 + 0.05, desc,
                ha="right", va="center", fontsize=desc_size,
                color=TEXT_MUTED, style="italic")


# ---------------------------------------------------------------------------
# LEFT: TemplateContext
# ---------------------------------------------------------------------------

LX, LW = 2.0, 44.5
LY_TOP = 92.5
CARD_BOTTOM = 6.5

# Outer card
ax.add_patch(FancyBboxPatch(
    (LX - 0.5, CARD_BOTTOM), LW + 1.0, LY_TOP - CARD_BOTTOM + 1.0,
    boxstyle="round,pad=0.02,rounding_size=0.8",
    linewidth=2.0, edgecolor=BOX_EDGE, facecolor="#fbfcfd",
))

# Card title
ax.text(LX + LW / 2, LY_TOP - 0.5,
        "@dataclass  TemplateContext",
        ha="center", va="top", fontsize=14, fontweight="bold",
        family="monospace", color=TEXT_DARK)
ax.text(LX + LW / 2, LY_TOP - 3.0,
        "everything a template may inspect at selection & render time",
        ha="center", va="top", fontsize=9.5, color=TEXT_MUTED, style="italic")

cur_y = LY_TOP - 4.6

# --- Required fields header ---
header_box(LX, cur_y, LW, 2.2,
           "Required fields", "always populated by caller",
           edge="#9a6700")
cur_y -= 2.5
for name, desc in CTX_REQUIRED:
    field_row(LX, cur_y, LW, 2.55, name, desc, CTX_REQ_BG)
    cur_y -= 2.6

cur_y -= 0.8

# --- Optional fields header ---
header_box(LX, cur_y, LW, 2.2,
           "Optional fields", "default to None; templates gate on presence",
           edge="#57606a")
cur_y -= 2.5
for name, desc in CTX_OPTIONAL:
    field_row(LX, cur_y, LW, 2.55, name, desc, CTX_OPT_BG)
    cur_y -= 2.6

cur_y -= 0.8

# --- Derived properties header ---
header_box(LX, cur_y, LW, 2.2,
           "Derived @property", "computed convenience accessors",
           edge="#0969da")
cur_y -= 2.5
for name, desc in CTX_DERIVED:
    field_row(LX, cur_y, LW, 2.55, name, desc, CTX_DERIV_BG)
    cur_y -= 2.6


# ---------------------------------------------------------------------------
# RIGHT: TemplateConfig
# ---------------------------------------------------------------------------

RX, RW = 51.5, 46.5
RY_TOP = 92.5

# Outer card
ax.add_patch(FancyBboxPatch(
    (RX - 0.5, CARD_BOTTOM), RW + 1.0, RY_TOP - CARD_BOTTOM + 1.0,
    boxstyle="round,pad=0.02,rounding_size=0.8",
    linewidth=2.0, edgecolor=BOX_EDGE, facecolor="#fbfcfd",
))

# Card title
ax.text(RX + RW / 2, RY_TOP - 0.5,
        "@dataclass  TemplateConfig",
        ha="center", va="top", fontsize=14, fontweight="bold",
        family="monospace", color=TEXT_DARK)
ax.text(RX + RW / 2, RY_TOP - 3.0,
        "hyperparameters that parameterise every gate; one config per call",
        ha="center", va="top", fontsize=9.5, color=TEXT_MUTED, style="italic")

cur_y = RY_TOP - 4.3

# Compact row heights so all five groups fit
GROUP_HEADER_H = 1.75
ROW_H = 1.72

for group_title, family_key, fields in CFG_GROUPS:
    color = PALETTE[family_key]
    # Group header strip with colored left bar and family-tag chip
    ax.add_patch(Rectangle((RX, cur_y - GROUP_HEADER_H), 0.55, GROUP_HEADER_H,
                           linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((RX + 0.55, cur_y - GROUP_HEADER_H),
                           RW - 0.55, GROUP_HEADER_H,
                           linewidth=0.7, edgecolor="#d0d7de",
                           facecolor=color + "22"))  # 22 ~ 13% alpha hex
    ax.text(RX + 1.2, cur_y - GROUP_HEADER_H / 2 + 0.05, group_title,
            ha="left", va="center", fontsize=10.5, fontweight="bold",
            color=TEXT_DARK)
    # Family chip on the right of the header
    chip_w, chip_h = 9.5, 1.4
    chip_x = RX + RW - chip_w - 0.6
    chip_y = cur_y - GROUP_HEADER_H / 2 - chip_h / 2 + 0.05
    ax.add_patch(FancyBboxPatch(
        (chip_x, chip_y), chip_w, chip_h,
        boxstyle="round,pad=0.02,rounding_size=0.35",
        linewidth=0, facecolor=color,
    ))
    ax.text(chip_x + chip_w / 2, chip_y + chip_h / 2 + 0.02,
            f"-> {family_key}.py templates",
            ha="center", va="center", fontsize=8.0,
            color="white", fontweight="bold", family="monospace")
    cur_y -= GROUP_HEADER_H + 0.15

    for fname in fields:
        field_row(RX, cur_y, RW, ROW_H, fname, "",
                  bg="white", left_bar_color=color,
                  mono_size=8.4)
        cur_y -= ROW_H + 0.04

    cur_y -= 0.45  # gap between groups


# ---------------------------------------------------------------------------
# Connector + footer legend
# ---------------------------------------------------------------------------

# Subtle arrow indicating "feeds into" relationship
ax.annotate(
    "", xy=(RX - 0.6, 50), xytext=(LX + LW + 0.6, 50),
    arrowprops=dict(arrowstyle="->", lw=1.6, color="#57606a",
                    connectionstyle="arc3,rad=0.0"),
)
ax.text((LX + LW + RX) / 2 + 0.0, 51.6,
        "(cfg, ctx)",
        ha="center", va="bottom", fontsize=9.5,
        family="monospace", color=TEXT_DARK,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#57606a", lw=0.7))
ax.text((LX + LW + RX) / 2 + 0.0, 48.3,
        "pure gate(cfg, ctx) -> bool",
        ha="center", va="top", fontsize=8.6,
        color=TEXT_MUTED, style="italic")

# Footer legends, placed below both cards
legend_y = 3.5
ax.text(LX, legend_y + 1.6, "Context field shading:",
        ha="left", va="center", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
legend_items = [
    (CTX_REQ_BG,   "required"),
    (CTX_OPT_BG,   "optional"),
    (CTX_DERIV_BG, "derived @property"),
]
lx = LX
for col, lab in legend_items:
    ax.add_patch(Rectangle((lx, legend_y - 1.0), 2.4, 1.5,
                           edgecolor="#d0d7de", facecolor=col, linewidth=0.6))
    ax.text(lx + 2.8, legend_y - 0.25, lab,
            ha="left", va="center", fontsize=9, color=TEXT_DARK)
    lx += 13.5

# Family color legend
ax.text(RX, legend_y + 1.6, "Config group -> template family:",
        ha="left", va="center", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
fx = RX
for key in ["confidence", "qtype", "evidence", "presentation", "hierarchy"]:
    ax.add_patch(Rectangle((fx, legend_y - 1.0), 1.3, 1.5,
                           facecolor=PALETTE[key], edgecolor="none"))
    ax.text(fx + 1.7, legend_y - 0.25, key,
            ha="left", va="center", fontsize=8.8, color=TEXT_DARK)
    fx += 9.3


out_path = (
    "/home/wangni/notion-figures/"
    "bridge-template-framework-pluggable-prefix-templates-for-cla/fig_005.png"
)
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
