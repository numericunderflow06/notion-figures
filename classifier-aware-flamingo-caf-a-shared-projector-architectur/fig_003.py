"""
Figure 3: Four Injection Surfaces for CAF
Stacked layouts of LLM input token sequences for prefix, interleaved,
metadata, and unified-ensemble surfaces.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# Consistent color palette for token block types
COLORS = {
    "prefix_emb":  "#5B8DEF",  # blue   - instruct/prefix tokens
    "cls_llm":     "#E67E22",  # orange - classifier verdict embedding
    "q_pre":       "#27AE60",  # green  - perceiver queries (pre / standard)
    "q_post":      "#16A085",  # teal   - perceiver queries (post / unified)
    "text":        "#BDC3C7",  # grey   - question / answer text tokens
    "delim":       "#34495E",  # dark   - reserved delimiter tokens
}

LABELS = {
    "prefix_emb": "prefix_emb",
    "cls_llm":    "cls_llm",
    "q_pre":      "q_pre",
    "q_post":     "q_post",
    "text":       "text tokens",
    "delim":      "⟨CLF⟩ / ⟨/CLF⟩",
}


def draw_block(ax, x, y, w, h, color, label, edge="#2C3E50", lw=1.0, fs=9, fc_text="white"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.05",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fs, color=fc_text, weight="bold")


def draw_lane(ax, y, lane_name, lane_desc, blocks, lane_h=0.9):
    """blocks: list of (width, color_key, label)."""
    # Lane label on left
    ax.text(-0.15, y + lane_h / 2, lane_name,
            ha="right", va="center", fontsize=13, weight="bold", color="#2C3E50")
    ax.text(-0.15, y - 0.18, lane_desc,
            ha="right", va="center", fontsize=8.5, style="italic", color="#7F8C8D")

    x = 0.0
    gap = 0.05
    for (w, ckey, label) in blocks:
        color = COLORS[ckey]
        # Text tokens lighter font color
        fc_text = "#2C3E50" if ckey in ("text",) else "white"
        fs = 8.5 if len(label) > 10 else 9.5
        draw_block(ax, x, y, w, lane_h, color, label, fs=fs, fc_text=fc_text)
        x += w + gap

    # Right arrow representing LLM input flow
    ax.annotate("", xy=(x + 0.25, y + lane_h / 2), xytext=(x + 0.02, y + lane_h / 2),
                arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.4))
    ax.text(x + 0.32, y + lane_h / 2, "→ frozen LLM",
            ha="left", va="center", fontsize=9, color="#2C3E50")


fig, ax = plt.subplots(figsize=(14.5, 8.6), dpi=200)
ax.set_xlim(-2.4, 12.0)
ax.set_ylim(-1.3, 9.5)
ax.set_aspect("auto")
ax.axis("off")

# Title
ax.text(4.8, 9.05, "Four Injection Surfaces",
        ha="center", va="center", fontsize=17, weight="bold", color="#1B2631")
ax.text(4.8, 8.55,
        "Token-sequence layouts fed to the frozen LLM. Colored blocks denote projected segments; "
        "⟨CLF⟩…⟨/CLF⟩ are reserved delimiters.",
        ha="center", va="center", fontsize=10.5, color="#566573")

# Lane vertical positions (top → bottom)
lane_ys = [6.7, 5.0, 3.3, 1.4]

# 1) Prefix surface
# prefix_emb (instruct) + cls_llm + q_pre + Q text + A text
draw_lane(
    ax, lane_ys[0],
    "Prefix",
    "instruct ⊕ classifier ⊕ perceiver before text",
    [
        (0.9, "prefix_emb", "prefix_emb"),
        (0.9, "cls_llm",    "cls_llm"),
        (1.7, "q_pre",      "q_pre  (perceiver)"),
        (1.6, "text",       "Question tokens"),
        (1.4, "text",       "Answer tokens"),
    ],
)

# 2) Interleaved surface
# q_pre tokens distributed inside text via gated cross-attention markers
draw_lane(
    ax, lane_ys[1],
    "Interleaved",
    "Flamingo gated cross-attention at LLM blocks",
    [
        (0.9, "prefix_emb", "prefix_emb"),
        (0.9, "cls_llm",    "cls_llm"),
        (1.1, "text",       "Question (part 1)"),
        (1.1, "q_pre",      "q_pre"),
        (1.1, "text",       "Question (part 2)"),
        (1.1, "q_pre",      "q_pre"),
        (1.2, "text",       "Answer"),
    ],
)

# 3) Metadata surface
# Reserved delimiters around classifier verdict; q_pre after
draw_lane(
    ax, lane_ys[2],
    "Metadata",
    "verdict wrapped by reserved tokens",
    [
        (0.9, "prefix_emb", "prefix_emb"),
        (0.55, "delim",     "⟨CLF⟩"),
        (1.1, "cls_llm",    "cls_llm"),
        (0.65, "delim",     "⟨/CLF⟩"),
        (1.5, "q_pre",      "q_pre"),
        (1.4, "text",       "Question"),
        (1.3, "text",       "Answer"),
    ],
)

# 4) Unified ensemble surface
# Pre-perceiver AND post-perceiver queries + metadata-style classifier wrap
draw_lane(
    ax, lane_ys[3],
    "Unified",
    "ensemble of all surfaces, shared P, shared params",
    [
        (0.9, "prefix_emb", "prefix_emb"),
        (0.55, "delim",     "⟨CLF⟩"),
        (1.0, "cls_llm",    "cls_llm"),
        (0.65, "delim",     "⟨/CLF⟩"),
        (1.3, "q_pre",      "q_pre"),
        (1.3, "text",       "Question"),
        (1.3, "q_post",     "q_post"),
        (1.2, "text",       "Answer"),
    ],
)

# Legend block (right side)
legend_x0 = 9.4
legend_y0 = 7.1
legend_items = [
    ("prefix_emb", "Instruct / prefix tokens"),
    ("cls_llm",    "Projected class embedding\n(P · E_cls of classifier verdict)"),
    ("q_pre",      "Pre-LLM perceiver queries\n(P · perceiver output)"),
    ("q_post",     "Post-context perceiver queries\n(unified ensemble only)"),
    ("delim",      "Reserved delimiters ⟨CLF⟩ ⟨/CLF⟩"),
    ("text",       "Question / answer text tokens"),
]
ax.text(legend_x0, legend_y0 + 0.95, "Legend",
        ha="left", va="center", fontsize=11.5, weight="bold", color="#1B2631")

ly = legend_y0 + 0.55
for ckey, desc in legend_items:
    swatch = FancyBboxPatch((legend_x0, ly - 0.13), 0.32, 0.28,
                            boxstyle="round,pad=0.0,rounding_size=0.04",
                            linewidth=0.8, edgecolor="#2C3E50",
                            facecolor=COLORS[ckey])
    ax.add_patch(swatch)
    ax.text(legend_x0 + 0.42, ly, desc,
            ha="left", va="center", fontsize=9, color="#2C3E50")
    ly -= 0.58

# Bottom caption — shared parameters note
caption = ("All four surfaces share the same trained parameters: "
           "perceiver + gated cross-attention + LoRA + E_cls + single shared projector P "
           "(+ optional FiLM, instruct tokens). The frozen LLM, frozen TS encoder Φ_TS, "
           "and frozen external classifier are identical across surfaces.")
ax.text(4.8, -0.7, caption,
        ha="center", va="center", fontsize=9.5, style="italic",
        color="#34495E", wrap=True,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F6F7",
                  edgecolor="#AEB6BF", linewidth=0.8))

plt.tight_layout()
out_path = "/home/wangni/notion-figures/classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_003.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
