"""Figure 2: Cascade Selection Algorithm for select_template."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

OUTPUT_PATH = (
    "/home/wangni/notion-figures/"
    "bridge-template-framework-pluggable-prefix-templates-for-cla/fig_002.png"
)

# Palette
CTRL_FILL = "#E3F2FD"      # control-flow box fill
CTRL_EDGE = "#1565C0"      # control-flow box edge
DEC_FILL = "#FFF8E1"       # decision diamond fill
DEC_EDGE = "#F57F17"       # decision diamond edge
RET_FILL = "#E8F5E9"       # return / success
RET_EDGE = "#2E7D32"
ERR_FILL = "#FFEBEE"       # error path
ERR_EDGE = "#C62828"
LOOP_FILL = "#F3E5F5"      # loop box
LOOP_EDGE = "#6A1B9A"
TEXT = "#212121"

CTRL_ARROW = "#1565C0"
ERR_ARROW = "#C62828"
RET_ARROW = "#2E7D32"

fig, ax = plt.subplots(figsize=(11, 14), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 18)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")


def rounded_box(x, y, w, h, label, fill, edge, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        linewidth=1.8, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT, weight=weight, wrap=True)


def diamond(x, y, w, h, label, fill, edge, fontsize=10):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    dia = Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.8)
    ax.add_patch(dia)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=TEXT)


def loop_container(x, y, w, h, label, fill, edge):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.25",
        linewidth=2.0, edgecolor=edge, facecolor=fill,
        linestyle="--",
    )
    ax.add_patch(box)
    ax.text(x - w / 2 + 0.15, y + h / 2 - 0.25, label,
            ha="left", va="center", fontsize=10,
            color=edge, weight="bold", style="italic")


def arrow(x1, y1, x2, y2, color=CTRL_ARROW, label=None,
          label_offset=(0.15, 0), ls="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color=color, linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha="left", va="center", fontsize=9,
                color=color, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", edgecolor=color, linewidth=0.8))


# ---------------- Title ----------------
ax.text(5, 17.5, "select_template(cfg, ctx): Cascade Selection Algorithm",
        ha="center", va="center", fontsize=14, weight="bold", color=TEXT)
ax.text(5, 17.05,
        "Section 4.1 — priority-ordered walk over REGISTRY ∪ ASSERTIVE_FALLBACK",
        ha="center", va="center", fontsize=10, style="italic", color="#555")

# ---------------- Nodes ----------------
# 1. Start
rounded_box(5, 16.2, 3.6, 0.7,
            "START  select_template(cfg, ctx)",
            CTRL_FILL, CTRL_EDGE, fontsize=11, weight="bold")

# 2. Union templates
rounded_box(5, 15.1, 5.2, 0.8,
            "candidates = REGISTRY ∪ ASSERTIVE_FALLBACK",
            CTRL_FILL, CTRL_EDGE, fontsize=10.5)

# 3. Sort
rounded_box(5, 14.0, 5.2, 0.8,
            "sorted = sort(candidates, key=priority, ascending)",
            CTRL_FILL, CTRL_EDGE, fontsize=10.5)

# Loop container background
loop_container(5, 8.55, 8.6, 9.5,
               "for tpl in sorted  (lowest priority first)",
               "#FAFAFA", LOOP_EDGE)

# 4. Take next template
rounded_box(5, 12.7, 4.6, 0.75,
            "tpl ← next(sorted)\npriority = tpl.priority",
            "#FFFFFF", LOOP_EDGE, fontsize=10)

# 5. try evaluate gate (control box)
rounded_box(5, 11.45, 4.6, 0.8,
            "try:  fired = tpl.gate(cfg, ctx)",
            "#FFFFFF", LOOP_EDGE, fontsize=10.5)

# 6. Exception decision
diamond(5, 10.0, 3.6, 1.5,
        "gate raised\nException ?",
        DEC_FILL, DEC_EDGE, fontsize=10)

# 7a. Error branch: raise RuntimeError
rounded_box(8.55, 10.0, 2.7, 1.4,
            "raise RuntimeError(\n  f\"gate '{tpl.name}' failed\"\n) from exc",
            ERR_FILL, ERR_EDGE, fontsize=9.5, weight="bold")

# 7b. fired? decision
diamond(5, 7.9, 3.2, 1.4,
        "fired == True ?",
        DEC_FILL, DEC_EDGE, fontsize=10.5)

# 8a. Return tpl (success)
rounded_box(8.55, 7.9, 2.7, 1.0,
            "return tpl\n(first match wins)",
            RET_FILL, RET_EDGE, fontsize=10.5, weight="bold")

# 8b. continue / more templates? decision
diamond(5, 5.85, 3.6, 1.4,
        "more templates\nin sorted ?",
        DEC_FILL, DEC_EDGE, fontsize=10)

# 9. Fallback (outside loop)
rounded_box(5, 3.55, 5.4, 0.85,
            "return ASSERTIVE_FALLBACK\n(unconditional gate, base.py)",
            RET_FILL, RET_EDGE, fontsize=10.5, weight="bold")

# 10. End
rounded_box(5, 2.2, 3.2, 0.7,
            "END  →  prefix = tpl.render(ctx)",
            CTRL_FILL, CTRL_EDGE, fontsize=10.5, weight="bold")

# ---------------- Arrows ----------------
# 1 -> 2
arrow(5, 15.85, 5, 15.5, CTRL_ARROW)
# 2 -> 3
arrow(5, 14.7, 5, 14.4, CTRL_ARROW)
# 3 -> 4 (enter loop)
arrow(5, 13.6, 5, 13.07, CTRL_ARROW)
# 4 -> 5
arrow(5, 12.32, 5, 11.85, CTRL_ARROW)
# 5 -> 6 (exception decision)
arrow(5, 11.05, 5, 10.75, CTRL_ARROW)

# 6 -> 7a (yes, raise)
arrow(6.8, 10.0, 7.2, 10.0, ERR_ARROW, label="yes",
      label_offset=(-0.1, 0.25))
# 6 -> 7b (no, check fired)
arrow(5, 9.25, 5, 8.6, CTRL_ARROW, label="no",
      label_offset=(0.15, 0))

# 7b -> 8a (yes, return)
arrow(6.6, 7.9, 7.2, 7.9, RET_ARROW, label="yes",
      label_offset=(-0.1, 0.25))
# 7b -> 8b (no, continue loop)
arrow(5, 7.2, 5, 6.55, CTRL_ARROW, label="no",
      label_offset=(0.15, 0))

# 8b -> back to 4 (loop continuation, curved arrow)
loop_back = FancyArrowPatch(
    (3.2, 5.85), (2.55, 12.7),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=LOOP_EDGE,
    connectionstyle="arc3,rad=-0.45",
    linestyle="--",
)
ax.add_patch(loop_back)
ax.text(1.4, 9.3, "yes\nnext\ntpl",
        ha="center", va="center", fontsize=9,
        color=LOOP_EDGE, weight="bold",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor=LOOP_EDGE, linewidth=0.8))

# 8b -> 9 (no, fallback exit)
arrow(5, 5.15, 5, 3.97, CTRL_ARROW, label="no",
      label_offset=(0.15, 0))

# 9 -> 10
arrow(5, 3.13, 5, 2.55, RET_ARROW)

# 8a -> 10 (success path joins end)
success_arrow = FancyArrowPatch(
    (8.55, 7.4), (8.55, 2.2),
    arrowstyle="-", mutation_scale=18,
    linewidth=1.8, color=RET_ARROW,
)
ax.add_patch(success_arrow)
success_arrow2 = FancyArrowPatch(
    (8.55, 2.2), (6.6, 2.2),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=RET_ARROW,
)
ax.add_patch(success_arrow2)

# 7a (error) flows out of the diagram - terminate
err_term = FancyArrowPatch(
    (8.55, 9.3), (8.55, 8.6),
    arrowstyle="-|>", mutation_scale=18,
    linewidth=1.8, color=ERR_ARROW,
)
ax.add_patch(err_term)
rounded_box(8.55, 8.25, 2.4, 0.55,
            "propagate to caller",
            ERR_FILL, ERR_EDGE, fontsize=9.5, weight="bold")

# ---------------- Legend ----------------
legend_elements = [
    Line2D([0], [0], color=CTRL_ARROW, lw=2.2, label="Control flow"),
    Line2D([0], [0], color=RET_ARROW, lw=2.2, label="Return / success"),
    Line2D([0], [0], color=ERR_ARROW, lw=2.2, label="Error flow (RuntimeError)"),
    Line2D([0], [0], color=LOOP_EDGE, lw=2.2, ls="--", label="Loop iteration"),
]
ax.legend(handles=legend_elements, loc="lower left",
          bbox_to_anchor=(0.02, 0.005), fontsize=9.5,
          frameon=True, framealpha=0.95,
          edgecolor="#999", title="Edge semantics",
          title_fontsize=10)

# Side annotation about Section 9
ax.text(0.15, 14.6,
        "Section 9:\nGate exceptions are\nlocalised — the offending\n"
        "template name is\nembedded in the\nRuntimeError message,\n"
        "with the original\nexception chained via\n'from exc'.",
        ha="left", va="top", fontsize=9, color=ERR_EDGE,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor=ERR_FILL, edgecolor=ERR_EDGE, linewidth=1.2))

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {OUTPUT_PATH}")
