"""
fig_008: Two-Stage Agentic Pipeline (Companion Paper Framing)
Shows how BTF fits as the Stage-2 action space of an Arrow-T template-routing
agent, with Stage-1 Arrow classifier routing and shared REINFORCE training.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Muted academic palette
COL_BG_S1   = "#E8EEF4"   # Stage 1 background
COL_BG_S2   = "#F1ECE3"   # Stage 2 background
COL_INPUT   = "#C9D6E2"   # input/data
COL_ROUTER  = "#6E89A8"   # Arrow / Arrow-T routers
COL_ACTION  = "#A8B4C4"   # action / choice
COL_OUTPUT  = "#B7A88F"   # classifier output
COL_TEMPLATE= "#C7B89A"   # BTF template
COL_PREFIX  = "#D6C9A8"   # rendered prefix
COL_LLM     = "#8C7B5C"   # frozen LLM
COL_ANSWER  = "#5C6E84"   # answer
COL_TRAIN   = "#3F4A5C"   # training annotation box
COL_EDGE    = "#3F4A5C"
COL_TEXT    = "#1F2530"

fig, ax = plt.subplots(figsize=(15, 8.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---------- Title ----------
ax.text(50, 53.5, "Two-Stage Agentic Pipeline: BTF as Stage-2 Action Space",
        ha="center", va="center", fontsize=15, fontweight="bold", color=COL_TEXT)
ax.text(50, 50.6,
        "Stage 1 (Arrow) routes classifiers · Stage 2 (Arrow-T) routes BTF templates · "
        "frozen LLM consumes the rendered prefix",
        ha="center", va="center", fontsize=10.5, style="italic", color="#3F4A5C")

# ---------- Stage backdrops ----------
stage1_bg = FancyBboxPatch((2.0, 13.5), 45.0, 32.5,
                           boxstyle="round,pad=0.5,rounding_size=1.2",
                           linewidth=1.2, edgecolor="#9AAEC2",
                           facecolor=COL_BG_S1, zorder=1)
stage2_bg = FancyBboxPatch((52.0, 13.5), 46.0, 32.5,
                           boxstyle="round,pad=0.5,rounding_size=1.2",
                           linewidth=1.2, edgecolor="#B7A88F",
                           facecolor=COL_BG_S2, zorder=1)
ax.add_patch(stage1_bg)
ax.add_patch(stage2_bg)

ax.text(24.5, 44.0, "STAGE 1  —  Classifier Routing",
        ha="center", va="center", fontsize=12.2, fontweight="bold", color="#34465B")
ax.text(75.0, 44.0, "STAGE 2  —  Template Routing  (BTF action space)",
        ha="center", va="center", fontsize=12.2, fontweight="bold", color="#6B5A3E")

# ---------- Helper to draw a labelled box ----------
def box(ax, x, y, w, h, label, sublabel=None, facecolor="#C9D6E2",
        edgecolor=COL_EDGE, fontsize=10.5, fontweight="bold",
        sub_fs=8.8, text_color=COL_TEXT, sub_color="#2F3946", rounding=0.6, lw=1.2):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                       linewidth=lw, edgecolor=edgecolor, facecolor=facecolor, zorder=3)
    ax.add_patch(p)
    if sublabel:
        ax.text(x, y + 0.65, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)
        ax.text(x, y - 1.05, sublabel, ha="center", va="center",
                fontsize=sub_fs, color=sub_color, zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=COL_EDGE, lw=1.6, style="-|>", mutation=18):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=mutation,
                        linewidth=lw, color=color, zorder=5)
    ax.add_patch(a)

# ===================== STAGE 1 =====================
# Row y-center for the main pipeline blocks
ROW = 32.5

# 1A. Time-series input
box(ax, 8.5, ROW, 11.5, 6.5,
    "Time-series\ninput  x",
    sublabel="(+ question, metadata)",
    facecolor=COL_INPUT, fontsize=10.8)

# 1B. Arrow router
box(ax, 22.5, ROW, 10.5, 6.5,
    "Arrow\nrouter  π_θ",
    sublabel="masked-softmax head",
    facecolor=COL_ROUTER, text_color="white", sub_color="#E8EEF4", fontsize=11)

# 1C. Classifier choice (action)
box(ax, 35.0, ROW, 9.5, 6.5,
    "Classifier\nchoice  c",
    sublabel="a₁ ∈ {C₁…C_K}",
    facecolor=COL_ACTION, fontsize=10.5)

# 1D. Classifier output (top-1 + top-k + confidence + entropy)
box(ax, 24.0, 20.0, 36.0, 7.5,
    "Classifier output",
    sublabel="top-1 ŷ · top-k distribution · confidence · entropy",
    facecolor=COL_OUTPUT, text_color="white", sub_color="#F4ECDF",
    fontsize=11.2, sub_fs=9.4)

# arrows in stage 1
arrow(ax, 14.3, ROW, 17.2, ROW)            # input -> Arrow
arrow(ax, 27.8, ROW, 30.2, ROW)            # Arrow -> choice
arrow(ax, 35.0, ROW - 3.4, 32.0, 23.9, lw=1.6)   # choice -> classifier output (down)
# label on the down arrow
ax.text(36.6, 27.5, "run\nclassifier c",
        ha="left", va="center", fontsize=8.6, style="italic", color="#34465B")

# ===================== STAGE 2 =====================
# 2A. TemplateContext (from classifier output + metadata)
box(ax, 60.5, ROW, 13.5, 6.5,
    "TemplateContext",
    sublabel="(ŷ, top-k, conf, H, qtype, meta)",
    facecolor=COL_INPUT, fontsize=10.6, sub_fs=8.6)

# 2B. Arrow-T router
box(ax, 75.5, ROW, 10.5, 6.5,
    "Arrow-T\nrouter  π_φ",
    sublabel="masked-softmax head",
    facecolor=COL_ROUTER, text_color="white", sub_color="#E8EEF4", fontsize=11)

# 2C. BTF template choice
box(ax, 89.5, ROW, 11.0, 6.5,
    "BTF template\nchoice  t",
    sublabel="a₂ ∈ 54+1 templates",
    facecolor=COL_TEMPLATE, fontsize=10.5)

# 2D. Rendered prefix
box(ax, 67.0, 20.0, 19.5, 6.8,
    "Rendered prefix  p = render_t(ctx)",
    sublabel="natural-language CoT opener",
    facecolor=COL_PREFIX, fontsize=10.6, sub_fs=8.8)

# 2E. Frozen LLM
box(ax, 84.5, 20.0, 11.0, 6.8,
    "Frozen LLM",
    sublabel="(no gradient)",
    facecolor=COL_LLM, text_color="white", sub_color="#F1ECE3",
    fontsize=11.2, sub_fs=8.8)

# arrows in stage 2
arrow(ax, 67.25, ROW, 70.2, ROW)             # ctx -> Arrow-T
arrow(ax, 80.8, ROW, 84.0, ROW)              # Arrow-T -> template
arrow(ax, 89.5, ROW - 3.4, 70.5, 23.7, lw=1.6)  # template -> rendered prefix
ax.text(85.0, 27.0, "render\ntemplate t",
        ha="center", va="center", fontsize=8.6, style="italic", color="#6B5A3E")
arrow(ax, 76.8, 20.0, 79.0, 20.0)            # prefix -> LLM

# ---------- Cross-stage handoff arrow (classifier output -> TemplateContext) ----------
# big curved arrow from right side of "Classifier output" box up-and-over to TemplateContext
handoff = FancyArrowPatch(
    (42.05, 20.0), (53.75, 30.0),
    connectionstyle="arc3,rad=-0.25",
    arrowstyle="-|>", mutation_scale=22,
    linewidth=2.0, color="#34465B", zorder=5,
)
ax.add_patch(handoff)
ax.text(48.0, 26.7, "classifier output\n+ metadata", ha="center", va="center",
        fontsize=9.2, style="italic", color="#2F3946")

# ---------- Final answer arrow leaving Stage 2 ----------
box(ax, 92.0, 9.5, 11.0, 4.5,
    "Answer  ŷ_LLM",
    facecolor=COL_ANSWER, text_color="white",
    fontsize=10.8, fontweight="bold", rounding=0.5)
arrow(ax, 84.5, 16.6, 90.4, 11.6, lw=1.8)

# ---------- Reward arrow back to Arrow-T (and back to Arrow via Stage 1) ----------
# Reward signal: dashed line from Answer up to both routers
reward_color = "#7A4A2F"
# Answer -> Arrow-T (reward r)
r1 = FancyArrowPatch((92.0, 11.8), (75.5, 28.0),
                     connectionstyle="arc3,rad=0.35",
                     arrowstyle="-|>", mutation_scale=16,
                     linewidth=1.4, color=reward_color, linestyle=(0,(5,3)), zorder=4)
ax.add_patch(r1)
ax.text(96.5, 22.0, "reward  r\n(task metric)", ha="center", va="center",
        fontsize=8.8, color=reward_color, rotation=90)

# Reward also propagates to Stage-1 Arrow (shared signal)
r2 = FancyArrowPatch((87.5, 9.5), (22.5, 8.6),
                     connectionstyle="arc3,rad=0.18",
                     arrowstyle="-|>", mutation_scale=16,
                     linewidth=1.3, color=reward_color, linestyle=(0,(5,3)), zorder=4)
ax.add_patch(r2)
ax.text(55.0, 5.4, "shared reward r updates both routers",
        ha="center", va="center", fontsize=9.0, style="italic", color=reward_color)
# small upward arrow from the r2 path into Arrow router
arrow(ax, 22.5, 8.7, 22.5, 28.7, color=reward_color, lw=1.3, style="-|>", mutation=14)

# ---------- Training annotation banner (applies to BOTH stages) ----------
banner_x, banner_y, banner_w, banner_h = 6.0, 47.5, 88.0, 1.6
banner = FancyBboxPatch((banner_x, banner_y - banner_h/2), banner_w, banner_h,
                        boxstyle="round,pad=0.02,rounding_size=0.4",
                        linewidth=1.0, edgecolor=COL_TRAIN, facecolor="#2F3946", zorder=2)
# (Disabled top banner — keep things uncluttered. Use bottom training box instead.)

# Bottom training panel
train_x, train_y = 2.0, 1.0
train_w, train_h = 49.0, 3.4
train = FancyBboxPatch((train_x, train_y), train_w, train_h,
                       boxstyle="round,pad=0.02,rounding_size=0.5",
                       linewidth=1.1, edgecolor=COL_TRAIN, facecolor="#EDEEF2", zorder=2)
ax.add_patch(train)
ax.text(train_x + train_w/2, train_y + train_h/2 + 0.75,
        "Both stages trained with REINFORCE",
        ha="center", va="center", fontsize=10.6, fontweight="bold", color=COL_TRAIN)
ax.text(train_x + train_w/2, train_y + train_h/2 - 0.85,
        "EMA baseline · temperature-annealed exploration · masked-softmax constrained action head",
        ha="center", va="center", fontsize=9.3, style="italic", color="#34465B")

# ---------- Legend (small) ----------
legend_handles = [
    patches.Patch(facecolor=COL_ROUTER, edgecolor=COL_EDGE, label="Learned policy (router)"),
    patches.Patch(facecolor=COL_ACTION, edgecolor=COL_EDGE, label="Discrete action"),
    patches.Patch(facecolor=COL_LLM,    edgecolor=COL_EDGE, label="Frozen module"),
    Line2D([0], [0], color=reward_color, lw=1.6, linestyle=(0,(5,3)),
           label="Reward / gradient signal"),
]
leg = ax.legend(handles=legend_handles, loc="lower right",
                bbox_to_anchor=(0.985, 0.018),
                fontsize=9.0, frameon=True, framealpha=0.95,
                edgecolor="#9AAEC2", ncol=1)
leg.get_frame().set_linewidth(0.8)

# ---------- Save ----------
out_path = "/home/wangni/notion-figures/bridge-template-framework-pluggable-prefix-templates-for-cla/fig_008.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
