"""
fig_003: OpenTSLM Multimodal Architecture
Architecture diagram showing the full OpenTSLM pipeline with trainable vs frozen components.
Shows both Flamingo and SP (Soft Prompt) variants as two parallel paths.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Color palette ──────────────────────────────────────────────
C_TRAINABLE       = "#388E3C"   # green border – trainable
C_TRAINABLE_LIGHT = "#C8E6C9"   # light green fill
C_FROZEN          = "#616161"   # gray border – frozen
C_FROZEN_LIGHT    = "#E0E0E0"   # light gray fill
C_INPUT           = "#1565C0"   # blue border – input
C_INPUT_LIGHT     = "#BBDEFB"
C_OUTPUT          = "#E65100"   # orange border – output
C_OUTPUT_LIGHT    = "#FFE0B2"
C_ARROW           = "#424242"
C_PATH_FLAMINGO   = "#6A1B9A"   # purple – Flamingo path
C_PATH_SP         = "#00695C"   # teal – SP path
BG                = "white"

fig, ax = plt.subplots(figsize=(21, 10.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.8, 19.8)
ax.set_ylim(-1.8, 9.5)
ax.set_aspect('equal')
ax.axis('off')

# ── Helper functions ──────────────────────────────────────────
def draw_block(ax, x, y, w, h, label, sublabel=None,
               edge_color="#333", fill_color="#fff", text_color="#222",
               fontsize=12, sublabel_fontsize=9, lw=2.2, zorder=3):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.20",
                         facecolor=fill_color, edgecolor=edge_color,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w / 2, cy + 0.28, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, zorder=zorder + 1)
        ax.text(x + w / 2, cy - 0.35, sublabel,
                ha='center', va='center', fontsize=sublabel_fontsize,
                color=text_color, style='italic', zorder=zorder + 1)
    else:
        ax.text(x + w / 2, cy, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=text_color, zorder=zorder + 1)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0,
               style='-|>', connectionstyle="arc3,rad=0.0", zorder=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color,
                            linewidth=lw, mutation_scale=18,
                            connectionstyle=connectionstyle,
                            zorder=zorder)
    ax.add_patch(arrow)

# ── Layout ────────────────────────────────────────────────────
# Vertical positions
top_y    = 6.0    # Flamingo path
mid_y    = 3.8    # shared input/encoder/projector
bot_y    = 0.5    # SP path
bh       = 1.5    # block height

# ── 1. Time Series Input ─────────────────────────────────────
inp_x, inp_w = 0.0, 2.4
draw_block(ax, inp_x, mid_y, inp_w, bh,
           "Time Series\nInput",
           edge_color=C_INPUT, fill_color=C_INPUT_LIGHT,
           text_color=C_INPUT, fontsize=13)

# ── 2. CNN + Transformer Encoder (trainable) ─────────────────
enc_x, enc_w = 3.3, 3.0
draw_block(ax, enc_x, mid_y, enc_w, bh,
           "CNN + Transformer\nEncoder",
           sublabel="CNNTokenizer /\nTransformerCNNEncoder",
           edge_color=C_TRAINABLE, fill_color=C_TRAINABLE_LIGHT,
           text_color="#2E7D32", fontsize=12, sublabel_fontsize=8.5)

# ── 3. MLP Projector (trainable) ─────────────────────────────
proj_x, proj_w = 7.2, 2.3
draw_block(ax, proj_x, mid_y, proj_w, bh,
           "MLP\nProjector",
           sublabel="MLPProjector",
           edge_color=C_TRAINABLE, fill_color=C_TRAINABLE_LIGHT,
           text_color="#2E7D32", fontsize=13, sublabel_fontsize=9)

# ── Branch point ─────────────────────────────────────────────
branch_x = proj_x + proj_w + 0.4
branch_y = mid_y + bh / 2
ax.plot(branch_x, branch_y, 'o', color=C_ARROW, markersize=7, zorder=5)

# ── 4a. Flamingo Cross-Attention (trainable) ─────────────────
flam_x, flam_w = 10.8, 2.8
draw_block(ax, flam_x, top_y, flam_w, bh,
           "Flamingo\nCross-Attention",
           sublabel="Gated xattn-dense\nlayers (trainable)",
           edge_color=C_TRAINABLE, fill_color=C_TRAINABLE_LIGHT,
           text_color="#2E7D32", fontsize=12, sublabel_fontsize=8.5)

# ── 4b. SP Concatenation (trainable) ─────────────────────────
sp_x, sp_w = 10.8, 2.8
draw_block(ax, sp_x, bot_y, sp_w, bh,
           "SP\nConcatenation",
           sublabel="Prefix concatenation\nto LLM embeddings",
           edge_color=C_TRAINABLE, fill_color=C_TRAINABLE_LIGHT,
           text_color="#2E7D32", fontsize=12, sublabel_fontsize=8.5)

# ── 5a. Frozen LLM (Flamingo path) ──────────────────────────
llm_top_x, llm_w = 14.4, 2.5
draw_block(ax, llm_top_x, top_y, llm_w, bh,
           "Frozen LLM",
           sublabel="LLM backbone\n(weights frozen)",
           edge_color=C_FROZEN, fill_color=C_FROZEN_LIGHT,
           text_color=C_FROZEN, fontsize=13, sublabel_fontsize=8.5)

# ── 5b. Frozen LLM (SP path) ────────────────────────────────
draw_block(ax, llm_top_x, bot_y, llm_w, bh,
           "Frozen LLM",
           sublabel="LLM backbone\n(weights frozen)",
           edge_color=C_FROZEN, fill_color=C_FROZEN_LIGHT,
           text_color=C_FROZEN, fontsize=13, sublabel_fontsize=8.5)

# ── 6. CoT Output ───────────────────────────────────────────
out_x, out_w = 17.6, 1.2
draw_block(ax, out_x, top_y, out_w, bh,
           "CoT\nOutput",
           edge_color=C_OUTPUT, fill_color=C_OUTPUT_LIGHT,
           text_color=C_OUTPUT, fontsize=11)
draw_block(ax, out_x, bot_y, out_w, bh,
           "CoT\nOutput",
           edge_color=C_OUTPUT, fill_color=C_OUTPUT_LIGHT,
           text_color=C_OUTPUT, fontsize=11)

# ── LM Embeddings annotations ────────────────────────────────
for ly in [top_y, bot_y]:
    ax.text(llm_top_x + llm_w / 2, ly - 0.32,
            "LM embeddings: trainable",
            ha='center', va='center', fontsize=8,
            color=C_TRAINABLE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', fc='#E8F5E9',
                      ec=C_TRAINABLE, lw=0.8, alpha=0.9))

# ── Variant labels ───────────────────────────────────────────
ax.text(12.3, top_y + bh + 0.65, "Variant A: Flamingo Cross-Attention",
        ha='center', va='center', fontsize=12,
        fontweight='bold', color=C_PATH_FLAMINGO)
ax.text(12.3, bot_y - 0.5, "Variant B: Soft Prompt (SP) Concatenation",
        ha='center', va='center', fontsize=12,
        fontweight='bold', color=C_PATH_SP)

# ── Arrows: shared flow ─────────────────────────────────────
mid_cy = mid_y + bh / 2

# Input -> Encoder
draw_arrow(ax, inp_x + inp_w, mid_cy, enc_x, mid_cy)

# Encoder -> Projector
draw_arrow(ax, enc_x + enc_w, mid_cy, proj_x, mid_cy)

# Projector -> branch dot
draw_arrow(ax, proj_x + proj_w, mid_cy, branch_x - 0.12, mid_cy)

# ── Arrows: branch to Flamingo (upward) ─────────────────────
top_cy = top_y + bh / 2
draw_arrow(ax, branch_x, branch_y,
           flam_x, top_cy,
           connectionstyle="arc3,rad=-0.22", color=C_PATH_FLAMINGO, lw=2.2)

# ── Arrows: branch to SP (downward) ─────────────────────────
bot_cy = bot_y + bh / 2
draw_arrow(ax, branch_x, branch_y,
           sp_x, bot_cy,
           connectionstyle="arc3,rad=0.22", color=C_PATH_SP, lw=2.2)

# ── Arrows: Flamingo -> Frozen LLM (top) ────────────────────
draw_arrow(ax, flam_x + flam_w, top_cy,
           llm_top_x, top_cy,
           color=C_PATH_FLAMINGO, lw=2.0)

# ── Arrows: SP -> Frozen LLM (bottom) ───────────────────────
draw_arrow(ax, sp_x + sp_w, bot_cy,
           llm_top_x, bot_cy,
           color=C_PATH_SP, lw=2.0)

# ── Arrows: Frozen LLM -> CoT Output ────────────────────────
draw_arrow(ax, llm_top_x + llm_w, top_cy, out_x, top_cy, color=C_OUTPUT, lw=2.0)
draw_arrow(ax, llm_top_x + llm_w, bot_cy, out_x, bot_cy, color=C_OUTPUT, lw=2.0)

# ── "Injects into" annotation (Flamingo) ─────────────────────
# Dashed curved arrow above the main arrow showing injection relationship
inject_label_y = top_cy + 0.68
ax.annotate("",
            xy=(llm_top_x + 0.35, top_cy + 0.50),
            xytext=(flam_x + flam_w - 0.15, top_cy + 0.50),
            arrowprops=dict(arrowstyle='->', color=C_PATH_FLAMINGO,
                            lw=1.3, ls=(0, (4, 3)),
                            connectionstyle="arc3,rad=-0.25"))
ax.text((flam_x + flam_w + llm_top_x) / 2, inject_label_y + 0.15,
        "injects into LLM layers",
        ha='center', va='bottom', fontsize=8.5, color=C_PATH_FLAMINGO,
        fontstyle='italic')

# ── Dashed separator ─────────────────────────────────────────
sep_y = (top_y + bot_y + bh) / 2
ax.plot([10.2, 19.6], [sep_y, sep_y], '--', color='#BDBDBD', lw=1.2, zorder=1)

# ── Legend ────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_TRAINABLE_LIGHT, edgecolor=C_TRAINABLE,
                   linewidth=2.0, label='Trainable'),
    mpatches.Patch(facecolor=C_FROZEN_LIGHT, edgecolor=C_FROZEN,
                   linewidth=2.0, label='Frozen'),
    mpatches.Patch(facecolor=C_INPUT_LIGHT, edgecolor=C_INPUT,
                   linewidth=2.0, label='Input'),
    mpatches.Patch(facecolor=C_OUTPUT_LIGHT, edgecolor=C_OUTPUT,
                   linewidth=2.0, label='Output'),
]
ax.legend(handles=legend_items, loc='upper left',
          bbox_to_anchor=(-0.02, 1.06), ncol=4,
          fontsize=11, frameon=True, framealpha=0.95,
          edgecolor='#BDBDBD', fancybox=True,
          handlelength=1.6, handletextpad=0.5, columnspacing=1.5)

# ── Title ─────────────────────────────────────────────────────
ax.set_title("OpenTSLM Multimodal Architecture",
             fontsize=20, fontweight='bold', pad=35, color="#212121")

# ── Summary footnote ──────────────────────────────────────────
footnote = (
    "Trainable: Encoder, Projector, Flamingo xattn layers, LM embeddings   |   "
    "Frozen: LLM backbone"
)
ax.text(0.0, -1.4, footnote,
        fontsize=9.5, color="#666", va='top', fontfamily='monospace')

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig("/home/wangni/notion-figures/sdft/fig_003.png",
            dpi=200, bbox_inches='tight', facecolor=BG)
plt.close()

print("Figure saved to /home/wangni/notion-figures/sdft/fig_003.png")
