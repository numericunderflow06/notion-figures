"""
fig_002: OpenTSLMFlamingo Architecture for Stage 6
Left-to-right architecture diagram with dual-pathway inputs merging
into Flamingo-style cross-attention within the Llama 3.2 3B backbone.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Colour palette ──────────────────────────────────────────────────
TEXT_BLUE      = "#2563EB"
TEXT_BLUE_LT   = "#DBEAFE"
TS_ORANGE      = "#D97706"
TS_ORANGE_LT   = "#FEF3C7"
FUSE_PURPLE    = "#7C3AED"
FUSE_PURP_LT   = "#EDE9FE"
LLM_GREEN      = "#059669"
LLM_GREEN_LT   = "#D1FAE5"
OUT_RED        = "#DC2626"
OUT_RED_LT     = "#FEE2E2"
DARK           = "#1E293B"
MID            = "#475569"

fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Helpers ──────────────────────────────────────────────────────────
def box(x, y, w, h, label, sub=None, fc="#fff", ec="#333",
        fs=11, fw="bold", sfs=8.5, lw=1.6, tc=DARK, sc=MID):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.20, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4)
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=sfs, color=sc, fontstyle="italic", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, zorder=4)
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, color="#64748B", lw=1.8,
          rad=0.0, head="->", sA=6, sB=6):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=head,
                        connectionstyle=f"arc3,rad={rad}",
                        color=color, linewidth=lw, mutation_scale=16,
                        shrinkA=sA, shrinkB=sB, zorder=2)
    ax.add_patch(a)


def R(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def L(b):  return (b[0], b[1] + b[3] / 2)
def T(b):  return (b[0] + b[2] / 2, b[1] + b[3])
def B(b):  return (b[0] + b[2] / 2, b[1])


# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(10.0, 9.7,
        "OpenTSLMFlamingo Architecture — Stage 6: Financial Reports",
        fontsize=15, fontweight="bold", color=DARK, ha="center", va="center")


# ═══════════════════════════════════════════════════════════════════
# BACKGROUND REGIONS
# ═══════════════════════════════════════════════════════════════════
# Text pathway (top-left)
ax.add_patch(FancyBboxPatch((0.3, 5.2), 7.2, 4.2, boxstyle="round,pad=0.3",
             facecolor=TEXT_BLUE_LT, edgecolor=TEXT_BLUE,
             linewidth=1.0, alpha=0.22, zorder=0))
ax.text(0.75, 9.15, "Text Pathway", fontsize=10, fontweight="bold",
        color=TEXT_BLUE, alpha=0.8)

# Time-series pathway (bottom-left)
ax.add_patch(FancyBboxPatch((0.3, 0.3), 7.2, 4.6, boxstyle="round,pad=0.3",
             facecolor=TS_ORANGE_LT, edgecolor=TS_ORANGE,
             linewidth=1.0, alpha=0.22, zorder=0))
ax.text(0.75, 4.62, "Time-Series Pathway", fontsize=10, fontweight="bold",
        color=TS_ORANGE, alpha=0.8)


# ═══════════════════════════════════════════════════════════════════
# TEXT PATHWAY (top half)
# ═══════════════════════════════════════════════════════════════════
BH = 1.05

b_filing = box(0.7, 7.6, 2.4, BH, "Filing Text",
               "≤ 3 000 chars (markdown)", fc=TEXT_BLUE_LT, ec=TEXT_BLUE)

b_prompt = box(0.7, 5.9, 2.4, BH, "Pre / Post Prompts",
               "QA context + question", fc=TEXT_BLUE_LT, ec=TEXT_BLUE)

b_tok = box(3.8, 7.6, 2.5, BH, "LLM Tokenizer",
            "Llama 3.2 tokenizer", fc=TEXT_BLUE_LT, ec=TEXT_BLUE)

b_emb = box(3.8, 5.9, 2.5, BH, "Token Embeddings",
            "[B, T_txt, 3072]", fc=TEXT_BLUE_LT, ec=TEXT_BLUE)

# Text pathway arrows
arrow(*R(b_filing), *L(b_tok), color=TEXT_BLUE)
arrow(*B(b_tok), *T(b_emb), color=TEXT_BLUE)
arrow(*R(b_prompt), *L(b_emb), color=TEXT_BLUE)


# ═══════════════════════════════════════════════════════════════════
# TIME-SERIES PATHWAY (bottom half)
# ═══════════════════════════════════════════════════════════════════
b_stock = box(0.7, 2.5, 2.4, BH, "Stock Prices",
              "60 days, z-normalized", fc=TS_ORANGE_LT, ec=TS_ORANGE)

b_cnn = box(3.8, 3.15, 2.5, BH, "CNNTokenizer",
            "Conv1d, patch_size=4", fc=TS_ORANGE_LT, ec=TS_ORANGE)

b_per = box(3.8, 1.55, 2.5, BH, "Perceiver Resampler",
            "64 latents", fc=TS_ORANGE_LT, ec=TS_ORANGE, fs=10)

# Dimension annotations
ax.text(6.5, 3.65, "[B, N, 128]", fontsize=7.5, color=TS_ORANGE,
        ha="center", fontstyle="italic", zorder=4)
ax.text(6.5, 1.15, "[B, 64, 128]", fontsize=7.5, color=TS_ORANGE,
        ha="center", fontstyle="italic", zorder=4)

# EMBED_DIM callout
ax.text(5.05, 4.35, "EMBED_DIM = 128", fontsize=9, fontweight="bold",
        color=FUSE_PURPLE, ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=FUSE_PURPLE, alpha=0.9, linewidth=1.2), zorder=5)

# TS pathway arrows
arrow(*R(b_stock), *L(b_cnn), color=TS_ORANGE)
arrow(*B(b_cnn), *T(b_per), color=TS_ORANGE)


# ═══════════════════════════════════════════════════════════════════
# MLP PROJECTOR  (bridge between TS and LLM)
# ═══════════════════════════════════════════════════════════════════
b_proj = box(7.6, 1.55, 2.3, BH, "MLP Projector",
             "128 → 3072", fc=FUSE_PURP_LT, ec=FUSE_PURPLE)

arrow(*R(b_per), *L(b_proj), color=FUSE_PURPLE)


# ═══════════════════════════════════════════════════════════════════
# LLM BACKBONE (right side — tall box)
# ═══════════════════════════════════════════════════════════════════
lx, ly, lw_box, lh = 11.0, 0.5, 4.8, 8.8
ax.add_patch(FancyBboxPatch((lx, ly), lw_box, lh,
             boxstyle="round,pad=0.25", facecolor=LLM_GREEN_LT,
             edgecolor=LLM_GREEN, linewidth=2.2, zorder=1))
ax.text(lx + lw_box / 2, ly + lh - 0.45, "Llama 3.2 3B",
        fontsize=15, fontweight="bold", color=LLM_GREEN,
        ha="center", zorder=4)

# hidden_dim badge
ax.text(lx + lw_box / 2, ly + 0.3, "hidden_dim = 3072",
        fontsize=8.5, fontweight="bold", color=LLM_GREEN, ha="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=LLM_GREEN, alpha=0.9, linewidth=1.0), zorder=5)

# Internal blocks (bottom → top)
ix = lx + 0.55
iw = lw_box - 1.1

b_sa = box(ix, 1.3, iw, 1.2, "Self-Attention +\nFFN Layers",
           "28 decoder blocks", fc="#ECFDF5", ec=LLM_GREEN, fs=10, lw=1.2)

b_xattn = box(ix, 3.5, iw, 1.6, "Gated Cross-\nAttention",
              "interleaved every N layers",
              fc=FUSE_PURP_LT, ec=FUSE_PURPLE, fs=12, lw=1.5)

b_fused = box(ix, 6.0, iw, 1.1, "Fused Hidden States",
              "[B, T, 3072]", fc="#ECFDF5", ec=LLM_GREEN, fs=10, lw=1.2)

b_lmh = box(ix, 7.6, iw, 0.95, "LM Head",
            "→ next-token logits", fc="#ECFDF5", ec=LLM_GREEN, fs=10, lw=1.2)

# Internal arrows
arrow(*T(b_sa), *B(b_xattn), color=LLM_GREEN, lw=1.5)
arrow(*T(b_xattn), *B(b_fused), color=FUSE_PURPLE, lw=1.5)
arrow(*T(b_fused), *B(b_lmh), color=LLM_GREEN, lw=1.5)


# ═══════════════════════════════════════════════════════════════════
# CONNECTIONS INTO LLM
# ═══════════════════════════════════════════════════════════════════

# Text embeddings → Self-Attention layers (text tokens enter as main input)
arrow(6.3, 6.42, ix, 1.9, color=TEXT_BLUE, lw=2.2, rad=-0.18)

# Small label on the text arrow
ax.text(8.2, 4.3, "text tokens", fontsize=8.5, color=TEXT_BLUE,
        ha="center", fontstyle="italic", rotation=0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor=TEXT_BLUE, alpha=0.85, linewidth=0.8), zorder=5)

# MLP Projector → Gated Cross-Attention (time-series features)
arrow(9.9, 2.07, ix, 4.1, color=TS_ORANGE, lw=2.2, rad=-0.08)

# Small label on the TS arrow
ax.text(10.2, 3.0, "TS features", fontsize=8.5, color=TS_ORANGE,
        ha="center", fontstyle="italic", rotation=0,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor=TS_ORANGE, alpha=0.85, linewidth=0.8), zorder=5)


# ═══════════════════════════════════════════════════════════════════
# OUTPUT (right side)
# ═══════════════════════════════════════════════════════════════════
b_out = box(16.6, 3.5, 3.0, 2.2, "Classification",
            "(a) / (b) / (c)",
            fc=OUT_RED_LT, ec=OUT_RED, fs=14, fw="bold", lw=2.0)

# Task labels
ax.text(18.1, 3.1, "price return  ·  volatility  ·  direction",
        fontsize=8.5, color=MID, ha="center", fontstyle="italic", zorder=4)

# Arrow from LM Head → output
arrow(lx + lw_box, 8.07, 18.1, 5.7, color=OUT_RED, lw=2.2, rad=0.2)


# ═══════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(fc=TEXT_BLUE_LT, ec=TEXT_BLUE, lw=1.3, label="Text pathway"),
    mpatches.Patch(fc=TS_ORANGE_LT, ec=TS_ORANGE, lw=1.3, label="Time-series pathway"),
    mpatches.Patch(fc=FUSE_PURP_LT, ec=FUSE_PURPLE, lw=1.3, label="Fusion (cross-attention)"),
    mpatches.Patch(fc=LLM_GREEN_LT, ec=LLM_GREEN, lw=1.3, label="Llama 3.2 3B backbone"),
    mpatches.Patch(fc=OUT_RED_LT, ec=OUT_RED, lw=1.3, label="Output"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9.5,
          framealpha=0.95, edgecolor="#CBD5E1",
          bbox_to_anchor=(0.995, 0.005))

plt.tight_layout(pad=0.4)
plt.savefig("/home/wangni/notion-figures/finance/fig_002.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: /home/wangni/notion-figures/finance/fig_002.png")
