"""
CAF Architecture Overview (fig_001)

Three input paths:
  1. Time series -> TS encoder Phi_TS (frozen) -> Perceiver (trainable) -> P
  2. Time series -> Classifier (frozen) -> verdict -> E_cls (trainable) -> P
  3. Instruct tokens (trainable) -> P
All routed through the SAME shared projector P into the frozen LLM with
gated cross-attention. FiLM is shown as an optional dashed pathway from
the class embedding onto the perceiver queries.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ---------- palette ----------
FROZEN_FILL   = "#E8E8EC"   # light gray for frozen blocks
FROZEN_EDGE   = "#9A9AA3"
TRAIN_FILL    = "#FFD7A8"   # warm accent for trainable
TRAIN_EDGE    = "#D97706"
PROJ_FILL     = "#FCA5A5"   # stronger warm accent for P (the shared projector)
PROJ_EDGE     = "#B91C1C"
LLM_FILL      = "#DCDCE2"
LLM_EDGE      = "#6B7280"
GATED_FILL    = "#FED7AA"
GATED_EDGE    = "#C2410C"
ARROW_COLOR   = "#374151"
FILM_COLOR    = "#7C3AED"   # purple-ish for optional FiLM

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(15, 9.0), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, label, sub=None, fill=FROZEN_FILL, edge=FROZEN_EDGE,
        fontsize=11, lw=1.4, sub_fontsize=8.5, italic_sub=True, ls="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.18,rounding_size=0.8",
        linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls,
    )
    ax.add_patch(patch)
    if sub is None:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize, weight="bold")
    else:
        ax.text(x + w / 2, y + h / 2 + 0.9, label,
                ha="center", va="center", fontsize=fontsize, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 1.4, sub,
                ha="center", va="center", fontsize=sub_fontsize,
                style="italic" if italic_sub else "normal", color="#4B5563")
    return (x, y, w, h)


def arrow(p_from, p_to, color=ARROW_COLOR, lw=1.6, ls="-", rad=0.0,
          mutation_scale=14, zorder=2):
    a = FancyArrowPatch(
        p_from, p_to,
        arrowstyle="-|>", color=color, linewidth=lw, linestyle=ls,
        mutation_scale=mutation_scale,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(a)


def proj_badge(x, y, idx):
    """Draw the shared projector P as a small circular block with an index tag."""
    c = Circle((x, y), 1.9, facecolor=PROJ_FILL, edgecolor=PROJ_EDGE,
               linewidth=1.6, zorder=3)
    ax.add_patch(c)
    ax.text(x, y + 0.2, "P", ha="center", va="center",
            fontsize=12, weight="bold", color="#7F1D1D", zorder=4)
    # small "same instance" badge
    badge = Circle((x + 1.7, y + 1.7), 0.85,
                   facecolor="white", edgecolor=PROJ_EDGE, linewidth=1.0,
                   zorder=5)
    ax.add_patch(badge)
    ax.text(x + 1.7, y + 1.7, f"{idx}", ha="center", va="center",
            fontsize=7.5, weight="bold", color=PROJ_EDGE, zorder=6)


# ============================================================
# LEFT COLUMN: three input lanes
# ============================================================

# --- Lane 1: TS encoder -> Perceiver -> P1
ts_in   = box(2.5, 46.5, 13, 5,  "Time Series",
              sub="x : R^{T x C}", fill="white", edge="#94A3B8")
ts_enc  = box(2.5, 36.5, 13, 6,  r"TS Encoder  $\Phi_{TS}$",
              sub="frozen (patch 1D-conv)", fill=FROZEN_FILL, edge=FROZEN_EDGE)
percv   = box(2.5, 26.0, 13, 6,  "Perceiver",
              sub="trainable (Q tokens)", fill=TRAIN_FILL, edge=TRAIN_EDGE)

# --- Lane 2: Classifier -> verdict -> E_cls -> P2
clf_in  = box(20, 46.5, 13, 5, "Time Series",
              sub="(same x)", fill="white", edge="#94A3B8")
clf     = box(20, 36.5, 13, 6, "Classifier  $f_{cls}$",
              sub="frozen, independent", fill=FROZEN_FILL, edge=FROZEN_EDGE)
verdict = box(20, 30.0, 13, 4, r"verdict  $\hat{y} \in \{1,\dots,K\}$",
              fill="white", edge="#94A3B8", fontsize=10)
ecls    = box(20, 22.5, 13, 5.5, r"$E_{cls}$  (Mode V / P)",
              sub="trainable class table", fill=TRAIN_FILL, edge=TRAIN_EDGE)

# --- Lane 3: Instruct tokens -> P3
instr_in = box(37.5, 46.5, 13, 5, "Task / Instruct",
               sub="optional", fill="white", edge="#94A3B8")
instr    = box(37.5, 36.5, 13, 6, "Instruct Tokens",
               sub="trainable embeddings", fill=TRAIN_FILL, edge=TRAIN_EDGE)

# arrows down each lane
arrow((9.0, 46.5), (9.0, 42.5))     # ts_in -> ts_enc
arrow((9.0, 36.5), (9.0, 32.0))     # ts_enc -> perceiver
arrow((26.5, 46.5), (26.5, 42.5))   # clf_in -> classifier
arrow((26.5, 36.5), (26.5, 34.0))   # classifier -> verdict
arrow((26.5, 30.0), (26.5, 28.0))   # verdict -> E_cls
arrow((44.0, 46.5), (44.0, 42.5))   # instr_in -> instruct tokens


# ============================================================
# SHARED PROJECTOR P at three call sites
# ============================================================
# Lane 1 -> P1 (below perceiver)
proj_badge(9.0, 21.0, 1)
arrow((9.0, 26.0), (9.0, 22.9))

# Lane 2 -> P2 (below E_cls)
proj_badge(26.5, 17.5, 2)
arrow((26.5, 22.5), (26.5, 19.4))

# Lane 3 -> P3 (below instruct tokens)
proj_badge(44.0, 31.0, 3)
arrow((44.0, 36.5), (44.0, 32.9))


# ============================================================
# FiLM optional dashed pathway: from E_cls (class embedding)
# onto Perceiver queries.
# ============================================================
film_arrow = FancyArrowPatch(
    (20.0, 25.2),   # left edge of E_cls
    (15.5, 29.0),   # right edge of perceiver
    arrowstyle="-|>", color=FILM_COLOR, linewidth=1.8,
    linestyle=(0, (5, 3)),
    mutation_scale=14,
    connectionstyle="arc3,rad=-0.25",
    zorder=2,
)
ax.add_patch(film_arrow)
ax.text(17.2, 32.0, "FiLM\n(optional)",
        fontsize=9.5, color=FILM_COLOR, style="italic", weight="bold",
        ha="center", va="center")


# ============================================================
# RIGHT SIDE: Frozen LLM with gated cross-attention
# ============================================================
# Bus that collects the three projected streams into the LLM
bus_x = 56.0
ax.plot([bus_x, bus_x], [12, 36], color="#9CA3AF", linewidth=1.4,
        linestyle=(0, (2, 2)), zorder=1)

# arrows from each P into the bus
arrow((9.0 + 1.9, 21.0), (bus_x, 21.0), rad=0.0, lw=1.6)
arrow((26.5 + 1.9, 17.5), (bus_x, 17.5), rad=0.0, lw=1.6)
arrow((44.0 + 1.9, 31.0), (bus_x, 31.0), rad=0.0, lw=1.6)

# small text on bus
ax.text(53.5, 34.0, "shared latent\n$d_{lat}\\!\\to\\!d_{llm}$",
        fontsize=9, color="#4B5563", style="italic", ha="right")

# --- LLM block (frozen)
llm_x, llm_y, llm_w, llm_h = 60.5, 11.5, 32, 30
llm_patch = FancyBboxPatch(
    (llm_x, llm_y), llm_w, llm_h,
    boxstyle="round,pad=0.3,rounding_size=1.2",
    linewidth=1.6, edgecolor=LLM_EDGE, facecolor=LLM_FILL,
)
ax.add_patch(llm_patch)
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 2.0,
        "Frozen LLM backbone  +  LoRA",
        ha="center", va="center", fontsize=12, weight="bold")
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 4.4,
        "(LLaMA-style decoder, LoRA = trainable)",
        ha="center", va="center", fontsize=9, color="#4B5563", style="italic")

# Three stacked transformer layers with interleaved gated x-attn
layer_h = 4.6
gap = 1.2
for i in range(3):
    y0 = llm_y + 3.0 + i * (layer_h + gap)
    # self-attn / FFN block (frozen)
    box(llm_x + 2.0, y0, 14, layer_h,
        f"LLM block {3 - i}",
        sub="self-attn + FFN (frozen)",
        fill=FROZEN_FILL, edge=FROZEN_EDGE,
        fontsize=10, sub_fontsize=8)
    # gated cross-attention (trainable)
    box(llm_x + 17.5, y0, 12.5, layer_h,
        "Gated cross-attn",
        sub=r"$\tanh(\alpha) \cdot$ x-attn",
        fill=GATED_FILL, edge=GATED_EDGE,
        fontsize=10, sub_fontsize=8)
    # connector inside LLM: bus -> gated x-attn at this layer
    arrow((bus_x, 21.0), (llm_x + 17.5, y0 + layer_h / 2),
          color="#6B7280", lw=1.0, rad=0.0, mutation_scale=10)

# Output arrow from LLM
arrow((llm_x + llm_w, llm_y + llm_h / 2),
      (llm_x + llm_w + 4.5, llm_y + llm_h / 2),
      lw=1.8, mutation_scale=16)
ax.text(llm_x + llm_w + 5.0, llm_y + llm_h / 2 + 1.4,
        "Answer", fontsize=11, weight="bold")
ax.text(llm_x + llm_w + 5.0, llm_y + llm_h / 2 - 1.0,
        "next-token loss\non answer span",
        fontsize=8.5, color="#4B5563", style="italic")


# ============================================================
# Title and section labels
# ============================================================
ax.text(50, 60.5, "Classifier-Aware Flamingo (CAF) — Architecture Overview",
        ha="center", va="center", fontsize=15, weight="bold")
ax.text(50, 58.4,
        "Three input paths routed through the single shared projector  $P$  "
        "into the frozen LLM with gated cross-attention",
        ha="center", va="center", fontsize=10.5, color="#374151", style="italic")

ax.text(9.0, 53.6, "(1) Perceiver path", ha="center", fontsize=10,
        color="#374151", weight="bold")
ax.text(26.5, 53.6, "(2) Classifier path", ha="center", fontsize=10,
        color="#374151", weight="bold")
ax.text(44.0, 53.6, "(3) Instruct path", ha="center", fontsize=10,
        color="#374151", weight="bold")


# ============================================================
# Legend
# ============================================================
legend_handles = [
    mpatches.Patch(facecolor=FROZEN_FILL, edgecolor=FROZEN_EDGE,
                   label="Frozen component"),
    mpatches.Patch(facecolor=TRAIN_FILL, edgecolor=TRAIN_EDGE,
                   label="Trainable component"),
    mpatches.Patch(facecolor=PROJ_FILL, edgecolor=PROJ_EDGE,
                   label="Shared projector $P$  (same instance at all 3 sites)"),
    mpatches.Patch(facecolor=GATED_FILL, edgecolor=GATED_EDGE,
                   label="Gated cross-attention (trainable)"),
    Line2D([0], [0], color=ARROW_COLOR, lw=1.8, label="Default forward flow"),
    Line2D([0], [0], color=FILM_COLOR, lw=1.8, linestyle=(0, (5, 3)),
           label="FiLM conditioning (optional)"),
]
leg = ax.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True,
                fontsize=9.5, handlelength=2.2, columnspacing=1.6,
                borderpad=0.8)
leg.get_frame().set_edgecolor("#D1D5DB")
leg.get_frame().set_linewidth(0.8)


# ============================================================
# Save
# ============================================================
out = ("/home/wangni/notion-figures/"
       "classifier-aware-flamingo-caf-a-shared-projector-architectur/"
       "fig_001.png")
plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.06)
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved figure to {out}")
