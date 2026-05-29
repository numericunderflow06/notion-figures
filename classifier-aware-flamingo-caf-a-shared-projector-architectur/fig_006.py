"""
Figure 6: Forward Pass Flow
Linearized end-to-end flow for Classifier-Aware Flamingo (CAF).
Maps pseudocode variable names from ARCHITECTURE.md §4.4 onto the block diagram.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# ---------- Palette ----------
COL_INPUT  = "#E8F1F8"   # raw signal / instruction
COL_FROZEN = "#D9D9D9"   # frozen encoders / LLM
COL_TRAIN  = "#FFE2B7"   # trainable adapters
COL_SHARED = "#FFC7C7"   # shared projector P
COL_OUT    = "#D8E9D5"   # logits

EDGE = "#333333"
TEXT = "#111111"
ARROW_VAR = "#1F4E79"    # pseudocode variable color


def add_box(ax, x, y, w, h, text, color, fontsize=10, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color=TEXT, weight=weight, zorder=3)


def add_arrow(ax, x1, y1, x2, y2, label=None, label_side="right",
              label_offset=0.08, color=EDGE, lw=1.4):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, zorder=2,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = label_offset if label_side == "right" else -label_offset
        ha = "left" if label_side == "right" else "right"
        ax.text(mx + dx, my, label,
                ha=ha, va="center", fontsize=9.5,
                color=ARROW_VAR, weight="bold", style="italic",
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.85))


# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(11.5, 14), dpi=200)
ax.set_xlim(0, 12)
ax.set_ylim(0, 16)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(6, 15.55, "CAF Forward Pass — End-to-End Flow",
        ha="center", va="center", fontsize=15, weight="bold", color=TEXT)
ax.text(6, 15.15,
        "Variable names (italic blue) trace ARCHITECTURE.md §4.4 pseudocode",
        ha="center", va="center", fontsize=10, color="#555555", style="italic")

# ---------- Row 1: raw inputs ----------
# Raw signal x
add_box(ax, 0.5, 13.1, 3.0, 1.0,
        "Raw signal  x\n(time series)", COL_INPUT, fontsize=11, weight="bold")

# Instruction text
add_box(ax, 4.5, 13.1, 3.0, 1.0,
        "Instruction text\n(prompt / question)", COL_INPUT, fontsize=11, weight="bold")

# Optional instruct token ids (task tag)
add_box(ax, 8.5, 13.1, 3.0, 1.0,
        "Task tag (optional)\n→ instruct token ids", COL_INPUT, fontsize=11, weight="bold")

# ---------- Row 2: frozen encoders + trainable instruct embed ----------
# Φ_TS (frozen)
add_box(ax, 0.5, 11.2, 3.0, 1.0,
        "Φ_TS  (frozen)\npatch 1D-conv TS encoder", COL_FROZEN, fontsize=10.5)
add_arrow(ax, 2.0, 13.1, 2.0, 12.2)

# Φ_clf (frozen)
add_box(ax, 4.5, 11.2, 3.0, 1.0,
        "Φ_clf  (frozen)\nexternal classifier", COL_FROZEN, fontsize=10.5)
# x also feeds the classifier
add_arrow(ax, 3.5, 13.6, 4.5, 12.05, label=None)

# Instruct token embedding table (trainable)
add_box(ax, 8.5, 11.2, 3.0, 1.0,
        "Instruct token embed\n(trainable, d_lat)", COL_TRAIN, fontsize=10.5)
add_arrow(ax, 10.0, 13.1, 10.0, 12.2)

# ---------- Row 3: latents ----------
# Patch tokens z_TS
add_box(ax, 0.5, 9.5, 3.0, 0.95,
        "Patch tokens (d_lat)", COL_INPUT, fontsize=10.5)
add_arrow(ax, 2.0, 11.2, 2.0, 10.45)

# Predicted class id y_hat -> E_cls lookup (trainable)
add_box(ax, 4.5, 9.5, 3.0, 0.95,
        "E_cls lookup  (trainable)\nclass embedding table", COL_TRAIN, fontsize=10.5)
add_arrow(ax, 6.0, 11.2, 6.0, 10.45, label="ŷ (class id)", label_side="right")

# Instruct latents
add_box(ax, 8.5, 9.5, 3.0, 0.95,
        "Instruct latents (d_lat)", COL_INPUT, fontsize=10.5)
add_arrow(ax, 10.0, 11.2, 10.0, 10.45)

# ---------- Row 4: perceiver (+ optional FiLM) ----------
# Perceiver resampler (trainable)
add_box(ax, 0.5, 7.6, 3.0, 1.15,
        "Perceiver resampler\n(trainable)\nN_q learned queries", COL_TRAIN, fontsize=10.5)
add_arrow(ax, 2.0, 9.5, 2.0, 8.75, label="Z_TS_lat", label_side="left")

# FiLM modulation (optional, trainable) — placed between E_cls and perceiver queries
add_box(ax, 4.5, 7.6, 3.0, 1.15,
        "FiLM  (optional, trainable)\nγ, β = f(cls_lat)\nq ← γ⊙q + β", COL_TRAIN, fontsize=10.5)
# E_cls feeds FiLM
add_arrow(ax, 6.0, 9.5, 6.0, 8.75, label="cls_lat", label_side="right")
# FiLM modulates perceiver queries
add_arrow(ax, 4.5, 8.18, 3.5, 8.18, label="q_emb", label_side="right",
          label_offset=0.05)

# cls_lat shortcut down to projector (will draw later as separate path)
# ---------- Row 5: shared linear projector P ----------
# Big shared P box spanning the width
add_box(ax, 0.5, 5.7, 11.0, 1.05,
        "Shared linear projector  P : ℝ^d_lat → ℝ^d_llm   (trainable)",
        COL_SHARED, fontsize=12, weight="bold")

# Arrow: perceiver -> P  (Z_TS_lat -> Z_TS_llm side; perceiver output is what P consumes)
add_arrow(ax, 2.0, 7.6, 2.0, 6.75, label="perceiver out", label_side="left",
          label_offset=0.06)

# Arrow: E_cls (cls_lat) -> P  (bypassing FiLM)
# cls_lat takes a path from row-3 E_cls down past FiLM into P
add_arrow(ax, 7.2, 9.5, 7.2, 6.75, label="cls_lat", label_side="right")

# Arrow: instruct latents -> P
add_arrow(ax, 10.0, 9.5, 10.0, 6.75, label="ins_lat", label_side="right")

# ---------- Row 6: projected (LLM-space) embeddings ----------
add_box(ax, 0.3, 4.2, 3.2, 1.0,
        "Z_TS_llm\n(TS tokens in d_llm)", COL_INPUT, fontsize=10.5)
add_arrow(ax, 2.0, 5.7, 2.0, 5.2, label="Z_TS_llm", label_side="left")

add_box(ax, 4.4, 4.2, 3.2, 1.0,
        "cls_llm\n(class token(s) in d_llm)", COL_INPUT, fontsize=10.5)
add_arrow(ax, 6.0, 5.7, 6.0, 5.2, label="cls_llm", label_side="right")

add_box(ax, 8.5, 4.2, 3.2, 1.0,
        "ins_llm\n(instruct tokens in d_llm)", COL_INPUT, fontsize=10.5)
add_arrow(ax, 10.0, 5.7, 10.0, 5.2, label="ins_llm", label_side="right")

# ---------- Row 7: prefix assembly + text token embed ----------
add_box(ax, 0.5, 2.6, 7.0, 1.05,
        "Prefix assembly  (concat per injection surface)\n"
        "prefix_emb = [ cls_llm ‖ Z_TS_llm ‖ ins_llm ]",
        COL_TRAIN, fontsize=11, weight="bold")

# Funnel arrows from the three llm-space boxes into the prefix box
add_arrow(ax, 2.0, 4.2, 3.0, 3.65)
add_arrow(ax, 6.0, 4.2, 4.5, 3.65)
add_arrow(ax, 10.0, 4.2, 5.5, 3.65)

# Text token embedding (frozen LLM embedding matrix)
add_box(ax, 8.5, 2.6, 3.2, 1.05,
        "Text token embed\n(frozen LLM embed)", COL_FROZEN, fontsize=10.5)
# instruction text feeds text embedding directly (shortcut from row-1 instruction)
add_arrow(ax, 5.5, 13.6, 10.1, 3.65)

# ---------- Row 8: inputs_embeds + frozen LLM ----------
add_box(ax, 0.5, 0.9, 11.0, 1.05,
        "Frozen LLM  +  LoRA  +  Gated cross-attention   →   next-token logits",
        COL_FROZEN, fontsize=12, weight="bold")
# concat into inputs_embeds (label on the merge arrows)
add_arrow(ax, 4.0, 2.6, 5.0, 1.95, label="inputs_embeds", label_side="left",
          label_offset=0.05)
add_arrow(ax, 10.0, 2.6, 7.5, 1.95)

# Final output: logits
add_box(ax, 4.5, -0.5, 3.0, 0.95,
        "Logits  →  answer span", COL_OUT, fontsize=11, weight="bold")
add_arrow(ax, 6.0, 0.9, 6.0, 0.45, label="logits", label_side="right")

# ---------- Legend ----------
legend_handles = [
    Line2D([0], [0], marker="s", color="w", label="Input / activation",
           markerfacecolor=COL_INPUT, markeredgecolor=EDGE, markersize=14),
    Line2D([0], [0], marker="s", color="w", label="Frozen module",
           markerfacecolor=COL_FROZEN, markeredgecolor=EDGE, markersize=14),
    Line2D([0], [0], marker="s", color="w", label="Trainable adapter",
           markerfacecolor=COL_TRAIN, markeredgecolor=EDGE, markersize=14),
    Line2D([0], [0], marker="s", color="w", label="Shared projector P",
           markerfacecolor=COL_SHARED, markeredgecolor=EDGE, markersize=14),
    Line2D([0], [0], marker="s", color="w", label="Output",
           markerfacecolor=COL_OUT, markeredgecolor=EDGE, markersize=14),
]
ax.legend(handles=legend_handles, loc="lower left",
          bbox_to_anchor=(-0.02, -0.08), ncol=5, frameon=False, fontsize=9.5)

plt.tight_layout()
out_path = "/home/wangni/notion-figures/classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_006.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")
