"""
fig_005: Temporally-Biased Cross-Attention
Two-part diagram:
  (1) Block diagram showing Q/K/V flow with temporal bias addition
  (2) Small example attention matrix heatmap with soft diagonal pattern
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
BG       = "#FFFFFF"
BOX_TS   = "#3B82F6"   # blue – time-series path
BOX_TXT  = "#10B981"   # green – text path
BOX_ATTN = "#8B5CF6"   # purple – attention internals
BOX_OUT  = "#F59E0B"   # amber – output
BOX_BIAS = "#EF4444"   # red – temporal bias
ARROW_C  = "#374151"   # dark grey arrows
LABEL_C  = "#111827"   # near-black labels
LIGHT_BG = "#F9FAFB"   # very light grey for sub-panels
GATE_C   = "#6366F1"   # indigo for gate

fig = plt.figure(figsize=(16, 10.5), facecolor=BG, dpi=200)

# ── Layout: left 60 % block diagram, right 35 % heatmap ────────────────────
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.08,
                      left=0.03, right=0.97, bottom=0.14, top=0.90)

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Block diagram (left)
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis("off")
ax1.set_facecolor(BG)

def rounded_box(ax, xy, w, h, color, text, fontsize=11, fontcolor="white",
                alpha=1.0, lw=0, edgecolor=None, style="round,pad=0.15",
                fontweight="bold", text_y_offset=0):
    """Draw a rounded rectangle with centred text."""
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch(xy, w, h, boxstyle=style,
                         facecolor=color, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2 + text_y_offset
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, fontweight=fontweight, zorder=3)
    return box

def arrow(ax, start, end, color=ARROW_C, lw=1.8, style="-|>", shrinkA=4, shrinkB=4):
    a = FancyArrowPatch(start, end, arrowstyle=style,
                        color=color, lw=lw, shrinkA=shrinkA, shrinkB=shrinkB,
                        mutation_scale=14, zorder=4)
    ax.add_patch(a)
    return a

# ── Input boxes ─────────────────────────────────────────────────────────────
# TS patches input (left side)
rounded_box(ax1, (0.3, 8.2), 2.8, 0.85, BOX_TS, "TS Patches", 12)
ax1.text(1.7, 7.85, "(from projector)", ha="center", va="top",
         fontsize=8.5, color="#6B7280", style="italic")

# Text embeddings input (right side)
rounded_box(ax1, (6.0, 8.2), 3.2, 0.85, BOX_TXT, "Text Embeddings", 12)
ax1.text(7.6, 7.85, "(from LLM tokens)", ha="center", va="top",
         fontsize=8.5, color="#6B7280", style="italic")

# ── LayerNorm on queries ────────────────────────────────────────────────────
rounded_box(ax1, (0.7, 6.6), 2.0, 0.7, "#60A5FA", "LayerNorm", 10.5, fontweight="normal")
arrow(ax1, (1.7, 8.2), (1.7, 7.3))

# ── Q / K / V projection boxes ─────────────────────────────────────────────
# Q from TS
rounded_box(ax1, (0.5, 5.2), 1.3, 0.65, BOX_ATTN, "Q", 13)
ax1.text(1.15, 4.85, "4 heads", ha="center", va="top",
         fontsize=8, color="#6B7280")
arrow(ax1, (1.7, 6.6), (1.15, 5.85))

# K from text
rounded_box(ax1, (4.0, 5.2), 1.3, 0.65, BOX_ATTN, "K", 13)
arrow(ax1, (7.6, 8.2), (7.6, 6.15), color=BOX_TXT)
arrow(ax1, (7.6, 6.15), (4.65, 5.85), color=BOX_TXT)

# V from text
rounded_box(ax1, (6.5, 5.2), 1.3, 0.65, BOX_ATTN, "V", 13)
arrow(ax1, (7.6, 6.15), (7.15, 5.85), color=BOX_TXT)
# small dot at junction
ax1.plot(7.6, 6.15, 'o', color=BOX_TXT, markersize=5, zorder=5)

# ── Attention logits: Q K^T ─────────────────────────────────────────────────
rounded_box(ax1, (1.6, 3.7), 2.6, 0.7, "#7C3AED", "QK$^T$ / $\\sqrt{d_k}$", 11, fontweight="normal")
arrow(ax1, (1.15, 5.2), (2.5, 4.4))
arrow(ax1, (4.65, 5.2), (3.7, 4.4))

# ── Temporal bias box ──────────────────────────────────────────────────────
rounded_box(ax1, (5.5, 3.7), 3.6, 0.7, BOX_BIAS, "$-\\alpha\\,|\\,t_{ts} - t_{text}\\,|$", 11.5)
ax1.text(7.3, 3.45, "$\\alpha$ learnable (init 1.0)", ha="center", va="top",
         fontsize=8.5, color="#991B1B", fontweight="bold")

# ── "+" addition circle ────────────────────────────────────────────────────
plus_x, plus_y = 4.6, 2.6
circle = plt.Circle((plus_x, plus_y), 0.32, facecolor="#E0E7FF",
                     edgecolor=BOX_ATTN, linewidth=2, zorder=3)
ax1.add_patch(circle)
ax1.text(plus_x, plus_y, "+", ha="center", va="center",
         fontsize=18, fontweight="bold", color=BOX_ATTN, zorder=4)

arrow(ax1, (2.9, 3.7), (4.35, 2.92))
arrow(ax1, (7.3, 3.7), (4.85, 2.92))
ax1.text(4.6, 3.15, "add bias to logits", ha="center", va="bottom",
         fontsize=8, color="#6B7280", style="italic")

# ── Softmax ─────────────────────────────────────────────────────────────────
rounded_box(ax1, (3.8, 1.5), 1.6, 0.6, "#A78BFA", "Softmax", 11, fontweight="normal")
arrow(ax1, (plus_x, plus_y - 0.32), (plus_x, 2.1))

# ── Weighted sum with V ─────────────────────────────────────────────────────
rounded_box(ax1, (3.5, 0.3), 2.2, 0.65, BOX_ATTN, "Attn × V", 12)
arrow(ax1, (4.6, 1.5), (4.6, 0.95))
# V arrow down to weighted sum
arrow(ax1, (7.15, 5.2), (7.15, 1.1), color="#9CA3AF", lw=1.2, style="-")
arrow(ax1, (7.15, 1.1), (5.7, 0.62), color="#9CA3AF")
ax1.text(7.35, 3.1, "V", ha="left", va="center",
         fontsize=10, color="#7C3AED", fontweight="bold")

# ── Timestamps for TS and Text ──────────────────────────────────────────────
# TS timestamps
ax1.text(0.3, 9.35, "$t_{ts}$", ha="center", va="center",
         fontsize=11, color=BOX_TS, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#DBEAFE", ec=BOX_TS, lw=1))
arrow(ax1, (0.3, 9.05), (0.3, 4.2), color=BOX_TS, lw=1.2, style="-")
arrow(ax1, (0.3, 4.2), (5.5, 4.05), color=BOX_BIAS, lw=1.5, style="-|>")

# Text timestamps
ax1.text(9.7, 9.35, "$t_{text}$", ha="center", va="center",
         fontsize=11, color=BOX_TXT, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="#D1FAE5", ec=BOX_TXT, lw=1))
arrow(ax1, (9.7, 9.05), (9.7, 4.45), color=BOX_TXT, lw=1.2, style="-")
arrow(ax1, (9.7, 4.45), (9.1, 4.1), color=BOX_BIAS, lw=1.5, style="-|>")

# ── Title for left panel ───────────────────────────────────────────────────
ax1.text(5.0, 10.0, "Cross-Attention Data Flow", ha="center", va="center",
         fontsize=14, fontweight="bold", color=LABEL_C)

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Attention heatmap (right)
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])

# ── Build synthetic attention with temporal bias ────────────────────────────
n_ts   = 8    # TS patches (queries)
n_text = 12   # text tokens (keys)

# Timestamps: TS patches span [0,1] uniformly
t_ts   = np.linspace(0, 1, n_ts)
# Text tokens: most have no timestamp (set far away), a few are temporal refs
t_text = np.full(n_text, np.nan)
# Assign temporal anchors to a few text positions (simulating temporal refs)
temporal_text_indices = [1, 3, 5, 7, 9, 11]
temporal_text_times   = [0.0, 0.15, 0.35, 0.55, 0.75, 1.0]
for idx, t in zip(temporal_text_indices, temporal_text_times):
    t_text[idx] = t

# Compute bias matrix: -alpha * |t_ts_i - t_text_j|
alpha = 1.0
bias = np.zeros((n_ts, n_text))
for i in range(n_ts):
    for j in range(n_text):
        if not np.isnan(t_text[j]):
            bias[i, j] = -alpha * abs(t_ts[i] - t_text[j])
        else:
            bias[i, j] = -0.5  # moderate negative for non-temporal tokens

# Random base logits (small) + bias
np.random.seed(42)
logits = np.random.randn(n_ts, n_text) * 0.3 + bias

# Softmax along key dimension
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

attn = softmax(logits, axis=1)

# ── Plot heatmap ────────────────────────────────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap

# warm-cool colormap: cool (blue) for low, warm (orange/red) for high
colors_cmap = ["#1E3A5F", "#3B82F6", "#93C5FD", "#FDE68A", "#F59E0B", "#DC2626"]
cmap = LinearSegmentedColormap.from_list("temporal", colors_cmap, N=256)

im = ax2.imshow(attn, cmap=cmap, aspect="auto", interpolation="nearest")

# Axis labels
ts_labels = [f"P{i}" for i in range(n_ts)]
text_labels = []
text_token_names = ["[CLS]", "Jan", "rose", "Feb", "then", "Apr", "the",
                    "Jun", "trend", "Aug", "shows", "Dec"]
for j in range(n_text):
    text_labels.append(text_token_names[j])

ax2.set_xticks(range(n_text))
ax2.set_xticklabels(text_labels, fontsize=9, rotation=45, ha="right")
ax2.set_yticks(range(n_ts))
ax2.set_yticklabels(ts_labels, fontsize=10)

ax2.set_xlabel("Text Tokens (Keys)", fontsize=11, fontweight="bold", labelpad=8)
ax2.set_ylabel("TS Patches (Queries)", fontsize=11, fontweight="bold", labelpad=8)

# Highlight temporal-reference columns
for idx in temporal_text_indices:
    ax2.get_xticklabels()[idx].set_color(BOX_TXT)
    ax2.get_xticklabels()[idx].set_fontweight("bold")

# Title
ax2.set_title("Attention Weights After\nTemporal Proximity Bias", fontsize=13,
              fontweight="bold", color=LABEL_C, pad=12)

# Colorbar
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label("Attention Weight", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# ── Annotate the diagonal preference ───────────────────────────────────────
# Draw a subtle dashed line showing the "diagonal" where ts time ≈ text time
# Map temporal text indices to their positions
diag_ts_y = []
diag_txt_x = []
for idx, t in zip(temporal_text_indices, temporal_text_times):
    # Find closest TS patch
    closest_ts = np.argmin(np.abs(t_ts - t))
    diag_ts_y.append(closest_ts)
    diag_txt_x.append(idx)

ax2.plot(diag_txt_x, diag_ts_y, '--', color="white", linewidth=1.8, alpha=0.8, zorder=5)
# Place label in upper-left area of heatmap (away from crowded bottom-right)
ax2.text(0.5, 0.3, "temporal\nalignment\ndiagonal", fontsize=8.5, color="white",
         fontweight="bold", ha="center", va="center",
         bbox=dict(boxstyle="round,pad=0.25", fc="#00000088", ec="white", lw=0.5),
         zorder=6)

# ── Cell value annotations on high-attention cells ──────────────────────────
for i in range(n_ts):
    for j in range(n_text):
        val = attn[i, j]
        if val > 0.12:
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5, fontweight="bold",
                     color="white" if val > 0.15 else "#1E3A5F", zorder=6)

# ── Supra-title ─────────────────────────────────────────────────────────────
fig.suptitle("Temporally-Biased Cross-Attention", fontsize=17, fontweight="bold",
             color=LABEL_C, y=0.96)

# ── Bottom annotation: the full formula ─────────────────────────────────────
formula = (r"$\mathrm{Output} = \mathrm{ts\_embeds} \;+\; "
           r"\tanh(\mathrm{gate}) \;\cdot\; "
           r"\mathrm{CrossAttn}\!\left(\mathrm{LN}(\mathrm{ts\_embeds}),\; "
           r"\mathrm{text\_embeds}\right)$"
           r"$\qquad$"
           r"bias$(i,j) = -\alpha\,|\,t_{ts}^{(i)} - t_{text}^{(j)}\,|$")
fig.text(0.5, 0.025, formula, ha="center", va="bottom",
         fontsize=10.5, color="#374151",
         bbox=dict(boxstyle="round,pad=0.4", fc="#F3F4F6", ec="#D1D5DB", lw=0.8))

# ── Legend annotation on left panel ─────────────────────────────────────────
# Gated residual note near bottom
ax1.text(1.5, -0.15, "Gated residual connection (~4M params total)",
         ha="center", va="top", fontsize=9, color="#6B7280", style="italic")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/alignment/fig_005.png"
fig.savefig(out_path, dpi=200, facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
