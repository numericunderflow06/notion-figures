"""
fig_008: Representation Analysis Pipeline (CAF)
Schematic of planned probing workflow.
Methodology only — not measured.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---- style ----
PALETTE = {
    "ckpt":    "#2E4057",   # deep slate
    "node_bg": "#F4F6FA",   # very light gray-blue
    "node_ed": "#3B6FB6",   # blue
    "accent":  "#D9534F",   # red accent
    "soft":    "#7FB069",   # green
    "amber":   "#E0A458",   # amber
    "ink":     "#1F2933",
    "muted":   "#6B7280",
}

fig, ax = plt.subplots(figsize=(15.5, 7.6), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---- title + caption ----
ax.text(50, 53.6, "Representation Analysis Pipeline",
        ha="center", va="center", fontsize=17, fontweight="bold",
        color=PALETTE["ink"])
ax.text(50, 50.4,
        "Planned probing of a trained CAF checkpoint  ·  methodology only — not measured",
        ha="center", va="center", fontsize=11, style="italic",
        color=PALETTE["muted"])

# ---- checkpoint node (left) ----
ckpt_x, ckpt_y, ckpt_w, ckpt_h = 2.5, 19, 15, 16
ckpt = FancyBboxPatch((ckpt_x, ckpt_y), ckpt_w, ckpt_h,
                      boxstyle="round,pad=0.3,rounding_size=1.2",
                      linewidth=2.0, edgecolor=PALETTE["ckpt"],
                      facecolor=PALETTE["ckpt"])
ax.add_patch(ckpt)
ax.text(ckpt_x + ckpt_w/2, ckpt_y + ckpt_h - 2.2,
        "Trained CAF\nCheckpoint",
        ha="center", va="top", fontsize=13, fontweight="bold",
        color="white")

# trainable param chips inside checkpoint
chips = ["Perceiver", "Gated x-attn", "LoRA", r"$E_{cls}$ + P", "FiLM"]
for i, c in enumerate(chips):
    cy = ckpt_y + ckpt_h - 6.4 - i*1.7
    chip = FancyBboxPatch((ckpt_x + 1.4, cy - 0.7), ckpt_w - 2.8, 1.35,
                          boxstyle="round,pad=0.05,rounding_size=0.4",
                          linewidth=0.8, edgecolor="white",
                          facecolor="#43607F")
    ax.add_patch(chip)
    ax.text(ckpt_x + ckpt_w/2, cy, c, ha="center", va="center",
            fontsize=9, color="white")

# ---- four analysis nodes (right column) ----
node_x = 38
node_w = 22
node_h = 9.5
gap_y  = 2.0
top_y  = 39   # top edge of first node
positions = [top_y - i*(node_h + gap_y) for i in range(4)]

node_titles = [
    "Class-Embedding Geometry",
    "Perceiver Attention",
    "Gate Utilization",
    "Confidence & Surface Ablations",
]
node_subs = [
    r"cosine matrix · PCA · effective rank",
    r"query $\rightarrow$ patch attention heatmap",
    r"per-layer gated x-attn $\tanh(\alpha)$ profile",
    r"top-1 conf. vs. {prefix, interleaved, meta, ensemble}",
]
node_colors = [PALETTE["node_ed"], "#8E44AD", PALETTE["soft"], PALETTE["amber"]]

# helper: draw a small thumbnail inside the node
def draw_thumb_cosine(ax, x0, y0, w, h):
    """Cosine similarity matrix thumbnail."""
    rng = np.random.RandomState(3)
    n = 6
    M = rng.uniform(0.05, 0.45, size=(n, n))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    # off-diag block to suggest class clustering
    M[:3, :3] += 0.35; M[3:, 3:] += 0.3
    M = np.clip(M, 0, 1)
    extent = (x0, x0+w, y0, y0+h)
    ax.imshow(M, extent=extent, cmap="viridis", aspect="auto",
              vmin=0, vmax=1, zorder=2)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                               edgecolor="#333", linewidth=0.7, zorder=3))

def draw_thumb_attention(ax, x0, y0, w, h):
    """Attention grid heatmap thumbnail (queries x patches)."""
    rng = np.random.RandomState(7)
    Q, T = 4, 14
    A = rng.uniform(0, 0.2, size=(Q, T))
    # add focused bands
    for q in range(Q):
        c = rng.randint(2, T-2)
        A[q, c-1:c+2] += rng.uniform(0.6, 0.95)
    A = np.clip(A, 0, 1)
    extent = (x0, x0+w, y0, y0+h)
    ax.imshow(A, extent=extent, cmap="magma", aspect="auto",
              vmin=0, vmax=1, zorder=2)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                               edgecolor="#333", linewidth=0.7, zorder=3))

def draw_thumb_gate(ax, x0, y0, w, h):
    """Bar chart of per-layer gate magnitudes."""
    rng = np.random.RandomState(11)
    n = 8
    vals = np.abs(rng.normal(0.5, 0.25, size=n))
    vals = np.clip(vals, 0.05, 1.0)
    bw = w / (n * 1.4)
    for i, v in enumerate(vals):
        bx = x0 + 0.4 + i * (bw * 1.4)
        bh = v * (h - 0.8)
        ax.add_patch(plt.Rectangle((bx, y0 + 0.3), bw, bh,
                                   facecolor=PALETTE["soft"],
                                   edgecolor="#2E5E2E", linewidth=0.5,
                                   zorder=3))
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                               edgecolor="#333", linewidth=0.7, zorder=3))
    # baseline
    ax.plot([x0, x0+w], [y0+0.3, y0+0.3], color="#333", lw=0.6, zorder=4)

def draw_thumb_ablation(ax, x0, y0, w, h):
    """Grouped bars: 4 injection surfaces with confidence."""
    rng = np.random.RandomState(17)
    labels = ["pre", "int", "met", "ens"]
    vals = [0.55, 0.72, 0.5, 0.83]
    colors = ["#C97C5D", "#7E9CC4", "#B6B27F", "#D9534F"]
    bw = (w - 1.2) / 4
    for i, (lab, v, c) in enumerate(zip(labels, vals, colors)):
        bx = x0 + 0.6 + i * bw
        bh = v * (h - 1.6)
        ax.add_patch(plt.Rectangle((bx + bw*0.1, y0 + 0.3), bw*0.8, bh,
                                   facecolor=c, edgecolor="#333",
                                   linewidth=0.5, zorder=3))
        ax.text(bx + bw/2, y0 + 0.05, lab, ha="center", va="bottom",
                fontsize=6.5, color="#333", zorder=4)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                               edgecolor="#333", linewidth=0.7, zorder=3))

thumb_drawers = [draw_thumb_cosine, draw_thumb_attention,
                 draw_thumb_gate, draw_thumb_ablation]

# Draw the four analysis nodes
for i, (ny, ttl, sub, col, drawer) in enumerate(
        zip(positions, node_titles, node_subs, node_colors, thumb_drawers)):
    # outer box
    box = FancyBboxPatch((node_x, ny - node_h), node_w, node_h,
                         boxstyle="round,pad=0.2,rounding_size=0.8",
                         linewidth=1.6, edgecolor=col,
                         facecolor=PALETTE["node_bg"])
    ax.add_patch(box)
    # colored header strip
    hdr = FancyBboxPatch((node_x, ny - 2.2), node_w, 2.2,
                         boxstyle="round,pad=0.0,rounding_size=0.8",
                         linewidth=0, facecolor=col, alpha=0.95)
    ax.add_patch(hdr)
    ax.text(node_x + 0.8, ny - 1.1, ttl, ha="left", va="center",
            fontsize=11, fontweight="bold", color="white")
    ax.text(node_x + node_w - 0.8, ny - 1.1, f"A{i+1}",
            ha="right", va="center", fontsize=9, color="white",
            fontweight="bold")
    # subcaption
    ax.text(node_x + node_w/2, ny - 3.3, sub, ha="center", va="center",
            fontsize=9, style="italic", color=PALETTE["ink"])
    # thumbnail
    th_x = node_x + 0.9
    th_y = ny - node_h + 0.7
    th_w = node_w - 1.8
    th_h = node_h - 4.6
    drawer(ax, th_x, th_y, th_w, th_h)

# ---- arrows from checkpoint to nodes ----
ckpt_right_x = ckpt_x + ckpt_w
ckpt_mid_y   = ckpt_y + ckpt_h/2
for ny in positions:
    target_x = node_x
    target_y = ny - node_h/2
    arrow = FancyArrowPatch((ckpt_right_x + 0.3, ckpt_mid_y),
                            (target_x - 0.4, target_y),
                            arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.6, color=PALETTE["ckpt"],
                            connectionstyle="arc3,rad=0.15",
                            zorder=1)
    ax.add_patch(arrow)

# ---- right-side synthesis panel ----
syn_x, syn_w = 64, 33
syn_top, syn_bot = 39, 3.5
syn = FancyBboxPatch((syn_x, syn_bot), syn_w, syn_top - syn_bot,
                     boxstyle="round,pad=0.25,rounding_size=1.0",
                     linewidth=1.4, edgecolor=PALETTE["accent"],
                     facecolor="#FDF5F4")
ax.add_patch(syn)

ax.text(syn_x + syn_w/2, syn_top - 2.0,
        "Synthesis & Reporting",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color=PALETTE["accent"])

bullets = [
    r"$\bullet$  Does $E_{cls}$ separate classes in latent space?",
    r"$\bullet$  Do perceiver queries attend to class-relevant patches?",
    r"$\bullet$  Which layers actually use the cross-attn gate?",
    r"$\bullet$  Which injection surface yields highest confidence?",
    r"$\bullet$  Where does the classifier signal help vs. hurt?",
]
for j, b in enumerate(bullets):
    ax.text(syn_x + 1.5, syn_top - 5.5 - j*3.6, b,
            ha="left", va="center", fontsize=10,
            color=PALETTE["ink"])

# arrows from each node to synthesis
for ny in positions:
    src_x = node_x + node_w
    src_y = ny - node_h/2
    tgt_x = syn_x
    tgt_y = ny - node_h/2
    arrow = FancyArrowPatch((src_x + 0.2, src_y),
                            (tgt_x - 0.3, tgt_y),
                            arrowstyle="-|>", mutation_scale=11,
                            linewidth=1.0, color=PALETTE["muted"],
                            linestyle=(0, (3, 2)),
                            zorder=1)
    ax.add_patch(arrow)

# ---- footer banner: methodology only ----
banner = FancyBboxPatch((2.5, 0.4), 95, 2.4,
                        boxstyle="round,pad=0.1,rounding_size=0.6",
                        linewidth=1.0, edgecolor=PALETTE["accent"],
                        facecolor="#FDECEA")
ax.add_patch(banner)
ax.text(50, 1.6,
        "Methodology only — no checkpoints trained yet; thumbnails are illustrative placeholders.",
        ha="center", va="center", fontsize=10.5, fontweight="bold",
        color=PALETTE["accent"])

# ---- save ----
out = "/home/wangni/notion-figures/classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_008.png"
plt.tight_layout()
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
