"""
fig_001.py — OpenTSLM-Flamingo architecture diagram for Bosch CheckIt.

Renders the end-to-end forward path:
  8 z-scored channels -> CNN patch encoder -> Perceiver resampler
  -> gated cross-attention injected into every Llama-3.2-1B layer
  -> German free-form generation.
Trainable blocks in Bosch red, frozen blocks in gray.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ---------- style ----------
BOSCH_RED   = "#EE0000"   # trainable
FROZEN_GRAY = "#BFBFBF"   # frozen
EDGE_DARK   = "#2A2A2A"
TEXT_DARK   = "#1A1A1A"
MUTED       = "#5A5A5A"
SHAPE_GRAY  = "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

FIG_W, FIG_H = 19.0, 9.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ---------- title ----------
ax.text(FIG_W / 2, 8.55, "OpenTSLM-Flamingo Architecture",
        fontsize=17, fontweight="bold", ha="center", color=TEXT_DARK)
ax.text(FIG_W / 2, 8.10,
        "End-to-end forward path for German expert-commentary generation on heat-pump time series",
        fontsize=10.5, ha="center", color=MUTED, style="italic")


def rounded_box(x, y, w, h, facecolor, edgecolor=None, lw=1.4, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=edgecolor if edgecolor is not None else facecolor,
        linewidth=lw,
        alpha=alpha,
    ))


def shape_arrow(x0, x1, y, label):
    """Arrow with the tensor-shape label written ABOVE the arrow line."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=EDGE_DARK,
                                lw=1.7, mutation_scale=16))
    if label:
        ax.text((x0 + x1) / 2, y + 0.45, label,
                fontsize=9.5, ha="center", color=SHAPE_GRAY, family="monospace")


# ---------- horizontal layout (left -> right) ----------
# blocks sized so gaps fit the tensor-shape labels comfortably
in_x,  in_y,  in_w,  in_h  = 0.3,  2.5, 1.9, 4.4
cnn_x, cnn_y, cnn_w, cnn_h = 3.5,  2.5, 2.3, 4.4
prc_x, prc_y, prc_w, prc_h = 7.2,  2.5, 2.3, 4.4
llm_x, llm_y, llm_w, llm_h = 10.9, 1.0, 4.9, 6.8
out_x, out_y, out_w, out_h = 17.0, 2.5, 1.9, 4.4

ARROW_Y = 4.7   # vertical centerline of the forward path

# ---------- 1. Input: 8 time-series channels ----------
rounded_box(in_x, in_y, in_w, in_h, "white", edgecolor="#999999", lw=1.2)

rng = np.random.default_rng(7)
panel_h = in_h / 8
for i in range(8):
    base_y = in_y + i * panel_h + 0.10
    xs = np.linspace(in_x + 0.20, in_x + in_w - 0.20, 60)
    phase = i * 0.7
    freq = 1.5 + 0.4 * (i % 3)
    signal = 0.18 * np.sin(np.linspace(0, freq * np.pi, 60) + phase) \
             + 0.04 * rng.standard_normal(60)
    ys = base_y + panel_h * 0.45 + signal
    ax.plot(xs, ys, color=EDGE_DARK, linewidth=0.8, alpha=0.85)

ax.text(in_x + in_w / 2, in_y + in_h + 0.30, "Input",
        fontsize=12, ha="center", color=TEXT_DARK, fontweight="bold")
ax.text(in_x + in_w / 2, in_y - 0.20,
        "8 z-scored\ntime-series channels",
        fontsize=9.5, ha="center", va="top", color=TEXT_DARK)

# arrow into CNN
shape_arrow(in_x + in_w + 0.10, cnn_x - 0.10, ARROW_Y, "(B, L)")

# ---------- 2. CNN patch encoder (trainable) ----------
rounded_box(cnn_x, cnn_y, cnn_w, cnn_h, BOSCH_RED, edgecolor=BOSCH_RED, lw=1.6)

ax.text(cnn_x + cnn_w / 2, cnn_y + cnn_h - 0.45,
        "CNN Patch\nEncoder",
        fontsize=12.5, ha="center", va="top", color="white", fontweight="bold")
ax.text(cnn_x + cnn_w / 2, cnn_y + cnn_h / 2 - 0.15,
        "Conv1d(1 → 128)\nkernel=4, stride=4\n+ LayerNorm\n+ learnable PE\n+ dropout",
        fontsize=9, ha="center", va="center", color="white",
        family="monospace", linespacing=1.5)
ax.text(cnn_x + cnn_w / 2, cnn_y + 0.30,
        "embed_dim = 128",
        fontsize=8.5, ha="center", color="white", family="monospace",
        fontstyle="italic")

# arrow into Perceiver
shape_arrow(cnn_x + cnn_w + 0.10, prc_x - 0.10, ARROW_Y, "(B, 128, L/4)")

# ---------- 3. Perceiver resampler (trainable) ----------
rounded_box(prc_x, prc_y, prc_w, prc_h, BOSCH_RED, edgecolor=BOSCH_RED, lw=1.6)

ax.text(prc_x + prc_w / 2, prc_y + prc_h - 0.45,
        "Perceiver\nResampler",
        fontsize=12.5, ha="center", va="top", color="white", fontweight="bold")
ax.text(prc_x + prc_w / 2, prc_y + prc_h / 2 - 0.15,
        "PerceiverResampler\n(dim=128)\n\nlearnable latent\nqueries  →\nfixed-length\ntokens",
        fontsize=9, ha="center", va="center", color="white",
        family="monospace", linespacing=1.5)
ax.text(prc_x + prc_w / 2, prc_y + 0.30,
        "open_flamingo",
        fontsize=8.5, ha="center", color="white", family="monospace",
        fontstyle="italic")

# arrow into Llama (main signal)
shape_arrow(prc_x + prc_w + 0.10, llm_x - 0.10, ARROW_Y, "(B, N, 128)")

# ---------- 4. Llama-3.2-1B decoder with interleaved gated cross-attention ----------
rounded_box(llm_x, llm_y, llm_w, llm_h, FROZEN_GRAY, edgecolor="#7A7A7A", lw=1.4, alpha=0.85)

ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.35,
        "Llama-3.2-1B Decoder",
        fontsize=13, ha="center", va="top", color=TEXT_DARK, fontweight="bold")
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 0.78,
        "(frozen weights, FlamingoLMMixin)",
        fontsize=9.5, ha="center", va="top", color=TEXT_DARK, style="italic")
ax.text(llm_x + llm_w / 2, llm_y + llm_h - 1.18,
        "cross_attn_every_n_layers = 1",
        fontsize=9.5, ha="center", color=TEXT_DARK, family="monospace")

# inner stack: cross-attn + Llama layer pairs (showing 4 representative pairs)
n_pairs = 4
pair_total_h = 0.95
pair_w_attn = 1.85
pair_w_llm  = 2.25
pair_x_attn = llm_x + 0.25
pair_x_llm  = pair_x_attn + pair_w_attn + 0.20
stack_bottom_y = llm_y + 0.45

# "...more layers..." indicator at top of stack
ax.text(llm_x + llm_w / 2,
        stack_bottom_y + n_pairs * pair_total_h + 0.05,
        "⋮   (16 decoder layers total)   ⋮",
        fontsize=9, ha="center", color=MUTED, style="italic")

cross_attn_centers = []
for i in range(n_pairs):
    y = stack_bottom_y + i * pair_total_h
    pair_h = 0.75

    # gated cross-attention (trainable, red)
    rounded_box(pair_x_attn, y, pair_w_attn, pair_h,
                BOSCH_RED, edgecolor=BOSCH_RED, lw=1.2)
    ax.text(pair_x_attn + pair_w_attn / 2, y + pair_h / 2 + 0.10,
            "Gated", fontsize=8.8, ha="center", va="center",
            color="white", fontweight="bold")
    ax.text(pair_x_attn + pair_w_attn / 2, y + pair_h / 2 - 0.13,
            "Cross-Attn", fontsize=8.3, ha="center", va="center",
            color="white", family="monospace")

    # frozen Llama decoder layer (white card on gray background)
    rounded_box(pair_x_llm, y, pair_w_llm, pair_h,
                "white", edgecolor="#888888", lw=1.0)
    ax.text(pair_x_llm + pair_w_llm / 2, y + pair_h / 2,
            "Llama Decoder Layer", fontsize=9, ha="center", va="center",
            color=TEXT_DARK)

    cross_attn_centers.append((pair_x_attn, y + pair_h / 2))

# fan-in lines from Perceiver output into each gated cross-attention block (subtle)
trunk_x = prc_x + prc_w + 0.10
for cx, cy in cross_attn_centers:
    ax.annotate("",
                xy=(cx, cy), xytext=(trunk_x, cy),
                arrowprops=dict(arrowstyle="-", color="#8A8A8A",
                                lw=0.7, alpha=0.50,
                                connectionstyle="arc3,rad=0.0"))

# ---------- 5. Output: German free-form text ----------
rounded_box(out_x, out_y, out_w, out_h, "white", edgecolor=EDGE_DARK, lw=1.4)

ax.text(out_x + out_w / 2, out_y + out_h + 0.30, "Output",
        fontsize=12, ha="center", color=TEXT_DARK, fontweight="bold")
ax.text(out_x + out_w / 2, out_y + out_h - 0.40,
        "German Expert\nCommentary",
        fontsize=10.5, ha="center", va="top", color=TEXT_DARK, fontweight="bold")
ax.text(out_x + out_w / 2, out_y + out_h / 2 - 0.20,
        "“Die Wärmepumpe\nläuft überwiegend\nim stationären\nBetrieb …”",
        fontsize=9, ha="center", va="center", color="#404040", style="italic",
        linespacing=1.5)
ax.text(out_x + out_w / 2, out_y + 0.30,
        "max_new_tokens = 2000",
        fontsize=8.5, ha="center", color=MUTED, family="monospace")

# arrow from Llama into output
shape_arrow(llm_x + llm_w + 0.10, out_x - 0.05, ARROW_Y, "logits")

# ---------- footer annotations ----------
ax.text(FIG_W / 2, 0.55,
        "Special tokens:  <image>     <|endofchunk|>",
        fontsize=10.5, ha="center", color=TEXT_DARK, family="monospace")
ax.text(FIG_W / 2, 0.20,
        "Training signal: cross-entropy on the answer span only",
        fontsize=10.5, ha="center", color=MUTED, style="italic")

# ---------- legend ----------
leg_y = 7.55
leg_x = 0.4

ax.add_patch(Rectangle((leg_x, leg_y), 0.40, 0.28,
                       facecolor=BOSCH_RED, edgecolor=BOSCH_RED))
ax.text(leg_x + 0.50, leg_y + 0.14, "Trainable",
        fontsize=10.5, va="center", color=TEXT_DARK)

ax.add_patch(Rectangle((leg_x + 1.70, leg_y), 0.40, 0.28,
                       facecolor=FROZEN_GRAY, edgecolor="#7A7A7A"))
ax.text(leg_x + 2.20, leg_y + 0.14, "Frozen",
        fontsize=10.5, va="center", color=TEXT_DARK)

out_path = ("/home/wangni/notion-figures/"
            "bosch-checkit-expert-commentary-generation-for-heat-pump-tim/"
            "fig_001.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
