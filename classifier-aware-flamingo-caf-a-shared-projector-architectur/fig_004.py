"""
fig_004: Mode V vs Mode P Class Embeddings
Side-by-side comparison of verdict-slot (Mode V) and prefix-full (Mode P)
class embeddings in CAF.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_PATH = "/home/wangni/notion-figures/classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_004.png"

# Color palette
COL_FIXED = "#E5E5E5"        # fixed template token (gray)
COL_FIXED_EDGE = "#666666"
COL_SLOT = "#F4B860"          # learned class-slot in Mode V (orange)
COL_SLOT_EDGE = "#C97D1A"
COL_CLASS = ["#F4B860", "#7FBFE0", "#9BD49B", "#C0C0C0"]  # K classes in Mode P
COL_CLASS_EDGE = ["#C97D1A", "#3E7FA8", "#4F9F4F", "#777777"]
TEXT_COLOR = "#1A1A1A"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=200)
fig.patch.set_facecolor("white")

# Shared y range: 0 (top, title) -> -22 (bottom, params)
Y_TOP = 1.0
Y_BOT = -10.0
for ax in (axL, axR):
    ax.set_xlim(-1, 11)
    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

# ---------------------------------------------------------------
# Helper to draw a rounded token
# ---------------------------------------------------------------
def draw_token(ax, x, y, w, h, text, face, edge, fontsize=10, weight="normal",
               text_color=TEXT_COLOR):
    box = FancyBboxPatch(
        (x, y - h / 2.0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.0, facecolor=face, edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight)

# ===============================================================
# LEFT panel: Mode V (verdict-slot)
# ===============================================================
axL.text(5, 0.6, "Mode V: verdict-slot",
         ha="center", va="center", fontsize=15, weight="bold", color=TEXT_COLOR)
axL.text(5, -0.15,
         r"Fixed template with class-name slot $\leftarrow E_{\mathrm{cls}}(v)$",
         ha="center", va="center", fontsize=11, color="#555555",
         style="italic")

# ----- Template row 1: "Based on signal analysis , finding is [A] ."
row1_tokens = ["Based", "on", "signal", "analysis", "finding", "is"]
row1_y = -1.7
tok_h = 0.7
x = -0.4
gap = 0.15
widths_row1 = [1.05, 0.55, 1.05, 1.15, 1.05, 0.55]
for tok, w in zip(row1_tokens, widths_row1):
    draw_token(axL, x, row1_y, w, tok_h, tok, COL_FIXED, COL_FIXED_EDGE, fontsize=10)
    x += w + gap

# Slot [A] (Mode V learned class slot)
slot_w = 0.95
draw_token(axL, x, row1_y, slot_w, tok_h, "[A]",
           COL_SLOT, COL_SLOT_EDGE, fontsize=10, weight="bold")
x_slot1 = x
x += slot_w + gap
draw_token(axL, x, row1_y, 0.4, tok_h, ".", COL_FIXED, COL_FIXED_EDGE, fontsize=10)

# ----- Template row 2: "The motion patterns match [A] ."
row2_tokens = ["The", "motion", "patterns", "match"]
row2_y = -2.95
widths_row2 = [0.7, 1.05, 1.2, 1.0]
x = -0.4
for tok, w in zip(row2_tokens, widths_row2):
    draw_token(axL, x, row2_y, w, tok_h, tok, COL_FIXED, COL_FIXED_EDGE, fontsize=10)
    x += w + gap
draw_token(axL, x, row2_y, slot_w, tok_h, "[A]",
           COL_SLOT, COL_SLOT_EDGE, fontsize=10, weight="bold")
x_slot2 = x
x += slot_w + gap
draw_token(axL, x, row2_y, 0.4, tok_h, ".", COL_FIXED, COL_FIXED_EDGE, fontsize=10)

# ----- Annotation: substitution arrow from a "verdict v" pill
verdict_x, verdict_y = 5.0, -4.9
draw_token(axL, verdict_x - 0.85, verdict_y, 1.7, 0.65,
           r"$E_{\mathrm{cls}}(v)$",
           "#FFE6C7", COL_SLOT_EDGE, fontsize=11, weight="bold")

# arrow to slot row1
axL.annotate("", xy=(x_slot1 + slot_w / 2.0, row1_y - tok_h / 2.0 - 0.05),
             xytext=(verdict_x - 0.4, verdict_y + 0.35),
             arrowprops=dict(arrowstyle="->", color=COL_SLOT_EDGE,
                             lw=1.4, connectionstyle="arc3,rad=-0.25"))
# arrow to slot row2
axL.annotate("", xy=(x_slot2 + slot_w / 2.0, row2_y - tok_h / 2.0 - 0.05),
             xytext=(verdict_x + 0.4, verdict_y + 0.35),
             arrowprops=dict(arrowstyle="->", color=COL_SLOT_EDGE,
                             lw=1.4, connectionstyle="arc3,rad=0.25"))

axL.text(verdict_x, verdict_y - 0.7,
         "single learned vector per class, substituted into [A] slot",
         ha="center", va="center", fontsize=9.5, color="#444444", style="italic")

# ----- Legend
leg_y = -6.4
draw_token(axL, 0.4, leg_y, 0.45, 0.45, "", COL_FIXED, COL_FIXED_EDGE)
axL.text(1.0, leg_y, "frozen LLM token embedding", ha="left", va="center",
         fontsize=10, color=TEXT_COLOR)
draw_token(axL, 5.0, leg_y, 0.45, 0.45, "", COL_SLOT, COL_SLOT_EDGE)
axL.text(5.6, leg_y, "learned class slot", ha="left", va="center",
         fontsize=10, color=TEXT_COLOR)

# ----- Parameter count footer
axL.text(5.0, -8.4,
         r"$|\mathcal{V}_{\mathrm{cls}}| = K$  vectors of dim $d_{\mathrm{lat}}$",
         ha="center", va="center", fontsize=11, color=TEXT_COLOR)
axL.text(5.0, -9.2,
         r"trainable params: $K \cdot d_{\mathrm{lat}}$  $\approx$  98 K",
         ha="center", va="center", fontsize=12, weight="bold",
         color="#7A3E00",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF4E5",
                   edgecolor=COL_SLOT_EDGE, linewidth=1.2))

# ===============================================================
# RIGHT panel: Mode P (prefix-full)
# ===============================================================
axR.text(5, 0.6, "Mode P: prefix-full",
         ha="center", va="center", fontsize=15, weight="bold", color=TEXT_COLOR)
axR.text(5, -0.15,
         r"Fully-learned per-class soft prompt of length $L_{\mathrm{tmpl}}$",
         ha="center", va="center", fontsize=11, color="#555555",
         style="italic")

# K x L_tmpl grid of learned vectors
K = 4
L = 10
cell_w = 0.62
cell_h = 0.50
gap_x = 0.06
gap_y = 0.10
grid_total_w = L * cell_w + (L - 1) * gap_x
grid_x0 = (10 - grid_total_w) / 2.0 + 0.2  # center horizontally a bit
grid_y0 = -1.6

class_labels = ["class 1", "class 2", "class 3", r"class $K$"]
for r in range(K):
    fc = COL_CLASS[r]
    ec = COL_CLASS_EDGE[r]
    y = grid_y0 - r * (cell_h + gap_y)
    # row label
    axR.text(grid_x0 - 0.35, y, class_labels[r], ha="right", va="center",
             fontsize=10, color=TEXT_COLOR)
    for i in range(L):
        x = grid_x0 + i * (cell_w + gap_x)
        rect = Rectangle((x, y - cell_h / 2), cell_w, cell_h,
                         facecolor=fc, edgecolor=ec, linewidth=0.8)
        axR.add_patch(rect)

# Brace for L_tmpl (top)
brace_y = grid_y0 + cell_h / 2 + 0.25
axR.annotate("", xy=(grid_x0, brace_y),
             xytext=(grid_x0 + grid_total_w, brace_y),
             arrowprops=dict(arrowstyle="-", color="#333333", lw=1.1))
# tick marks
for xt in (grid_x0, grid_x0 + grid_total_w):
    axR.plot([xt, xt], [brace_y, brace_y - 0.12], color="#333333", lw=1.1)
axR.text(grid_x0 + grid_total_w / 2, brace_y + 0.35,
         r"$L_{\mathrm{tmpl}}$ learned positions",
         ha="center", va="center", fontsize=10.5, color=TEXT_COLOR)

# Brace for K classes (right)
last_y = grid_y0 - (K - 1) * (cell_h + gap_y)
brace_x = grid_x0 + grid_total_w + 0.25
axR.annotate("", xy=(brace_x, grid_y0 + cell_h / 2),
             xytext=(brace_x, last_y - cell_h / 2),
             arrowprops=dict(arrowstyle="-", color="#333333", lw=1.1))
for yt in (grid_y0 + cell_h / 2, last_y - cell_h / 2):
    axR.plot([brace_x, brace_x + 0.12], [yt, yt], color="#333333", lw=1.1)
axR.text(brace_x + 0.55,
         (grid_y0 + cell_h / 2 + last_y - cell_h / 2) / 2,
         r"$K$ classes", ha="left", va="center", fontsize=10.5,
         color=TEXT_COLOR, rotation=-90)

# Subtitle below grid
axR.text(5.0, last_y - cell_h / 2 - 0.55,
         "entire prefix learned per class (no fixed template)",
         ha="center", va="center", fontsize=10, color="#444444",
         style="italic")

# Legend (color = class)
leg_y = -6.4
leg_x = 0.4
for r in range(K):
    rect = Rectangle((leg_x, leg_y - 0.22), 0.45, 0.45,
                     facecolor=COL_CLASS[r], edgecolor=COL_CLASS_EDGE[r],
                     linewidth=0.8)
    axR.add_patch(rect)
    axR.text(leg_x + 0.6, leg_y, class_labels[r], ha="left", va="center",
             fontsize=9.5, color=TEXT_COLOR)
    leg_x += 2.3

# Parameter count footer
axR.text(5.0, -8.4,
         r"$K \cdot L_{\mathrm{tmpl}}$  vectors of dim $d_{\mathrm{lat}}$",
         ha="center", va="center", fontsize=11, color=TEXT_COLOR)
axR.text(5.0, -9.2,
         r"trainable params: $K \cdot L_{\mathrm{tmpl}} \cdot d_{\mathrm{lat}}$"
         r"  $\approx$  590 K",
         ha="center", va="center", fontsize=12, weight="bold",
         color="#1F4E79",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F1F8",
                   edgecolor="#3E7FA8", linewidth=1.2))

# ---------------------------------------------------------------
# Super title and divider
# ---------------------------------------------------------------
fig.suptitle("Class-Embedding Modes in CAF: Verdict-Slot vs Prefix-Full",
             fontsize=16, weight="bold", color=TEXT_COLOR, y=0.985)

# Light vertical divider
fig.lines.append(plt.Line2D([0.5, 0.5], [0.06, 0.92],
                            transform=fig.transFigure,
                            color="#CCCCCC", lw=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved figure to: {OUT_PATH}")
