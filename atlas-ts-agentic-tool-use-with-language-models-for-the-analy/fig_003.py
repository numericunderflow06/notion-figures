"""
fig_003: Two-Agent Architecture with Tensors on the Wire
ATLAS-TS pipeline block diagram naming tensor shapes and message types
on every edge.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_aspect('equal')

# --- Color palette (component families) ---
C_CHRONOS = '#4682B4'   # steel blue (frozen encoder)
C_AGENT   = '#9370DB'   # purple (trainable agents)
C_TOOL    = '#3CB371'   # green (tool registry)
C_BTF     = '#FF8C00'   # orange (BTF + LLM, frozen)
C_INOUT   = '#555555'   # dark gray (I/O)
C_EDGE    = '#222222'
C_DASH    = '#888888'

LBL_FONT  = 10
TXT_FONT  = 11
TTL_FONT  = 11
SHAPE_FONT = 9.5

# --- Helper: draw a rounded labeled box ---
def box(x, y, w, h, title, subtitle=None, color='#cccccc', edge='black',
        lw=1.6, hatch=None, italic_sub=False):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        linewidth=lw, edgecolor=edge, facecolor=color, alpha=0.92,
        hatch=hatch
    )
    ax.add_patch(rect)
    if subtitle is None:
        ax.text(x + w/2, y + h/2, title,
                ha='center', va='center', fontsize=TTL_FONT,
                fontweight='bold', color='white' if _is_dark(color) else 'black')
    else:
        ax.text(x + w/2, y + h*0.66, title,
                ha='center', va='center', fontsize=TTL_FONT,
                fontweight='bold', color='white' if _is_dark(color) else 'black')
        ax.text(x + w/2, y + h*0.30, subtitle,
                ha='center', va='center', fontsize=SHAPE_FONT,
                style='italic' if italic_sub else 'normal',
                color='white' if _is_dark(color) else 'black')
    return (x, y, w, h)

def _is_dark(hexc):
    hexc = hexc.lstrip('#')
    r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
    return (0.299*r + 0.587*g + 0.114*b) < 130

# --- Helper: arrow with label (solid = forward, dashed = cached/reuse) ---
def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.28),
          dashed=False, color=C_EDGE, lw=1.8, curve=0.0, label_bg=True,
          mono=True, fontsize=LBL_FONT):
    style = '->'
    linestyle = '--' if dashed else '-'
    connectionstyle = f"arc3,rad={curve}"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
        zorder=3
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        kwargs = dict(ha='center', va='center', fontsize=fontsize,
                      color=color,
                      family='monospace' if mono else 'sans-serif')
        if label_bg:
            kwargs['bbox'] = dict(boxstyle='round,pad=0.20',
                                  facecolor='white', edgecolor='none',
                                  alpha=0.95)
        ax.text(mx, my, label, **kwargs)

# ============================================================
# COMPONENTS (left to right)
# ============================================================

# 0. Raw input
box(0.2, 4.2, 1.6, 1.8,
    "Raw input",
    r"$x \in \mathbb{R}^{C \times T}$",
    color='#EAEAEA', edge=C_INOUT, italic_sub=False)

# 1. Chronos-2 frozen encoder
box(2.6, 3.8, 2.4, 2.6,
    "Chronos-2",
    "frozen encoder",
    color=C_CHRONOS, edge='#243B53', italic_sub=True)
# snowflake-ish freeze badge
ax.text(2.78, 6.18, "❄",  # snowflake symbol
        ha='left', va='center', fontsize=14, color='white')

# 2. ATLAS-T (decoder)
box(6.0, 6.4, 2.6, 1.9,
    "ATLAS-T",
    "tool-user decoder",
    color=C_AGENT, edge='#3B1F5E', italic_sub=True)

# 3. Tool registry
box(9.4, 6.4, 2.6, 1.9,
    "Tool registry",
    r"$\{c_1, \ldots, c_K, \mathrm{NO\_TOOL}\}$",
    color=C_TOOL, edge='#1F5E3B', italic_sub=False)

# 4. ATLAS-A (with 4-token memory)
box(6.0, 0.8, 2.6, 2.6,
    "ATLAS-A",
    "analytics decoder",
    color=C_AGENT, edge='#3B1F5E', italic_sub=True)
# 4-token memory pill underneath the title
ax.text(7.30, 1.55, r"memory  $m \in \mathbb{R}^{4 \times d}$",
        ha='center', va='center', fontsize=SHAPE_FONT, color='white',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.18', facecolor='#3B1F5E',
                  edgecolor='white', linewidth=0.8))

# 5. BTF render registry
box(9.4, 0.8, 2.6, 2.6,
    "BTF renders",
    "registry (55 entries)",
    color=C_BTF, edge='#7A3E00', italic_sub=True)
ax.text(10.70, 1.30,
        r"snippet templates",
        ha='center', va='center', fontsize=SHAPE_FONT, color='black',
        bbox=dict(boxstyle='round,pad=0.16', facecolor='white',
                  edgecolor='#7A3E00', linewidth=0.8))

# 6. Frozen LLM Φ_LLM
box(13.4, 3.8, 2.8, 2.6,
    r"$\Phi_{\mathrm{LLM}}$",
    "frozen language model",
    color=C_BTF, edge='#7A3E00', italic_sub=True)
ax.text(13.6, 6.18, "❄",
        ha='left', va='center', fontsize=14, color='white')

# 7. Answer
box(16.8, 4.2, 2.7, 1.8,
    "Answer",
    r"$\hat{y}$",
    color='#EAEAEA', edge=C_INOUT, italic_sub=False)

# ============================================================
# EDGES (forward = solid; cached / reuse = dashed)
# ============================================================

# x -> Chronos-2
arrow(1.8, 5.1, 2.6, 5.1,
      label=r"$x \in \mathbb{R}^{C \times T}$",
      label_offset=(0, 0.35))

# Chronos-2 -> ATLAS-T  (e in R^768)
arrow(5.0, 5.6, 6.0, 7.0,
      label=r"$e \in \mathbb{R}^{768}$",
      label_offset=(-0.15, 0.30), curve=-0.10)

# Chronos-2 -> ATLAS-A  (shared embedding, cached -> dashed)
arrow(5.0, 4.6, 6.0, 2.6,
      label=r"$e$  (shared, cached)",
      label_offset=(-0.10, -0.30), curve=0.10,
      dashed=True, color=C_DASH)

# ATLAS-T -> Tool registry  (action c_k)
arrow(8.6, 7.35, 9.4, 7.35,
      label=r"action $c_k$",
      label_offset=(0, 0.32), mono=False)

# Tool registry -> ATLAS-A  (p in Delta^K)
arrow(10.7, 6.4, 7.30, 3.4,
      label=r"$p \in \Delta^{K}$",
      label_offset=(0.55, 0.05), curve=-0.18)

# ATLAS-A self-loop for 4-token memory  (dashed)
arrow(6.05, 2.55, 6.05, 1.55,
      label=None,
      curve=-0.55, dashed=True, color=C_DASH, lw=1.4)
ax.text(5.45, 2.05, r"$m$",
        ha='right', va='center', fontsize=LBL_FONT,
        family='monospace', color=C_DASH)

# Cached p (dashed loop on tool registry -> ATLAS-A note)
arrow(10.7, 6.4, 10.7, 3.4,
      label=r"$p$ cached for reuse",
      label_offset=(1.20, 0.0), dashed=True, color=C_DASH, lw=1.4)

# ATLAS-A -> BTF renders  (state s^(2))
arrow(8.6, 2.10, 9.4, 2.10,
      label=None)
ax.text(9.0, 0.42,
        r"state $s^{(2)} = (e,\ p,\ v,\ \rho,\ \eta,\ Q,\ \psi(q))$",
        ha='center', va='center', fontsize=LBL_FONT,
        family='monospace', color=C_EDGE,
        bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                  edgecolor='#bbbbbb', linewidth=0.8))

# BTF renders -> LLM  (snippet text, <=55)
arrow(12.0, 2.4, 13.6, 4.2,
      label=r"snippet text  ($\leq$ 55 renders)",
      label_offset=(0.55, 0.05), curve=0.15, mono=False)

# LLM -> Answer
arrow(16.2, 5.1, 16.8, 5.1,
      label=r"$\hat{y}$",
      label_offset=(0, 0.32))

# LLM cache (dashed self-loop, keyed by (s, q, y*))
arrow(14.6, 6.4, 15.3, 6.4,
      label=None, dashed=True, color=C_DASH, lw=1.2,
      curve=-0.9)
ax.text(14.95, 7.20, r"LLM cache by $(s, q, y^{*})$",
        ha='center', va='bottom', fontsize=SHAPE_FONT,
        color=C_DASH, family='monospace',
        bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                  edgecolor='none', alpha=0.95))

# ============================================================
# Legend
# ============================================================
legend_elems = [
    mpatches.Patch(facecolor=C_CHRONOS, edgecolor='black', label='Frozen encoder (Chronos-2)'),
    mpatches.Patch(facecolor=C_AGENT,   edgecolor='black', label='Trainable agent (decoder)'),
    mpatches.Patch(facecolor=C_TOOL,    edgecolor='black', label='Tool registry'),
    mpatches.Patch(facecolor=C_BTF,     edgecolor='black', label='BTF renders / frozen LLM'),
    Line2D([0], [0], color=C_EDGE,  lw=1.8, linestyle='-',  label='Forward flow'),
    Line2D([0], [0], color=C_DASH,  lw=1.6, linestyle='--', label='Cached / reused value'),
]
leg = ax.legend(handles=legend_elems, loc='lower center',
                bbox_to_anchor=(0.5, -0.02), ncol=6,
                frameon=True, fontsize=9.5, handlelength=2.0,
                columnspacing=1.2, borderpad=0.6)
leg.get_frame().set_edgecolor('#999999')

# ============================================================
# Title
# ============================================================
ax.text(10.0, 9.55,
        "ATLAS-TS: Two-Agent Architecture with Tensors on the Wire",
        ha='center', va='center', fontsize=15, fontweight='bold')
ax.text(10.0, 9.05,
        "Solid arrows: forward flow.  Dashed arrows: cached or reused values.  "
        "Tensor shapes use monospace.",
        ha='center', va='center', fontsize=10, style='italic', color='#444444')

# ============================================================
# Save
# ============================================================
out_path = "/home/wangni/notion-figures/atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_003.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Saved figure to: {out_path}")
