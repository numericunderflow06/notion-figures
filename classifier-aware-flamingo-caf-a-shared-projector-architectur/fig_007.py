"""fig_007: Trainable Parameter Budget for CAF (Mode V).

Data source: paper/paper.tex, tab:budget (lines 282-298).
Default config: Llama-3.2-3B backbone, d_lat=d_llm=3072, N_q=64,
K=8, L_cls=4, L_tmpl=24, M=16, FiLM on.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = Path(
    "/home/wangni/notion-figures/"
    "classifier-aware-flamingo-caf-a-shared-projector-architectur/fig_007.png"
)

# (component label, parameter count in millions, % of trainable total)
# Pulled verbatim from tab:budget; Mode V row used for E_cls (total ~99M).
components = [
    ("Gated cross-attention", 50.0, 50.5),
    (r"FiLM ($\gamma,\beta$)", 19.0, 19.2),
    ("LoRA adapters ($r{=}16$)", 12.0, 12.1),
    ("Shared projector $P$", 9.4, 9.5),
    (r"Perceiver $\Pi$", 8.4, 8.5),
    (r"$E_\mathrm{cls}$ (Mode P)", 0.590, 0.6),
    (r"$E_\mathrm{cls}$ (Mode V)", 0.098, 0.1),
    ("Instruct tokens $I$", 0.049, 0.05),
]

# Sorted descending by parameter count (already sorted above).
labels = [c[0] for c in components]
params = np.array([c[1] for c in components])
shares = [c[2] for c in components]

# Frozen LLM backbone shown as a faint reference for scale (~3B parameters).
llm_params = 3000.0  # millions

ACCENT = "#2563eb"  # consistent blue accent for trainable components
FROZEN = "#cfd5dd"  # light gray for the frozen LLM reference bar

fig, ax = plt.subplots(figsize=(10, 6.6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.subplots_adjust(top=0.86, left=0.22, right=0.97, bottom=0.10)

# Build y positions: frozen LLM at the top (separated visually),
# trainable components below, top→bottom largest→smallest.
all_labels = ["LLM backbone (frozen)"] + labels
y_pos = np.arange(len(all_labels))[::-1]  # so first label is at top
y_frozen = y_pos[0]
y_trainable = y_pos[1:]

# Frozen reference bar.
ax.barh(
    y_frozen,
    llm_params,
    color=FROZEN,
    edgecolor="#a8b0bb",
    linewidth=0.8,
    height=0.7,
    zorder=2,
)
ax.text(
    llm_params,
    y_frozen,
    r"  $\sim$3000 M  (frozen, $\approx$3 B)",
    va="center",
    ha="left",
    fontsize=10,
    color="#4b5563",
)

# Trainable bars.
bars = ax.barh(
    y_trainable,
    params,
    color=ACCENT,
    edgecolor="#1e3a8a",
    linewidth=0.6,
    height=0.7,
    zorder=3,
)

# Annotate each trainable bar with parameter count and share-of-trainable.
for y, p, s in zip(y_trainable, params, shares):
    if p >= 1.0:
        param_str = f"{p:.1f} M"
    else:
        param_str = f"{p*1000:.0f} K"
    ax.text(
        p,
        y,
        f"  {param_str}  ({s:.2g}%)",
        va="center",
        ha="left",
        fontsize=10,
        color="#111827",
    )

# Log x-axis so small components remain readable next to the 3B reference.
ax.set_xscale("log")
ax.set_xlim(0.02, 2.5e5)

ax.set_yticks(y_pos)
ax.set_yticklabels(all_labels, fontsize=11)
ax.set_xlabel("Trainable parameters (millions, log scale)", fontsize=11)
ax.tick_params(axis="x", labelsize=10)

# Separator between frozen reference and trainable rows.
sep_y = (y_frozen + y_trainable[0]) / 2
ax.axhline(sep_y, color="#9ca3af", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)

# Light vertical grid for readability on log scale.
ax.grid(axis="x", which="major", color="#e5e7eb", linewidth=0.6, zorder=0)
ax.grid(axis="x", which="minor", color="#f3f4f6", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["left"].set_color("#9ca3af")
ax.spines["bottom"].set_color("#9ca3af")

# Title + subtitle (subtitle uses figtext to mimic two-line heading).
fig.suptitle(
    "CAF Trainable Parameter Budget",
    fontsize=14,
    fontweight="bold",
    x=0.22,
    y=0.965,
    ha="left",
)
fig.text(
    0.22,
    0.925,
    r"Default config (Llama-3.2-3B, $d_\mathrm{lat}{=}d_\mathrm{llm}{=}3072$, "
    r"$N_q{=}64$, FiLM on); total $\sim$99 M $\approx$ 3% of LLM (Mode V)",
    fontsize=10,
    color="#4b5563",
)

# Legend.
from matplotlib.patches import Patch

legend_handles = [
    Patch(facecolor=ACCENT, edgecolor="#1e3a8a", label="Trainable (CAF)"),
    Patch(facecolor=FROZEN, edgecolor="#a8b0bb", label="Frozen reference"),
]
ax.legend(
    handles=legend_handles,
    loc="lower right",
    frameon=False,
    fontsize=10,
)

plt.savefig(OUT_PATH, dpi=200, facecolor="white")
plt.close(fig)

print(f"Saved: {OUT_PATH}")
