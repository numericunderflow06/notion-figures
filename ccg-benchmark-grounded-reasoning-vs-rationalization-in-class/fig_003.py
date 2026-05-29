"""
fig_003: Benchmark Pipeline Architecture
End-to-end dataflow for the CCG D&B benchmark.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Mermaid-style palette
COLOR_DATASET   = "#E1F5FE"  # light blue
COLOR_DATASET_E = "#0277BD"
COLOR_RUNNER    = "#FFF3E0"  # light orange
COLOR_RUNNER_E  = "#E65100"
COLOR_RAW       = "#FFFDE7"  # light yellow (the shared raw.json)
COLOR_RAW_E     = "#F57F17"
COLOR_JUDGE     = "#F3E5F5"  # light purple
COLOR_JUDGE_E   = "#6A1B9A"
COLOR_SUM       = "#E8F5E9"  # light green
COLOR_SUM_E     = "#2E7D32"
COLOR_OUT       = "#FCE4EC"  # light pink
COLOR_OUT_E     = "#AD1457"
COLOR_ARROW     = "#455A64"

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_axis_off()
fig.patch.set_facecolor("white")

# ---- Title ----
ax.text(8.0, 8.55, "Benchmark Pipeline Architecture",
        ha="center", va="center", fontsize=17, fontweight="bold")
ax.text(8.0, 8.10,
        "Surface-agnostic dataflow: both generation runners emit a common raw.json,"
        " feeding two parallel judges then a summarizer.",
        ha="center", va="center", fontsize=10.5, style="italic", color="#37474F")

def box(x, y, w, h, text, facecolor, edgecolor, fontsize=10.5, fontweight="bold",
        text_color="#102027"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.6, facecolor=facecolor, edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color)

def arrow(x0, y0, x1, y1, label=None, label_offset=(0, 0.22),
          label_color="#1A237E", label_fontsize=9, style="->"):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        linewidth=1.6, color=COLOR_ARROW,
    )
    ax.add_patch(a)
    if label:
        lx = (x0 + x1) / 2 + label_offset[0]
        ly = (y0 + y1) / 2 + label_offset[1]
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=label_fontsize, color=label_color,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.18",
                          facecolor="white", edgecolor="#B0BEC5", linewidth=0.6))

# ---- Stage 1: Datasets ----
ds_x, ds_w, ds_h = 0.3, 2.4, 0.85
ax.text(ds_x + ds_w/2, 7.05, "Datasets", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLOR_DATASET_E)
box(ds_x, 5.55, ds_w, ds_h, "ECG-QA",  COLOR_DATASET, COLOR_DATASET_E)
box(ds_x, 4.55, ds_w, ds_h, "HAR",     COLOR_DATASET, COLOR_DATASET_E)
box(ds_x, 3.55, ds_w, ds_h, "Sleep",   COLOR_DATASET, COLOR_DATASET_E)

# ---- Stage 2: Two generation runners ----
r_x, r_w, r_h = 3.7, 3.2, 1.2
ax.text(r_x + r_w/2, 7.05, "Generation Runners", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLOR_RUNNER_E)
box(r_x, 5.5, r_w, r_h,
    "Prompt-Injection\n(text class signal in prompt)",
    COLOR_RUNNER, COLOR_RUNNER_E, fontsize=10)
box(r_x, 3.55, r_w, r_h,
    "Residual-Injection\n(activation-level class signal)",
    COLOR_RUNNER, COLOR_RUNNER_E, fontsize=10)

# Bracket label: "surface-agnostic"
ax.annotate("", xy=(r_x + r_w + 0.15, 5.5),  xytext=(r_x + r_w + 0.15, 6.7),
            arrowprops=dict(arrowstyle="-", color="#90A4AE", lw=1.0))
ax.annotate("", xy=(r_x + r_w + 0.15, 3.55), xytext=(r_x + r_w + 0.15, 4.75),
            arrowprops=dict(arrowstyle="-", color="#90A4AE", lw=1.0))

# Dataset -> runners arrows (fan-in)
for cy in (5.97, 4.97, 3.97):
    arrow(ds_x + ds_w, cy, r_x, 6.10, label=None)
    arrow(ds_x + ds_w, cy, r_x, 4.15, label=None)

# ---- Stage 3: Common raw.json ----
raw_x, raw_w, raw_h = 7.45, 1.85, 1.4
box(raw_x, 4.55, raw_w, raw_h,
    "raw.json\n(shared schema)",
    COLOR_RAW, COLOR_RAW_E, fontsize=11)
# Emphasis under the shared raw
ax.text(raw_x + raw_w/2, 4.32,
        "both runners emit\nthe same record format",
        ha="center", va="top", fontsize=8.5, style="italic", color="#5D4037")

# runners -> raw.json
arrow(r_x + r_w, 6.10, raw_x, 5.55,
      label="raw.json", label_offset=(0, 0.22))
arrow(r_x + r_w, 4.15, raw_x, 4.95,
      label="raw.json", label_offset=(0, -0.30))

# ---- Stage 4: Two parallel judges ----
j_x, j_w, j_h = 10.0, 3.0, 1.1
ax.text(j_x + j_w/2, 7.05, "Parallel Judges", ha="center", va="center",
        fontsize=11, fontweight="bold", color=COLOR_JUDGE_E)
box(j_x, 5.65, j_w, j_h,
    "5-regime judge\npoc_regime_judge.py",
    COLOR_JUDGE, COLOR_JUDGE_E, fontsize=10)
box(j_x, 3.65, j_w, j_h,
    "Factorized 8-cell judge\npoc_rationale_effect_judge.py",
    COLOR_JUDGE, COLOR_JUDGE_E, fontsize=10)

# raw -> judges
arrow(raw_x + raw_w, 5.45, j_x, 6.20,
      label="raw.json", label_offset=(0, 0.25))
arrow(raw_x + raw_w, 4.95, j_x, 4.20,
      label="raw.json", label_offset=(0, -0.30))

# ---- Stage 5: Summarizer ----
s_x, s_w, s_h = 13.55, 2.15, 1.1
box(s_x, 4.65, s_w, s_h,
    "Summarizer",
    COLOR_SUM, COLOR_SUM_E, fontsize=11)

# judges -> summarizer
arrow(j_x + j_w, 6.20, s_x, 5.45,
      label="regime_labels.json", label_offset=(0, 0.25))
arrow(j_x + j_w, 4.20, s_x, 4.95,
      label="rationale_effect.json", label_offset=(0, -0.30))

# ---- Stage 6: Output JSON artifacts ----
o_x, o_w, o_h = 13.45, 2.25, 0.78
box(o_x, 3.10, o_w, o_h, "cell_counts.json", COLOR_OUT, COLOR_OUT_E, fontsize=10)
box(o_x, 2.15, o_w, o_h, "cross_tab.json",   COLOR_OUT, COLOR_OUT_E, fontsize=10)

# summarizer -> outputs
arrow(s_x + s_w/2, 4.65, o_x + o_w/2, 3.88, label=None)
arrow(s_x + s_w/2, 4.65, o_x + o_w/2, 2.93, label=None)

# ---- Canonical path schema banner ----
schema_y = 1.30
ax.add_patch(FancyBboxPatch(
    (0.5, schema_y - 0.45), 15.0, 0.85,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=1.3, facecolor="#ECEFF1", edgecolor="#455A64"))
ax.text(8.0, schema_y + 0.18,
        "Canonical output path schema",
        ha="center", va="center", fontsize=10.5, fontweight="bold", color="#263238")
ax.text(8.0, schema_y - 0.18,
        "runs/<model>/<dataset>/<method>/<run-tag>/{raw.json, regime_labels.json, "
        "rationale_effect.json, cell_counts.json, cross_tab.json}",
        ha="center", va="center", fontsize=10.5, family="monospace", color="#1A237E")

# ---- Legend ----
legend_items = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_DATASET,
           markeredgecolor=COLOR_DATASET_E, markersize=13, label="Dataset"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_RUNNER,
           markeredgecolor=COLOR_RUNNER_E, markersize=13, label="Generation runner"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_RAW,
           markeredgecolor=COLOR_RAW_E, markersize=13, label="Shared raw record"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_JUDGE,
           markeredgecolor=COLOR_JUDGE_E, markersize=13, label="Judge"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_SUM,
           markeredgecolor=COLOR_SUM_E, markersize=13, label="Summarizer"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=COLOR_OUT,
           markeredgecolor=COLOR_OUT_E, markersize=13, label="Output artifact"),
]
ax.legend(handles=legend_items, loc="lower center",
          bbox_to_anchor=(0.5, -0.005), ncol=6, frameon=False, fontsize=9.5)

plt.tight_layout()
out_path = "/home/wangni/notion-figures/ccg-benchmark-grounded-reasoning-vs-rationalization-in-class/fig_003.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
