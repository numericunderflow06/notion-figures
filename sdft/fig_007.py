"""
fig_007: Memory Layout — Student and Teacher Model Components
Side-by-side stacked bar chart comparing memory footprint breakdown.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- Data from Fact §4.9 ----------
# Frozen LLM: ~6 GB float32 each
# Trainable encoder: ~2M params → ~8 MB fp32
# Projector: ~0.3M params → ~1.2 MB fp32
# Flamingo cross-attention: ~50M params → ~200 MB fp32
# Adam optimizer stores grad + m + v → ~3x extra for trainable params

# --- Student model ---
student_frozen_llm = 6.0        # GB, frozen — no gradients
student_encoder_w = 0.008       # ~2M params fp32
student_projector_w = 0.0012    # ~0.3M params fp32
student_flamingo_w = 0.2        # ~50M params fp32
student_grad_optim = (0.008 + 0.0012 + 0.2) * 3  # grad + 2 optimizer states
student_activations = 1.5       # forward+backward activations estimate

# --- Teacher model ---
teacher_frozen_llm = 6.0
teacher_encoder_w = 0.008
teacher_projector_w = 0.0012
teacher_flamingo_w = 0.2
teacher_grad_optim = 0.0        # NO gradient storage
teacher_activations = 0.6       # forward-only activations

# ---------- Build stacked bars ----------
# Stack order (bottom to top):
#   0. Frozen LLM weights
#   1. Encoder weights
#   2. Projector weights
#   3. Flamingo weights
#   4. Gradients & optimizer states (student only)
#   5. Activations / KV-cache

student_stack = [
    student_frozen_llm,
    student_encoder_w,
    student_projector_w,
    student_flamingo_w,
    student_grad_optim,
    student_activations,
]

teacher_stack = [
    teacher_frozen_llm,
    teacher_encoder_w,
    teacher_projector_w,
    teacher_flamingo_w,
    teacher_grad_optim,
    teacher_activations,
]

labels = [
    "Frozen LLM (~6 GB)",
    "Encoder (~2M params)",
    "Projector (~0.3M params)",
    "Flamingo layers (~50M params)",
    "Gradients + optimizer states",
    "Activations / KV-cache",
]

colors = [
    "#3D6FA0",  # deep blue    — frozen LLM
    "#5BAE5B",  # green        — encoder
    "#F2A93B",  # amber        — projector
    "#D45B5B",  # coral red    — flamingo
    "#8E6FBF",  # purple       — gradients
    "#5BBFB0",  # teal         — activations
]

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_width = 0.55
x_positions = [0.0, 1.2]
bar_labels_x = ["Student Model", "Teacher Model\n(EMA copy)"]

for idx, (stack, xpos) in enumerate(zip([student_stack, teacher_stack], x_positions)):
    bottom = 0.0
    for j, (val, color) in enumerate(zip(stack, colors)):
        if val == 0:
            bottom += val
            continue
        ax.bar(
            xpos, val, width=bar_width, bottom=bottom, color=color,
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        # Annotate inside the bar if segment is tall enough
        mid = bottom + val / 2
        if val >= 0.45:
            display = f"{val:.1f} GB"
            ax.text(xpos, mid, display, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=4)
        bottom += val

    # Total annotation on top
    total = sum(stack)
    ax.text(xpos, total + 0.18, f"Total: ~{total:.1f} GB",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#222222", zorder=4)

# ---------- Callout annotations ----------

student_x = x_positions[0]
teacher_x = x_positions[1]
mid_x = (student_x + teacher_x) / 2  # midpoint between bars

# Flamingo weight bands — single label in the center with arrows to both bars
flamingo_bot_s = sum(student_stack[:3])
flamingo_mid_s = flamingo_bot_s + student_stack[3] / 2
flamingo_bot_t = sum(teacher_stack[:3])
flamingo_mid_t = flamingo_bot_t + teacher_stack[3] / 2

# Place "Flamingo ~200 MB" text centrally, below the flamingo band
ax.annotate(
    "Flamingo ~200 MB (each)",
    xy=(mid_x, flamingo_mid_s),
    xytext=(mid_x, flamingo_mid_s - 0.45),
    fontsize=9, ha="center", va="top", color="#D45B5B", fontweight="bold",
    zorder=4,
)
# Arrows from text to student flamingo bar
ax.annotate(
    "",
    xy=(student_x + bar_width / 2, flamingo_mid_s),
    xytext=(mid_x - 0.08, flamingo_mid_s - 0.2),
    arrowprops=dict(arrowstyle="-|>", color="#D45B5B", lw=1.0),
    zorder=4,
)
# Arrow from text to teacher flamingo bar
ax.annotate(
    "",
    xy=(teacher_x - bar_width / 2, flamingo_mid_t),
    xytext=(mid_x + 0.08, flamingo_mid_t - 0.2),
    arrowprops=dict(arrowstyle="-|>", color="#D45B5B", lw=1.0),
    zorder=4,
)

# Gradient+optim bar on student — label to the left
grad_top_s = sum(student_stack[:5])
grad_bot_s = sum(student_stack[:4])
grad_mid_s = (grad_top_s + grad_bot_s) / 2
ax.annotate(
    f"~{student_grad_optim*1000:.0f} MB\n(grad + optimizer)",
    xy=(student_x - bar_width / 2, grad_mid_s),
    xytext=(student_x - bar_width / 2 - 0.15, grad_mid_s),
    fontsize=9, ha="right", va="center", color="#6B50A0", fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color="#8E6FBF", lw=1.0),
    zorder=4,
)

# "No gradient storage" annotation on teacher — to the right
teacher_grad_y = sum(teacher_stack[:4])
ax.annotate(
    "No gradient\nstorage",
    xy=(teacher_x + bar_width / 2, teacher_grad_y + 0.02),
    xytext=(teacher_x + bar_width / 2 + 0.15, teacher_grad_y + 0.55),
    fontsize=10, fontstyle="italic", color="#8E6FBF", fontweight="bold",
    ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color="#8E6FBF", lw=1.5),
    zorder=4,
)

# ---------- Axes formatting ----------
ax.set_xticks(x_positions)
ax.set_xticklabels(bar_labels_x, fontsize=13, fontweight="bold")
ax.set_ylabel("Memory (GB)", fontsize=13, fontweight="bold")
ax.set_xlim(-0.9, 2.3)
ax.set_ylim(0, max(sum(student_stack), sum(teacher_stack)) + 1.0)
ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)

# ---------- Legend (below chart) ----------
legend_patches = [mpatches.Patch(facecolor=c, edgecolor="white", label=l)
                  for c, l in zip(colors, labels)]
ax.legend(
    handles=legend_patches,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    fontsize=9.5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    title="Components",
    title_fontsize=10.5,
    handlelength=1.5,
    columnspacing=1.2,
)

# ---------- Title ----------
ax.set_title(
    "Memory Layout: Student and Teacher Model Components",
    fontsize=15, fontweight="bold", pad=16,
)

# ---------- Subtitle / note ----------
fig.text(
    0.5, 0.005,
    "Frozen LLM ~6 GB (float32) each  |  Trainable: encoder ~2M, projector ~0.3M, Flamingo ~50M params  |  Teacher maintained via EMA",
    ha="center", fontsize=9, color="#666666", style="italic",
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("/home/wangni/notion-figures/sdft/fig_007.png", dpi=200,
            facecolor="white", bbox_inches="tight")
plt.close()
print("Saved: /home/wangni/notion-figures/sdft/fig_007.png")
