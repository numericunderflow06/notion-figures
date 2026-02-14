import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

# ── Configuration ──────────────────────────────────────────────────────────
BLUE = "#2563EB"
BLUE_LIGHT = "#93C5FD"
ORANGE = "#EA580C"
ORANGE_LIGHT = "#FDBA74"
GRAY = "#6B7280"
GRAY_LIGHT = "#E5E7EB"
BG = "#FFFFFF"
TEXT_DARK = "#1F2937"
SHADE_COLOR = "#F97316"

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5),
                                         gridspec_kw={"wspace": 0.35})
fig.patch.set_facecolor(BG)

# ══════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Classical Surprisal
# ══════════════════════════════════════════════════════════════════════════
ax_left.set_facecolor(BG)
ax_left.set_title("Classical Surprisal", fontsize=15, fontweight="bold",
                   color=BLUE, pad=18)

# Vocabulary words and their probabilities (conceptual)
words = ["the", "a", "cat", "dog", "sat", "ran"]
probs = [0.30, 0.18, 0.12, 0.10, 0.08, 0.05]
colors = [GRAY_LIGHT] * len(words)

# Highlight the observed word
observed_idx = 2  # "cat"
colors[observed_idx] = BLUE

bars = ax_left.barh(range(len(words)), probs, color=colors, edgecolor="white",
                    height=0.6, zorder=2)

# Add probability labels
for i, (w, p) in enumerate(zip(words, probs)):
    ax_left.text(p + 0.008, i, f"{p:.2f}", va="center", ha="left",
                 fontsize=10, color=TEXT_DARK,
                 fontweight="bold" if i == observed_idx else "normal")

ax_left.set_yticks(range(len(words)))
ax_left.set_yticklabels(words, fontsize=11, fontfamily="monospace")
ax_left.set_xlabel("P(word | context)", fontsize=11, color=TEXT_DARK)
ax_left.set_xlim(0, 0.42)
ax_left.invert_yaxis()

# Remove spines
for spine in ["top", "right"]:
    ax_left.spines[spine].set_visible(False)
ax_left.spines["left"].set_linewidth(0.5)
ax_left.spines["bottom"].set_linewidth(0.5)
ax_left.tick_params(axis="x", labelsize=9, colors=GRAY)
ax_left.tick_params(axis="y", length=0)

# Arrow pointing to observed word
ax_left.annotate("observed word $w_t$",
                 xy=(probs[observed_idx], observed_idx),
                 xytext=(0.32, observed_idx + 1.4),
                 fontsize=10, color=BLUE, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5),
                 ha="center")

# Formula box
formula_text = r"$S_{\mathrm{classical}}(w_t) = -\log\, P(w_t \mid \mathrm{context})$"
ax_left.text(0.50, -0.14, formula_text, transform=ax_left.transAxes,
             fontsize=11.5, ha="center", va="top", color=BLUE,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BLUE_LIGHT,
                       edgecolor=BLUE, alpha=0.25, linewidth=1.2))

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Divergence-Based Surprisal
# ══════════════════════════════════════════════════════════════════════════
ax_right.set_facecolor(BG)
ax_right.set_title("Divergence-Based Surprisal", fontsize=15,
                    fontweight="bold", color=ORANGE, pad=18)

# Create two predictive distributions (conceptual Gaussians over vocab space)
x = np.linspace(-4, 6, 500)

# Before observing w_t: broader, centered earlier
mu_before, sigma_before = 1.0, 1.4
pdf_before = norm.pdf(x, mu_before, sigma_before)

# After observing w_t: shifted and tighter
mu_after, sigma_after = 2.2, 1.0
pdf_after = norm.pdf(x, mu_after, sigma_after)

# Plot the two distributions
ax_right.plot(x, pdf_before, color=GRAY, lw=2.5, label=r"$P(\cdot \mid w_1 \ldots w_{t-1})$  (before)",
              zorder=3)
ax_right.plot(x, pdf_after, color=ORANGE, lw=2.5, label=r"$P(\cdot \mid w_1 \ldots w_t)$  (after)",
              zorder=3)

# Fill distributions lightly
ax_right.fill_between(x, pdf_before, alpha=0.12, color=GRAY, zorder=1)
ax_right.fill_between(x, pdf_after, alpha=0.12, color=ORANGE, zorder=1)

# Shade the KL divergence region (between the two curves where they overlap)
# Use a hatch pattern in the overlap region for visual emphasis
y_min = np.minimum(pdf_before, pdf_after)
y_max = np.maximum(pdf_before, pdf_after)

# Shade the area between the two curves to represent divergence
ax_right.fill_between(x, pdf_before, pdf_after,
                      where=(pdf_before > 0.01) | (pdf_after > 0.01),
                      alpha=0.18, color=SHADE_COLOR, zorder=2,
                      label="KL divergence region")

# Axis formatting
ax_right.set_xlabel("Vocabulary space (next-word predictions)", fontsize=11,
                    color=TEXT_DARK)
ax_right.set_ylabel("Probability density", fontsize=11, color=TEXT_DARK)
ax_right.set_xlim(-3.5, 5.5)
ax_right.set_ylim(0, max(pdf_after.max(), pdf_before.max()) * 1.25)
ax_right.set_xticks([])
ax_right.set_yticks([])

for spine in ["top", "right"]:
    ax_right.spines[spine].set_visible(False)
ax_right.spines["left"].set_linewidth(0.5)
ax_right.spines["bottom"].set_linewidth(0.5)

# Legend
leg = ax_right.legend(fontsize=9.5, loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor=GRAY_LIGHT, fancybox=True)
leg.get_frame().set_linewidth(0.8)

# Annotation: KL divergence arrow
# Point to the middle of the divergence region
mid_x = 1.6
mid_y_before = norm.pdf(mid_x, mu_before, sigma_before)
mid_y_after = norm.pdf(mid_x, mu_after, sigma_after)
mid_y = (mid_y_before + mid_y_after) / 2

ax_right.annotate("distribution\nshift after\nobserving $w_t$",
                  xy=(mid_x, mid_y),
                  xytext=(-2.0, 0.30),
                  fontsize=9.5, color=ORANGE, fontweight="bold",
                  ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))

# Formula box
formula_div = (r"$S_{\mathrm{div}}(w_t) = D_{\mathrm{KL}}\!("
               r"P(\cdot \mid w_1\!\ldots\!w_t)\;\|\;"
               r"P(\cdot \mid w_1\!\ldots\!w_{t-1}))$")
ax_right.text(0.50, -0.14, formula_div, transform=ax_right.transAxes,
              fontsize=11, ha="center", va="top", color=ORANGE,
              bbox=dict(boxstyle="round,pad=0.4", facecolor=ORANGE_LIGHT,
                        edgecolor=ORANGE, alpha=0.25, linewidth=1.2))

# ── Global title ───────────────────────────────────────────────────────────
fig.suptitle("Classical vs Divergence-Based Surprisal: Conceptual Comparison",
             fontsize=16, fontweight="bold", color=TEXT_DARK, y=1.01)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig("/home/wangni/notion-figures/conll/paper3_fig1.png",
            dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
plt.close()
print("Saved: /home/wangni/notion-figures/conll/paper3_fig1.png")
