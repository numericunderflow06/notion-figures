"""
fig_006: Question Types and Label Computation
Three-panel diagram illustrating the three question types derived from each filing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colour palette ──────────────────────────────────────────────────────────
GREEN  = "#2E8B57"   # increase / bullish
RED    = "#C0392B"   # decrease / bearish
GRAY   = "#5D6D7E"   # stable / sideways
BG_PANELS = ["#F0F7F4", "#FDF0EF", "#F0F2F7"]
BORDER_COLORS = [GREEN, RED, "#4A6FA5"]
TITLE_BG = "#2C3E50"
FORMULA_BG = "#FAFAFA"

# ── Synthetic price data for schematics ─────────────────────────────────────
np.random.seed(42)
days_pre = np.arange(60)
days_post = np.arange(60, 65)

# Panel 1 – price return: clear upward move post-filing
pre1 = 50 + np.cumsum(np.random.randn(60) * 0.3)
post1_up = pre1[-1] + np.cumsum(np.random.randn(5) * 0.3 + 0.4)

# Panel 2 – volatility: low-vol pre, high-vol post
pre2 = 30 + np.cumsum(np.random.randn(60) * 0.15)
post2 = pre2[-1] + np.cumsum(np.random.randn(5) * 0.6)

# Panel 3 – direction: clear bearish slope
pre3 = 20 + np.cumsum(np.random.randn(60) * 0.2)
post3 = pre3[-1] + np.cumsum(np.random.randn(5) * 0.2 - 0.5)


def draw_price_chart(ax, pre, post, annotation_fn):
    """Draw a mini price chart with pre/post shading."""
    all_prices = np.concatenate([pre, post])
    ymin, ymax = all_prices.min(), all_prices.max()
    yrng = ymax - ymin
    pad_lo = ymin - yrng * 0.12
    pad_hi = ymax + yrng * 0.22

    # Shaded regions
    ax.fill_between(days_pre, pad_lo, pad_hi,
                    color="#E8E8E8", alpha=0.35, zorder=0)
    ax.fill_between(days_post, pad_lo, pad_hi,
                    color="#FFF9C4", alpha=0.5, zorder=0)

    # Price line
    ax.plot(days_pre, pre, color="#666666", linewidth=1.1, zorder=2)
    ax.plot(np.concatenate([[days_pre[-1]], days_post]),
            np.concatenate([[pre[-1]], post]),
            color="#333333", linewidth=1.6, zorder=2)

    # Filing date marker
    ax.axvline(x=59.5, color="#E67E22", linewidth=1.4, linestyle="--", zorder=3)
    ax.text(59.5, ymax + yrng * 0.10, "Filing\nDate", fontsize=7,
            ha="center", va="bottom", color="#E67E22", fontweight="bold")

    # Axis
    ax.set_xlim(-2, 69)
    ax.set_ylim(pad_lo, pad_hi)
    ax.set_xticks([0, 30, 60, 65])
    ax.set_xticklabels(["t\u221260", "t\u221230", "t\u2080", "t\u2080+5"], fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Close Price", fontsize=7.5, labelpad=2)

    # Region labels at bottom
    ax.text(30, pad_lo + yrng * 0.02, "60 pre-filing days",
            fontsize=6.5, ha="center", va="bottom", color="#888888", style="italic")
    ax.text(62.5, pad_lo + yrng * 0.02, "5 post",
            fontsize=6.5, ha="center", va="bottom", color="#888888", style="italic")

    # Annotation callback
    annotation_fn(ax, pre, post, ymin, ymax, yrng)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_return(ax, pre, post, ymin, ymax, yrng):
    p0, p5 = pre[-1], post[-1]
    ax.annotate("", xy=(64, p5), xytext=(60, p0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.2))
    ret = (p5 - p0) / p0
    ax.text(64.5, (p0 + p5) / 2, f"r = {ret:+.1%}",
            fontsize=8, color=GREEN, fontweight="bold", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GREEN, alpha=0.9))


def annotate_volatility(ax, pre, post, ymin, ymax, yrng):
    pre_returns = np.diff(pre) / pre[:-1]
    full_post = np.concatenate([[pre[-1]], post])
    post_returns = np.diff(full_post) / full_post[:-1]
    pre_vol = np.std(pre_returns)
    post_vol = np.std(post_returns)
    ratio = post_vol / pre_vol

    # σ_pre bracket
    mid_pre = np.mean(pre[-10:])
    half_pre = pre_vol * mid_pre * 8
    ax.annotate("", xy=(55, mid_pre + half_pre),
                xytext=(55, mid_pre - half_pre),
                arrowprops=dict(arrowstyle="<->", color="#5D6D7E", lw=1.4))
    ax.text(54, mid_pre, r"$\sigma_{pre}$", fontsize=7, ha="right", color="#5D6D7E")

    # σ_post bracket
    mid_post = np.mean(post)
    half_post = post_vol * mid_post * 8
    ax.annotate("", xy=(62, mid_post + half_post),
                xytext=(62, mid_post - half_post),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    ax.text(63.2, mid_post, r"$\sigma_{post}$", fontsize=7, ha="left", color=RED)

    ax.text(62, ymax + yrng * 0.10, f"ratio = {ratio:.2f}",
            fontsize=7.5, color=RED, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, alpha=0.9))


def annotate_direction(ax, pre, post, ymin, ymax, yrng):
    x = np.arange(len(post))
    coeffs = np.polyfit(x, post, 1)
    slope = coeffs[0]
    mean_p = np.mean(post)
    norm_slope = slope / mean_p

    fit_y = np.polyval(coeffs, x)
    ax.plot(days_post, fit_y, color=RED, linewidth=2.2, linestyle="-", zorder=4)

    # Place label above regression line, centered in the post region
    ax.text(62, ymax + yrng * 0.06,
            f"s = {norm_slope:+.4f}/day",
            fontsize=7.5, color=RED, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=RED, alpha=0.9))


# ── Build figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10.5), facecolor="white")

# Title block — use two separate text calls to avoid overlap
fig.text(0.5, 0.975,
         "Question Types and Label Computation",
         fontsize=17, fontweight="bold", ha="center", va="center", color=TITLE_BG)
fig.text(0.5, 0.952,
         "Each filing generates 3 question\u2013answer pairs based on the 5-day post-filing stock price window",
         fontsize=10.5, ha="center", va="center", color="#666666")

# Grid: chart row (top) + info row (bottom)
gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.28,
                      left=0.04, right=0.96, top=0.895, bottom=0.04,
                      height_ratios=[1.0, 1.1])

panels = [
    {
        "title": "1  Price Return",
        "subtitle": "Binary \u2014 5-day endpoint return",
        "pre": pre1, "post": post1_up,
        "annotate": annotate_return,
        "formula": r"$r = \frac{P_{t_0+5}\; -\; P_{t_0}}{P_{t_0}}$",
        "rule": "r \u2265 0    \u2192  Increase\nr < 0    \u2192  Decrease",
        "answers": [("(a) Increase", GREEN), ("(b) Decrease", RED)],
        "border": GREEN,
        "bg": BG_PANELS[0],
    },
    {
        "title": "2  Volatility",
        "subtitle": "Ternary \u2014 std ratio with thresholds",
        "pre": pre2, "post": post2,
        "annotate": annotate_volatility,
        "formula": (r"$\mathrm{ratio} = \sigma_{\mathrm{post}}\; /\; \sigma_{\mathrm{pre}}$"
                    + "\n"
                    + r"$\sigma = \mathrm{std}(\mathrm{daily\; returns})$"),
        "rule": "ratio > 1.5     \u2192  Increase\nratio < 0.667   \u2192  Decrease\notherwise         \u2192  Stable",
        "answers": [("(a) Increase", GREEN), ("(b) Decrease", RED),
                    ("(c) Remain stable", GRAY)],
        "border": RED,
        "bg": BG_PANELS[1],
    },
    {
        "title": "3  Market Direction",
        "subtitle": "Ternary \u2014 linear regression slope",
        "pre": pre3, "post": post3,
        "annotate": annotate_direction,
        "formula": (r"$\mathrm{slope} = \mathrm{linreg}(P_{t_0+1\,..\,5})$"
                    + "\n"
                    + r"$s = \mathrm{slope}\; /\; \bar{P}$   (fraction/day)"),
        "rule": "s > +0.002   \u2192  Bullish\ns < \u22120.002   \u2192  Bearish\notherwise       \u2192  Sideways",
        "answers": [("(a) Bullish", GREEN), ("(b) Bearish", RED),
                    ("(c) Sideways", GRAY)],
        "border": BORDER_COLORS[2],
        "bg": BG_PANELS[2],
    },
]

for col, p in enumerate(panels):
    # ── Top row: price chart ──
    ax_chart = fig.add_subplot(gs[0, col])
    ax_chart.set_facecolor(p["bg"])

    # Title + subtitle
    ax_chart.set_title(p["title"], fontsize=13, fontweight="bold",
                       color=p["border"], pad=18, loc="center")
    ax_chart.text(0.5, 1.025, p["subtitle"], transform=ax_chart.transAxes,
                  fontsize=8.5, ha="center", va="bottom", color="#888888",
                  style="italic")

    draw_price_chart(ax_chart, p["pre"], p["post"], p["annotate"])

    for spine in ax_chart.spines.values():
        spine.set_edgecolor(p["border"])
        spine.set_linewidth(1.2)

    # ── Bottom row: computation info ──
    ax_info = fig.add_subplot(gs[1, col])
    ax_info.set_facecolor(p["bg"])
    ax_info.set_xlim(0, 10)
    ax_info.set_ylim(0, 10)
    ax_info.axis("off")

    # --- Computation box ---
    box_comp = FancyBboxPatch((0.2, 6.1), 9.6, 3.5,
                              boxstyle="round,pad=0.3",
                              facecolor=FORMULA_BG, edgecolor=p["border"],
                              linewidth=1.2, zorder=2)
    ax_info.add_patch(box_comp)

    ax_info.text(0.55, 9.25, "Computation", fontsize=9.5, fontweight="bold",
                 color=p["border"], va="top")
    ax_info.text(5.0, 7.65, p["formula"], fontsize=10,
                 ha="center", va="center", color="#333333")

    # --- Thresholds box ---
    box_thresh = FancyBboxPatch((0.2, 2.5), 9.6, 3.2,
                                boxstyle="round,pad=0.3",
                                facecolor="white", edgecolor=p["border"],
                                linewidth=1.2, linestyle="--", zorder=2)
    ax_info.add_patch(box_thresh)

    ax_info.text(0.55, 5.4, "Thresholds", fontsize=9.5, fontweight="bold",
                 color=p["border"], va="top")
    ax_info.text(5.0, 3.85, p["rule"], fontsize=9.5,
                 ha="center", va="center", color="#333333",
                 family="monospace", linespacing=1.6)

    # --- Answer choice badges ---
    ax_info.text(0.4, 2.1, "Answer choices:", fontsize=9, fontweight="bold",
                 color="#444444", va="top")

    n_ans = len(p["answers"])
    total_width = 9.2
    badge_width = total_width / n_ans
    x_start = 0.4

    for i, (label, clr) in enumerate(p["answers"]):
        bx = x_start + i * badge_width
        badge = FancyBboxPatch((bx, 0.25), badge_width - 0.3, 1.3,
                               boxstyle="round,pad=0.2",
                               facecolor=clr, edgecolor=clr,
                               alpha=0.12, linewidth=0.8, zorder=2)
        ax_info.add_patch(badge)
        ax_info.text(bx + (badge_width - 0.3) / 2, 0.9, label,
                     fontsize=9, fontweight="bold", color=clr,
                     ha="center", va="center")

# ── Save ────────────────────────────────────────────────────────────────────
out_path = "/home/wangni/notion-figures/finance/fig_006.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved figure to {out_path}")
