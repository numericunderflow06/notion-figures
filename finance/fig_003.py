"""
fig_003: Data Pipeline — From Raw Filings to Training Samples
Vertical flowchart showing the 7 preprocessing stages in financial_reports_loader.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Color palette ──
BG_COLOR = "#FFFFFF"
STEP_FILL = "#E8F0FE"
STEP_EDGE = "#4A7EC2"
HEADER_BG = "#3B6DAE"
HEADER_FG = "#FFFFFF"
ARROW_COLOR = "#5B5B5B"
ANNOT_COLOR = "#C0392B"
PARAM_COLOR = "#555555"
FUNC_COLOR = "#2C6BAC"

# ── Pipeline steps ──
# Each step: num, title, func, desc, params, annot_right, annot_left, extra_height
steps = [
    {
        "num": "1",
        "title": "Download Stock Prices",
        "func": "download_stock_prices()",
        "desc": "Fetch daily OHLCV from Yahoo Finance for 14 European real estate tickers",
        "params": 'period="max", auto_adjust=True',
        "annot_right": None,
        "annot_left": "14 tickers\n(6 countries)",
        "extra_h": 0,
    },
    {
        "num": "2",
        "title": "Read Filing Metadata",
        "func": "csv.DictReader(METADATA_CSV)",
        "desc": "Parse CSV: company_isin, filing_date, filing_type_class, relative_path",
        "params": None,
        "annot_right": "19,632 filings",
        "annot_left": None,
        "extra_h": 0,
    },
    {
        "num": "3",
        "title": "Match Filings to Price Windows",
        "func": "_get_price_window()",
        "desc": "Extract 60-day pre-filing price series and 5-day post-filing return",
        "params": "PRE_FILING_DAYS=60, POST_FILING_DAYS=5",
        "annot_right": "17,631 usable",
        "annot_left": "~2,001 skipped",
        "extra_h": 0,
    },
    {
        "num": "4",
        "title": "Strip Boilerplate Text",
        "func": "_strip_boilerplate() + _read_filing_text()",
        "desc": "Remove disclaimers, image placeholders, metadata; truncate to char limit",
        "params": "MAX_FILING_TEXT_CHARS=3,000",
        "annot_right": None,
        "annot_left": None,
        "extra_h": 0,
    },
    {
        "num": "5",
        "title": "Compute Labels (3 types)",
        "func": "_compute_volatility_label()  _compute_direction_label()",
        "desc": "price_return: (a) up / (b) down\nvolatility: ratio thresholds 1.5 / 0.667\ndirection: lin. reg. slope vs \u00b10.002",
        "params": None,
        "annot_right": "17,631 samples\n\u00d7 3 label columns",
        "annot_left": None,
        "extra_h": 0.35,
    },
    {
        "num": "6",
        "title": "Cache as Pickle",
        "func": "build_preprocessed_dataset()",
        "desc": "Serialize full DataFrame to preprocessed_dataset.pkl",
        "params": None,
        "annot_right": None,
        "annot_left": None,
        "extra_h": -0.15,
    },
    {
        "num": "7",
        "title": "Time-Based Split + 3\u00d7 Expansion",
        "func": "load_financial_reports_splits() + _expand()",
        "desc": "Sort by filing_date \u2192 80/10/10 split\nExpand each sample into 3 QA pairs\n(price_return, volatility, direction)",
        "params": "TEST_FRAC=0.1, VAL_FRAC=0.1",
        "annot_right": "52,893 QA pairs\n(train / val / test)",
        "annot_left": None,
        "extra_h": 0.50,
    },
]

# ── Layout parameters ──
box_w = 5.8
base_box_h = 1.35
x_center = 5.0
x_left = x_center - box_w / 2
arrow_gap = 0.38

# Compute total height needed
total_h = sum(base_box_h + s["extra_h"] + arrow_gap for s in steps) - arrow_gap + 1.8
fig_h = max(total_h + 1.0, 16)

# ── Figure setup ──
fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=200)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, fig_h)
ax.axis("off")

# ── Title ──
title_y = fig_h - 0.5
ax.text(
    5, title_y, "Data Pipeline: From Raw Filings to Training Samples",
    ha="center", va="center", fontsize=15, fontweight="bold",
    color="#1A1A1A", family="sans-serif"
)
ax.text(
    5, title_y - 0.45,
    "financial_reports_loader.py  \u2192  FinancialReportsQADataset.py",
    ha="center", va="center", fontsize=9.5, color=PARAM_COLOR,
    family="monospace", style="italic"
)

# ── Draw steps ──
cursor_y = title_y - 1.3  # top of first box

for i, step in enumerate(steps):
    box_h = base_box_h + step["extra_h"]
    y_top = cursor_y
    y_bot = y_top - box_h
    y_mid = (y_top + y_bot) / 2

    # ── Rounded rectangle ──
    rect = FancyBboxPatch(
        (x_left, y_bot), box_w, box_h,
        boxstyle="round,pad=0.10",
        facecolor=STEP_FILL, edgecolor=STEP_EDGE, linewidth=1.5,
        zorder=2
    )
    ax.add_patch(rect)

    # ── Step number circle ──
    cx = x_left + 0.38
    cy = y_top - 0.32
    circle = plt.Circle(
        (cx, cy), 0.25,
        facecolor=HEADER_BG, edgecolor="none", zorder=3
    )
    ax.add_patch(circle)
    ax.text(
        cx, cy, step["num"],
        ha="center", va="center", fontsize=12.5, fontweight="bold",
        color=HEADER_FG, zorder=4
    )

    # ── Title ──
    ax.text(
        cx + 0.45, cy, step["title"],
        ha="left", va="center", fontsize=10.5, fontweight="bold",
        color="#1A1A1A", zorder=4
    )

    # ── Function name (monospace, below title) ──
    func_y = cy - 0.35
    ax.text(
        x_left + 0.30, func_y, step["func"],
        ha="left", va="center", fontsize=7.5, color=FUNC_COLOR,
        family="monospace", zorder=4
    )

    # ── Description (below function) ──
    n_desc_lines = step["desc"].count("\n") + 1
    desc_y = func_y - 0.2 - (n_desc_lines - 1) * 0.12
    ax.text(
        x_left + 0.30, desc_y, step["desc"],
        ha="left", va="top", fontsize=8, color="#333333",
        zorder=4, linespacing=1.45
    )

    # ── Parameter annotation (at bottom of box) ──
    if step["params"]:
        ax.text(
            x_left + 0.30, y_bot + 0.12, step["params"],
            ha="left", va="bottom", fontsize=7, color=PARAM_COLOR,
            style="italic", zorder=4
        )

    # ── Right-side annotation ──
    if step["annot_right"]:
        ax.text(
            x_left + box_w + 0.3, y_mid,
            step["annot_right"],
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=ANNOT_COLOR, zorder=4, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FDECEA",
                      edgecolor=ANNOT_COLOR, linewidth=0.8, alpha=0.95)
        )

    # ── Left-side annotation ──
    if step["annot_left"]:
        ax.text(
            x_left - 0.3, y_mid,
            step["annot_left"],
            ha="right", va="center", fontsize=8, color="#666666",
            zorder=4, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", linewidth=0.7, alpha=0.9)
        )

    # ── Arrow to next step ──
    if i < len(steps) - 1:
        arr_start = y_bot
        arr_end = y_bot - arrow_gap
        ax.annotate(
            "", xy=(x_center, arr_end), xytext=(x_center, arr_start),
            arrowprops=dict(
                arrowstyle="-|>", color=ARROW_COLOR,
                lw=2, mutation_scale=16
            ),
            zorder=1
        )

    cursor_y = y_bot - arrow_gap

# ── Footer ──
ax.text(
    5, cursor_y + arrow_gap - 0.5,
    "Source: financial_reports_loader.py (436 lines) + FinancialReportsQADataset.py (167 lines)",
    ha="center", va="center", fontsize=8, color="#999999", style="italic"
)

plt.tight_layout(pad=0.3)
plt.savefig(
    "/home/wangni/notion-figures/finance/fig_003.png",
    dpi=200, bbox_inches="tight", facecolor=BG_COLOR
)
plt.close()
print("Saved: /home/wangni/notion-figures/finance/fig_003.png")
