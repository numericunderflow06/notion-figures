#!/usr/bin/env python3
"""
fig_005: Experiment Design — Datasets, Baselines, and Ablations
Structured infographic with three rows:
  Top:    6 dataset cards with modality icons, sample counts, task types
  Middle: 5 baselines listed
  Bottom: 6 ablation configurations as a matrix
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.linewidth": 0,
})

# ── Color palette ─────────────────────────────────────────────────────
COL_ECG   = "#C93C3E"   # warm red
COL_EEG   = "#7046A0"   # purple
COL_ACC   = "#2E8B3E"   # green
COL_ECG_L = "#FAEAEB"   # light red bg
COL_EEG_L = "#EDE5F5"   # light purple bg
COL_ACC_L = "#E3F5E6"   # light green bg

COL_TITLE   = "#1A1A2E"
COL_TEXT    = "#2C2C54"
COL_SUBTEXT = "#666688"
COL_BG_CARD = "#F5F5FA"
COL_BG_ROW  = "#F0F0F8"
COL_ACCENT  = "#3366CC"
COL_ACCENT_L = "#DCE8F8"
COL_HIGHLIGHT = "#FFD166"
COL_HL_DARK  = "#E69B20"
COL_HL_BG   = "#FFF7E0"

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 17), facecolor="white")

# ── Data ──────────────────────────────────────────────────────────────
datasets = [
    {"name": "ECG-QA",            "mod": "ECG",   "detail": "12-lead, 10 s",
     "task": "Free-form QA\n(3,138 question types)", "samples": "~240k",
     "color": COL_ECG, "bg": COL_ECG_L},
    {"name": "PTB-XL",            "mod": "ECG",   "detail": "12-lead, 10 s",
     "task": "Multi-label rhythm\n+ morphology (71 labels)", "samples": "~21.8k",
     "color": COL_ECG, "bg": COL_ECG_L},
    {"name": "CPSC 2018",         "mod": "ECG",   "detail": "12-lead, variable",
     "task": "9-class arrhythmia\nclassification", "samples": "~6.9k",
     "color": COL_ECG, "bg": COL_ECG_L},
    {"name": "Chapman-\nShaoxing","mod": "ECG",   "detail": "12-lead, 10 s",
     "task": "Multi-label rhythm\nclassification", "samples": "~10.6k",
     "color": COL_ECG, "bg": COL_ECG_L},
    {"name": "Sleep-EDF",         "mod": "EEG",   "detail": "1-ch Fpz-Cz, 30 s",
     "task": "5-class sleep staging\n(W, REM, N1-N3)", "samples": "~7.4k",
     "color": COL_EEG, "bg": COL_EEG_L},
    {"name": "HAR-CoT",           "mod": "Accel", "detail": "3-axis, 2.56 s",
     "task": "8-class activity\nrecognition", "samples": "~68k",
     "color": COL_ACC, "bg": COL_ACC_L},
]

baselines = [
    ("B1", "OpenTSLM-Flamingo",     "LLaMA-3.2-3B\nConv1D encoder\nSupervised training"),
    ("B2", "OpenTSLM-SoftPrompt",   "LLaMA-3.2-3B\nToken concatenation\nfusion"),
    ("B3", "QoQ-Med-VL-7B",        "Released model\nEvaluated on overlapping\nECG tasks"),
    ("B4", "Text-only LLaMA-3.2",  "No TS encoder\nText descriptions\nonly"),
    ("B5", "GPT-4o",               "TS plots + text\n(as reported in\nOpenTSLM)"),
]

ablation_cols = ["Encoder", "Fusion", "Training"]
ablations = [
    ("A0", "Full system (ours)",   ["ECG-JEPA\n(frozen)", "Flamingo\ncross-attn", "Supervised\n\u2192 DRPO"], None),
    ("A1", "Encoder ablation",     ["Conv1D\n(OpenTSLM)", "Flamingo\ncross-attn", "Supervised\n\u2192 DRPO"], 0),
    ("A2", "DRPO ablation",        ["ECG-JEPA\n(frozen)", "Flamingo\ncross-attn", "Supervised\nonly"], 2),
    ("A3", "Fusion ablation",      ["ECG-JEPA\n(frozen)", "Token concat\n(SoftPrompt)", "Supervised\n\u2192 DRPO"], 1),
    ("A4", "Curriculum ablation",  ["ECG-JEPA\n(frozen)", "Flamingo\ncross-attn", "DRPO only\n(no curriculum)"], 2),
    ("A5", "Freeze ablation",      ["ECG-JEPA\n(unfrozen)", "Flamingo\ncross-attn", "Supervised\n\u2192 DRPO"], 0),
]

# ═══════════════════════════════════════════════════════════════
# Layout constants (figure-fraction coordinates)
# ═══════════════════════════════════════════════════════════════
L = 0.03;  R = 0.97;  W = R - L

# Vertical layout
TITLE_Y    = 0.985
S1_TITLE   = 0.960   # Datasets section title
S1_CARDS_T = 0.925   # top of card row
S1_CARDS_B = 0.595   # bottom of card row
S1_LEG_Y   = 0.575   # modality legend
DIV1_Y     = 0.555
S2_TITLE   = 0.540   # Baselines section title
S2_CARDS_T = 0.505
S2_CARDS_B = 0.395
DIV2_Y     = 0.375
S3_TITLE   = 0.360   # Ablation section title
S3_MAT_T   = 0.310   # top of matrix
S3_MAT_B   = 0.040   # bottom of matrix
S3_LEG_Y   = 0.020   # ablation legend

# ═══════════════════════════════════════════════════════════════
# MAIN TITLE
# ═══════════════════════════════════════════════════════════════
fig.text(0.50, TITLE_Y, "Experiment Design: Datasets, Baselines, and Ablations",
         fontsize=19, fontweight="bold", color=COL_TITLE, ha="center", va="top")

# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATASETS
# ═══════════════════════════════════════════════════════════════
fig.text(0.50, S1_TITLE, "DATASETS", fontsize=15, fontweight="bold",
         color=COL_TITLE, ha="center", va="top")
fig.text(0.50, S1_TITLE - 0.020, "6 benchmarks across 3 modalities",
         fontsize=10.5, color=COL_SUBTEXT, ha="center", va="top")

n_cards = 6
card_gap = 0.012
card_w = (W - card_gap * (n_cards - 1)) / n_cards
card_h = S1_CARDS_T - S1_CARDS_B

for i, ds in enumerate(datasets):
    x0 = L + i * (card_w + card_gap)
    y0 = S1_CARDS_B
    cx = x0 + card_w / 2

    # Card background
    fig.patches.append(FancyBboxPatch(
        (x0, y0), card_w, card_h, boxstyle="round,pad=0.008",
        facecolor=ds["bg"], edgecolor=ds["color"], linewidth=1.8,
        transform=fig.transFigure, zorder=2))

    # ── Modality badge ──
    bw, bh = 0.052, 0.020
    by = y0 + card_h - 0.016
    fig.patches.append(FancyBboxPatch(
        (cx - bw/2, by - bh/2), bw, bh, boxstyle="round,pad=0.004",
        facecolor=ds["color"], edgecolor="none",
        transform=fig.transFigure, zorder=3))
    fig.text(cx, by, ds["mod"], fontsize=9, fontweight="bold",
             color="white", ha="center", va="center", zorder=4)

    # ── Waveform icon ──
    icon_cy = by - 0.042
    icon_w = card_w * 0.68
    icon_h = 0.038

    if ds["mod"] == "ECG":
        t = np.linspace(0, 1, 150)
        y_w = np.zeros_like(t)
        for beat_off in [0.0, 0.50]:
            m = (t > 0.04+beat_off) & (t < 0.14+beat_off)
            y_w[m] = 0.12 * np.sin(np.pi * (t[m] - 0.04 - beat_off) / 0.10)
            m = (t > 0.17+beat_off) & (t < 0.20+beat_off)
            y_w[m] = -0.10
            m = (t > 0.20+beat_off) & (t < 0.27+beat_off)
            y_w[m] = 0.85 * np.sin(np.pi * (t[m] - 0.20 - beat_off) / 0.07)
            m = (t > 0.27+beat_off) & (t < 0.31+beat_off)
            y_w[m] = -0.08
            m = (t > 0.36+beat_off) & (t < 0.48+beat_off)
            y_w[m] = 0.22 * np.sin(np.pi * (t[m] - 0.36 - beat_off) / 0.12)
        xs = cx - icon_w/2 + t * icon_w
        ys = icon_cy + y_w * icon_h * 0.5
        fig.lines.append(Line2D(xs, ys, color=ds["color"], linewidth=1.5,
                                 transform=fig.transFigure, zorder=3))

    elif ds["mod"] == "EEG":
        t = np.linspace(0, 1, 120)
        y_w = (np.sin(2*np.pi*3.5*t) * 0.35
               + 0.20 * np.sin(2*np.pi*8*t + 0.7)
               + 0.10 * np.sin(2*np.pi*13*t + 1.2))
        xs = cx - icon_w/2 + t * icon_w
        ys = icon_cy + y_w * icon_h * 0.5
        fig.lines.append(Line2D(xs, ys, color=ds["color"], linewidth=1.5,
                                 transform=fig.transFigure, zorder=3))

    elif ds["mod"] == "Accel":
        rng = np.random.RandomState(42)
        t = np.linspace(0, 1, 80)
        colors_ax = [ds["color"], "#5DC96A", "#1E6B2E"]
        alphas = [0.95, 0.65, 0.40]
        for ax_i, (c_ax, al) in enumerate(zip(colors_ax, alphas)):
            y_w = (np.sin(2*np.pi*2.5*t + ax_i*1.3) * 0.5
                   + 0.25 * rng.randn(80))
            y_w = y_w / (np.abs(y_w).max() + 1e-9) * 0.38
            shift = (ax_i - 1) * 0.18
            xs = cx - icon_w/2 + t * icon_w
            ys = icon_cy + (y_w + shift) * icon_h * 0.5
            fig.lines.append(Line2D(xs, ys, color=c_ax, linewidth=1.2,
                                     alpha=al, transform=fig.transFigure, zorder=3))

    # ── Dataset name ──
    name_y = icon_cy - 0.030
    fig.text(cx, name_y, ds["name"], fontsize=11.5, fontweight="bold",
             color=COL_TITLE, ha="center", va="center", zorder=4,
             linespacing=0.95)

    # ── Modality detail ──
    det_y = name_y - 0.026
    fig.text(cx, det_y, ds["detail"], fontsize=8.5,
             color=COL_SUBTEXT, ha="center", va="center", zorder=4)

    # ── Task description ──
    task_y = det_y - 0.038
    fig.text(cx, task_y, ds["task"], fontsize=8.5,
             color=COL_TEXT, ha="center", va="center", zorder=4,
             linespacing=1.1)

    # ── Sample count ──
    samp_y = y0 + 0.032
    fig.text(cx, samp_y, ds["samples"], fontsize=14, fontweight="bold",
             color=ds["color"], ha="center", va="center", zorder=4)
    fig.text(cx, samp_y - 0.018, "samples", fontsize=8,
             color=COL_SUBTEXT, ha="center", va="center", zorder=4)

# ── Modality legend ──
legend_items = [
    (COL_ECG, "ECG (12-lead)"),
    (COL_EEG, "EEG (single-channel)"),
    (COL_ACC, "Accelerometer (3-axis)"),
]
leg_start = 0.28
for j, (col, lab) in enumerate(legend_items):
    lx = leg_start + j * 0.17
    fig.patches.append(FancyBboxPatch(
        (lx, S1_LEG_Y - 0.006), 0.012, 0.012,
        boxstyle="round,pad=0.002", facecolor=col, edgecolor="none",
        transform=fig.transFigure, zorder=3))
    fig.text(lx + 0.017, S1_LEG_Y, lab, fontsize=9.5,
             color=COL_TEXT, ha="left", va="center", zorder=4)

# ── Divider ──
fig.lines.append(Line2D([L+0.02, R-0.02], [DIV1_Y, DIV1_Y],
                         color="#D0D0E0", linewidth=1.0,
                         transform=fig.transFigure, zorder=1))

# ═══════════════════════════════════════════════════════════════
# SECTION 2: BASELINES
# ═══════════════════════════════════════════════════════════════
fig.text(0.50, S2_TITLE, "BASELINES", fontsize=15, fontweight="bold",
         color=COL_TITLE, ha="center", va="top")
fig.text(0.50, S2_TITLE - 0.020, "5 comparison methods",
         fontsize=10.5, color=COL_SUBTEXT, ha="center", va="top")

n_bl = len(baselines)
bl_gap = 0.010
bl_w = (W - bl_gap * (n_bl - 1)) / n_bl
bl_h = S2_CARDS_T - S2_CARDS_B

for i, (bid, bname, bdesc) in enumerate(baselines):
    x0 = L + i * (bl_w + bl_gap)
    y0 = S2_CARDS_B
    cx = x0 + bl_w / 2

    # Card bg
    fig.patches.append(FancyBboxPatch(
        (x0, y0), bl_w, bl_h, boxstyle="round,pad=0.006",
        facecolor=COL_BG_CARD, edgecolor=COL_ACCENT, linewidth=1.2,
        transform=fig.transFigure, zorder=2))

    # ID badge
    bw2, bh2 = 0.025, 0.017
    bx2 = x0 + 0.010
    by2 = y0 + bl_h - 0.012
    fig.patches.append(FancyBboxPatch(
        (bx2, by2 - bh2/2), bw2, bh2, boxstyle="round,pad=0.003",
        facecolor=COL_ACCENT, edgecolor="none",
        transform=fig.transFigure, zorder=3))
    fig.text(bx2 + bw2/2, by2, bid, fontsize=8, fontweight="bold",
             color="white", ha="center", va="center", zorder=4)

    # Name (right of badge)
    fig.text(bx2 + bw2 + 0.008, by2, bname, fontsize=9.5, fontweight="bold",
             color=COL_TITLE, ha="left", va="center", zorder=4)

    # Description (centered in lower part of card)
    desc_y = y0 + bl_h * 0.35
    fig.text(cx, desc_y, bdesc, fontsize=8.5, color=COL_SUBTEXT,
             ha="center", va="center", zorder=4, linespacing=1.15)

# ── Divider ──
fig.lines.append(Line2D([L+0.02, R-0.02], [DIV2_Y, DIV2_Y],
                         color="#D0D0E0", linewidth=1.0,
                         transform=fig.transFigure, zorder=1))

# ═══════════════════════════════════════════════════════════════
# SECTION 3: ABLATION MATRIX
# ═══════════════════════════════════════════════════════════════
fig.text(0.50, S3_TITLE, "ABLATION STUDIES", fontsize=15, fontweight="bold",
         color=COL_TITLE, ha="center", va="top")
fig.text(0.50, S3_TITLE - 0.020,
         "6 configurations isolating each component — highlighted cells show the ablated component",
         fontsize=10.5, color=COL_SUBTEXT, ha="center", va="top")

# Matrix geometry
mat_L = L + 0.155      # room for row labels
mat_R = R - 0.015
n_rows = len(ablations)
n_cols = len(ablation_cols)
col_w = (mat_R - mat_L) / n_cols
row_h = (S3_MAT_T - S3_MAT_B) / (n_rows + 1)   # +1 for header

# ── Header row ──
for j, cn in enumerate(ablation_cols):
    cx = mat_L + j * col_w + col_w / 2
    cy = S3_MAT_T - row_h / 2
    fig.patches.append(FancyBboxPatch(
        (mat_L + j * col_w + 0.003, S3_MAT_T - row_h + 0.003),
        col_w - 0.006, row_h - 0.006,
        boxstyle="round,pad=0.005", facecolor=COL_ACCENT, edgecolor="none",
        transform=fig.transFigure, zorder=2))
    fig.text(cx, cy, cn, fontsize=12, fontweight="bold",
             color="white", ha="center", va="center", zorder=4)

# Header for row-label column
fig.text(L + 0.005, S3_MAT_T - row_h / 2, "ID / Purpose", fontsize=10,
         fontweight="bold", color=COL_ACCENT, ha="left", va="center")

header_bot = S3_MAT_T - row_h

# ── Data rows ──
for i, (aid, alabel, vals, changed) in enumerate(ablations):
    ry_top = header_bot - i * row_h
    ry_bot = ry_top - row_h
    cy = ry_top - row_h / 2

    # Alternating row stripe
    if i % 2 == 0:
        fig.patches.append(FancyBboxPatch(
            (L, ry_bot + 0.002), R - L, row_h - 0.004,
            boxstyle="round,pad=0.003", facecolor=COL_BG_ROW, edgecolor="none",
            transform=fig.transFigure, zorder=1))

    # Row label: badge + text
    bw3, bh3 = 0.028, 0.019
    bx3 = L + 0.007
    badge_col = COL_HL_DARK if aid == "A0" else COL_ACCENT
    fig.patches.append(FancyBboxPatch(
        (bx3, cy - bh3/2), bw3, bh3, boxstyle="round,pad=0.003",
        facecolor=badge_col, edgecolor="none",
        transform=fig.transFigure, zorder=3))
    fig.text(bx3 + bw3/2, cy, aid, fontsize=9, fontweight="bold",
             color="white", ha="center", va="center", zorder=4)

    fig.text(bx3 + bw3 + 0.008, cy, alabel,
             fontsize=10 if aid == "A0" else 9.5,
             fontweight="bold" if aid == "A0" else "normal",
             color=COL_TITLE if aid == "A0" else COL_TEXT,
             ha="left", va="center", zorder=4)

    # Cells
    for j, val in enumerate(vals):
        cx_c = mat_L + j * col_w + col_w / 2
        cell_x = mat_L + j * col_w + 0.004
        cell_y = ry_bot + 0.004
        cell_w = col_w - 0.008
        cell_h = row_h - 0.008

        is_changed = (changed is not None and j == changed)
        is_ours = (aid == "A0")

        if is_ours:
            fc, ec, lw = COL_ACCENT_L, COL_ACCENT, 1.6
            tc, tw = COL_ACCENT, "bold"
        elif is_changed:
            fc, ec, lw = COL_HL_BG, COL_HL_DARK, 2.2
            tc, tw = "#8B6914", "bold"
        else:
            fc, ec, lw = "white", "#CCCCDD", 0.8
            tc, tw = COL_TEXT, "normal"

        fig.patches.append(FancyBboxPatch(
            (cell_x, cell_y), cell_w, cell_h,
            boxstyle="round,pad=0.005", facecolor=fc, edgecolor=ec,
            linewidth=lw, transform=fig.transFigure, zorder=2))

        # Small change marker
        if is_changed:
            mx = cell_x + cell_w - 0.010
            my = cell_y + cell_h - 0.007
            fig.text(mx, my, "\u25B2", fontsize=7, color=COL_HL_DARK,
                     ha="center", va="center", zorder=5)

        fig.text(cx_c, cy, val, fontsize=9.5, color=tc, fontweight=tw,
                 ha="center", va="center", zorder=4, linespacing=1.05)

# ── Ablation legend ──
abl_leg = [
    (COL_ACCENT_L, COL_ACCENT, "Full system (A0)"),
    (COL_HL_BG, COL_HL_DARK, "Ablated component"),
    ("white", "#CCCCDD", "Unchanged component"),
]
alx_start = 0.30
for k, (fc, ec, lab) in enumerate(abl_leg):
    alx = alx_start + k * 0.16
    fig.patches.append(FancyBboxPatch(
        (alx, S3_LEG_Y - 0.006), 0.012, 0.012,
        boxstyle="round,pad=0.002", facecolor=fc, edgecolor=ec,
        linewidth=1.2, transform=fig.transFigure, zorder=3))
    fig.text(alx + 0.017, S3_LEG_Y, lab, fontsize=9.5,
             color=COL_TEXT, ha="left", va="center", zorder=4)

# ═══════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════
out = "/home/wangni/notion-figures/qoq-med/fig_005.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white",
            edgecolor="none", pad_inches=0.15)
plt.close(fig)
print(f"Saved: {out}")
