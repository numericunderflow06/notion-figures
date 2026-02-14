"""
fig_007: Data Processing Pipeline
Flow diagram showing the data processing pipeline from raw ZuCo .mat files
through preprocessing, subject-based splitting, z-score normalization,
TextTimeSeriesPrompt formatting, to model input.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────
C_RAW1   = "#E8D5B7"   # warm tan – ZuCo 1.0 raw
C_RAW2   = "#D5C4E8"   # soft lavender – ZuCo 2.0 raw
C_LOAD1  = "#F5C882"   # warm gold – scipy loader
C_LOAD2  = "#B8A0D8"   # muted purple – h5py loader
C_CACHE  = "#A8D8B9"   # sage green – cached .pkl
C_MERGE  = "#89C9E0"   # sky blue – merged processing
C_SPLIT  = "#7BB5D6"   # medium blue – split
C_NORM   = "#6DA0C4"   # deeper blue – normalization
C_PROMPT = "#5B8CB2"   # steel blue – prompt formatting
C_MODEL  = "#4A78A0"   # dark steel – model input
C_DETAIL = "#F0F0F0"   # light grey – detail boxes
C_TEXT   = "#2C3E50"   # dark slate – text
C_ARROW  = "#5D6D7E"   # grey-blue – arrows

fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(0, 22)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── helper: rounded box ────────────────────────────────────────────
def draw_box(ax, x, y, w, h, color, text_lines, fontsize=10,
             text_color=C_TEXT, bold_first=True, corner=0.15, alpha=1.0,
             border_color=None):
    """Draw a rounded rectangle with multi-line centred text."""
    bc = border_color if border_color else color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={corner}",
        facecolor=color, edgecolor=bc,
        linewidth=1.4, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    n = len(text_lines)
    line_gap = min(0.30, h / (n + 1))
    top = y + h / 2 + (n - 1) * line_gap / 2
    for i, line in enumerate(text_lines):
        weight = "bold" if (bold_first and i == 0) else "normal"
        fs = fontsize if i == 0 else fontsize - 1
        ax.text(
            x + w / 2, top - i * line_gap, line,
            ha="center", va="center", fontsize=fs,
            fontweight=weight, color=text_color, zorder=3,
        )

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.8,
               connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=16, zorder=4,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)

# ── coordinates ─────────────────────────────────────────────────────
# column centres (x)
col_raw   = 1.0
col_load  = 4.2
col_cache = 7.4
col_merge = 10.2
col_split = 13.0
col_norm  = 16.0
col_model = 19.2

box_w = 2.6
box_h = 1.6

# row centres (y) – two parallel rows for ZuCo 1.0 / 2.0
row_top = 7.2   # ZuCo 1.0
row_bot = 4.4   # ZuCo 2.0
row_mid = 5.8   # merged path centre

# ── STAGE 1: Raw .mat files ────────────────────────────────────────
draw_box(ax, col_raw, row_top, box_w, box_h, C_RAW1, [
    "ZuCo 1.0 .mat",
    "12 subjects",
    "NR + TSR tasks",
    "MATLAB v5 format",
], fontsize=10)

draw_box(ax, col_raw, row_bot, box_w, box_h, C_RAW2, [
    "ZuCo 2.0 .mat",
    "16 subjects",
    "NR + TSR tasks",
    "MATLAB v7.3 (HDF5)",
], fontsize=10)

# ── STAGE 2: Loaders ───────────────────────────────────────────────
draw_box(ax, col_load, row_top, box_w, box_h, C_LOAD1, [
    "scipy.io.loadmat",
    "Extract per-word",
    "5 ET metrics",
], fontsize=10)

draw_box(ax, col_load, row_bot, box_w, box_h, C_LOAD2, [
    "h5py (HDF5)",
    "Extract per-word",
    "5 ET metrics",
], fontsize=10)

# ── STAGE 3: Cache (.pkl) ──────────────────────────────────────────
draw_box(ax, col_cache, row_top, box_w, box_h, C_CACHE, [
    "Cache .pkl",
    "preprocessed_et.pkl",
    "{task: {subj: [sent]}}",
], fontsize=10)

draw_box(ax, col_cache, row_bot, box_w, box_h, C_CACHE, [
    "Cache .pkl",
    "preprocessed_et_zuco2.pkl",
    "{task: {subj: [sent]}}",
], fontsize=10)

# ── STAGE 4: Merge + flatten ───────────────────────────────────────
draw_box(ax, col_merge, row_mid, box_w, box_h + 0.4, C_MERGE, [
    "Build sample list",
    "et_channels: 5 metrics",
    "label: NR / TSR",
    "subject, sentence_text",
], fontsize=10)

# ── STAGE 5: Subject-based split ───────────────────────────────────
draw_box(ax, col_split, row_mid, box_w, box_h + 0.4, C_SPLIT, [
    "Subject-based Split",
    "Train / Val / Test",
    "No data leakage",
    "HF Dataset objects",
], fontsize=10, text_color="white")

# ── STAGE 6: Z-score + prompt ──────────────────────────────────────
draw_box(ax, col_norm, row_mid, box_w, box_h + 0.4, C_NORM, [
    "Per-channel z-score",
    "& TextTimeSeriesPrompt",
    "(text, norm_series)",
    "mean/std in label text",
], fontsize=10, text_color="white")

# ── STAGE 7: Model input ───────────────────────────────────────────
draw_box(ax, col_model, row_mid, box_w, box_h + 0.4, C_MODEL, [
    "PromptWithAnswer",
    "pre_prompt (task desc)",
    "5 x TTS prompts",
    "post_prompt + answer",
], fontsize=10, text_color="white")

# ── ARROWS ──────────────────────────────────────────────────────────
# Raw -> Load (top)
draw_arrow(ax, col_raw + box_w, row_top + box_h/2,
           col_load, row_top + box_h/2)
# Raw -> Load (bottom)
draw_arrow(ax, col_raw + box_w, row_bot + box_h/2,
           col_load, row_bot + box_h/2)

# Load -> Cache (top)
draw_arrow(ax, col_load + box_w, row_top + box_h/2,
           col_cache, row_top + box_h/2)
# Load -> Cache (bottom)
draw_arrow(ax, col_load + box_w, row_bot + box_h/2,
           col_cache, row_bot + box_h/2)

# Cache top -> Merge (converging)
merge_mid_y = row_mid + (box_h + 0.4) / 2
draw_arrow(ax, col_cache + box_w, row_top + box_h/2,
           col_merge, merge_mid_y + 0.35,
           connectionstyle="arc3,rad=-0.12")
# Cache bottom -> Merge (converging)
draw_arrow(ax, col_cache + box_w, row_bot + box_h/2,
           col_merge, merge_mid_y - 0.35,
           connectionstyle="arc3,rad=0.12")

# Merge -> Split
draw_arrow(ax, col_merge + box_w, row_mid + (box_h+0.4)/2,
           col_split, row_mid + (box_h+0.4)/2)
# Split -> Norm
draw_arrow(ax, col_split + box_w, row_mid + (box_h+0.4)/2,
           col_norm, row_mid + (box_h+0.4)/2)
# Norm -> Model
draw_arrow(ax, col_norm + box_w, row_mid + (box_h+0.4)/2,
           col_model, row_mid + (box_h+0.4)/2)

# ── ET metrics detail box (bottom) ─────────────────────────────────
detail_x, detail_y, detail_w, detail_h = 1.0, 0.6, 9.2, 2.6
detail_box = FancyBboxPatch(
    (detail_x, detail_y), detail_w, detail_h,
    boxstyle="round,pad=0.15",
    facecolor=C_DETAIL, edgecolor="#CCCCCC",
    linewidth=1.2, alpha=0.9, zorder=1,
)
ax.add_patch(detail_box)

ax.text(detail_x + 0.3, detail_y + detail_h - 0.3,
        "5 Eye-Tracking Metrics (per word)", fontsize=10.5,
        fontweight="bold", color=C_TEXT, va="top", zorder=3)

metrics_info = [
    ("FFD", "First Fixation Duration (ms)"),
    ("GD",  "Gaze Duration (ms)"),
    ("GPT", "Go-Past Time (ms)"),
    ("TRT", "Total Reading Time (ms)"),
    ("nFix", "Number of fixations"),
]
for i, (abbr, desc) in enumerate(metrics_info):
    yy = detail_y + detail_h - 0.75 - i * 0.38
    ax.text(detail_x + 0.5, yy, f"{abbr}", fontsize=9.5,
            fontweight="bold", color="#3A6B8C", family="monospace", zorder=3)
    ax.text(detail_x + 1.3, yy, f"— {desc}", fontsize=9.5,
            color=C_TEXT, zorder=3)

# ── Prompt structure detail box (bottom right) ─────────────────────
prompt_x, prompt_y, prompt_w, prompt_h = 11.0, 0.6, 10.0, 2.6
prompt_box = FancyBboxPatch(
    (prompt_x, prompt_y), prompt_w, prompt_h,
    boxstyle="round,pad=0.15",
    facecolor=C_DETAIL, edgecolor="#CCCCCC",
    linewidth=1.2, alpha=0.9, zorder=1,
)
ax.add_patch(prompt_box)

ax.text(prompt_x + 0.3, prompt_y + prompt_h - 0.3,
        "Final Prompt Structure (PromptWithAnswer)", fontsize=10.5,
        fontweight="bold", color=C_TEXT, va="top", zorder=3)

prompt_lines = [
    ('pre_prompt:',  '"You are given word-level eye-tracking measurements..."'),
    ('TTS #1:',      '"This is the First Fixation Duration..." + [z-scored series]'),
    ('TTS #2:',      '"This is the Gaze Duration..." + [z-scored series]'),
    ('  ...',        '(5 channels total: FFD, GD, GPT, TRT, nFixations)'),
    ('post_prompt:', '"Possible labels: Normal Reading, Task-Specific Reading.\\nAnswer:"'),
]
for i, (lbl, desc) in enumerate(prompt_lines):
    yy = prompt_y + prompt_h - 0.75 - i * 0.38
    ax.text(prompt_x + 0.4, yy, lbl, fontsize=9,
            fontweight="bold", color="#3A6B8C", family="monospace", zorder=3)
    ax.text(prompt_x + 2.3, yy, desc, fontsize=8.5,
            color=C_TEXT, zorder=3)

# ── Subject split detail annotation ────────────────────────────────
# ZuCo 1.0 split
annot_x = col_split + 0.1
ax.text(annot_x, row_mid - 0.2, "ZuCo 1.0: 8 train / 2 val / 2 test subjects",
        fontsize=8.5, color="#555555", style="italic", zorder=3)
ax.text(annot_x, row_mid - 0.55, "ZuCo 2.0: 10 train / 3 val / 3 test subjects",
        fontsize=8.5, color="#555555", style="italic", zorder=3)

# ── Title ───────────────────────────────────────────────────────────
ax.text(11, 9.55, "Data Processing Pipeline: ZuCo Eye-Tracking → OpenTSLM Model Input",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_TEXT)

# ── Save ────────────────────────────────────────────────────────────
fig.tight_layout(pad=0.5)
out = "/home/wangni/notion-figures/zuco/fig_007.png"
fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
