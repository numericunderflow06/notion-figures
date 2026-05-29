"""
fig_002: Data Pipeline — CSVs to Training Samples
Shows the flow from raw CSVs through join, cleaning, grouping, splitting,
and task expansion to the final per-task QADataset samples.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Color palette
C_INPUT = "#4A6FA5"      # blue - raw inputs
C_PROCESS = "#6B8E23"    # olive - processing steps
C_RECORDS = "#C97B3B"    # warm orange - system records
C_TRAIN = "#2E7D5C"      # green - train
C_VAL = "#D4A52A"        # gold - val
C_TEST = "#A83232"       # red - test
C_TEXT = "#1A1A1A"
C_ARROW = "#444444"
C_BG_NOTE = "#F4F0E8"    # soft cream for annotation boxes

fig, ax = plt.subplots(figsize=(20, 11), dpi=200)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# ---------- Title ----------
ax.text(10, 10.55, "Data Pipeline: CSVs to Training Samples",
        ha="center", va="center", fontsize=17, fontweight="bold", color=C_TEXT)
ax.text(10, 10.15,
        "bosch_heatpump_loader.py  +  BoschHeatPumpQADataset._expand_by_task",
        ha="center", va="center", fontsize=10.5, style="italic", color="#555555")


def box(x, y, w, h, label, sub=None, color=C_PROCESS, fc=None, fontsize=10.5,
        sub_fontsize=9, text_color="white"):
    """Draw a rounded box with a title and optional subtitle."""
    if fc is None:
        fc = color
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=1.4, edgecolor=color, facecolor=fc)
    ax.add_patch(patch)
    if sub is None:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color)
    else:
        ax.text(x + w / 2, y + h * 0.66, label,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color)
        ax.text(x + w / 2, y + h * 0.30, sub,
                ha="center", va="center",
                fontsize=sub_fontsize, color=text_color)


def arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.22),
          label_color=C_TEXT, label_fontsize=9.5, lw=1.7, color=C_ARROW):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label is not None:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=label_fontsize, color=label_color,
                fontweight="bold")


def note(x, y, w, h, text, fontsize=8.5):
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            linewidth=0.9, edgecolor="#B5AC95",
                            facecolor=C_BG_NOTE)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fontsize, color="#333333")


# ===== Stage 1: Two input CSVs =====
sx = 0.3
box(sx, 6.7, 2.6, 1.4, "df_feats_train.csv",
    sub="21,553 rows\n13 channels + sysid + date",
    color=C_INPUT, fc=C_INPUT)
box(sx, 4.6, 2.6, 1.4, "df_metadata_main_train.csv",
    sub="256 rows\nsysid + metadata + 2 comments",
    color=C_INPUT, fc=C_INPUT)
ax.text(sx + 1.3, 8.45, "Raw CSVs",
        ha="center", fontsize=10.5, fontweight="bold", color=C_INPUT)

# ===== Stage 2: Join =====
join_x = 4.0
box(join_x, 5.7, 2.0, 1.4, "_load_and_join",
    sub="validate\nsysid join",
    color=C_PROCESS, fc=C_PROCESS)

# Arrows from CSV boxes to Join
arrow(sx + 2.6, 7.4, join_x, 6.7,
      label="21,553 rows", label_offset=(0.05, 0.28),
      label_fontsize=8.5)
arrow(sx + 2.6, 5.3, join_x, 6.1,
      label="256 rows", label_offset=(0.05, -0.28),
      label_fontsize=8.5)

# ===== Stage 3: Cleaning =====
clean_x = 6.7
box(clean_x, 5.7, 2.1, 1.4, "_clean_features",
    sub="±inf → NaN\nffill + bfill + 0",
    color=C_PROCESS, fc=C_PROCESS)
arrow(join_x + 2.0, 6.4, clean_x, 6.4, label="joined df",
      label_offset=(0, 0.28), label_fontsize=8.5)

# ===== Stage 4: Group by sysid sorted by date =====
grp_x = 9.5
box(grp_x, 5.7, 2.0, 1.4, "_build_system_records",
    sub="group by sysid\nsorted by date",
    color=C_PROCESS, fc=C_PROCESS)
arrow(clean_x + 2.1, 6.4, grp_x, 6.4, label="clean df",
      label_offset=(0, 0.28), label_fontsize=8.5)

# ===== Stage 5: 256 system records =====
rec_x = 12.1
box(rec_x, 5.4, 2.2, 2.0, "256 system records",
    sub="each system:\n• 13 channel arrays\n• metadata\n• 2 gold comments",
    color=C_RECORDS, fc=C_RECORDS,
    fontsize=10.5, sub_fontsize=8.8)
arrow(grp_x + 2.0, 6.4, rec_x, 6.4, label="256 systems",
      label_offset=(0, 0.28), label_fontsize=8.5)

# ===== Stage 6: Per-system split =====
split_x = 14.9
# Split header
ax.text(split_x + 0.85, 8.45,
        "load_bosch_heatpump_splits",
        ha="center", fontsize=10, fontweight="bold", color=C_PROCESS)
ax.text(split_x + 0.85, 8.13,
        "seed=42  •  val_frac=0.15  •  test_frac=0.15",
        ha="center", fontsize=8.5, color="#555555")

# three colored split boxes
box(split_x, 7.0, 1.7, 0.9, "Train", sub="180 systems",
    color=C_TRAIN, fc=C_TRAIN, fontsize=11, sub_fontsize=9)
box(split_x, 5.95, 1.7, 0.9, "Val", sub="38 systems",
    color=C_VAL, fc=C_VAL, fontsize=11, sub_fontsize=9, text_color="#3A2D00")
box(split_x, 4.90, 1.7, 0.9, "Test", sub="38 systems",
    color=C_TEST, fc=C_TEST, fontsize=11, sub_fontsize=9)

# Arrows from system records to split
arrow(rec_x + 2.2, 6.8, split_x, 7.45,
      label="70%", label_offset=(0.18, 0.20),
      label_fontsize=8.5, label_color=C_TRAIN)
arrow(rec_x + 2.2, 6.4, split_x, 6.40,
      label="15%", label_offset=(0.18, 0.20),
      label_fontsize=8.5, label_color="#8A6F00")
arrow(rec_x + 2.2, 6.0, split_x, 5.35,
      label="15%", label_offset=(0.18, -0.22),
      label_fontsize=8.5, label_color=C_TEST)

# ===== Stage 7: Task expansion x2 =====
exp_x = 17.6
ax.text(exp_x + 1.0, 8.45,
        "_expand_by_task",
        ha="center", fontsize=10, fontweight="bold", color=C_PROCESS)
ax.text(exp_x + 1.0, 8.13,
        "x 2 tasks  (space heating + DHW)",
        ha="center", fontsize=8.5, color="#555555")

box(exp_x, 7.0, 2.0, 0.9, "Train samples", sub="360 (180 × 2)",
    color=C_TRAIN, fc=C_TRAIN, fontsize=10.5, sub_fontsize=9)
box(exp_x, 5.95, 2.0, 0.9, "Val samples", sub="76 (38 × 2)",
    color=C_VAL, fc=C_VAL, fontsize=10.5, sub_fontsize=9, text_color="#3A2D00")
box(exp_x, 4.90, 2.0, 0.9, "Test samples", sub="76 (38 × 2)",
    color=C_TEST, fc=C_TEST, fontsize=10.5, sub_fontsize=9)

arrow(split_x + 1.7, 7.45, exp_x, 7.45, label="×2",
      label_offset=(0, 0.22), label_fontsize=9, label_color=C_TRAIN)
arrow(split_x + 1.7, 6.40, exp_x, 6.40, label="×2",
      label_offset=(0, 0.22), label_fontsize=9, label_color="#8A6F00")
arrow(split_x + 1.7, 5.35, exp_x, 5.35, label="×2",
      label_offset=(0, 0.22), label_fontsize=9, label_color=C_TEST)

# ===== Bottom: annotations / notes per stage =====
note(0.3, 3.0, 5.7, 1.3,
     "Stage 1 → 2: join\n"
     "Inner-join feature time series (21,553 rows of\n"
     "per-day records) with per-system metadata (256\n"
     "rows) on sysid; validate every sysid has metadata.",
     fontsize=8.6)

note(6.3, 3.0, 5.1, 1.3,
     "Stage 3: cleaning per sysid\n"
     "Replace ±inf → NaN, then ffill + bfill + zero-fill\n"
     "inside each system group to handle missing\n"
     "channel readings without leakage across systems.",
     fontsize=8.6)

note(11.7, 3.0, 4.4, 1.3,
     "Stage 4 → 5: per-system records\n"
     "Sort each group by date, stack 13 channels into\n"
     "fixed-length arrays, attach metadata and the\n"
     "two German expert comments (gold labels).",
     fontsize=8.6)

note(16.4, 3.0, 3.3, 1.3,
     "Stage 6 → 7: split & expand\n"
     "Split SYSTEMS (not samples) with seed=42 to\n"
     "avoid leakage, then expand each system into\n"
     "one sample per commentary task.",
     fontsize=8.6)

# ===== Legend (very compact) =====
legend_handles = [
    mpatches.Patch(color=C_INPUT, label="Raw CSV input"),
    mpatches.Patch(color=C_PROCESS, label="Processing step"),
    mpatches.Patch(color=C_RECORDS, label="System records"),
    mpatches.Patch(color=C_TRAIN, label="Train"),
    mpatches.Patch(color=C_VAL, label="Val"),
    mpatches.Patch(color=C_TEST, label="Test"),
]
leg = ax.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, 0.005),
                ncol=6, frameon=False, fontsize=9.5,
                handlelength=1.4, handleheight=1.0, columnspacing=1.6)

# ===== Final totals strip =====
ax.text(10, 2.05,
        "Final dataset: 360 train  +  76 val  +  76 test  =  512 (system, task) samples",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#EEF2F5",
                  edgecolor="#B5BFC8", linewidth=1.0))

plt.tight_layout()
out = "/home/wangni/notion-figures/bosch-checkit-expert-commentary-generation-for-heat-pump-tim/fig_002.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}")
