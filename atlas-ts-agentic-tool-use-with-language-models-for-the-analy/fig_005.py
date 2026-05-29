"""
fig_005: Where the 25-Point Gain Lives — Improvement vs Strongest Baseline
Horizontal bar chart of accuracy delta per dataset (ATLAS-TS Llama-3.2-3B minus
strongest non-agent baseline).
"""
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---------------------------------------------------------------
# Sorted ascending by gain (smallest at top of chart, largest at bottom).
# matplotlib barh: position 0 -> bottom, increasing y -> upward, so we
# put the LARGEST gain at position 0 and the SMALLEST at the top.
datasets = ['ECG-QA-CoT', 'Sleep-CoT', 'HAR-CoT', 'TSQA']
deltas    = [25.25,        9.87,        6.76,      0.0]
from_vals = [46.25,        80.69,       72.20,     None]
to_vals   = [71.50,        90.56,       78.96,     None]
baselines = [
    'vs OpenTSLM-Flamingo (Llama-3.2-3B)',
    'vs OpenTSLM-Chronos2 (Gemma3-270M)',
    'vs OpenTSLM-Chronos2 (Llama-3.2-3B)',
    'baseline already saturated',
]

# --- Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.2))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')

ACCENT = '#2E86AB'        # consistent blue accent
ACCENT_EDGE = '#1a5680'
HEADLINE = '#C73E1D'      # warm red for headline call-out

y_pos = np.arange(len(datasets))
bars = ax.barh(
    y_pos, deltas,
    color=ACCENT, edgecolor=ACCENT_EDGE, linewidth=1.1,
    height=0.55, zorder=3,
)

# Highlight headline bar (ECG-QA-CoT, position 0)
bars[0].set_color('#1F6FA0')
bars[0].set_edgecolor('#0F4A75')
bars[0].set_linewidth(1.6)

# --- Annotations next to each bar --------------------------------------
for i, (delta, fv, tv, base) in enumerate(zip(deltas, from_vals, to_vals, baselines)):
    if delta > 0.5:
        main_txt = f'{fv:.2f}%  →  {tv:.2f}%     (+{delta:.2f} pp)'
        ax.text(
            delta + 0.7, i + 0.13,
            main_txt,
            va='center', ha='left',
            fontsize=11.5, fontweight='bold', color='#1a1a1a',
        )
        ax.text(
            delta + 0.7, i - 0.20,
            base,
            va='center', ha='left',
            fontsize=9.5, color='#666',
        )
    else:
        # TSQA case: saturated baseline
        ax.text(
            0.4, i + 0.13,
            '≈ 0 pp     (TSLM baseline already near ceiling)',
            va='center', ha='left',
            fontsize=11.5, fontstyle='italic', color='#444',
        )
        ax.text(
            0.4, i - 0.20,
            base,
            va='center', ha='left',
            fontsize=9.5, color='#777',
        )

# --- Axes & styling ----------------------------------------------------
ax.set_yticks(y_pos)
ax.set_yticklabels(datasets, fontsize=13, fontweight='bold')
ax.set_xlabel(
    'Accuracy improvement over strongest non-agent baseline  (percentage points)',
    fontsize=12, labelpad=10,
)
ax.set_xlim(0, 38)
ax.set_ylim(-0.6, len(datasets) - 0.4)

ax.set_title(
    'Where the 25-Point Gain Lives — Improvement vs Strongest Baseline',
    fontsize=14.5, fontweight='bold', pad=14, loc='left',
)

ax.grid(axis='x', linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)
for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#888')
ax.spines['bottom'].set_color('#888')
ax.tick_params(axis='x', colors='#444')
ax.tick_params(axis='y', colors='#222')

# --- Headline call-out arrow ------------------------------------------
# Place text in empty space above HAR-CoT row and arc down to the ECG-QA-CoT bar.
ax.annotate(
    'headline 25-pp gain',
    xy=(25.0, 0.30), xytext=(22.0, 2.70),
    fontsize=13, fontweight='bold', color=HEADLINE,
    ha='center', va='center',
    arrowprops=dict(
        arrowstyle='->,head_length=0.6,head_width=0.4',
        color=HEADLINE, lw=2.0,
        connectionstyle='arc3,rad=0.28',
    ),
)

# --- Caption ----------------------------------------------------------
fig.text(
    0.5, 0.015,
    'ATLAS-TS (Llama-3.2-3B) minus the best non-agent baseline per dataset.  '
    'Source: §6 Table 1.',
    ha='center', fontsize=10, color='#555', style='italic',
)

plt.tight_layout(rect=[0, 0.035, 1, 0.99])

OUT = '/home/wangni/notion-figures/atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_005.png'
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {OUT}')
