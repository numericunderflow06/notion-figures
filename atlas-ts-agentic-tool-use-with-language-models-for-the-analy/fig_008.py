"""
Figure 008: Tool Pool Taxonomy for ATLAS-T
Four column-cards showing the families of tools in ATLAS-T's tool pool.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# ---- Configure figure ----
fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#fbfbfd')
ax.set_facecolor('#fbfbfd')

# ---- Title ----
ax.text(8, 8.55, 'Tool Pool Taxonomy',
        ha='center', va='center',
        fontsize=20, fontweight='bold', color='#1a1a2e',
        family='sans-serif')
ax.text(8, 8.10,
        'Four families of pretrained specialists exposed as discrete actions to ATLAS-T',
        ha='center', va='center',
        fontsize=11, color='#555570', style='italic',
        family='sans-serif')

# ---- Palette (four distinct hues) ----
families = [
    {
        'name': 'Frozen Encoder\n+ Trainable Head',
        'header': '#3b6ea5',
        'body':   '#eaf1f9',
        'border': '#2c5282',
        'icon':   'FE',
        'badges': ['frozen trunk', 'lightweight'],
        'subsections': [
            ('Encoders',
             ['Chronos-2  (d = 768)',
              'MOMENT-1-large  (385 M, d = 1024)',
              'Random-Transformer  (ablation)']),
            ('Heads (× 5 generic)',
             ['mlp', 'deep_mlp', 'gated',
              'multiscale', 'attention']),
        ],
    },
    {
        'name': 'End-to-End\nSpecialists',
        'header': '#2f8f6e',
        'body':   '#e8f4ee',
        'border': '#1f6b4f',
        'icon':   'E2E',
        'badges': ['trained per task'],
        'subsections': [
            ('Models',
             ['InceptionTime',
              '   ~ 474 K params',
              'MiniRocket',
              '   84 kernels × 20 dilations',
              '   + Ridge classifier']),
        ],
    },
    {
        'name': 'Domain\nFoundation Models',
        'header': '#c46a1e',
        'body':   '#fbf0e3',
        'border': '#9c4f10',
        'icon':   'DFM',
        'badges': ['domain-specific', 'pretrained'],
        'subsections': [
            ('Models',
             ['HARNet5     ~ 10 M',
              'U-Sleep',
              'xresnet1d101',
              'BIOT',
              'ConvTran']),
        ],
    },
    {
        'name': 'ECG-QA\nPer-Template Heads',
        'header': '#8b3a8f',
        'body':   '#f3e8f4',
        'border': '#6b2870',
        'icon':   'QA',
        'badges': ['42 templates', 'shared trunk'],
        'subsections': [
            ('Structure',
             ['42 × mlp heads',
              'one head per ECG-QA',
              '   question template',
              'Shared frozen trunk',
              '   feeds every head']),
        ],
    },
]

# ---- Card geometry ----
card_left = 0.45
card_right = 15.55
gap = 0.30
n = len(families)
card_w = (card_right - card_left - (n - 1) * gap) / n
card_top = 7.55
card_bottom = 1.45
header_h = 1.05

for i, fam in enumerate(families):
    x0 = card_left + i * (card_w + gap)
    x1 = x0 + card_w
    yc_bot = card_bottom
    yc_top = card_top

    # Card body (rounded background)
    body = FancyBboxPatch(
        (x0, yc_bot), card_w, yc_top - yc_bot,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.0,
        edgecolor=fam['border'],
        facecolor=fam['body'],
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(body)

    # Colored header strip
    header = FancyBboxPatch(
        (x0, yc_top - header_h), card_w, header_h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=0,
        facecolor=fam['header'],
        zorder=2,
    )
    ax.add_patch(header)
    # Square off the bottom of the header so it joins the card cleanly
    header_cap = Rectangle(
        (x0 + 0.03, yc_top - header_h),
        card_w - 0.06, 0.18,
        linewidth=0,
        facecolor=fam['header'],
        zorder=2,
    )
    ax.add_patch(header_cap)

    # Family index pill (top-left of header)
    pill_cx = x0 + 0.42
    pill_cy = yc_top - 0.32
    pill = mpatches.FancyBboxPatch(
        (pill_cx - 0.30, pill_cy - 0.18),
        0.60, 0.36,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0,
        facecolor='white',
        alpha=0.92,
        zorder=3,
    )
    ax.add_patch(pill)
    ax.text(pill_cx, pill_cy, f'{i+1}',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=fam['header'],
            family='sans-serif',
            zorder=4)

    # Family name (header text)
    ax.text(x0 + card_w / 2, yc_top - 0.55, fam['name'],
            ha='center', va='center',
            fontsize=13, fontweight='bold',
            color='white', family='sans-serif',
            zorder=4)

    # Metadata badges (along bottom of header, on header strip)
    badge_y = yc_top - header_h + 0.20
    total_w = 0
    badge_widths = []
    for b in fam['badges']:
        w = 0.10 + 0.085 * len(b)
        badge_widths.append(w)
        total_w += w
    total_w += 0.18 * (len(fam['badges']) - 1)
    bx = x0 + card_w / 2 - total_w / 2
    for b, w in zip(fam['badges'], badge_widths):
        badge = mpatches.FancyBboxPatch(
            (bx, badge_y - 0.13), w, 0.26,
            boxstyle="round,pad=0.01,rounding_size=0.08",
            linewidth=0.8,
            edgecolor='white',
            facecolor=fam['header'],
            alpha=0.55,
            zorder=4,
        )
        ax.add_patch(badge)
        ax.text(bx + w / 2, badge_y, b,
                ha='center', va='center',
                fontsize=8.5, color='white',
                family='sans-serif',
                zorder=5)
        bx += w + 0.18

    # ---- Body content: subsections with bullets ----
    cursor_y = yc_top - header_h - 0.40
    for sub_title, items in fam['subsections']:
        # subsection header
        ax.text(x0 + 0.30, cursor_y, sub_title,
                ha='left', va='top',
                fontsize=10.5, fontweight='bold',
                color=fam['border'],
                family='sans-serif',
                zorder=3)
        cursor_y -= 0.36
        # underline under subsection title
        ax.plot([x0 + 0.30, x1 - 0.30],
                [cursor_y + 0.18, cursor_y + 0.18],
                color=fam['header'], linewidth=0.6,
                alpha=0.4, zorder=3)

        for item in items:
            indent = 0
            text = item
            if text.startswith('   '):
                indent = 0.30
                text = text.lstrip()
            # bullet
            if not item.startswith('   '):
                ax.plot(x0 + 0.36 + indent, cursor_y + 0.04,
                        marker='o', markersize=3.2,
                        color=fam['header'], zorder=3)
                tx_x = x0 + 0.55 + indent
            else:
                tx_x = x0 + 0.55 + indent
            ax.text(tx_x, cursor_y, text,
                    ha='left', va='center',
                    fontsize=10, color='#222230',
                    family='sans-serif',
                    zorder=3)
            cursor_y -= 0.32
        cursor_y -= 0.18  # gap between subsections

# ---- Connector bracket: "All exposed as discrete actions to ATLAS-T" ----
bracket_y = 1.18
ax.plot([card_left + 0.10, card_right - 0.10],
        [bracket_y, bracket_y],
        color='#777788', linewidth=1.2, zorder=2)
for i in range(n):
    x0 = card_left + i * (card_w + gap) + card_w / 2
    ax.plot([x0, x0], [card_bottom, bracket_y],
            color='#777788', linewidth=1.0,
            linestyle=(0, (3, 2)), alpha=0.7, zorder=2)
    ax.plot(x0, bracket_y, marker='v', markersize=6,
            color='#444455', zorder=3)

# Central ATLAS-T node
atlas_box = FancyBboxPatch(
    (6.6, 0.45), 2.8, 0.55,
    boxstyle="round,pad=0.02,rounding_size=0.12",
    linewidth=1.4,
    edgecolor='#1a1a2e',
    facecolor='#1a1a2e',
    zorder=4,
)
ax.add_patch(atlas_box)
ax.text(8.0, 0.725, 'ATLAS-T  (tool user)',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color='white', family='sans-serif',
        zorder=5)

# Tie atlas-t box to bracket
ax.plot([8.0, 8.0], [1.0, bracket_y],
        color='#444455', linewidth=1.2, zorder=3)

# ---- Bottom caption ----
caption = (
    'A new tool drops in as one extra discrete action for ATLAS-T '
    'without retraining ATLAS-A or the frozen LLM backbone.'
)
ax.text(8, 0.16, caption,
        ha='center', va='center',
        fontsize=10.5, style='italic',
        color='#444455',
        family='sans-serif')

# ---- Save ----
out_path = '/home/wangni/notion-figures/atlas-ts-agentic-tool-use-with-language-models-for-the-analy/fig_008.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight',
            facecolor='#fbfbfd', edgecolor='none')
plt.close(fig)
print(f'Saved: {out_path}')
