"""Generate fig_005: GRPO Training Step sequence diagram."""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
})

OUT = ('/home/wangni/notion-figures/'
       'autoreward-symbolic-regression-for-adaptive-reward-discovery/fig_005.png')

# ---------- canvas ----------
fig, ax = plt.subplots(figsize=(13.5, 10.5))
ax.set_xlim(-0.2, 16.0)
ax.set_ylim(0.0, 14.4)
ax.set_aspect('equal')
ax.axis('off')

# ---------- colors ----------
LANE_FILL = '#e8eef7'
LANE_EDGE = '#3a5d8a'
LANE_TEXT = '#1f3a5f'
LIFELINE  = '#a0a0a0'
ARROW     = '#2b2b2b'
ARROW_RET = '#a23b3b'   # red for gradient return
ACT_FILL  = '#ffd58c'
ACT_EDGE  = '#8a5d2a'
EQ_COLOR  = '#a23b3b'
NOTE_FILL = '#fff4d9'

# ---------- lanes ----------
lanes = [
    ("Prompt\nbatch",            1.4),
    ("Policy $\\pi_\\theta$\n(LoRA only)", 4.6),
    ("Heuristic\nRubric $r(\\cdot)$",   7.8),
    ("Reward\nFormula $R^*$", 11.0),
    ("GRPO\nAdvantage",            14.2),
]

y_top, y_bot = 13.4, 0.6
for label, x in lanes:
    box = FancyBboxPatch((x - 1.25, y_top - 0.55), 2.5, 1.1,
                         boxstyle="round,pad=0.04",
                         facecolor=LANE_FILL, edgecolor=LANE_EDGE, linewidth=1.1)
    ax.add_patch(box)
    ax.text(x, y_top, label, ha='center', va='center',
            fontsize=10.5, weight='bold', color=LANE_TEXT)
    ax.plot([x, x], [y_top - 0.55, y_bot], color=LIFELINE,
            linestyle=(0, (4, 3)), linewidth=0.7)


# ---------- helpers ----------
def msg(x0, x1, y, top_label, eq=None, color=ARROW, lw=1.0):
    """Horizontal sequence message."""
    arr = FancyArrowPatch((x0, y), (x1, y),
                          arrowstyle='-|>', mutation_scale=11,
                          color=color, linewidth=lw)
    ax.add_patch(arr)
    mx = (x0 + x1) / 2.0
    ax.text(mx, y + 0.18, top_label, ha='center', va='bottom',
            fontsize=9.0, family='monospace', color='#1a1a1a')
    if eq:
        ax.text(mx, y - 0.18, eq, ha='center', va='top',
                fontsize=8.5, color=EQ_COLOR, style='italic')


def self_action(x, y_high, y_low, note, note_offset=1.05):
    """Self-loop arrow on a lifeline with a side note."""
    # activation bar
    bar = Rectangle((x - 0.13, y_low), 0.26, y_high - y_low,
                    facecolor=ACT_FILL, edgecolor=ACT_EDGE,
                    linewidth=0.7, alpha=0.9)
    ax.add_patch(bar)
    # self-loop arrow: out to the right then back
    arr = FancyArrowPatch((x + 0.13, y_high - 0.15),
                          (x + 0.13, y_low + 0.15),
                          connectionstyle="arc3,rad=-0.55",
                          arrowstyle='-|>', mutation_scale=9,
                          color=ARROW, linewidth=0.9)
    ax.add_patch(arr)
    ax.text(x + note_offset, (y_high + y_low) / 2.0, note,
            ha='left', va='center', fontsize=9.0,
            family='monospace', color='#1a1a1a')


# ---------- sequence (top -> bottom) ----------
# 1. Prompt -> Policy: sample a prompt
msg(1.4, 4.6, 12.2, "sample  $x_i \\sim \\mathcal{D}$")

# 2. Policy self-action: generate G=3 completions
self_action(
    x=4.6, y_high=11.4, y_low=10.2,
    note=("generate $G=3$ completions  $\\{y_1,y_2,y_3\\}$\n"
          "temp = 0.7,  max_new_tok = 384"),
    note_offset=0.55,
)

# 3. Policy -> Rubric: send completions
msg(4.6, 7.8, 9.4, "for $i = 1..G$:  send  $y_i$")

# 4. Rubric self-action: per-dimension heuristic scoring
self_action(
    x=7.8, y_high=8.6, y_low=7.4,
    note=("score  $r(y_i)\\in[0,1]^8$\n"
          "(8 TS-CoTQual dims, heuristics)"),
    note_offset=0.55,
)

# 5. Rubric -> Reward Formula: rubric vectors
msg(7.8, 11.0, 6.6, "$r(y_i)$  (8-dim rubric)", eq="Eq. 4")

# 6. Reward Formula self-action: apply discovered R*
self_action(
    x=11.0, y_high=5.8, y_low=4.6,
    note=("$R_i = R^{*}(\\,r(y_i)\\,)$\n"
          "discovered symbolic formula"),
    note_offset=0.55,
)

# 7. Reward Formula -> GRPO: group of rewards
msg(11.0, 14.2, 3.8, "$\\{R_1, R_2, R_3\\}$", eq="Eq. 5")

# 8. GRPO self-action: group-relative advantage
self_action(
    x=14.2, y_high=3.0, y_low=1.8,
    note=("$A_i = \\dfrac{R_i - \\mu_G}{\\sigma_G + \\varepsilon}$\n"
          "group-relative norm."),
    note_offset=-3.4,   # place note to the LEFT (closer to figure center)
)

# 9. GRPO -> Policy: REINFORCE-style gradient (return arrow)
msg(14.2, 4.6, 1.0,
    "$\\nabla_\\theta J \\approx \\mathbb{E}[\\,A_i\\,\\nabla_\\theta \\log \\pi_\\theta(y_i \\mid x_i)\\,]$  (REINFORCE)",
    eq="Eq. 6", color=ARROW_RET, lw=1.2)

# 10. Policy update note (sits under Policy lane)
ax.text(
    4.6, 0.30,
    "AdamW update on LoRA params   (lr = 1e-5,  grad-clip = 1.0)",
    ha='center', va='center', fontsize=9.5, family='monospace',
    color='#1a1a1a',
    bbox=dict(boxstyle='round,pad=0.30', facecolor=NOTE_FILL,
              edgecolor=ACT_EDGE, linewidth=0.8),
)

# ---------- title-less; small caption-like footer omitted (caption lives in LaTeX) ----------

plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}")
