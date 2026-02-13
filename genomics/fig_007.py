import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(18, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('white')

# Color palette
C_ROOT = '#2D3748'
C_OPENTSLM = '#2B6CB0'
C_GENOMIC = '#C05621'
C_NEW_FILE = '#276749'
C_DIR = '#6B46C1'
C_FILE = '#718096'
C_OPENTSLM_BG = '#EBF8FF'
C_GENOMIC_BG = '#FFFAF0'
C_NEW_BG = '#F0FFF4'

FT_COLORS = {
    '.py': '#3182CE', '.yaml': '#DD6B20', '.sub': '#805AD5',
    '.sh': '#38A169', '.md': '#E53E3E', '.log': '#718096',
    '.png': '#D69E2E', '.json': '#00B5D8', '.txt': '#ED64A6',
    'other': '#A0AEC0',
}

def draw_box(x, y, w, h, text, ec='#4A5568', bg='white',
             fs=10, bold=False, tc=None, lw=1.5):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25",
                         facecolor=bg, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + 0.5, y + h/2, text, fontsize=fs,
            color=tc or ec, fontweight=weight, va='center', zorder=3)

# ============================================================
# TITLE
# ============================================================
ax.text(50, 99, 'Project Repository Structure',
        fontsize=20, fontweight='bold', color='#1A202C',
        ha='center', va='top', zorder=5)
ax.text(50, 96.8, 'OpenTSLM-Genomics  |  555 files  ·  135 directories',
        fontsize=12, color='#4A5568', ha='center', va='top', zorder=5)

# ============================================================
# TOP TREE: /genomics/ -> about.txt, OpenTSLM-Genomics/ -> docs
# ============================================================
# Root box
rx, ry, rw, rh = 2, 92, 15, 3
draw_box(rx, ry, rw, rh, '/genomics/', C_ROOT, '#E2E8F0',
         fs=12, bold=True, tc='#1A202C')

trunk_x = rx + 2  # left-aligned trunk

# Vertical trunk from root down
ax.plot([trunk_x, trunk_x], [ry, 69], color='#A0AEC0', lw=1.8, zorder=1)

# about.txt
ab_mid_y = 90
ax.plot([trunk_x, trunk_x + 3], [ab_mid_y, ab_mid_y],
        color='#A0AEC0', lw=1.5, zorder=1)
draw_box(trunk_x + 3, ab_mid_y - 1.1, 10, 2.2, 'about.txt',
         C_FILE, '#F7FAFC', fs=9.5)

# OpenTSLM-Genomics/
otg_mid_y = 87
ax.plot([trunk_x, trunk_x + 3], [otg_mid_y, otg_mid_y],
        color='#A0AEC0', lw=1.5, zorder=1)
draw_box(trunk_x + 3, otg_mid_y - 1.1, 19, 2.2, 'OpenTSLM-Genomics/',
         C_ROOT, '#E2E8F0', fs=10, bold=True, tc='#1A202C')

# Docs under OpenTSLM-Genomics
docs = [
    ('FINAL_SUMMARY.md', '.md'),
    ('GENOMICS_OPENTSLM_README.md', '.md'),
    ('SETUP_GUIDE.md', '.md'),
    ('setup_env.sh', '.sh'),
    ('activate_env.sh', '.sh'),
]
doc_trunk_x = trunk_x + 5
doc_x = trunk_x + 7
doc_sp = 2.3
doc_top_y = 84.5

ax.plot([doc_trunk_x, doc_trunk_x],
        [doc_top_y, doc_top_y - len(docs) * doc_sp + 1],
        color='#CBD5E0', lw=1.2, zorder=1)

for i, (name, ext) in enumerate(docs):
    dy = doc_top_y - (i + 0.5) * doc_sp
    ax.plot([doc_trunk_x, doc_x], [dy, dy],
            color='#CBD5E0', lw=1.2, zorder=1)
    c = FT_COLORS.get(ext, C_FILE)
    draw_box(doc_x, dy - 0.9, 21, 1.8, name, c, '#F7FAFC', fs=8.5, lw=1.0)

# ============================================================
# TWO BRANCH CONNECTORS down to panels
# ============================================================
# Branch point below the doc list
branch_y = 69
left_target_x = 24.5
right_target_x = 75
panel_top = 62

# Left branch (blue - OpenTSLM)
ax.plot([trunk_x, left_target_x], [branch_y, branch_y],
        color=C_OPENTSLM, lw=2.5, zorder=1)
ax.plot([left_target_x, left_target_x], [branch_y, panel_top],
        color=C_OPENTSLM, lw=2.5, zorder=1)
# Arrow
ax.annotate('', xy=(left_target_x, panel_top + 0.1),
            xytext=(left_target_x, panel_top + 2.5),
            arrowprops=dict(arrowstyle='->', color=C_OPENTSLM, lw=2.2))

# Right branch (orange - genomic_llm)
ax.plot([trunk_x, right_target_x], [branch_y, branch_y],
        color=C_GENOMIC, lw=2.5, zorder=1)
ax.plot([right_target_x, right_target_x], [branch_y, panel_top],
        color=C_GENOMIC, lw=2.5, zorder=1)
ax.annotate('', xy=(right_target_x, panel_top + 0.1),
            xytext=(right_target_x, panel_top + 2.5),
            arrowprops=dict(arrowstyle='->', color=C_GENOMIC, lw=2.2))

# Branch labels along the horizontal lines
ax.text((trunk_x + left_target_x) / 2, branch_y + 0.8, 'OpenTSLM/',
        fontsize=10, color=C_OPENTSLM, fontweight='bold',
        ha='center', va='bottom', zorder=3,
        bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.8))
ax.text((trunk_x + right_target_x) / 2, branch_y + 0.8,
        'genomic_llm_architectures/',
        fontsize=10, color=C_GENOMIC, fontweight='bold',
        ha='center', va='bottom', zorder=3,
        bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.8))

# ============================================================
# LEFT PANEL: OpenTSLM/ (Adapted Flamingo Framework)
# ============================================================
lp_x, lp_y = 2, 10
lp_w = 46
lp_h = panel_top - lp_y

left_bg = FancyBboxPatch((lp_x, lp_y), lp_w, lp_h,
                          boxstyle="round,pad=0.5",
                          facecolor=C_OPENTSLM_BG, edgecolor=C_OPENTSLM,
                          linewidth=2.5, zorder=0, alpha=0.5)
ax.add_patch(left_bg)

ax.text(lp_x + lp_w/2, panel_top - 1.2,
        'OpenTSLM/', fontsize=15, fontweight='bold',
        color=C_OPENTSLM, ha='center', va='top', zorder=3)
ax.text(lp_x + lp_w/2, panel_top - 4,
        'Adapted Flamingo Framework', fontsize=10,
        color='#4A5568', ha='center', va='top', zorder=3, style='italic')

new_files = [
    ('train_genomics_lrb.py', 'Training script', True),
    ('test_genomics_dataset.py', 'Test script', True),
    ('download_lrb_dataset.py', 'Dataset downloader', True),
    ('demo_dna_encoding.py', 'Encoding demo', True),
    ('requirements.txt', '21 packages', False),
]

ny = 53.5
item_sp = 3.3
for name, desc, is_new in new_files:
    bg = C_NEW_BG if is_new else '#F7FAFC'
    ec = C_NEW_FILE if is_new else C_FILE
    draw_box(6, ny, 26, 2.5, name, ec, bg, fs=9.5, bold=is_new, lw=1.5)
    ax.text(33, ny + 1.25, desc, fontsize=8.5, color='#718096',
            va='center', zorder=3)
    if is_new:
        ax.plot(4.8, ny + 1.25, '*', color=C_NEW_FILE, markersize=13,
                markeredgewidth=0, zorder=3)
    ny -= item_sp

# Key NEW directory
dir_y = ny - 1
draw_box(6, dir_y, 38, 3,
         'src/time_series_datasets/genomics_lrb/',
         C_NEW_FILE, C_NEW_BG, fs=10, bold=True, lw=1.8)
ax.plot(4.8, dir_y + 1.5, '*', color=C_NEW_FILE, markersize=13,
        markeredgewidth=0, zorder=3)
ax.text(6.5, dir_y - 1.3,
        'NEW: Genomics dataset module (DNA to time series encoding)',
        fontsize=9, color=C_NEW_FILE, va='top', style='italic', zorder=3)

ax.text(6.5, dir_y - 3.5,
        '[inherited OpenTSLM modules: models, configs, utils, assets]',
        fontsize=9, color='#A0AEC0', va='top', style='italic', zorder=3)

# ============================================================
# RIGHT PANEL: genomic_llm_architectures/
# ============================================================
rp_x, rp_y = 52, 10
rp_w = 46
rp_h = panel_top - rp_y

right_bg = FancyBboxPatch((rp_x, rp_y), rp_w, rp_h,
                           boxstyle="round,pad=0.5",
                           facecolor=C_GENOMIC_BG, edgecolor=C_GENOMIC,
                           linewidth=2.5, zorder=0, alpha=0.5)
ax.add_patch(right_bg)

ax.text(rp_x + rp_w/2, panel_top - 1.2,
        'genomic_llm_architectures/', fontsize=15, fontweight='bold',
        color=C_GENOMIC, ha='center', va='top', zorder=3)
ax.text(rp_x + rp_w/2, panel_top - 4,
        'Baseline Architecture Comparison', fontsize=10,
        color='#4A5568', ha='center', va='top', zorder=3, style='italic')

right_items = [
    ('train.py', 'PyTorch Lightning training', False),
    ('validate_lrb_models.py', 'Multi-model validation', False),
    ('src/', 'Source code', True),
    ('configs/', '129 YAML Hydra configs', True),
    ('condor_scripts/', 'HTCondor job scripts', True),
    ('slurm_scripts/', 'SLURM job scripts', True),
    ('lrb_validation_results/', 'Condor submission files', True),
    ('outputs_lrb_resnet1d/', 'ResNet1D training outputs', True),
    ('plotting/', 'Metrics extraction & plotting', True),
]

ry_pos = 53.5
for name, desc, is_dir in right_items:
    ec = C_DIR if is_dir else C_FILE
    bg = '#FAF5FF' if is_dir else '#F7FAFC'
    draw_box(55, ry_pos, 23, 2.5, name, ec, bg, fs=9.5, bold=is_dir, lw=1.5)
    ax.text(79.5, ry_pos + 1.25, desc, fontsize=8.5, color='#718096',
            va='center', zorder=3)
    ry_pos -= item_sp

# ============================================================
# FILE COUNT STACKED BAR
# ============================================================
ax.text(50, 8.5, 'File Counts by Extension (555 total)',
        fontsize=12, fontweight='bold', color='#2D3748',
        ha='center', va='bottom', zorder=5)

file_counts = [
    ('.py', 194), ('.yaml', 129), ('.sub', 80), ('.sh', 61),
    ('.md', 28), ('.log', 17), ('.png', 13), ('.json', 8),
    ('.txt', 8), ('other', 6),
]
total = sum(c for _, c in file_counts)
bar_x0, bar_y0, bar_w, bar_h = 5, 5, 90, 2.8

cum = bar_x0
for ext, count in file_counts:
    seg_w = (count / total) * bar_w
    color = FT_COLORS[ext]
    rect = FancyBboxPatch((cum, bar_y0), seg_w, bar_h,
                           boxstyle="square,pad=0",
                           facecolor=color, edgecolor='white',
                           linewidth=0.8, zorder=4)
    ax.add_patch(rect)
    if seg_w > 5:
        ax.text(cum + seg_w/2, bar_y0 + bar_h/2,
                f'{ext}  {count}', fontsize=8.5, color='white',
                ha='center', va='center', fontweight='bold', zorder=5)
    elif seg_w > 3:
        ax.text(cum + seg_w/2, bar_y0 + bar_h/2,
                f'{ext}\n{count}', fontsize=7.5, color='white',
                ha='center', va='center', fontweight='bold', zorder=5)
    elif seg_w > 1.8:
        ax.text(cum + seg_w/2, bar_y0 + bar_h/2,
                f'{count}', fontsize=7, color='white',
                ha='center', va='center', fontweight='bold', zorder=5)
    cum += seg_w

# ============================================================
# LEGEND (bottom)
# ============================================================
leg_y = 2.5
legend_data = [
    (5, 'NEW (genomics-specific)', C_NEW_FILE, '*', 13),
    (24, 'Directory', C_DIR, 's', 8),
    (36, 'File', C_FILE, 's', 8),
]
for lx, label, color, marker, ms in legend_data:
    ax.plot(lx, leg_y, marker, color=color, markersize=ms,
            markeredgewidth=0, zorder=5)
    ax.text(lx + 1.2, leg_y, label, fontsize=9, color='#4A5568',
            va='center', zorder=5)

# Color legend for bar
ftl_x = 50
ax.text(ftl_x, leg_y, 'Colors:', fontsize=8.5, color='#4A5568',
        va='center', fontweight='bold', zorder=5)
ftl_cx = ftl_x + 5
for ext, count in file_counts:
    color = FT_COLORS[ext]
    ax.plot(ftl_cx, leg_y, 's', color=color, markersize=7, zorder=5)
    ax.text(ftl_cx + 0.8, leg_y, ext, fontsize=7.5, color='#4A5568',
            va='center', zorder=5)
    ftl_cx += len(ext) * 0.55 + 2.5

# Stats badge top-right
ax.text(95, 97, '555 files\n135 dirs',
        fontsize=10, color='#2D3748', fontweight='bold',
        va='top', ha='center', zorder=5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EDF2F7',
                  edgecolor='#CBD5E0', linewidth=1))

plt.tight_layout(pad=0.5)
plt.savefig('/home/wangni/notion-figures/genomics/fig_007.png',
            dpi=200, facecolor='white', bbox_inches='tight')
plt.close()
print("Figure saved to /home/wangni/notion-figures/genomics/fig_007.png")
