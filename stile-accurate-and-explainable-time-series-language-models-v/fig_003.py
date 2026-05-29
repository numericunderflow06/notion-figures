#!/usr/bin/env python3
"""fig_003: Two-Stage Training Pipeline (SFT -> STILE) for the STILE paper.

Stage 1 SFT pipeline (Chronos-2 -> Perceiver -> Gated x-attn -> LoRA-Llama)
produces an SFT-trained TSLM checkpoint that initialises BOTH the teacher and
student branches of Stage 2 STILE self-distillation. The teacher additionally
observes the predictor's y-hat (dashed pathway). A small g_theta adapter
generates the rationale opening tokens for the student and absorbs the
forward-KL distillation gradient.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.5), dpi=200)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ---- soft palette ----
    TSLM_EDGE    = '#4A82B4'   # steel blue
    TSLM_FILL    = '#D9E6F1'
    ADAPTER_EDGE = '#D97E2B'   # warm orange
    ADAPTER_FILL = '#FBE0BF'
    FROZEN_EDGE  = '#717B88'   # slate
    FROZEN_FILL  = '#E3E5E8'
    LOSS_EDGE    = '#A0522D'   # sienna
    LOSS_FILL    = '#FFEFDC'
    CKPT_FILL    = '#EAF1F7'
    YHAT_COLOR   = '#B22222'
    TEXT         = '#1A1A1A'
    MUTED        = '#4A4A4A'

    # =====================  TITLE  =====================
    ax.text(7.75, 8.05, 'Two-Stage Training Pipeline',
            ha='center', va='center', fontsize=18, fontweight='bold', color=TEXT)

    # ---- stage headers ----
    ax.text(3.5, 7.30, 'STAGE 1  —  Supervised Fine-Tuning (SFT)',
            ha='center', va='center', fontsize=12.5, fontweight='bold',
            color='#2F4F4F')
    ax.text(11.45, 7.30, 'STAGE 2  —  STILE Self-Distillation',
            ha='center', va='center', fontsize=12.5, fontweight='bold',
            color='#2F4F4F')

    # ---- subtle vertical divider between stages ----
    ax.plot([7.4, 7.4], [0.55, 7.05], linestyle=(0, (3, 3)),
            color='gray', alpha=0.35, lw=1.0)

    # =====================  STAGE 1  =====================
    y_s1 = 5.55
    box_w, box_h = 1.45, 0.95
    s1_modules = [
        (1.05, 'Frozen\nChronos-2', FROZEN_FILL, FROZEN_EDGE),
        (2.70, 'Perceiver\nResampler', TSLM_FILL, TSLM_EDGE),
        (4.35, 'Gated\nCross-Attn',   TSLM_FILL, TSLM_EDGE),
        (6.00, 'LoRA-Llama',          TSLM_FILL, TSLM_EDGE),
    ]
    for cx, lbl, fc, ec in s1_modules:
        box = FancyBboxPatch((cx - box_w / 2, y_s1 - box_h / 2),
                             box_w, box_h,
                             boxstyle='round,pad=0.02,rounding_size=0.10',
                             fc=fc, ec=ec, lw=1.8)
        ax.add_patch(box)
        ax.text(cx, y_s1, lbl, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=TEXT)

    # arrows between stage 1 boxes
    for i in range(len(s1_modules) - 1):
        x_s = s1_modules[i][0]     + box_w / 2 + 0.03
        x_e = s1_modules[i + 1][0] - box_w / 2 - 0.03
        ax.add_patch(FancyArrowPatch((x_s, y_s1), (x_e, y_s1),
                                     arrowstyle='-|>', mutation_scale=14,
                                     lw=1.5, color='#333'))

    # time-series x input
    ax.text(0.20, y_s1, 'time-\nseries x',
            ha='left', va='center', fontsize=9.5, color=MUTED, style='italic')
    ax.add_patch(FancyArrowPatch((0.32, y_s1),
                                  (s1_modules[0][0] - box_w / 2 - 0.03, y_s1),
                                  arrowstyle='-|>', mutation_scale=11,
                                  lw=1.2, color='#777'))

    # stage 1 hyperparameter caption
    cap1 = (r'AdamW,  $\mathrm{lr}=2\times 10^{-4}$,  linear + warmup,  bf16'
            '\n'
            r'seed 42,  200 epochs,  early-stop patience 5')
    ax.text(3.5, 3.85, cap1, ha='center', va='center',
            fontsize=10, style='italic', color='#222',
            bbox=dict(boxstyle='round,pad=0.42', fc='#F7F7F2',
                      ec='#999', lw=0.8))

    # ---- SFT-trained TSLM checkpoint ----
    ckpt_w, ckpt_h = 2.25, 0.90
    ckpt_cx, ckpt_cy = 3.5, 2.32
    ckpt_box = FancyBboxPatch((ckpt_cx - ckpt_w / 2, ckpt_cy - ckpt_h / 2),
                              ckpt_w, ckpt_h,
                              boxstyle='round,pad=0.04,rounding_size=0.12',
                              fc=CKPT_FILL, ec=TSLM_EDGE, lw=2.3)
    ax.add_patch(ckpt_box)
    ax.text(ckpt_cx, ckpt_cy + 0.16, 'SFT-trained',
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color=TSLM_EDGE)
    ax.text(ckpt_cx, ckpt_cy - 0.16, 'TSLM checkpoint',
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color=TSLM_EDGE)

    # arrow stage-1 modules -> SFT checkpoint
    ax.add_patch(FancyArrowPatch((ckpt_cx, y_s1 - box_h / 2 - 0.03),
                                  (ckpt_cx, ckpt_cy + ckpt_h / 2 + 0.03),
                                  arrowstyle='-|>', mutation_scale=14,
                                  lw=1.5, color='#333'))

    # arrow checkpoint -> stage 2
    ax.add_patch(FancyArrowPatch((ckpt_cx + ckpt_w / 2 + 0.05, ckpt_cy),
                                  (7.95, ckpt_cy),
                                  arrowstyle='-|>', mutation_scale=18,
                                  lw=2.0, color=TSLM_EDGE))
    ax.text(5.95, ckpt_cy + 0.30,
            'same checkpoint initialises\nboth teacher and student',
            ha='center', va='center', fontsize=9.5,
            style='italic', color=TSLM_EDGE)

    # =====================  STAGE 2  =====================
    # ---- teacher ----
    t_cx, t_cy = 9.20, 6.05
    tw, th = 2.20, 1.05
    ax.add_patch(FancyBboxPatch((t_cx - tw / 2, t_cy - th / 2), tw, th,
                                 boxstyle='round,pad=0.02,rounding_size=0.10',
                                 fc=TSLM_FILL, ec=TSLM_EDGE, lw=2))
    ax.text(t_cx, t_cy + 0.22, 'Teacher TSLM', ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEXT)
    ax.text(t_cx, t_cy - 0.22, r'(observes $\hat{y}$)',
            ha='center', va='center', fontsize=10, style='italic', color=MUTED)

    # ---- student ----
    s_cx, s_cy = 9.20, 3.00
    sw, sh = 2.20, 1.05
    ax.add_patch(FancyBboxPatch((s_cx - sw / 2, s_cy - sh / 2), sw, sh,
                                 boxstyle='round,pad=0.02,rounding_size=0.10',
                                 fc=TSLM_FILL, ec=TSLM_EDGE, lw=2))
    ax.text(s_cx, s_cy + 0.22, 'Student TSLM', ha='center', va='center',
            fontsize=11, fontweight='bold', color=TEXT)
    ax.text(s_cx, s_cy - 0.22, '(no label)', ha='center', va='center',
            fontsize=10, style='italic', color=MUTED)

    # ---- masked forward-KL loss in the middle ----
    kl_cx, kl_cy = 9.20, 4.50
    klw, klh = 2.05, 0.85
    ax.add_patch(FancyBboxPatch((kl_cx - klw / 2, kl_cy - klh / 2), klw, klh,
                                 boxstyle='round,pad=0.03,rounding_size=0.10',
                                 fc=LOSS_FILL, ec=LOSS_EDGE, lw=1.8))
    ax.text(kl_cx, kl_cy + 0.16, 'masked forward-KL',
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color=LOSS_EDGE)
    ax.text(kl_cx, kl_cy - 0.20, r'$\mathcal{L}_{\mathrm{STILE}}$',
            ha='center', va='center', fontsize=13, color=LOSS_EDGE)

    # arrows teacher -> KL and student -> KL
    ax.add_patch(FancyArrowPatch((t_cx, t_cy - th / 2 - 0.02),
                                  (kl_cx, kl_cy + klh / 2 + 0.02),
                                  arrowstyle='-|>', mutation_scale=13,
                                  lw=1.5, color=LOSS_EDGE))
    ax.add_patch(FancyArrowPatch((s_cx, s_cy + sh / 2 + 0.02),
                                  (kl_cx, kl_cy - klh / 2 - 0.02),
                                  arrowstyle='-|>', mutation_scale=13,
                                  lw=1.5, color=LOSS_EDGE))
    ax.text(t_cx + 0.55, 5.30, 'teacher\nlogits',
            ha='left', va='center', fontsize=9, style='italic', color=LOSS_EDGE)
    ax.text(s_cx + 0.55, 3.70, 'student\nlogits',
            ha='left', va='center', fontsize=9, style='italic', color=LOSS_EDGE)

    # ---- predictor (top-right): Chronos-2 + linear head ----
    p_cx, p_cy = 12.85, 6.05
    pw, ph = 2.50, 1.05
    ax.add_patch(FancyBboxPatch((p_cx - pw / 2, p_cy - ph / 2), pw, ph,
                                 boxstyle='round,pad=0.02,rounding_size=0.10',
                                 fc=FROZEN_FILL, ec=FROZEN_EDGE, lw=1.8))
    ax.text(p_cx, p_cy + 0.24, 'Chronos-2 + linear head',
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            color=TEXT)
    ax.text(p_cx, p_cy - 0.04, 'upstream predictor',
            ha='center', va='center', fontsize=10, style='italic', color=MUTED)
    ax.text(p_cx, p_cy - 0.32, r'(produces $\hat{y}$)',
            ha='center', va='center', fontsize=9.5, color='#666')

    # ---- g_theta adapter (bottom-right) ----
    a_cx, a_cy = 12.85, 3.00
    aw, ah = 2.50, 1.05
    ax.add_patch(FancyBboxPatch((a_cx - aw / 2, a_cy - ah / 2), aw, ah,
                                 boxstyle='round,pad=0.02,rounding_size=0.10',
                                 fc=ADAPTER_FILL, ec=ADAPTER_EDGE, lw=2))
    ax.text(a_cx, a_cy + 0.24, r'$g_\theta$ adapter',
            ha='center', va='center', fontsize=11.5, fontweight='bold',
            color=TEXT)
    ax.text(a_cx, a_cy - 0.04, 'rationale-opening generator',
            ha='center', va='center', fontsize=9.5, style='italic', color=MUTED)
    ax.text(a_cx, a_cy - 0.32, r'($\sim$140 K trainable params)',
            ha='center', va='center', fontsize=9.5, color='#666')

    # ---- dashed y-hat arrow predictor -> teacher (teacher-only pathway) ----
    yhat = FancyArrowPatch((p_cx - pw / 2 - 0.02, p_cy),
                            (t_cx + tw / 2 + 0.02, t_cy),
                            arrowstyle='-|>', mutation_scale=15, lw=1.9,
                            color=YHAT_COLOR, linestyle=(0, (5, 3)))
    ax.add_patch(yhat)
    ax.text((p_cx - pw / 2 + t_cx + tw / 2) / 2, t_cy + 0.34,
            r'$\hat{y}$', ha='center', va='center',
            fontsize=14, fontweight='bold', color=YHAT_COLOR, style='italic')

    # ---- adapter -> student arrow ----
    ax.add_patch(FancyArrowPatch((a_cx - aw / 2 - 0.02, a_cy),
                                  (s_cx + sw / 2 + 0.02, s_cy),
                                  arrowstyle='-|>', mutation_scale=14,
                                  lw=1.6, color=ADAPTER_EDGE))
    ax.text((a_cx - aw / 2 + s_cx + sw / 2) / 2, s_cy + 0.34,
            'opening tokens', ha='center', va='center',
            fontsize=9.5, color=ADAPTER_EDGE, style='italic')

    # ---- stage 2 hyperparameter caption ----
    cap2 = (r'Per-module learning rates $\eta_m$ (validated search),  trace length $L$'
            '\n'
            r'shared optimiser / scheduler / gradient clipping with Stage 1')
    ax.text(9.05, 1.55, cap2, ha='center', va='center',
            fontsize=10, style='italic', color='#222',
            bbox=dict(boxstyle='round,pad=0.42', fc='#F7F7F2',
                      ec='#999', lw=0.8))

    # =====================  LEGEND  =====================
    legend_handles = [
        mpatches.Patch(facecolor=TSLM_FILL,    edgecolor=TSLM_EDGE,
                       lw=1.5, label='TSLM module'),
        mpatches.Patch(facecolor=ADAPTER_FILL, edgecolor=ADAPTER_EDGE,
                       lw=1.5, label=r'Trainable $g_\theta$ adapter'),
        mpatches.Patch(facecolor=FROZEN_FILL,  edgecolor=FROZEN_EDGE,
                       lw=1.5, label='Frozen / external'),
        mpatches.Patch(facecolor=LOSS_FILL,    edgecolor=LOSS_EDGE,
                       lw=1.5, label='Distillation loss'),
        Line2D([0], [0], color=YHAT_COLOR, linestyle=(0, (5, 3)),
               lw=1.9, label=r'$\hat{y}$ pathway (teacher only)'),
    ]
    leg = ax.legend(handles=legend_handles, loc='lower right',
                    fontsize=10, framealpha=0.96, edgecolor='#888',
                    bbox_to_anchor=(0.995, 0.01), borderpad=0.6,
                    handlelength=2.2, handleheight=1.1)
    leg.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out_path = ('/home/wangni/notion-figures/'
                'stile-accurate-and-explainable-time-series-language-models-v/'
                'fig_003.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    print(f'Saved {out_path}')
    plt.close()


if __name__ == '__main__':
    main()
