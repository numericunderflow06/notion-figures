"""
fig_003: Prompt Structure — Bilingual Multimodal Frame

Anatomy of one OpenTSLM-Flamingo training sample built by BoschHeatPumpQADataset:
  ① pre_prompt   : English instruction + filtered German metadata + German cue
  ② text_time_series_prompt_list : N channels — text label (carries abs mean/std)
                                   + z-scored sparkline (values carrier)
  ③ post_prompt  : German task cue (Heizbetrieb | Warmwasserbereitung)
All three feed the frozen-backbone OpenTSLM-Flamingo (Llama-3.2-1B).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ------------------------------------------------------------------ palette
EN_FILL   = "#E4EEF9"   # English / blue tint
EN_EDGE   = "#3D6CA8"
DE_FILL   = "#FFF1D6"   # German  / yellow-orange tint
DE_EDGE   = "#D9821F"
PANEL_FC  = "#FFFFFF"
PANEL_EC  = "#888888"
ARROW_C   = "#555555"
SPARK_FC  = "#FAFAFA"
SPARK_EC  = "#BBBBBB"
SPARK_LN  = "#1F4E8C"
MODEL_FC  = "#BEE3F8"
MODEL_EC  = "#2C5282"


def lang_chip(ax, x, y, label, edge, fc=None):
    """Tiny rounded 'EN' / 'DE' chip."""
    ax.text(
        x, y, label,
        fontsize=7.5, fontweight="bold", color="white",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc=edge, ec="none"),
        zorder=4,
    )


def main():
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ---------------------------------------------------------- title block
    ax.text(7, 13.55, "Prompt Structure: Bilingual Multimodal Frame",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(7, 13.05,
            "One training sample → OpenTSLM-Flamingo  "
            "(BoschHeatPumpQADataset)",
            ha="center", va="center", fontsize=10.5,
            style="italic", color="#555555")

    # =================================================== ① pre_prompt
    p1_y0, p1_y1 = 10.0, 12.55
    p1 = FancyBboxPatch((0.5, p1_y0), 13, p1_y1 - p1_y0,
                        boxstyle="round,pad=0.05",
                        lw=1.6, ec=PANEL_EC, fc=PANEL_FC, zorder=1)
    ax.add_patch(p1)
    ax.text(0.75, p1_y1 - 0.30, "①  pre_prompt",
            fontsize=12.5, fontweight="bold", color="#222")
    ax.text(3.05, p1_y1 - 0.30, "— _get_pre_prompt()",
            fontsize=9.5, style="italic", color="#777")

    # English instruction line
    en1 = FancyBboxPatch((0.8, p1_y1 - 1.05), 12.4, 0.55,
                         boxstyle="round,pad=0.03",
                         lw=1.0, ec=EN_EDGE, fc=EN_FILL, zorder=2)
    ax.add_patch(en1)
    lang_chip(ax, 1.15, p1_y1 - 0.78, "EN", EN_EDGE)
    ax.text(1.55, p1_y1 - 0.78,
            '"You are a heat-pump expert. Analyze the following '
            "time series and write an assessment ...\"",
            fontsize=10, family="monospace", va="center")

    # German metadata line
    de1 = FancyBboxPatch((0.8, p1_y1 - 1.70), 12.4, 0.55,
                         boxstyle="round,pad=0.03",
                         lw=1.0, ec=DE_EDGE, fc=DE_FILL, zorder=2)
    ax.add_patch(de1)
    lang_chip(ax, 1.15, p1_y1 - 1.43, "DE", DE_EDGE)
    ax.text(1.55, p1_y1 - 1.43,
            "Anlagen-Metadaten:  Geräte-Typ, Baujahr, Heizfläche, "
            "Speicher-Volumen, ...  (gefiltert)",
            fontsize=10, family="monospace", va="center")

    # German closing cue (style instruction)
    en2 = FancyBboxPatch((0.8, p1_y1 - 2.35), 12.4, 0.55,
                         boxstyle="round,pad=0.03",
                         lw=1.0, ec=EN_EDGE, fc=EN_FILL, zorder=2)
    ax.add_patch(en2)
    lang_chip(ax, 1.15, p1_y1 - 2.08, "EN", EN_EDGE)
    ax.text(1.55, p1_y1 - 2.08,
            '"Write your assessment as a single, natural paragraph '
            'in German."',
            fontsize=10, family="monospace", va="center",
            fontstyle="italic")

    # =================================================== ② text_time_series
    p2_y0, p2_y1 = 4.30, 9.55
    p2 = FancyBboxPatch((0.5, p2_y0), 13, p2_y1 - p2_y0,
                        boxstyle="round,pad=0.05",
                        lw=1.6, ec=PANEL_EC, fc=PANEL_FC, zorder=1)
    ax.add_patch(p2)
    ax.text(0.75, p2_y1 - 0.30,
            "②  text_time_series_prompt_list   (N channels)",
            fontsize=12.5, fontweight="bold", color="#222")
    ax.text(6.55, p2_y1 - 0.30,
            "— _get_text_time_series_prompt_list()",
            fontsize=9.5, style="italic", color="#777")

    # Header note about dual carrier
    ax.text(7, p2_y1 - 0.78,
            "Each channel = TEXT label (carries absolute mean / std)  +  "
            "z-scored VALUES (carries shape).",
            fontsize=9.7, color="#444", ha="center", style="italic")

    # Three sample channels
    channels = [
        ("outdoor_temp_mean",            12.88, 6.30),
        ("flow_temp_mean",               38.42, 4.51),
        ("compressor_modulation_mean",   47.10, 18.20),
    ]
    rng = np.random.default_rng(7)
    row_ys = np.linspace(p2_y1 - 1.65, p2_y0 + 0.95, 3)

    for i, ((name, mu, sd), y) in enumerate(zip(channels, row_ys)):
        # text label box (German-style cue, fits in the German tint)
        lbl = FancyBboxPatch((0.8, y - 0.32), 7.4, 0.66,
                             boxstyle="round,pad=0.03",
                             lw=1.0, ec=DE_EDGE, fc=DE_FILL, zorder=2)
        ax.add_patch(lbl)
        lang_chip(ax, 1.15, y, "DE", DE_EDGE)
        ax.text(1.55, y,
                f'"{name}, it has mean {mu:.2f} and std {sd:.2f}:"',
                fontsize=9.8, family="monospace", va="center")

        # arrow text→values
        ax.annotate("", xy=(9.20, y), xytext=(8.30, y),
                    arrowprops=dict(arrowstyle="->",
                                    color=ARROW_C, lw=1.1))

        # sparkline frame
        sp = FancyBboxPatch((9.3, y - 0.32), 4.0, 0.66,
                            boxstyle="round,pad=0.02",
                            lw=1.0, ec=SPARK_EC, fc=SPARK_FC, zorder=2)
        ax.add_patch(sp)
        # zero line (z-score)
        ax.plot([9.4, 13.2], [y, y], color="#CCCCCC", lw=0.7,
                linestyle="--", zorder=2.5)
        # synthetic z-scored signal
        t = np.linspace(0, 1, 100)
        sig = (np.sin(2 * np.pi * (1.3 + 0.4 * i) * t + i)
               + 0.35 * rng.standard_normal(100))
        sig = (sig - sig.mean()) / sig.std()
        sig = np.clip(sig, -2.4, 2.4)
        sx = 9.4 + t * 3.8
        sy = y + sig * 0.22
        ax.plot(sx, sy, color=SPARK_LN, lw=1.1, zorder=3)
        ax.text(13.18, y + 0.20, "z-score",
                fontsize=7.5, color="#666",
                ha="right", style="italic", zorder=3)

    # ellipsis indicating N channels
    ax.text(7, p2_y0 + 0.35, "·  ·  ·   (N channels)   ·  ·  ·",
            ha="center", fontsize=10, color="#777", style="italic")

    # =================================================== ③ post_prompt
    p3_y0, p3_y1 = 2.45, 3.95
    p3 = FancyBboxPatch((0.5, p3_y0), 13, p3_y1 - p3_y0,
                        boxstyle="round,pad=0.05",
                        lw=1.6, ec=PANEL_EC, fc=PANEL_FC, zorder=1)
    ax.add_patch(p3)
    ax.text(0.75, p3_y1 - 0.25, "③  post_prompt",
            fontsize=12.5, fontweight="bold", color="#222")
    ax.text(2.85, p3_y1 - 0.25, "— _get_post_prompt()   (one of two cues)",
            fontsize=9.5, style="italic", color="#777")

    pa = FancyBboxPatch((0.8, p3_y0 + 0.20), 6.0, 0.55,
                        boxstyle="round,pad=0.03",
                        lw=1.0, ec=DE_EDGE, fc=DE_FILL, zorder=2)
    ax.add_patch(pa)
    lang_chip(ax, 1.15, p3_y0 + 0.475, "DE", DE_EDGE)
    ax.text(1.55, p3_y0 + 0.475,
            "Expertenbewertung Heizbetrieb:",
            fontsize=10.5, family="monospace", fontweight="bold",
            va="center")

    pb = FancyBboxPatch((7.2, p3_y0 + 0.20), 6.0, 0.55,
                        boxstyle="round,pad=0.03",
                        lw=1.0, ec=DE_EDGE, fc=DE_FILL, zorder=2)
    ax.add_patch(pb)
    lang_chip(ax, 7.55, p3_y0 + 0.475, "DE", DE_EDGE)
    ax.text(7.95, p3_y0 + 0.475,
            "Expertenbewertung Warmwasserbereitung:",
            fontsize=10.5, family="monospace", fontweight="bold",
            va="center")

    # =================================================== merge arrows + model
    # OpenTSLM-Flamingo box
    mb_x0, mb_y0, mb_w, mb_h = 4.3, 0.55, 5.4, 1.30
    mb = FancyBboxPatch((mb_x0, mb_y0), mb_w, mb_h,
                        boxstyle="round,pad=0.05",
                        lw=2.0, ec=MODEL_EC, fc=MODEL_FC, zorder=3)
    ax.add_patch(mb)
    ax.text(mb_x0 + mb_w / 2, mb_y0 + mb_h - 0.40,
            "OpenTSLM-Flamingo",
            fontsize=13.5, fontweight="bold",
            ha="center", va="center")
    ax.text(mb_x0 + mb_w / 2, mb_y0 + mb_h - 0.85,
            "(frozen Llama-3.2-1B backbone  +  TS encoder + cross-attn)",
            fontsize=9, ha="center", va="center",
            color="#333", style="italic")

    # three arrows merging into model
    arrow_target = (mb_x0 + mb_w / 2, mb_y0 + mb_h)
    sources = [
        (2.8, p3_y0),   # from ① side
        (7.0, p3_y0),   # from ② (center)
        (11.2, p3_y0),  # from ③ side
    ]
    rads = [0.18, 0.0, -0.18]
    for (sx, sy), r in zip(sources, rads):
        ax.annotate(
            "", xy=arrow_target, xytext=(sx, sy),
            arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=1.6,
                            connectionstyle=f"arc3,rad={r}",
                            shrinkA=2, shrinkB=4),
            zorder=2.5,
        )

    # legend
    leg_en = mpatches.Patch(facecolor=EN_FILL, edgecolor=EN_EDGE,
                            label="English  (task instructions)")
    leg_de = mpatches.Patch(facecolor=DE_FILL, edgecolor=DE_EDGE,
                            label="German  (metadata, labels, target cues)")
    ax.legend(handles=[leg_en, leg_de], loc="lower left",
              bbox_to_anchor=(0.005, 0.0), fontsize=9,
              frameon=True, framealpha=0.95)

    plt.tight_layout()
    out = ("/home/wangni/notion-figures/"
           "bosch-checkit-expert-commentary-generation-for-heat-pump-tim/"
           "fig_003.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
