import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "scratch" / "ppt_v4_assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CLEAN CORPORATE LIGHT THEME PALETTE (McKinsey / Executive Boardroom Style)
# ==============================================================================
BG_LIGHT      = RGBColor(248, 250, 252)  # #F8FAFC Crisp Alabaster White
BG_WHITE      = RGBColor(255, 255, 255)  # #FFFFFF Pure White Card
BG_SOFT_BLUE  = RGBColor(239, 246, 255)  # #EFF6FF Soft Blue Highlight
BG_SOFT_GREEN = RGBColor(236, 253, 245)  # #ECFDF5 Soft Emerald Highlight
BG_SOFT_RED   = RGBColor(255, 241, 242)  # #FFF1F2 Soft Rose Highlight
BG_SOFT_AMBER = RGBColor(255, 251, 235)  # #FFFBEB Soft Amber Highlight
BG_SOFT_PURP  = RGBColor(245, 243, 255)  # #F5F3FF Soft Purple Highlight
BG_TERM       = RGBColor(15, 23, 42)     # #0F172A Dark Terminal ONLY for Code/Prompts

BORDER_GRAY   = RGBColor(203, 213, 225)  # #CBD5E1 Clean Hairline Border
NAVY_PRIMARY  = RGBColor(15, 23, 42)     # #0F172A Deep Executive Navy
BLUE_BRAND    = RGBColor(37, 99, 235)    # #2563EB Corporate Royal Blue
INDIGO_BRAND  = RGBColor(79, 70, 229)    # #4F46E5 Deep Indigo
PURPLE_BRAND  = RGBColor(124, 58, 237)   # #7C3AED Purple
TEAL_ACCENT   = RGBColor(13, 148, 136)   # #0D9488 Teal Accent
GREEN_POS     = RGBColor(5, 150, 105)    # #059669 Positive Green
AMBER_WARN    = RGBColor(217, 119, 6)    # #D97706 Prosody Amber
RED_NEG       = RGBColor(225, 29, 72)    # #E11D48 Negative Crimson

TEXT_DARK     = RGBColor(15, 23, 42)     # #0F172A Primary Ink
TEXT_BODY     = RGBColor(51, 65, 85)     # #334155 Secondary Ink
TEXT_MUTED    = RGBColor(100, 116, 139)  # #64748B Muted Caption
TEXT_WHITE    = RGBColor(255, 255, 255)

FONT_SANS = "Segoe UI"
FONT_MONO = "Consolas"


# ==============================================================================
# STEP 1: GENERATE HIGH-DPI VISUAL DIAGRAMS & CHARTS VIA MATPLOTLIB
# ==============================================================================
def generate_visual_assets():
    plt.rcParams["font.sans-serif"] = "Segoe UI"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # 1. Three Intonations Pitch Contour Diagram (Slide 2)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.7), dpi=250)
    t = np.linspace(0, 2.5, 200)

    f0_a = 175 + 25 * np.sin(2.2 * t) + 35 * (t / 2.5) ** 1.5
    axes[0].plot(t, f0_a, color="#059669", linewidth=3)
    axes[0].fill_between(t, 120, f0_a, color="#059669", alpha=0.12)
    axes[0].set_title("1. Genuine Gratitude (Positive)\nRising Pitch (225 Hz) + Lively Pace", fontsize=10.5, fontweight="bold", color="#059669", pad=8)

    f0_b = 160 + 6 * np.sin(3.0 * t)
    axes[1].plot(t, f0_b, color="#2563EB", linewidth=3)
    axes[1].fill_between(t, 120, f0_b, color="#2563EB", alpha=0.10)
    axes[1].set_title("2. Routine Closing (Neutral)\nFlat Pitch (160 Hz) + Steady Pace", fontsize=10.5, fontweight="bold", color="#2563EB", pad=8)

    f0_c = 215 - 45 * (t / 2.5) + 8 * np.sin(2.0 * t)
    axes[2].plot(t, f0_c, color="#E11D48", linewidth=3)
    axes[2].fill_between(t, 120, f0_c, color="#E11D48", alpha=0.12)
    axes[2].set_title("3. Sarcastic / Sad (Negative)\nFalling Pitch (193→155 Hz) + Slow Pace", fontsize=10.5, fontweight="bold", color="#E11D48", pad=8)

    for ax in axes:
        ax.set_ylim(120, 260)
        ax.set_xlim(0, 2.5)
        ax.set_xlabel("Time (seconds) — 'Thank you so much for the help today'", fontsize=8.5, color="#475569")
        ax.set_ylabel("Voice Pitch (Hz)", fontsize=8.5, color="#475569")
        ax.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
        ax.set_facecolor("#FFFFFF")

    fig.patch.set_facecolor("#F8FAFC")
    plt.tight_layout()
    p_intonation = ASSETS_DIR / "intonation_curves.png"
    plt.savefig(p_intonation, dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 2. Dual-Modal Architecture Diagram (Slide 4, excluding SenseVoice, COMPACT & NO OVERLAP)
    fig, ax = plt.subplots(figsize=(12.2, 3.15), dpi=250)
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 32)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    def draw_node(x, y, w, h, title, sub="", bg="#EFF6FF", border="#2563EB", title_col="#0F172A", sub_col="#475569"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", facecolor=bg, edgecolor=border, linewidth=1.4)
        ax.add_patch(rect)
        cx = x + w / 2
        if sub:
            ax.text(cx, y + h * 0.62, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=title_col)
            ax.text(cx, y + h * 0.28, sub, ha="center", va="center", fontsize=7.0, color=sub_col)
        else:
            ax.text(cx, y + h * 0.50, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=title_col)

    def draw_arrow(x1, y1, x2, y2, col="#64748B"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.3, shrinkA=2, shrinkB=2))

    # Audio Input & Upload
    draw_node(1, 12, 10, 7.5, "Audio Recording", "Mic / Upload", "#F8FAFC", "#64748B")
    draw_arrow(11, 15.75, 13.5, 15.75)
    draw_node(13.5, 12, 12, 7.5, "Upload & Analysis", "REST API", "#EFF6FF", "#2563EB")

    # Split to Audio (top) and Text (bottom)
    draw_arrow(25.5, 17.5, 29.5, 24.5)
    draw_arrow(25.5, 14.0, 29.5, 9.5)

    # Audio Branch (Emotion2Vec+ Large)
    draw_node(29.5, 21, 13, 7.2, "Audio Decoding &", "Normalization", "#EFF6FF", "#2563EB")
    draw_arrow(42.5, 24.6, 45.5, 24.6)
    draw_node(45.5, 21, 13, 7.2, "Emotion2Vec+", "Plus Large (Audio)", "#FDF2F8", "#DB2777", title_col="#831843")
    draw_arrow(58.5, 24.6, 61.5, 24.6)
    draw_node(61.5, 21, 14, 7.2, "Audio Score", "Distributions", "#EFF6FF", "#2563EB")

    # Text Branch (ElevenLabs Scribe -> Two Text Analyzers)
    draw_node(29.5, 5.5, 13, 7.5, "ElevenLabs Scribe", "ASR Transcription", "#F0FDF4", "#16A34A", title_col="#14532D")
    draw_arrow(42.5, 10.5, 45.5, 13.5)
    draw_arrow(42.5, 8.0, 45.5, 4.8)
    draw_node(45.5, 11.2, 13, 6.2, "Transcript", "Sentiment Model", "#FEF3C7", "#D97706", title_col="#78350F")
    draw_node(45.5, 1.8, 13, 6.2, "Transcript", "Emotion Model", "#FEF3C7", "#D97706", title_col="#78350F")

    draw_arrow(58.5, 4.9, 61.5, 4.9)
    draw_node(61.5, 1.8, 14, 6.2, "Emotion Category", "Mapping", "#FEF3C7", "#D97706")

    # Gated Late Fusion Blocks
    draw_arrow(75.5, 24.0, 79.5, 19.5)
    draw_arrow(58.5, 14.3, 79.5, 18.0)
    draw_node(79.5, 15.5, 14, 7.2, "Sentiment Gated", "Late Fusion", "#F5F3FF", "#7C3AED", title_col="#4C1D95")

    draw_arrow(75.5, 22.0, 79.5, 10.5)
    draw_arrow(75.5, 4.9, 79.5, 8.5)
    draw_node(79.5, 6.5, 14, 7.2, "Emotion Gated", "Late Fusion", "#F5F3FF", "#7C3AED", title_col="#4C1D95")

    # Assembly into Persisted Response
    draw_arrow(93.5, 19.1, 97.5, 16.5)
    draw_arrow(93.5, 10.1, 97.5, 14.5)
    draw_node(97.5, 12, 12, 7.5, "Persisted Analysis", "Response JSON", "#EFF6FF", "#2563EB")

    draw_arrow(109.5, 15.75, 112.5, 15.75)
    draw_node(112.5, 12, 8.5, 7.5, "Live Report", "& History", "#F8FAFC", "#0F172A", title_col="#0F172A")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    p_dual_modal = ASSETS_DIR / "dual_modal_diagram_v4.png"
    plt.savefig(p_dual_modal, dpi=250)
    plt.close(fig)

    # 3. Stage 2 Divergence: Dual-Modal vs. Tri-Modal System One Arbitrage (Slide 5)
    fig, ax = plt.subplots(figsize=(12.2, 3.0), dpi=250)
    ax.set_xlim(0, 122)
    ax.set_ylim(0, 31)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    divider = patches.Rectangle((60.8, 1), 0.4, 29, facecolor="#E2E8F0", edgecolor="none")
    ax.add_patch(divider)

    def draw_cmp_box(x, y, w, h, title, sub="", bg="#EFF6FF", border="#2563EB", title_col="#0F172A", sub_col="#475569"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.9", facecolor=bg, edgecolor=border, linewidth=1.3)
        ax.add_patch(rect)
        cx = x + w / 2
        if sub:
            ax.text(cx, y + h * 0.60, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color=title_col)
            ax.text(cx, y + h * 0.28, sub, ha="center", va="center", fontsize=6.6, color=sub_col)
        else:
            ax.text(cx, y + h * 0.50, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color=title_col)

    def draw_cmp_arrow(x1, y1, x2, y2, col="#64748B"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.3, shrinkA=2, shrinkB=2))

    # LEFT: Dual-Modal Gated Late Fusion
    ax.text(30.5, 29, "APPROACH 1: DUAL-MODAL GATED LATE FUSION", ha="center", fontsize=9.2, fontweight="bold", color="#4F46E5")
    ax.text(30.5, 27.2, "2 Modalities (Audio + Text)  |  Post-Hoc Mathematical Gating", ha="center", fontsize=7.4, color="#64748B")

    draw_cmp_box(2, 14, 11, 6.5, "Audio Wave", "Mic Recording", "#F8FAFC", "#64748B")
    draw_cmp_arrow(13, 17.5, 16, 21.5)
    draw_cmp_arrow(13, 17.0, 16, 8.5)

    draw_cmp_box(16, 19, 15, 6.5, "Emotion2Vec+ Large", "Audio Distribution", "#FDF2F8", "#DB2777", title_col="#831843")
    draw_cmp_box(16, 5.5, 15, 6.5, "ElevenLabs Scribe", "Text Transcript", "#F0FDF4", "#16A34A", title_col="#14532D")

    draw_cmp_arrow(31, 9.5, 34, 12.5)
    draw_cmp_arrow(31, 7.5, 34, 5.0)
    draw_cmp_box(34, 10.5, 12, 5.5, "Text Sentiment", "Analyzer 1", "#FEF3C7", "#D97706")
    draw_cmp_box(34, 2.5, 12, 5.5, "Text Emotion", "Analyzer 2", "#FEF3C7", "#D97706")

    draw_cmp_arrow(31, 22.0, 48.5, 18.5)
    draw_cmp_arrow(46, 13.5, 48.5, 17.0)
    draw_cmp_box(48.5, 15.5, 10.5, 6.0, "Sentiment Gated", "Late Fusion", "#F5F3FF", "#7C3AED", title_col="#4C1D95")

    draw_cmp_arrow(31, 20.0, 48.5, 8.5)
    draw_cmp_arrow(46, 5.5, 48.5, 6.5)
    draw_cmp_box(48.5, 4.5, 10.5, 6.0, "Emotion Gated", "Late Fusion", "#F5F3FF", "#7C3AED", title_col="#4C1D95")

    # RIGHT: Tri-Modal System One Arbitrage
    ax.text(91.5, 29, "APPROACH 2: TRI-MODAL SYSTEM ONE ARBITRAGE", ha="center", fontsize=9.2, fontweight="bold", color="#059669")
    ax.text(91.5, 27.2, "3 Modalities (+ Physical Prosody)  |  Early System One Attention Arbitrage", ha="center", fontsize=7.4, color="#64748B")

    draw_cmp_box(63, 13, 10, 6.5, "Audio Wave", "Raw & Norm", "#F8FAFC", "#64748B")
    draw_cmp_arrow(73, 17.5, 76, 23)
    draw_cmp_arrow(73, 16.25, 76, 16.25)
    draw_cmp_arrow(73, 15.0, 76, 8.5)

    draw_cmp_box(76, 20.5, 14, 5.8, "Whisper Turbo INT8", "ASR Transcript", "#F0FDF4", "#16A34A", title_col="#14532D")
    draw_cmp_box(76, 13.5, 14, 5.8, "Emotion2Vec+ Large", "Acoustic Prior", "#FDF2F8", "#DB2777", title_col="#831843")
    draw_cmp_box(76, 5.5, 14, 5.8, "Librosa Prosody DSP", "Pitch, Speed, dB", "#EFF6FF", "#2563EB", title_col="#1E3A8A")

    draw_cmp_arrow(90, 23.5, 93.5, 18.5)
    draw_cmp_arrow(90, 16.4, 93.5, 16.4)
    draw_cmp_arrow(90, 8.5, 93.5, 14.0)

    draw_cmp_box(93.5, 9.5, 15, 13.5, "System One Arbitrator", "Early Prompt Attention\nSingle-Pass PointerHead\n(0 Generated Tokens)", "#ECFDF5", "#059669", title_col="#047857")

    draw_cmp_arrow(108.5, 16.25, 111.5, 16.25)
    draw_cmp_box(111.5, 11, 8.5, 10.5, "Unified Output", "8 Emotions\nSarcasm (True)\n3-Sentiment", "#F0FDF4", "#059669", title_col="#047857")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    p_approach_cmp = ASSETS_DIR / "approach_comparison_diagram_v4.png"
    plt.savefig(p_approach_cmp, dpi=250)
    plt.close(fig)

    # 4. Slide 8: System One vs Generative LLM Diagram
    fig, ax = plt.subplots(figsize=(12.0, 2.7), dpi=250)
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 28)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    divider2 = patches.Rectangle((61.5, 1), 0.4, 26, facecolor="#E2E8F0", edgecolor="none")
    ax.add_patch(divider2)

    # LEFT: Generative LLM
    ax.text(30.5, 26, "STANDARD GENERATIVE LLM (Autoregressive Token Loop)", ha="center", fontsize=8.8, fontweight="bold", color="#DC2626")
    ax.text(30.5, 23.8, "Generates text token-by-token (Slow, High Compute, Hallucination Risk)", ha="center", fontsize=7.2, color="#64748B")

    draw_cmp_box(2, 9, 10, 8, "Input Prompt", "Audio + Words", "#F8FAFC", "#64748B")
    draw_cmp_arrow(12, 13, 14.5, 13)
    draw_cmp_box(14.5, 8.5, 13, 9, "Transformer\nLayer Stack", "Full KV-Cache", "#EFF6FF", "#2563EB")
    draw_cmp_arrow(27.5, 13, 30, 13)
    draw_cmp_box(30, 7.5, 14, 11, "Sequential\nToken Loop", "Tok 1 -> Tok 2 -> ...\n(15-30ms / token)", "#FEF2F2", "#DC2626", title_col="#991B1B")
    draw_cmp_arrow(44, 13, 46.5, 13)
    draw_cmp_box(46.5, 8.5, 13.5, 9, "Generated\nJSON String", "JSON syntax risk\nTrailing commas", "#FFF1F2", "#E11D48", title_col="#9F1239")

    # RIGHT: System One PointerHead
    ax.text(92.5, 26, "SYSTEM ONE ARBITRATOR (Kev PointerHead Engine)", ha="center", fontsize=8.8, fontweight="bold", color="#059669")
    ax.text(92.5, 23.8, "0 Generated Tokens  |  1 Forward Pass  |  100% Strict Typed Schema", ha="center", fontsize=7.2, color="#047857")

    draw_cmp_box(63.5, 9, 10, 8, "State Prompt", "Tri-Modal State", "#F8FAFC", "#64748B")
    draw_cmp_arrow(73.5, 13, 76, 13)
    draw_cmp_box(76, 8.5, 13, 9, "Transformer\nLayer Stack", "Self-Attention", "#EFF6FF", "#2563EB")
    draw_cmp_arrow(89, 13, 91.5, 13)
    draw_cmp_box(91.5, 6.5, 14.5, 13, "Single-Pass\nPointerHead", "Direct Softmax &\nContinuous Pointers\n(0 Generated Tokens)", "#ECFDF5", "#059669", title_col="#065F46")
    draw_cmp_arrow(106, 15.5, 108.5, 17.5)
    draw_cmp_arrow(106, 10.5, 108.5, 8.5)
    draw_cmp_box(108.5, 14.5, 13.5, 6.5, "type: choice", "8 Emotion Classes\nNormalized Softmax", "#F0FDF4", "#16A34A", title_col="#14532D")
    draw_cmp_box(108.5, 5.5, 13.5, 6.5, "type: noul", "Continuous [0, 1]\nSarcasm Scalar", "#F0FDF4", "#16A34A", title_col="#14532D")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    p_sysone = ASSETS_DIR / "system_one_comparison.png"
    plt.savefig(p_sysone, dpi=250)
    plt.close(fig)

    # 5. Slide 15: VRAM Staging Timeline Chart (NVIDIA GeForce GTX 1650 explicitly highlighted)
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=250)
    stages = ["Stage 1:\nWhisper Turbo\n(INT8)", "Evict\nVRAM", "Stage 2:\nEmotion2Vec+\n+ Librosa", "Evict\nVRAM", "Stage 3:\nKev-4B INT4\n(GGUF)", "Stage 3 Alt:\nKev-0.8B\n(GGUF Q8_0)"]
    vram_mb = [1120, 50, 1240, 50, 2410, 803]
    colors = ["#2563EB", "#94A3B8", "#D97706", "#94A3B8", "#4F46E5", "#059669"]

    bars = ax.bar(stages, vram_mb, color=colors, width=0.55, edgecolor="#0F172A", linewidth=0.6)
    ax.axhline(4096, color="#E11D48", linestyle="--", linewidth=2, label="4,096 MB Limit (NVIDIA GeForce GTX 1650)")
    ax.axhline(5200, color="#991B1B", linestyle=":", linewidth=1.5, label="5,200 MB If Loaded Simultaneously (OOM Crash)")
    ax.set_ylim(0, 5800)
    ax.set_ylabel("GPU Memory Usage (MB)", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.set_title("NVIDIA GTX 1650 (4GB) Sequential Load-Infer-Evict Profile", fontsize=10.5, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.tick_params(axis="x", labelsize=8)
    for bar, val in zip(bars, vram_mb):
        y_pos = bar.get_height() + 110
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val} MB", ha="center", fontsize=8.2, fontweight="bold", color="#0F172A")
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.9)
    plt.tight_layout()
    p_vram = ASSETS_DIR / "vram_chart.png"
    plt.savefig(p_vram, dpi=250)
    plt.close(fig)

    # 6. Slide 16: Enterprise Benchmark Bar Chart
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=250)
    labels = ["60-Clip\nOverall", "60-Clip\nEnglish", "60-Clip\nJapanese", "2,000-Clip\nEnglish", "2,000-Clip\nOverall"]
    ser_acc = [63.3, 93.3, 33.3, 82.7, 55.2]
    vocalis_acc = [83.3, 96.7, 76.7, 84.3, 67.6]
    x_idx = np.arange(len(labels))
    w = 0.35

    ax.bar(x_idx - w/2, ser_acc, width=w, label="Standalone SER (Emotion2Vec+)", color="#94A3B8", edgecolor="none")
    bars_voc = ax.bar(x_idx + w/2, vocalis_acc, width=w, label="Vocalis System One (Kev)", color="#2563EB", edgecolor="none")
    ax.set_ylim(0, 110)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, fontsize=8.5, fontweight="bold")
    ax.set_ylabel("3-Way Sentiment Accuracy (%)", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.set_title("3-Way Sentiment: Standalone SER vs. System One", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.legend(loc="upper right", fontsize=8.0)

    for i, (s, v) in enumerate(zip(ser_acc, vocalis_acc)):
        ax.text(i - w/2, s + 2, f"{s:.1f}%", ha="center", fontsize=7.6, color="#475569")
        ax.text(i + w/2, v + 2, f"{v:.1f}%", ha="center", fontsize=8.2, fontweight="bold", color="#1D4ED8")

    plt.tight_layout()
    p_bench1 = ASSETS_DIR / "enterprise_bench_chart.png"
    plt.savefig(p_bench1, dpi=250)
    plt.close(fig)

    # 7. Slide 17: SER Validation Corpora Chart
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=250)
    corpora = ["JVNV-JA\n(Japanese\nEmotion)", "JVNV-JA\n(Japanese\nSentiment)", "TESS\n(English\nEmotion)", "SAVEE\n(English\nSentiment)"]
    kev08_acc = [92.0, 96.0, 36.1, 48.0]
    kev4b_acc = [96.0, 100.0, 91.7, 72.0]
    x_idx = np.arange(len(corpora))
    w = 0.35

    ax.bar(x_idx - w/2, kev08_acc, width=w, label="Vocalis + Kev-0.8B (Q8_0)", color="#F59E0B", edgecolor="none")
    ax.bar(x_idx + w/2, kev4b_acc, width=w, label="Vocalis + Kev-4B (INT4)", color="#059669", edgecolor="none")
    ax.set_ylim(0, 112)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(corpora, fontsize=8.2, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=9.5, fontweight="bold", color="#0F172A")
    ax.set_title("Kev-0.8B vs. Kev-4B on SER Validation Corpora", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.legend(loc="upper right", fontsize=8.0)

    for i, (k0, k4) in enumerate(zip(kev08_acc, kev4b_acc)):
        ax.text(i - w/2, k0 + 2, f"{k0:.1f}%", ha="center", fontsize=7.6, color="#B45309")
        ax.text(i + w/2, k4 + 2, f"{k4:.1f}%", ha="center", fontsize=8.2, fontweight="bold", color="#047857")

    plt.tight_layout()
    p_bench2 = ASSETS_DIR / "ser_val_chart.png"
    plt.savefig(p_bench2, dpi=250)
    plt.close(fig)

    return {
        "intonation": p_intonation,
        "dual_modal": p_dual_modal,
        "approach_cmp": p_approach_cmp,
        "system_one": p_sysone,
        "vram": p_vram,
        "bench1": p_bench1,
        "bench2": p_bench2
    }


# ==============================================================================
# STEP 2: HELPER FUNCTIONS FOR CLEAN BOARDROOM PPTX GENERATION
# ==============================================================================
def set_bg(slide, color=BG_LIGHT):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, category, title, subtitle=None, page=1, total=19):
    tb_kick = slide.shapes.add_textbox(Inches(0.65), Inches(0.48), Inches(10.5), Inches(0.24))
    p_kick = tb_kick.text_frame.paragraphs[0]
    p_kick.text = category.upper()
    p_kick.font.name = FONT_SANS
    p_kick.font.size = Pt(9.5)
    p_kick.font.bold = True
    p_kick.font.color.rgb = BLUE_BRAND

    tb_title = slide.shapes.add_textbox(Inches(0.65), Inches(0.74), Inches(11.0), Inches(0.42))
    p_title = tb_title.text_frame.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_SANS
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY_PRIMARY

    if subtitle:
        tb_sub = slide.shapes.add_textbox(Inches(0.65), Inches(1.18), Inches(11.5), Inches(0.26))
        p_sub = tb_sub.text_frame.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = FONT_SANS
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = TEXT_MUTED

    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.65), Inches(7.04), Inches(12.033), Inches(0.015))
    div.fill.solid()
    div.fill.fore_color.rgb = BORDER_GRAY
    div.line.fill.background()

    tb_fl = slide.shapes.add_textbox(Inches(0.65), Inches(7.14), Inches(8.5), Inches(0.25))
    p_fl = tb_fl.text_frame.paragraphs[0]
    p_fl.text = "Vocalis POC  |  Tri-Modal System One Architecture & Dual UIs  |  Vaibhav & Balram"
    p_fl.font.name = FONT_SANS
    p_fl.font.size = Pt(8.5)
    p_fl.font.color.rgb = TEXT_MUTED

    tb_fr = slide.shapes.add_textbox(Inches(11.0), Inches(7.14), Inches(1.68), Inches(0.25))
    p_fr = tb_fr.text_frame.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.text = f"{page} / {total}"
    p_fr.font.name = FONT_SANS
    p_fr.font.size = Pt(8.5)
    p_fr.font.bold = True
    p_fr.font.color.rgb = TEXT_BODY


def add_box(slide, left, top, width, height, bg=BG_WHITE, border=BORDER_GRAY, top_bar=None, left_bar=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.adjustments[0] = 0.025
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg
    if border:
        shape.line.color.rgb = border
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()

    if top_bar:
        b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.06))
        b.fill.solid()
        b.fill.fore_color.rgb = top_bar
        b.line.fill.background()

    if left_bar:
        b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.08), Inches(height))
        b.fill.solid()
        b.fill.fore_color.rgb = left_bar
        b.line.fill.background()

    return shape


def add_badge(slide, left, top, width, height, text, bg_color, text_color=TEXT_WHITE, size=9.5):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    b.adjustments[0] = 0.5
    b.fill.solid()
    b.fill.fore_color.rgb = bg_color
    b.line.fill.background()
    tf = b.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = text
    p.font.name = FONT_SANS
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = text_color
    return b


def add_paragraphs(slide, left, top, width, height, items, default_size=12, default_color=TEXT_BODY):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for idx, it in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.line_spacing = it.get("line_spacing", 1.2) if isinstance(it, dict) else 1.2
        if isinstance(it, str):
            p.text = it
            p.font.name = FONT_SANS
            p.font.size = Pt(default_size)
            p.font.color.rgb = default_color
            p.space_after = Pt(4)
        else:
            p.space_before = Pt(it.get("space_before", 0))
            p.space_after = Pt(it.get("space_after", 5))
            if "align" in it:
                p.alignment = it["align"]
            p.text = it.get("text", "")
            p.font.name = it.get("font", FONT_SANS)
            p.font.size = Pt(it.get("size", default_size))
            p.font.bold = it.get("bold", False)
            p.font.color.rgb = it.get("color", default_color)
    return tb


def format_cell(cell, text, size=10.5, bold=False, color=TEXT_DARK, bg=None, align=PP_ALIGN.LEFT, font=FONT_SANS):
    if bg:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.12)
    cell.margin_right = Inches(0.12)
    cell.margin_top = Inches(0.06)
    cell.margin_bottom = Inches(0.06)
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = text
    p.font.name = font
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color


# ==============================================================================
# STEP 3: BUILD THE 19 HIGH-END, PERFECTLY PROPORTIONED SLIDES
# ==============================================================================
def build_v4_deck():
    assets = generate_visual_assets()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    total = 19

    # --------------------------------------------------------------------------
    # SLIDE 1: CLEAN EXECUTIVE COVER SLIDE
    # --------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, BG_WHITE)

    left_band = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.35), Inches(7.5))
    left_band.fill.solid()
    left_band.fill.fore_color.rgb = BLUE_BRAND
    left_band.line.fill.background()

    add_badge(s1, 0.95, 0.95, 3.8, 0.36, "TECHNICAL POC & ARCHITECTURE BRIEFING", BG_SOFT_BLUE, BLUE_BRAND, 10)

    add_paragraphs(s1, 0.95, 1.55, 11.4, 2.3, [
        {"text": "VOCALIS", "size": 44, "bold": True, "color": NAVY_PRIMARY, "space_after": 2},
        {"text": "Multimodal Speech Emotion & Sentiment Intelligence", "size": 26, "bold": True, "color": BLUE_BRAND, "space_after": 14},
        {"text": "Comparative Architecture Review: Dual-Modal Gated Late Fusion vs. Tri-Modal System One Arbitration on NVIDIA GTX 1650 (4 GB VRAM)", "size": 15, "color": TEXT_MUTED},
    ])

    add_box(s1, 0.95, 3.95, 11.4, 1.05, bg=BG_LIGHT, border=BORDER_GRAY, left_bar=NAVY_PRIMARY)
    add_paragraphs(s1, 1.25, 4.12, 4.5, 0.75, [
        {"text": "PRESENTED BY", "size": 9.5, "bold": True, "color": TEXT_MUTED, "space_after": 2},
        {"text": "Vaibhav & Balram", "size": 20, "bold": True, "color": NAVY_PRIMARY},
    ])
    add_paragraphs(s1, 6.0, 4.12, 6.1, 0.75, [
        {"text": "PRESENTATION SCOPE", "size": 9.5, "bold": True, "color": BLUE_BRAND, "space_after": 2},
        {"text": "1. Acoustic & SER Engineering Foundations   |   2. Dual-Modal vs. Tri-Modal Boundary\n3. Tri-Modal System One Deep Dive (Prosody DSP & PointerHead)   |   4. GTX 1650 & Benchmarks", "size": 11, "color": TEXT_BODY},
    ])

    kpis = [
        ("83.3% / 96.7%", "3-Way Sentiment Accuracy", "Audited EN+JA / EN Call Benchmark", GREEN_POS),
        ("100% Sentiment", "Japanese SER Validation", "96% Emotion on JVNV-JA Corpus", BLUE_BRAND),
        ("System One Arbitrage", "Zero-Generation Engine", "Single-Pass PointerHead (0 Tokens)", PURPLE_BRAND),
        ("GTX 1650 (4 GB)", "Edge Hardware Footprint", "Zero-OOM 4-Stage Eviction", AMBER_WARN),
    ]
    for i, (val, title, sub, col) in enumerate(kpis):
        x = 0.95 + i * 2.90
        add_box(s1, x, 5.30, 2.70, 1.55, bg=BG_WHITE, border=BORDER_GRAY, top_bar=col)
        add_paragraphs(s1, x + 0.18, 5.48, 2.35, 1.25, [
            {"text": val, "size": 19, "bold": True, "color": col, "space_after": 2},
            {"text": title, "size": 11.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 2},
            {"text": sub, "size": 9.5, "color": TEXT_MUTED},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 2: CORE FOUNDATIONS — SPEECH VS. LEXICAL DYNAMICS
    # --------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    set_bg(s2)
    add_header(
        s2,
        "1. Core Foundations  |  Speech vs. Lexical Dynamics",
        "Why Text Transcripts Alone Fail: 1 Sentence, 3 Contradictory Meanings",
        "Acoustic speech conveys pitch contours, tempo, and vocal timbre that are completely lost in text transcripts.",
        page=2, total=total
    )

    add_box(s2, 0.65, 1.48, 12.033, 0.78, bg=BG_WHITE, border=BORDER_GRAY, left_bar=BLUE_BRAND)
    add_paragraphs(s2, 0.90, 1.60, 11.6, 0.55, [
        {"text": "Spoken Speech  =  Lexical Content (Transcribed Text)   +   Acoustic Delivery (Pitch + Speed + Energy)", "size": 15, "bold": True, "color": NAVY_PRIMARY, "space_after": 2},
        {"text": "A standard NLP model observes:  \"Thank you so much for the help today.\"   It cannot discern genuine delight from sarcastic exhaustion.", "size": 11.5, "color": TEXT_BODY},
    ])

    s2.shapes.add_picture(str(assets["intonation"]), Inches(0.65), Inches(2.40), width=Inches(12.033))

    cards2 = [
        ("Genuine Gratitude → POSITIVE", "Terminal pitch inflection rises warmly (225 Hz) with brisk conversational tempo (5.1 syl/sec).", BG_SOFT_GREEN, GREEN_POS),
        ("Routine Transaction → NEUTRAL", "Flat, measured pitch contour (160 Hz) and baseline enterprise cadence (4.5 syl/sec).", BG_SOFT_BLUE, BLUE_BRAND),
        ("Sarcastic / Dejected → NEGATIVE", "Downward pitch drop (193→155 Hz) coupled with drawn-out, subdued delivery (3.4 syl/sec).", BG_SOFT_RED, RED_NEG),
    ]
    for i, (ct, cb, cbg, ccol) in enumerate(cards2):
        x = 0.65 + i * 4.08
        add_box(s2, x, 5.62, 3.88, 1.25, bg=cbg, border=ccol, left_bar=ccol)
        add_paragraphs(s2, x + 0.20, 5.74, 3.55, 1.0, [
            {"text": ct, "size": 12, "bold": True, "color": ccol, "space_after": 4},
            {"text": cb, "size": 10.5, "color": TEXT_DARK},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 3: ACOUSTIC INTELLIGENCE — HOW SER NEURAL MODELS WORK
    # --------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    set_bg(s3)
    add_header(
        s3,
        "1. Acoustic Intelligence  |  How SER Neural Models Work",
        "Principles of Speech Emotion Recognition (SER) Neural Models",
        "Emotion2Vec+ Large extracts high-dimensional latent acoustic representations directly from speech waveforms.",
        page=3, total=total
    )

    ser_steps = [
        ("STAGE 1", "Frame Decomposition", "Segments continuous audio into 20 ms frames (200 temporal windows for 4s clip).", "Temporal resolution.", BLUE_BRAND),
        ("STAGE 2", "Spectral Representation", "Extracts log-mel spectrograms capturing formant resonance and vocal timbre.", "Agnostic to text tokens.", TEAL_ACCENT),
        ("STAGE 3", "Latent Neural Matching", "Projects acoustic vectors against representations learned from 10,000+ hours of speech.", "1,024-dim latent embedding.", INDIGO_BRAND),
        ("STAGE 4", "Acoustic Prior Output", "Outputs normalized probability distribution across baseline emotional categories.", "Passed downstream as prior.", GREEN_POS),
    ]
    for i, (st, tit, desc, note, col) in enumerate(ser_steps):
        x = 0.65 + i * 3.08
        add_box(s3, x, 1.55, 2.75, 3.25, bg=BG_WHITE, border=BORDER_GRAY, top_bar=col)
        add_badge(s3, x + 0.20, 1.78, 1.0, 0.28, st, col, TEXT_WHITE, 9)
        add_paragraphs(s3, x + 0.20, 2.20, 2.35, 2.3, [
            {"text": tit, "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 8},
            {"text": desc, "size": 11, "color": TEXT_BODY, "space_after": 12},
            {"text": f"💡 {note}", "size": 10.5, "bold": True, "color": col},
        ])
        if i < 3:
            arr = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.81), Inches(2.95), Inches(0.22), Inches(0.30))
            arr.fill.solid()
            arr.fill.fore_color.rgb = BLUE_BRAND
            arr.line.fill.background()

    add_box(s3, 0.65, 5.10, 12.033, 1.75, bg=BG_SOFT_RED, border=RED_NEG, left_bar=RED_NEG)
    add_paragraphs(s3, 0.95, 5.25, 11.5, 1.45, [
        {"text": "Architectural Limitations of Standalone SER (Two Critical Blind Spots)", "size": 13.5, "bold": True, "color": RED_NEG, "space_after": 6},
        {"text": "1. Lexical Blindness: Models evaluate acoustic resonance but lack contextual comprehension of customer intent, complaints, or explicit instructions.", "size": 11.5, "color": TEXT_DARK, "space_after": 4},
        {"text": "2. Cross-Lingual Pitch Bias: In pitch-accent languages such as Japanese, natural polite inflections trigger false 'happy' classifications (74% error rate).", "size": 11.5, "color": TEXT_DARK},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 4: APPROACH 1 — DUAL-MODAL GATED LATE FUSION (PERFECT POSITIONING)
    # --------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    set_bg(s4)
    add_header(
        s4,
        "2. Architecture Track 1  |  Dual-Modal Gated Late Fusion",
        "Approach 1: Dual-Modal Architecture with Gated Late Fusion",
        "A decoupled dual-track pipeline uniting Emotion2Vec+ Large acoustic distributions with ElevenLabs Scribe lexical models.",
        page=4, total=total
    )

    # Clean diagram container card + embedded picture (fits in 3.20 inches, ends at Y=4.68)
    add_box(s4, 0.65, 1.48, 12.033, 3.20, bg=BG_WHITE, border=BORDER_GRAY)
    s4.shapes.add_picture(str(assets["dual_modal"]), Inches(0.76), Inches(1.52), width=Inches(11.8))

    # Bottom 3 Explanatory Cards (placed safely at top=4.85, height=2.15, ends at Y=7.00 -> ZERO OVERLAP!)
    cards4 = [
        (
            "AUDIO TRACK (SER BACKBONE)",
            "Emotion2Vec+ Plus Large",
            "Decodes normalized audio waveforms to generate baseline acoustic score distributions across emotion classes.\n\n📌 Note: Emotion2Vec+ Large is the common acoustic foundation across both POC architectures.",
            BLUE_BRAND, BG_SOFT_BLUE
        ),
        (
            "TEXT TRACK (TWO NLP MODELS)",
            "ElevenLabs Scribe + 2 Analyzers",
            "Transcribes spoken audio into text, then forks text representations into TWO separate downstream analyzers:\n1. Transcript Sentiment Model (Polarity)\n2. Transcript Emotion Model (Categorical)",
            AMBER_WARN, BG_SOFT_AMBER
        ),
        (
            "GATED LATE FUSION",
            "Dual Mathematical Gating",
            "Merges audio distributions and text predictions via confidence-weighted mathematical gating:\n• Sentiment Gated Late Fusion\n• Emotion Gated Late Fusion\nOutputs persisted report JSON.",
            PURPLE_BRAND, BG_SOFT_PURP
        ),
    ]
    for i, (ct, cs, cb, col, sbg) in enumerate(cards4):
        x = 0.65 + i * 4.08
        add_box(s4, x, 4.85, 3.88, 2.15, bg=BG_WHITE, border=col, top_bar=col)
        add_paragraphs(s4, x + 0.18, 5.02, 3.52, 1.90, [
            {"text": ct, "size": 9.5, "bold": True, "color": col, "space_after": 2},
            {"text": cs, "size": 11.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 4},
            {"text": cb, "size": 9.5, "color": TEXT_BODY},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 5: STAGE-2 DIVERGENCE — DUAL-MODAL VS. TRI-MODAL SYSTEM ONE
    # --------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    set_bg(s5)
    add_header(
        s5,
        "2. Architecture Comparison  |  Stage-2 Divergence",
        "Stage-2 Divergence: Dual-Modal Late Fusion vs. Tri-Modal System One",
        "Both backends share Emotion2Vec+ Large as the acoustic foundation—the fundamental architectural divergence occurs in Stage 2.",
        page=5, total=total
    )

    # Embedded Comparison Diagram container card + picture (top=1.48, height=3.05, ends at Y=4.53)
    add_box(s5, 0.65, 1.48, 12.033, 3.05, bg=BG_WHITE, border=BORDER_GRAY)
    s5.shapes.add_picture(str(assets["approach_cmp"]), Inches(0.76), Inches(1.52), width=Inches(11.8))

    # Structured Comparison Table Below (top=4.70, height=2.25, ends at Y=6.95 -> ZERO OVERLAP!)
    t5_shape = s5.shapes.add_table(3, 5, Inches(0.65), Inches(4.70), Inches(12.033), Inches(2.25))
    t5 = t5_shape.table
    for i, w in enumerate([Inches(2.4), Inches(2.3), Inches(2.4), Inches(2.4), Inches(2.533)]):
        t5.columns[i].width = w

    headers5 = ["Architecture Dimension", "Inputs / Modalities", "Second-Stage Mechanism", "Sarcasm & Tension Handling", "Multilingual Robustness"]
    for c_idx, h in enumerate(headers5):
        format_cell(t5.cell(0, c_idx), h, size=9.5, bold=True, color=TEXT_WHITE, bg=NAVY_PRIMARY)

    r5_1 = ["Approach 1:\nDual-Modal Gated Late Fusion", "2 Modalities:\nAudio + Scribe Text", "Gated Late Fusion:\nSeparate sentiment & emotion math gating", "Heuristic:\nRelies on confidence thresholds between models", "Good on English;\nCan inherit SER bias on foreign accents"]
    r5_2 = ["Approach 2:\nTri-Modal System One Arbitrage", "3 Modalities:\nText + SER Prior + Librosa Prosody", "Single-Pass System One:\nEarly prompt attention in transformer layers", "Natively Arbitrated:\nSpots polite words vs. falling pitch / sad prior", "High (+32% on Japanese):\nSystem One overrides false happy SER priors"]

    for c_idx, val in enumerate(r5_1):
        format_cell(t5.cell(1, c_idx), val, size=8.8, color=TEXT_BODY, bg=BG_LIGHT)
    for c_idx, val in enumerate(r5_2):
        format_cell(t5.cell(2, c_idx), val, size=8.8, bold=True, color=GREEN_POS if c_idx == 0 else TEXT_DARK, bg=BG_SOFT_GREEN)

    # --------------------------------------------------------------------------
    # SLIDE 6: SECTION BOUNDARY — ARCHITECTURAL MILESTONE & DUAL UIS
    # --------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    set_bg(s6)
    add_header(
        s6,
        "2. POC Evaluation  |  Architectural Milestone & Dual UIs",
        "Architectural Decision Milestone & Dual UI Experiences",
        "Why Vocalis standardizes on Tri-Modal System One Arbitration and how users interact with both POC paradigms.",
        page=6, total=total
    )

    # Left Card: Architectural Transition Boundary
    add_box(s6, 0.65, 1.52, 5.85, 5.30, bg=BG_WHITE, border=GREEN_POS, top_bar=GREEN_POS)
    add_badge(s6, 0.90, 1.78, 3.8, 0.32, "PRODUCTION SELECTION: TRI-MODAL SYSTEM ONE", GREEN_POS, TEXT_WHITE, 9.5)
    add_paragraphs(s6, 0.90, 2.25, 5.35, 4.3, [
        {"text": "Why Vocalis Moves Forward with Tri-Modal System One:", "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 8},
        {"text": "• Cross-Lingual Robustness: Standalone SER fails on 74% of Japanese speech by mistaking polite pitch-accent for happiness. System One cross-attends words with vocal acoustics, achieving 100% sentiment (+32% gain).", "size": 10.5, "color": TEXT_BODY, "space_after": 6},
        {"text": "• Real-World Sarcasm Arbitration: Dual-Modal late fusion cannot detect when polite words hide contempt. Tri-Modal captures falling pitch and slow cadence to trigger the sarcasm flag.", "size": 10.5, "color": TEXT_BODY, "space_after": 6},
        {"text": "• Zero-Generation Determinism: System One uses a PointerHead instead of autoregressive text generation—guaranteeing 100% schema validity at 5x–10x higher speed.", "size": 10.5, "color": TEXT_BODY, "space_after": 6},
        {"text": "• 4 GB Edge Feasibility: Fully compatible with consumer GPUs via sequential load-infer-evict staging.", "size": 10.5, "bold": True, "color": GREEN_POS},
    ])

    # Right Card: The Two Dedicated UI Experiences
    add_box(s6, 6.83, 1.52, 5.85, 5.30, bg=BG_WHITE, border=PURPLE_BRAND, top_bar=PURPLE_BRAND)
    add_badge(s6, 7.08, 1.78, 3.8, 0.32, "DUAL INTERACTIVE UIS IMPLEMENTED IN VOCALIS", PURPLE_BRAND, TEXT_WHITE, 9.5)
    add_paragraphs(s6, 7.08, 2.25, 5.35, 4.3, [
        {"text": "Two Tailored Enterprise Interfaces Built for the POC:", "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 8},
        {"text": "1. UI 1 — Dual-Modal Batch Analytics & Archive Explorer:", "size": 11.5, "bold": True, "color": PURPLE_BRAND, "space_after": 2},
        {"text": "• Designed for QA compliance teams, call center audits, and historical recording archives.\n• Visualizes decoupled Transcript Sentiment alongside Emotion2Vec+ curves and late-fusion gating shifts.", "size": 10.2, "color": TEXT_BODY, "space_after": 8},
        {"text": "2. UI 2 — Tri-Modal Live Studio & Deep Inspector:", "size": 11.5, "bold": True, "color": BLUE_BRAND, "space_after": 2},
        {"text": "• Built for real-time agent call monitoring, live microphone capture, and multi-modal debugging.\n• Displays live prosody telemetry (Pitch Hz, Tempo onsets/s, Energy dB), 8-emotion distribution, and interactive timeline sarcasm flip detection.", "size": 10.2, "color": TEXT_BODY},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 7: TRI-MODAL PROSODIC FEATURE EXTRACTION (MOVED HERE AS REQUESTED)
    # --------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    set_bg(s7)
    add_header(
        s7,
        "3. Tri-Modal Architecture  |  Prosodic Feature Extraction",
        "How Prosodic Feature Extraction Decodes Vocal Delivery",
        "Prosody captures the macro-physical dynamics of speech—4 objective acoustic signals extracted deterministically via Librosa DSP.",
        page=7, total=total
    )

    prosody_cards = [
        (
            "1. PITCH (F0 Fundamental Frequency — Hz)",
            "Fundamental frequency contour tracking whether voice inflections rise or fall at phrase boundaries.",
            "📈 Rising / Elevated Pitch (>210 Hz): Excitement, Satisfaction, or High Agitation",
            "📉 Depressed / Falling Pitch: Disappointment, Defeat, or Ironic Cynicism",
            BLUE_BRAND, BG_SOFT_BLUE
        ),
        (
            "2. TEMPO (Speaking Rate — Syllables / Onsets per Sec)",
            "Temporal speech cadence computed across silence-trimmed audio frames.",
            "⚡ Accelerated Tempo (>5.5 syl/sec): High Urgency, Panic, or Confrontation",
            "🐢 Decelerated Tempo (<3.5 syl/sec): Subdued Sadness, Hesitation, or Sarcastic Drag",
            AMBER_WARN, BG_SOFT_AMBER
        ),
        (
            "3. ENERGY (Vocal Sound Pressure — Decibels dB)",
            "Root-mean-square amplitude measuring physical air pressure project into the microphone.",
            "🔊 Elevated Energy (>78 dB): Escalated Hostility, Outrage, or Aggressive Outbursts",
            "🔉 Diminished Energy (<65 dB): Subdued Resignation, Grief, or Low Engagement",
            RED_NEG, BG_SOFT_RED
        ),
        (
            "4. VOICE QUALITY (Harmonic Stability & Jitter)",
            "Perturbation quotient across glottal pulse cycles revealing vocal stability versus micro-tremors.",
            "〰️ Elevated Perturbation: Acute Anxiety, Fear, or Suppressed Frustration",
            "✨ Steady Harmonic Phonation: Controlled Business Delivery or Neutral Compliance",
            TEAL_ACCENT, BG_SOFT_GREEN
        ),
    ]

    for idx, (ptit, pdesc, phigh, plow, col, soft_bg) in enumerate(prosody_cards):
        c = idx % 2
        r = idx // 2
        x = 0.65 + c * 6.12
        y = 1.52 + r * 2.72
        add_box(s7, x, y, 5.91, 2.50, bg=BG_WHITE, border=BORDER_GRAY, top_bar=col)
        add_paragraphs(s7, x + 0.22, y + 0.18, 5.45, 2.15, [
            {"text": ptit, "size": 13.5, "bold": True, "color": col, "space_after": 4},
            {"text": pdesc, "size": 10.5, "color": TEXT_MUTED, "space_after": 10},
            {"text": phigh, "size": 11, "bold": True, "color": NAVY_PRIMARY, "space_after": 6},
            {"text": plow, "size": 11, "bold": True, "color": TEXT_BODY},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 8: UNDERSTANDING SYSTEM ONE — HOW IT DIFFERS FROM A GENERATIVE LLM
    # --------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank)
    set_bg(s8)
    add_header(
        s8,
        "3. Tri-Modal Architecture  |  System One Engine Deep Dive",
        "Understanding 'System One': How It Differs from a Generative LLM",
        "System One uses transformer representations but replaces slow autoregressive token loops with instant PointerHead classifications.",
        page=8, total=total
    )

    # Embedded Comparison Diagram (top=1.48, height=2.70, ends at Y=4.18)
    add_box(s8, 0.65, 1.48, 12.033, 2.75, bg=BG_WHITE, border=BORDER_GRAY)
    s8.shapes.add_picture(str(assets["system_one"]), Inches(0.76), Inches(1.52), width=Inches(11.8))

    # Bottom 3 Deep Dive Cards (top=4.38, height=2.60, ends at Y=6.98 -> ZERO OVERLAP!)
    sysone_cards = [
        (
            "1. ZERO-GENERATION (0 TOKENS)",
            "Single Forward Pass vs. Token Loop",
            "• Standard LLMs generate responses token-by-token (15–30 ms/token), taking seconds.\n• System One generates ZERO text tokens. It evaluates the prompt in exactly 1 forward pass through transformer attention weights.\n• Result: 5x–10x faster execution and drastically lower GPU memory footprint.",
            RED_NEG, BG_SOFT_RED
        ),
        (
            "2. TYPED CLASSES (`type: choice`)",
            "Normalized Attention Pointers",
            "• Direct categorical classification over fixed taxonomy schemas.\n• Normalizes attention logits into exact softmax probabilities across the 8 canonical emotions.\n• Result: Eliminates word hallucination and guarantees calibrated confidence scores.",
            BLUE_BRAND, BG_SOFT_BLUE
        ),
        (
            "3. CONTINUOUS SCALARS (`type: noul`)",
            "Floating-Point Regression Head",
            "• Generative LLMs struggle to output calibrated continuous numbers reliably.\n• System One includes a `noul` head that directly reads out continuous floating-point probabilities in [0.0, 1.0].\n• Powers the `sarcasm_detected` trigger (`0.5715`) with 100% JSON schema compliance guarantee.",
            GREEN_POS, BG_SOFT_GREEN
        ),
    ]
    for i, (ct, cs, cb, col, sbg) in enumerate(sysone_cards):
        x = 0.65 + i * 4.08
        add_box(s8, x, 4.38, 3.88, 2.60, bg=BG_WHITE, border=col, top_bar=col)
        add_paragraphs(s8, x + 0.18, 4.54, 3.52, 2.30, [
            {"text": ct, "size": 9.5, "bold": True, "color": col, "space_after": 2},
            {"text": cs, "size": 11.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 6},
            {"text": cb, "size": 9.8, "color": TEXT_BODY},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 9: EMOTION TAXONOMY (8 EMOTIONS -> 3 SENTIMENTS)
    # --------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank)
    set_bg(s9)
    add_header(
        s9,
        "3. Project Taxonomy  |  Emotions & Business Sentiment",
        "8 Canonical Emotions Mapped to 3 Business Sentiments",
        "Vocalis detects 8 specific emotional categories and rolls them up into Positive, Neutral, or Negative sentiment for business KPIs.",
        page=9, total=total
    )

    add_box(s9, 0.65, 1.52, 3.65, 5.30, bg=BG_WHITE, border=GREEN_POS, top_bar=GREEN_POS)
    add_badge(s9, 0.90, 1.75, 2.1, 0.34, "POSITIVE SENTIMENT", GREEN_POS, TEXT_WHITE, 10)
    add_paragraphs(s9, 0.90, 2.25, 3.15, 1.6, [
        {"text": "1. satisfaction_gratitude", "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 6},
        {"text": "Genuine happiness, relief, warm appreciation, and client satisfaction.", "size": 11, "color": TEXT_BODY},
    ])
    add_box(s9, 0.88, 3.95, 3.19, 2.60, bg=BG_SOFT_RED, border=RED_NEG, left_bar=RED_NEG)
    add_paragraphs(s9, 1.08, 4.10, 2.85, 2.3, [
        {"text": "⚡ Sarcasm Polarity Flip", "size": 12, "bold": True, "color": RED_NEG, "space_after": 6},
        {"text": "If the speaker says polite words ('Thank you') with a sad/angry voice and System One flags sarcasm (≥ 45%):", "size": 10.5, "color": TEXT_DARK, "space_after": 8},
        {"text": "Positive Emotion ➔ Inverts to NEGATIVE Sentiment", "size": 11, "bold": True, "color": RED_NEG},
    ])

    add_box(s9, 4.52, 1.52, 3.40, 5.30, bg=BG_WHITE, border=BLUE_BRAND, top_bar=BLUE_BRAND)
    add_badge(s9, 4.77, 1.75, 2.0, 0.34, "NEUTRAL SENTIMENT", BLUE_BRAND, TEXT_WHITE, 10)
    add_paragraphs(s9, 4.77, 2.25, 2.90, 4.2, [
        {"text": "2. neutral_inquiry", "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 6},
        {"text": "Calm routine inquiries, informational questions, and objective business updates.", "size": 11, "color": TEXT_BODY, "space_after": 16},
        {"text": "Acoustic Profile:", "size": 11, "bold": True, "color": BLUE_BRAND, "space_after": 4},
        {"text": "• Steady pitch contour\n• Baseline conversational pace\n• Moderate sound pressure", "size": 10.5, "color": TEXT_BODY},
    ])

    add_box(s9, 8.14, 1.52, 4.54, 5.30, bg=BG_WHITE, border=RED_NEG, top_bar=RED_NEG)
    add_badge(s9, 8.39, 1.75, 2.8, 0.34, "NEGATIVE SENTIMENT (6 EMOTIONS)", RED_NEG, TEXT_WHITE, 10)

    neg_list = [
        ("3. frustration_irritation", "Impatient complaints & vocal annoyance"),
        ("4. outrage_hostility", "Shouting, fury & aggressive threats"),
        ("5. disappointed_sad", "Deep sorrow, sighing & falling pitch"),
        ("6. sarcasm_passive_aggression", "Mocking politeness hiding hostility"),
        ("7. anxiety_confusion", "Shaky, nervous & worried distress"),
        ("8. urgency_panic", "Frantic crisis & immediate emergency request"),
    ]
    for i, (ne, nd) in enumerate(neg_list):
        y = 2.25 + i * 0.74
        add_box(s9, 8.38, y, 4.06, 0.64, bg=BG_SOFT_RED, border=BORDER_GRAY, left_bar=RED_NEG)
        add_paragraphs(s9, 8.56, y + 0.08, 3.78, 0.50, [
            {"text": ne, "size": 11, "bold": True, "color": NAVY_PRIMARY, "space_after": 1},
            {"text": nd, "size": 9.5, "color": TEXT_BODY},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 10: TRI-MODAL SYSTEM ONE ARBITRATION IN ACTION
    # --------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank)
    set_bg(s10)
    add_header(
        s10,
        "3. Tri-Modal Architecture  |  System One Arbitration in Action",
        "How System One Arbitrates Sarcasm & Contradictions in Action",
        "Instead of trusting any single model, Kev acts as a System One arbitrator—weighing lexical words against SER tone and prosody.",
        page=10, total=total
    )

    inputs10 = [
        ("SIGNAL 1: SPOKEN WORDS (Whisper ASR)", "「Thank you so much for the help today.」", "Surface Lexical Meaning: Polite & Grateful (+)", BLUE_BRAND),
        ("SIGNAL 2: ACOUSTIC TONE (Emotion2Vec+)", "Acoustic Prior: sad (82% confidence)", "Vocal Timbre: Pronounced Sadness (-)", RED_NEG),
        ("SIGNAL 3: PHYSICAL PROSODY (Librosa DSP)", "Pitch: 193 Hz (downward falling terminal contour)\nTempo: 3.4 onsets/sec (slow, subdued cadence)", "Physical Delivery: Defeated / Dragging (-)", AMBER_WARN),
    ]
    for i, (itit, ival, imean, col) in enumerate(inputs10):
        y = 1.55 + i * 1.75
        add_box(s10, 0.65, y, 5.2, 1.55, bg=BG_WHITE, border=col, left_bar=col)
        add_paragraphs(s10, 0.88, y + 0.12, 4.8, 1.3, [
            {"text": itit, "size": 10, "bold": True, "color": col, "space_after": 3},
            {"text": ival, "size": 11.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 4},
            {"text": imean, "size": 10, "bold": True, "color": TEXT_MUTED},
        ])
        arr = s10.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(5.98), Inches(y + 0.62), Inches(0.45), Inches(0.32))
        arr.fill.solid()
        arr.fill.fore_color.rgb = INDIGO_BRAND
        arr.line.fill.background()

    add_box(s10, 6.55, 1.55, 6.133, 5.05, bg=BG_WHITE, border=INDIGO_BRAND, top_bar=INDIGO_BRAND)
    add_badge(s10, 6.85, 1.80, 3.8, 0.34, "THE ARBITRATOR: KEV-4B SYSTEM ONE ENGINE", INDIGO_BRAND, TEXT_WHITE, 9.5)

    add_paragraphs(s10, 6.85, 2.30, 5.55, 2.1, [
        {"text": "How System One Resolves Tension in 1 Forward Pass:", "size": 13, "bold": True, "color": NAVY_PRIMARY, "space_after": 8},
        {"text": "1. Cross-Attends Lexicon vs. Acoustics: Detects that polite words ('Thank you') contradict the 82% sad tone and falling pitch contour.", "size": 11, "color": TEXT_BODY, "space_after": 6},
        {"text": "2. Suppresses Literal Confidence: Down-weights 'satisfaction_gratitude' confidence from 91% down to 36.8%.", "size": 11, "color": TEXT_BODY, "space_after": 6},
        {"text": "3. Triggers Sarcasm Detector: Directly outputs `sarcasm_detected = 57.2%` (`sarcasm_flag = True`) via the `noul` head.", "size": 11, "bold": True, "color": RED_NEG},
    ])

    add_box(s10, 6.85, 4.55, 5.53, 1.80, bg=BG_SOFT_RED, border=RED_NEG, left_bar=RED_NEG)
    add_paragraphs(s10, 7.10, 4.70, 5.1, 1.5, [
        {"text": "FINAL ARBITRATED DECISION (rec_1791807581.wav)", "size": 10.5, "bold": True, "color": RED_NEG, "space_after": 4},
        {"text": "• Sarcasm Flag: TRUE (57.2% probability)\n• Emotion Breakdown: 44.7% Ironic Gratitude | 19.0% Sad | 7.8% Sarcasm\n• Calibrated Sentiment: NEGATIVE (62.3% polarity)", "size": 11, "bold": True, "color": NAVY_PRIMARY},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 11: EXACT SYSTEM ONE PROMPT PASSED TO KEV & JSON RESPONSE
    # --------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank)
    set_bg(s11)
    add_header(
        s11,
        "3. Tri-Modal Architecture  |  Exact System One Prompt & Output",
        "Exact Prompt Passed to Kev System One (`/v1/systemone`) & Raw Output",
        "Below is the exact verbatim prompt (`state` + `questions`) sent to Kev-4B and its single-pass PointerHead JSON response.",
        page=11, total=total
    )

    add_box(s11, 0.65, 1.48, 6.85, 5.38, bg=BG_TERM, border=NAVY_PRIMARY, top_bar=BLUE_BRAND)
    add_paragraphs(s11, 0.88, 1.62, 6.4, 5.1, [
        {"text": "1. EXACT `state` STRING PASSED TO SYSTEM ONE:", "size": 10.5, "bold": True, "color": RGBColor(56, 189, 248), "space_after": 4},
        {"text": "Speaker transcript: 「Thank you so much for the help today.」\n"
                 "Acoustic speech emotion prior: sad (confidence: 82%)\n"
                 "Prosodic acoustic features:\n"
                 "- Pitch contour: 193.0 Hz (std: 49.0 Hz, downward falling terminal contour)\n"
                 "- Speech tempo: 3.4 onsets/sec (slow / measured / subdued pace)\n"
                 "- Energy & vocal intensity: 73.5 dB (moderate baseline business volume)\n"
                 "- Voice quality: standard modal phonation, clean harmonic structure",
         "font": FONT_MONO, "size": 9.8, "color": TEXT_WHITE, "space_after": 10},
        {"text": "2. EXACT `questions` TYPED SCHEMA PASSED TO KEV:", "size": 10.5, "bold": True, "color": RGBColor(251, 191, 36), "space_after": 4},
        {"text": "{\n"
                 "  \"detected_emotion\": {\n"
                 "    \"type\": \"choice\",\n"
                 "    \"instructions\": \"Analyze the caller's spoken statement in the full\n"
                 "      context of their vocal acoustics and delivery. Resolve any tension\n"
                 "      between the words and vocal tone (e.g. sarcasm, polite complaints)...\",\n"
                 "    \"criteria\": {\n"
                 "      \"satisfaction_gratitude\": \"Delight, genuine satisfaction...\",\n"
                 "      \"neutral_inquiry\": \"Calm routine business inquiry...\",\n"
                 "      \"frustration_irritation\": \"Annoyance, impatient complaints...\",\n"
                 "      \"outrage_hostility\": \"Extreme fury, aggressive hostility...\",\n"
                 "      \"disappointed_sad\": \"Sorrow, deep sadness, heavy sighing...\",\n"
                 "      \"sarcasm_passive_aggression\": \"Ironic praise, mocking cynicism...\",\n"
                 "      \"anxiety_confusion\": \"Nervous panic, trembling uncertainty...\",\n"
                 "      \"urgency_panic\": \"Critical crisis, urgent emergency...\"\n"
                 "    }\n"
                 "  },\n"
                 "  \"sarcasm_detected\": {\n"
                 "    \"type\": \"noul\",\n"
                 "    \"instructions\": \"Is the speaker using sarcasm, irony, or passive-\n"
                 "      aggressive politeness that contradicts their vocal tone?\"\n"
                 "  }\n"
                 "}",
         "font": FONT_MONO, "size": 8.2, "color": RGBColor(203, 213, 225)},
    ])

    add_box(s11, 7.70, 1.48, 4.98, 3.25, bg=BG_TERM, border=NAVY_PRIMARY, top_bar=GREEN_POS)
    add_paragraphs(s11, 7.92, 1.62, 4.55, 2.95, [
        {"text": "3. EXACT `/v1/systemone` JSON OUTPUT:", "size": 10.5, "bold": True, "color": RGBColor(52, 211, 153), "space_after": 4},
        {"text": "{\n"
                 "  \"answers\": {\n"
                 "    \"detected_emotion\": {\n"
                 "      \"choice\": \"satisfaction_gratitude\",\n"
                 "      \"confidence\": 0.3683,\n"
                 "      \"probabilities\": {\n"
                 "        \"satisfaction_gratitude\": 0.4472,\n"
                 "        \"neutral_inquiry\": 0.2391,\n"
                 "        \"disappointed_sad\": 0.1900,\n"
                 "        \"sarcasm_passive_aggression\": 0.0778\n"
                 "      }\n"
                 "    },\n"
                 "    \"sarcasm_detected\": {\n"
                 "      \"noul\": 0.5715\n"
                 "    }\n"
                 "  }\n"
                 "}",
         "font": FONT_MONO, "size": 9.2, "color": TEXT_WHITE},
    ])

    add_box(s11, 7.70, 4.90, 4.98, 1.96, bg=BG_WHITE, border=AMBER_WARN, left_bar=AMBER_WARN)
    add_paragraphs(s11, 7.92, 5.02, 4.6, 1.75, [
        {"text": "Why Include Both Numbers & Natural Language?", "size": 11.5, "bold": True, "color": AMBER_WARN, "space_after": 4},
        {"text": "• Semantic Labels ('downward falling contour', 'slow pace') trigger Kev's contextual language models.\n• Numerical Anchors ('82%', '193 Hz') provide exact physical scaling at zero extra latency.", "size": 10, "color": TEXT_BODY},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 12: MAJOR FOCUS #1 — SHORTLISTED OPTIONS & TRADE-OFFS
    # --------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank)
    set_bg(s12)
    add_header(
        s12,
        "4. Major Focus #1  |  Research & Shortlisted Options",
        "Shortlisted Architectures Evaluated & Trade-Off Matrix",
        "We evaluated 5 distinct model architectures across accuracy, Japanese robustness, sarcasm detection, and 4 GB GPU speed.",
        page=12, total=total
    )

    t12_shape = s12.shapes.add_table(6, 5, Inches(0.65), Inches(1.50), Inches(12.033), Inches(5.30))
    t12 = t12_shape.table
    for i, w in enumerate([Inches(2.4), Inches(2.1), Inches(1.7), Inches(3.533), Inches(2.3)]):
        t12.columns[i].width = w

    for c_idx, h in enumerate(["Option Evaluated", "Models Tested", "Speed & VRAM", "Key Trade-Off Observed", "Decision"]):
        format_cell(t12.cell(0, c_idx), h, size=10.5, bold=True, color=TEXT_WHITE, bg=NAVY_PRIMARY)

    rows12 = [
        (
            "1. Standalone SER",
            "Emotion2Vec+ Large,\nehcalabres XLSR, Bagus",
            "203 ms / clip\n1.24 GB VRAM",
            "Strong on acted English (86%), but deaf to words and fails on 74% of Japanese clips (26.1% acc).",
            "❌ Reject Standalone\n(Retained as Stage-2 Prior)",
            RED_NEG, BG_WHITE
        ),
        (
            "2. Encoder Router",
            "Laya mmBERT\n(Multilingual Encoder)",
            "112 ms / clip\n0.95 GB VRAM",
            "Fast and lightweight (66.7% acc), but lacks reasoning depth to resolve polite sarcasm.",
            "⚠️ Superseded\n(Beat by Kev +11.6%)",
            AMBER_WARN, BG_LIGHT
        ),
        (
            "3. Native Audio LLM",
            "Qwen2.5-Omni-3B\n(Q4_K_M + mmproj GGUF)",
            "23,670 ms / clip\n3.64 GB Disk/VRAM",
            "10x too slow; experimental llama.cpp audio projector loses subtle tone and predicts 'happy'.",
            "❌ Reject for 4 GB Edge\n(Excessive Latency)",
            RED_NEG, BG_WHITE
        ),
        (
            "4. Post-LLM Math Fusion",
            "Confidence-Weighted\nSER + Kev Fusion",
            "650–3,000 ms\n1.2–2.4 GB VRAM",
            "Double-counts SER and gets overpowered by Emotion2Vec+'s 95%-confident Japanese errors (60% acc).",
            "❌ Reject Late Fusion\n(Biased toward SER)",
            RED_NEG, BG_LIGHT
        ),
        (
            "5. Tri-Modal System One Arbitrage",
            "Whisper INT8 + E2V+ +\nLibrosa + Kev-4B / 0.8B",
            "0.8B: 592 ms\n4B: 2,950 ms\n(<4 GB Staged)",
            "Single forward-pass PointerHead; catches sarcasm natively; 83.3% call sentiment & 100% JVNV-JA.",
            "✅ SELECTED WINNER\n(Production Architecture)",
            GREEN_POS, BG_SOFT_GREEN
        ),
    ]
    for r_idx, r in enumerate(rows12, start=1):
        bg = r[6]
        is_win = (r_idx == 5)
        format_cell(t12.cell(r_idx, 0), r[0], size=10.5, bold=True, color=GREEN_POS if is_win else NAVY_PRIMARY, bg=bg)
        format_cell(t12.cell(r_idx, 1), r[1], size=10, color=TEXT_BODY, bg=bg)
        format_cell(t12.cell(r_idx, 2), r[2], size=10, color=TEXT_BODY, bg=bg)
        format_cell(t12.cell(r_idx, 3), r[3], size=10, color=TEXT_DARK, bg=bg)
        format_cell(t12.cell(r_idx, 4), r[4], size=10, bold=True, color=r[5], bg=bg)

    # --------------------------------------------------------------------------
    # SLIDE 13: WHY PROMPT PRIMING BEAT LATE FUSION & QWEN-OMNI
    # --------------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank)
    set_bg(s13)
    add_header(
        s13,
        "4. Major Focus #1  |  Architectural Trade-Offs",
        "Why Tri-Modal Prompt Priming Beat Late Fusion & Qwen-Omni",
        "Three critical experiments proved why early prompt priming inside Kev is superior to mathematical score fusion.",
        page=13, total=total
    )

    exp_cards13 = [
        (
            "EXPERIMENT 1: WITHOUT SER PRIOR IN PROMPT",
            "Why Prosody Labels Alone Aren't Enough",
            RED_NEG, BG_SOFT_RED,
            [
                "• Tested feeding Kev-4B ONLY natural-language prosody labels (no 'sad 82%' SER prior).",
                "• Result: Polite words ('Thank you so much') overwhelmed soft prosody adjectives.",
                "• Kev became 91.2% overconfident in 'Gratitude' and missed sarcasm (15.9%).",
                "➔ Takeaway: Kev requires the explicit SER prior inside its prompt attention to doubt polite words."
            ]
        ),
        (
            "EXPERIMENT 2: POST-LLM MATHEMATICAL FUSION",
            "Why Averaging Probabilities After Kev Fails",
            AMBER_WARN, BG_SOFT_AMBER,
            [
                "• Tested combining Kev's output probabilities with Emotion2Vec+ via confidence-weighted math.",
                "• Result: Dropped 60-clip multilingual accuracy from 78.3% down to 60.0% (-18.3%).",
                "• Root Cause: Emotion2Vec+ outputs 85%–99% confidence when hallucinating 'happy' on Japanese.",
                "➔ Takeaway: Late fusion double-counts SER and overrides linguistic common sense."
            ]
        ),
        (
            "EXPERIMENT 3: NATIVE AUDIO LLM (QWEN-OMNI-3B)",
            "Why End-to-End Audio GGUF Failed on Edge",
            BLUE_BRAND, BG_SOFT_BLUE,
            [
                "• Tested quantized `Qwen2.5-Omni-3B-Q4_K_M.gguf` + `mmproj-Q8_0.gguf` (3.64 GB).",
                "• Result: Required 23.7 seconds for a 4.5s clip and predicted 'happy / positive' (0% sarcasm).",
                "• Root Cause: Quantized `mtmd` audio projector lost subtle pitch cues and mis-transcribed words.",
                "➔ Takeaway: Specialized staged models beat a single compressed omni model on 4 GB edge hardware."
            ]
        ),
    ]

    for i, (etag, etit, col, sbg, bullets) in enumerate(exp_cards13):
        x = 0.65 + i * 4.08
        add_box(s13, x, 1.52, 3.88, 5.25, bg=BG_WHITE, border=col, top_bar=col)
        add_badge(s13, x + 0.20, 1.78, 3.48, 0.30, etag, sbg, col, 8.8)
        add_paragraphs(s13, x + 0.22, 2.25, 3.45, 4.3, [
            {"text": etit, "size": 13.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 12},
            *[{"text": b, "size": 10.8, "bold": b.startswith("➔"), "color": col if b.startswith("➔") else TEXT_BODY, "space_after": 10} for b in bullets]
        ])

    # --------------------------------------------------------------------------
    # SLIDE 14: MAJOR FOCUS #2 — CONCISE SETUP & TECH STACK
    # --------------------------------------------------------------------------
    s14 = prs.slides.add_slide(blank)
    set_bg(s14)
    add_header(
        s14,
        "5. Major Focus #2  |  Tech Stack & Setup",
        "Production Tech Stack & Concise 4-Step Setup",
        "A modular Python + C++ CUDA stack engineered to run entirely locally on a standard 4 GB consumer/edge GPU.",
        page=14, total=total
    )

    stack_pillars14 = [
        ("1. SPEECH-TO-TEXT (ASR)", "Faster-Whisper Large-v3 Turbo", "• INT8-FP16 (`CTranslate2`)\n• Silero VAD voice segmentation\n• Unified EN + JA transcription\n• 0.35x RTF (2.9x real-time)", BLUE_BRAND),
        ("2. ACOUSTIC & PROSODY", "Emotion2Vec+ & Librosa DSP", "• `iic/emotion2vec_plus_large`\n• `Librosa` YIN F0 pitch & slope\n• Silence-trimmed onset tempo\n• Raw dB intensity & jitter", AMBER_WARN),
        ("3. SYSTEM ONE ARBITRATOR", "Kev-4B & Kev-0.8B (GGUF)", "• `Kev-4B-Q4_K_M.gguf` (2.4 GB)\n• `Kev-0.8B-Q8_0.gguf` (803 MB)\n• Custom C++ `llama-server.exe`\n• `/v1/systemone` PointerHead", INDIGO_BRAND),
        ("4. FULL-STACK STUDIO", "FastAPI + React 19 UI", "• Async FastAPI REST backend\n• React 19 + TypeScript + Vite\n• Live Mic Studio & Clip Inspector\n• 3-Sentiment + 8-Emotion views", GREEN_POS),
    ]
    for i, (head, sub, body, col) in enumerate(stack_pillars14):
        x = 0.65 + i * 3.08
        add_box(s14, x, 1.52, 2.88, 2.75, bg=BG_WHITE, border=BORDER_GRAY, top_bar=col)
        add_paragraphs(s14, x + 0.18, 1.70, 2.52, 2.45, [
            {"text": head, "size": 9.5, "bold": True, "color": col, "space_after": 3},
            {"text": sub, "size": 12.5, "bold": True, "color": NAVY_PRIMARY, "space_after": 8},
            {"text": body, "size": 10.2, "color": TEXT_BODY},
        ])

    add_box(s14, 0.65, 4.52, 12.033, 2.30, bg=BG_WHITE, border=BLUE_BRAND, left_bar=BLUE_BRAND)
    add_paragraphs(s14, 0.95, 4.68, 11.5, 2.0, [
        {"text": "CONCISE 4-STEP SETUP WORKFLOW", "size": 12.5, "bold": True, "color": BLUE_BRAND, "space_after": 8},
        {"text": "Step 1 — Python & CUDA Environment:  Install Python 3.11 venv with PyTorch 2.6 (CUDA 12.4), `faster-whisper`, `funasr`, and `librosa`.", "size": 11, "color": TEXT_DARK, "space_after": 5},
        {"text": "Step 2 — Pre-Quantized GGUF Models:  Place `Kev-4B-Q4_K_M.gguf` & `Kev-0.8B-Q8_0.gguf` in `data/models/` alongside `tools/llama/llama-server.exe`.", "size": 11, "color": TEXT_DARK, "space_after": 5},
        {"text": "Step 3 — Start Backend Server:       Run `python -m uvicorn backend.app:app --port 8000` (auto-manages GPU model loading & eviction).", "size": 11, "color": TEXT_DARK, "space_after": 5},
        {"text": "Step 4 — Start React Studio UI:      Run `npm run dev -- --port 5173` inside `web/` to launch Live Studio, Inspector & Benchmark Explorer.", "size": 11, "color": TEXT_DARK},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 15: 4 GB EDGE HARDWARE ENGINEERING (NVIDIA GEFORCE GTX 1650)
    # --------------------------------------------------------------------------
    s15 = prs.slides.add_slide(blank)
    set_bg(s15)
    add_header(
        s15,
        "5. Major Focus #2  |  4 GB Edge Hardware Engineering",
        "Fitting 5.2 GB of Models into an NVIDIA GeForce GTX 1650 (4 GB VRAM)",
        "Zero-OOM execution on a 4,096 MB GPU achieved through deterministic sequential load-infer-evict staging.",
        page=15, total=total
    )

    add_box(s15, 0.65, 1.50, 6.35, 5.30, bg=BG_WHITE, border=BORDER_GRAY)
    s15.shapes.add_picture(str(assets["vram"]), Inches(0.75), Inches(1.65), width=Inches(6.15))

    add_box(s15, 7.20, 1.50, 5.48, 5.30, bg=BG_WHITE, border=INDIGO_BRAND, top_bar=INDIGO_BRAND)
    add_paragraphs(s15, 7.45, 1.72, 5.0, 4.9, [
        {"text": "Deterministic VRAM Staging on NVIDIA GTX 1650", "size": 14, "bold": True, "color": NAVY_PRIMARY, "space_after": 10},
        {"text": "• Hardware Specification: NVIDIA GeForce GTX 1650 (4 GB GDDR6, Turing Architecture, 4,096 MB hardware ceiling).", "size": 11, "bold": True, "color": BLUE_BRAND, "space_after": 8},
        {"text": "1. Stage 1 (ASR): Load Faster-Whisper INT8 (1,120 MB) ➔ Transcribe speech ➔ `del model` + `torch.cuda.empty_cache()`.", "size": 10.5, "color": TEXT_BODY, "space_after": 8},
        {"text": "2. Stage 2 (SER + Prosody): Load Emotion2Vec+ (1,240 MB) ➔ Extract emotion priors while Librosa runs on CPU ➔ Evict GPU to 0 MB.", "size": 10.5, "color": TEXT_BODY, "space_after": 8},
        {"text": "3. Stage 3 (Isolated C++ Process): Spawn `llama-server.exe` with `Kev-4B INT4` (2,410 MB) ➔ Execute single-pass System One ➔ Terminate process to free 100% VRAM.", "size": 10.5, "color": TEXT_BODY, "space_after": 10},
        {"text": "⚡ Key Engineering Benefit:", "size": 11.5, "bold": True, "color": GREEN_POS, "space_after": 4},
        {"text": "By executing Kev in an isolated C++ CUDA subprocess (`llama-server.exe`), we bypass Windows PyTorch Triton slowdowns (3.7x faster) and eliminate VRAM fragmentation.", "size": 10.2, "color": TEXT_DARK},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 16: BENCHMARK RESULTS PART 1 — ENTERPRISE 60 & 2,000 CLIPS
    # --------------------------------------------------------------------------
    s16 = prs.slides.add_slide(blank)
    set_bg(s16)
    add_header(
        s16,
        "6. Benchmark Evaluation  |  Multilingual Enterprise Corpora",
        "Benchmark Results (Part 1): 60-Clip Audited & 2,000-Clip Enterprise Scale",
        "Evaluated on real corporate earnings calls and financial broadcasts across English and Japanese.",
        page=16, total=total
    )

    # Left chart
    add_box(s16, 0.65, 1.50, 5.85, 5.30, bg=BG_WHITE, border=BORDER_GRAY)
    s16.shapes.add_picture(str(assets["bench1"]), Inches(0.75), Inches(1.65), width=Inches(5.65))

    # Right Card 1: Benchmark Origin & Dataset Structure
    add_box(s16, 6.70, 1.50, 5.98, 1.78, bg=BG_WHITE, border=BLUE_BRAND, left_bar=BLUE_BRAND)
    add_paragraphs(s16, 6.90, 1.62, 5.65, 1.55, [
        {"text": "DATASET ORIGIN & GROUND-TRUTH STRUCTURE", "size": 10.5, "bold": True, "color": BLUE_BRAND, "space_after": 4},
        {"text": "• Source Corpora: `Earnings22` (English quarterly earnings calls) + `ReazonSpeech` (Japanese financial broadcasts).", "size": 9.8, "color": TEXT_DARK, "space_after": 3},
        {"text": "• Structure per Clip: Audio (`.wav`) + Ground-truth human transcript (`.txt`) + Categorical Emotion & Sentiment ground-truth generated and audited using Gemma 3 Flash LLM (with manual human audit on the 60-clip set to ensure 100% precision).", "size": 9.5, "color": TEXT_BODY},
    ])

    # Right Card 2: NVIDIA GTX 1650 Execution Latencies
    add_box(s16, 6.70, 3.42, 5.98, 1.58, bg=BG_SOFT_GREEN, border=GREEN_POS, left_bar=GREEN_POS)
    add_paragraphs(s16, 6.90, 3.52, 5.65, 1.38, [
        {"text": "NVIDIA GEFORCE GTX 1650 LATENCY (wrt 4.5s – 5.0s Audio Clip)", "size": 10.5, "bold": True, "color": GREEN_POS, "space_after": 3},
        {"text": "• Whisper Turbo INT8: ~180 ms (0.04x RTF)  |  Emotion2Vec+ Large: ~203 ms  |  Librosa Prosody: ~35 ms\n• Kev-0.8B (Q8_0 GGUF): ~170 ms  ➔  Total Pipeline: 592 ms (0.13x RTF, 7.6x faster than real-time)\n• Kev-4B (INT4 GGUF): ~2,530 ms  ➔  Total Pipeline: 2,950 ms (0.65x RTF, 1.5x real-time)", "size": 9.5, "color": NAVY_PRIMARY},
    ])

    # Right Table: Accuracy Summary
    t16_shape = s16.shapes.add_table(5, 4, Inches(6.70), Inches(5.12), Inches(5.98), Inches(1.68))
    t16 = t16_shape.table
    for i, w in enumerate([Inches(2.38), Inches(1.15), Inches(1.20), Inches(1.25)]):
        t16.columns[i].width = w
    for c_idx, h in enumerate(["Benchmark / Model", "8-Emotion", "3-Sentiment", "EN / JA Sent."]):
        format_cell(t16.cell(0, c_idx), h, size=9.0, bold=True, color=TEXT_WHITE, bg=NAVY_PRIMARY)

    rows16 = [
        ("60-Clip: SER Only", "55.0%", "63.3%", "93.3% / 33.3%"),
        ("60-Clip: Kev-0.8B INT4", "70.0%", "75.0%", "96.7% / 53.3%"),
        ("60-Clip: Kev-4B INT4", "78.3%", "83.3%", "90.0% / 76.7%"),
        ("2,000-Clip: Vocalis Best", "63.1%", "67.6%", "84.3% / 50.9%"),
    ]
    for r_idx, r in enumerate(rows16, start=1):
        hl = r_idx in (3, 4)
        bg = BG_SOFT_GREEN if hl else (BG_WHITE if r_idx % 2 == 1 else BG_LIGHT)
        format_cell(t16.cell(r_idx, 0), r[0], size=8.8, bold=hl, color=GREEN_POS if hl else TEXT_DARK, bg=bg)
        format_cell(t16.cell(r_idx, 1), r[1], size=8.8, bold=hl, color=TEXT_DARK, bg=bg, align=PP_ALIGN.CENTER)
        format_cell(t16.cell(r_idx, 2), r[2], size=8.8, bold=True, color=BLUE_BRAND if hl else TEXT_BODY, bg=bg, align=PP_ALIGN.CENTER)
        format_cell(t16.cell(r_idx, 3), r[3], size=8.8, color=TEXT_BODY, bg=bg, align=PP_ALIGN.CENTER)

    # --------------------------------------------------------------------------
    # SLIDE 17: BENCHMARK RESULTS PART 2 — ACTUAL SER VALIDATION DATASETS
    # --------------------------------------------------------------------------
    s17 = prs.slides.add_slide(blank)
    set_bg(s17)
    add_header(
        s17,
        "6. Benchmark Evaluation  |  Gold-Standard SER Validation",
        "Benchmark Results (Part 2): 122-Clip SER Validation Suite",
        "Tested across 4 standard emotional datasets: JVNV-JA (Japanese), TESS, SAVEE, and RAVDESS.",
        page=17, total=total
    )

    add_box(s17, 0.65, 1.50, 6.15, 5.30, bg=BG_WHITE, border=BORDER_GRAY)
    s17.shapes.add_picture(str(assets["bench2"]), Inches(0.75), Inches(1.65), width=Inches(5.95))

    t17_shape = s17.shapes.add_table(6, 4, Inches(7.0), Inches(1.50), Inches(5.68), Inches(3.40))
    t17 = t17_shape.table
    for i, w in enumerate([Inches(2.18), Inches(1.15), Inches(1.15), Inches(1.20)]):
        t17.columns[i].width = w
    for c_idx, h in enumerate(["SER Corpus", "Emotion2Vec+", "Kev-0.8B", "Kev-4B INT4"]):
        format_cell(t17.cell(0, c_idx), h, size=9.5, bold=True, color=TEXT_WHITE, bg=NAVY_PRIMARY)

    rows17 = [
        ("JVNV-JA (25 Japanese)\nNatural Emotional", "64.0% Emo\n88.0% Sent", "92.0% Emo\n96.0% Sent", "96.0% Emo ✅\n100.0% Sent ✅"),
        ("TESS (36 English)\n'Say the word <X>'", "100.0% Emo\n100.0% Sent", "36.1% Emo\n41.7% Sent", "91.7% Emo ✅\n86.1% Sent ✅"),
        ("SAVEE (25 English)\nNeutral Carrier Text", "84.0% Emo\n88.0% Sent", "20.0% Emo\n48.0% Sent", "48.0% Emo\n72.0% Sent"),
        ("RAVDESS (36 English)\n'Dogs are sitting...'", "88.9% Emo\n97.2% Sent", "30.6% Emo\n30.6% Sent", "52.8% Emo\n55.6% Sent"),
        ("OVERALL (122 Clips)", "86.1% Emo\n94.3% Sent", "42.6% Emo\n50.8% Sent", "72.1% Emo\n77.1% Sent"),
    ]
    for r_idx, r in enumerate(rows17, start=1):
        hl = r_idx in (1, 2, 5)
        bg = BG_SOFT_GREEN if hl else (BG_WHITE if r_idx % 2 == 1 else BG_LIGHT)
        format_cell(t17.cell(r_idx, 0), r[0], size=9.2, bold=True, color=NAVY_PRIMARY, bg=bg)
        format_cell(t17.cell(r_idx, 1), r[1], size=9.0, color=TEXT_BODY, bg=bg, align=PP_ALIGN.CENTER)
        format_cell(t17.cell(r_idx, 2), r[2], size=9.0, color=TEXT_MUTED, bg=bg, align=PP_ALIGN.CENTER)
        format_cell(t17.cell(r_idx, 3), r[3], size=9.2, bold=True, color=GREEN_POS if hl else TEXT_DARK, bg=bg, align=PP_ALIGN.CENTER)

    add_box(s17, 7.0, 5.08, 5.68, 1.72, bg=BG_SOFT_GREEN, border=GREEN_POS, left_bar=GREEN_POS)
    add_paragraphs(s17, 7.22, 5.18, 5.3, 1.55, [
        {"text": "Why Kev-4B Dominates Kev-0.8B (+29.5% Accuracy):", "size": 11, "bold": True, "color": GREEN_POS, "space_after": 4},
        {"text": "• On `JVNV-JA` (real emotional speech), Kev-4B reaches 96% Emotion & 100% Sentiment (+32% over SER alone).\n• On `TESS` ('Say the word ripe'), Kev-0.8B gets deceived by neutral text (36.1%), whereas Kev-4B correctly weights the acoustic prior (91.7%).", "size": 9.8, "color": TEXT_DARK},
    ])

    # --------------------------------------------------------------------------
    # SLIDE 18: MAJOR FOCUS #2 — CHALLENGES FACED & POC OBSERVATIONS
    # --------------------------------------------------------------------------
    s18 = prs.slides.add_slide(blank)
    set_bg(s18)
    add_header(
        s18,
        "7. Major Focus #2  |  Challenges & Observations",
        "5 Key Engineering Challenges Faced & POC Observations",
        "Real-world issues uncovered during our POC and the clean engineering fixes applied to the production pipeline.",
        page=18, total=total
    )

    ch_list18 = [
        ("1. 300x Audio Gain Boost Erased Quiet Prosody",
         "Auto-normalizing quiet mic recordings (300x boost) inflated baseline volume to 75.8 dB and distorted background hiss into rapid false speech.",
         "Routed un-normalized raw audio to Librosa (+ silence trimming), while sending normalized audio only to Whisper & Emotion2Vec+.",
         RED_NEG),
        ("2. Bilingual SER Label Leakage ('难过/sad')",
         "Emotion2Vec+ outputs Chinese/English labels ('难过/sad'). Passing raw unsplit strings into the English Live Studio prompt weakened Kev.",
         "Added `clean_e2v_label()` to strip Chinese prefixes before formatting the Kev `/v1/systemone` prompt.",
         AMBER_WARN),
        ("3. 4 GB VRAM Ceiling & PyTorch CUDA Overhead",
         "Loading all models together requires 5.2 GB VRAM, and Windows PyTorch NF4 lacked Triton kernels (running 3.7x slower).",
         "Built 4-stage GPU eviction and offloaded Kev to an isolated C++ `llama-server.exe` GGUF process (3.7x speedup, zero OOMs).",
         BLUE_BRAND),
        ("4. Japanese Pitch-Accent SER Hallucinations",
         "Standalone SER failed on 73.9% of Japanese clips, predicting 'happy' with 95% confidence due to natural Japanese pitch-accent.",
         "Used Early Prompt Priming (instead of Late Math Fusion) so Kev can override false 'happy' SER priors using the transcript.",
         INDIGO_BRAND),
        ("5. Model Size Threshold for Sarcasm (0.8B vs 4B)",
         "On polite sarcastic audio ('Thank you' spoken sadly), Kev-0.8B trusted the polite words (83% conf), missing the vocal contradiction.",
         "Standardized Live Studio on `Kev-4B INT4 (Q4_K_M.gguf)`, which detects sarcasm (57.2%) and flips sentiment to Negative.",
         GREEN_POS),
    ]

    for idx, (ctit, cprob, cfix, col) in enumerate(ch_list18):
        y = 1.48 + idx * 1.08
        add_box(s18, 0.65, y, 12.033, 0.96, bg=BG_WHITE, border=BORDER_GRAY, left_bar=col)
        add_paragraphs(s18, 0.88, y + 0.10, 3.25, 0.78, [
            {"text": f"CHALLENGE #{idx+1}", "size": 8.5, "bold": True, "color": col, "space_after": 2},
            {"text": ctit, "size": 11, "bold": True, "color": NAVY_PRIMARY},
        ])
        add_paragraphs(s18, 4.25, y + 0.10, 4.15, 0.78, [
            {"text": "OBSERVATION / ROOT CAUSE", "size": 8.5, "bold": True, "color": TEXT_MUTED, "space_after": 2},
            {"text": cprob, "size": 9.8, "color": TEXT_BODY},
        ])
        add_paragraphs(s18, 8.55, y + 0.10, 4.0, 0.78, [
            {"text": "POC SOLUTION APPLIED", "size": 8.5, "bold": True, "color": GREEN_POS, "space_after": 2},
            {"text": cfix, "size": 9.8, "bold": True, "color": TEXT_DARK},
        ])

    # --------------------------------------------------------------------------
    # SLIDE 19: MAJOR FOCUS #1 — FINAL RECOMMENDATIONS
    # --------------------------------------------------------------------------
    s19 = prs.slides.add_slide(blank)
    set_bg(s19)
    add_header(
        s19,
        "8. Major Focus #1  |  Final Recommendations",
        "Final Recommendations & Production Rollout Strategy",
        "Four concrete recommendations from the Vocalis POC for enterprise deployment.",
        page=19, total=total
    )

    recs19 = [
        ("RECOMMENDATION 1", "Deploy Kev-4B INT4 (GGUF) as Primary Arbitrator",
         "• Use `Kev-4B-Q4_K_M.gguf` via C++ `llama-server.exe` for Live Studio and customer-facing analytics.\n"
         "• Essential for resolving sarcasm, passive-aggression, and vocal-lexical conflicts (83.3% call sentiment, 100% JVNV-JA sentiment).",
         GREEN_POS, BG_SOFT_GREEN),
        ("RECOMMENDATION 2", "Use Kev-0.8B (Q8_0 GGUF) for High-Speed Batch Archival",
         "• Deploy `Kev-0.8B-Q8_0.gguf` when processing large backlogs of historical recordings where sub-second speed is priority.\n"
         "• Runs at 1.51 clips/sec (592 ms/clip) in just 803 MB VRAM with 83.4% English sentiment accuracy.",
         BLUE_BRAND, BG_SOFT_BLUE),
        ("RECOMMENDATION 3", "Surface 3-Way Sentiment KPIs + 8-Emotion Drill-Down",
         "• Present 3-Way Sentiment (`Positive`, `Neutral`, `Negative`) with the automatic `sarcasm_flag` flip as the top-level business metric.\n"
         "• Retain the 8 canonical emotions (`outrage_hostility`, `urgency_panic`, `disappointed_sad`) for compliance escalation.",
         INDIGO_BRAND, BG_WHITE),
        ("RECOMMENDATION 4", "Enforce Dual-Stream Audio DSP & Early System One Priming",
         "• Never run prosody extraction on gain-normalized audio, and avoid post-LLM probability averaging.\n"
         "• Keep fusion strictly inside System One's prompt attention to protect multilingual accuracy from SER accent bias.",
         AMBER_WARN, BG_SOFT_AMBER),
    ]

    for idx, (rtag, rtit, rbody, col, sbg) in enumerate(recs19):
        c = idx % 2
        r = idx // 2
        x = 0.65 + c * 6.12
        y = 1.52 + r * 2.38
        add_box(s19, x, y, 5.91, 2.18, bg=BG_WHITE, border=col, top_bar=col)
        add_badge(s19, x + 0.22, y + 0.18, 1.9, 0.28, rtag, sbg, col, 8.8)
        add_paragraphs(s19, x + 0.22, y + 0.54, 5.45, 1.5, [
            {"text": rtit, "size": 13, "bold": True, "color": NAVY_PRIMARY, "space_after": 6},
            {"text": rbody, "size": 10.2, "color": TEXT_BODY},
        ])

    add_box(s19, 0.65, 6.32, 12.033, 0.58, bg=NAVY_PRIMARY, border=None)
    add_paragraphs(s19, 0.90, 6.44, 11.5, 0.35, [
        {"text": "Thank You  |  Vocalis POC Presentation by Vaibhav & Balram  |  Live Demo Running at http://localhost:5173", "size": 11.5, "bold": True, "color": TEXT_WHITE, "align": PP_ALIGN.CENTER}
    ])

    out_path = ROOT / "Vocalis_Business_Presentation_v4_Vaibhav_Balram.pptx"
    prs.save(str(out_path))
    print(f"[✓] Created complete 19-slide v4 presentation: {out_path}")


if __name__ == "__main__":
    build_v4_deck()
