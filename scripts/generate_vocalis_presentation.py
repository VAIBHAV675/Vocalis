import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent.parent

# ==============================================================================
# EXECUTIVE COLOR PALETTE (SLEEK DARK SLATE / INDIGO / EMERALD / ROSE)
# ==============================================================================
BG_DARK       = RGBColor(11, 17, 32)     # #0B1120 Deep Executive Navy
BG_CARD       = RGBColor(19, 28, 49)     # #131C31 Elevated Card Slate
BG_CARD_ALT   = RGBColor(26, 38, 66)     # #1A2642 Highlighted Card Slate
BG_CODE       = RGBColor(15, 23, 42)     # #0F172A Code / Prompt Block
BORDER_SLATE  = RGBColor(51, 65, 85)     # #334155 Subtle Card Border

TEXT_WHITE    = RGBColor(248, 250, 252)  # #F8FAFC Primary Heading
TEXT_LIGHT    = RGBColor(203, 213, 225)  # #CBD5E1 Body Copy
TEXT_MUTED    = RGBColor(148, 163, 184)  # #94A3B8 Captions / Subtitles

ACCENT_INDIGO = RGBColor(99, 102, 241)   # #6366F1 Primary Brand Indigo
ACCENT_VIOLET = RGBColor(139, 92, 246)   # #8B5CF6 Secondary Violet
ACCENT_CYAN   = RGBColor(56, 189, 248)   # #38BDF8 Neutral / Info Cyan
ACCENT_GREEN  = RGBColor(16, 185, 129)   # #10B981 Positive / Success Emerald
ACCENT_AMBER  = RGBColor(245, 158, 11)   # #F59E0B Warning / Prosody Amber
ACCENT_ROSE   = RGBColor(244, 63, 94)    # #F43F5E Negative / Sarcasm Rose

FONT_MAIN = "Segoe UI"
FONT_MONO = "Consolas"


def set_slide_bg(slide, color=BG_DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, category: str, title: str, subtitle: str = "", page_num: int = 1, total_pages: int = 11):
    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT_INDIGO
    bar.line.fill.background()

    # Category pill / eyebrow
    cat_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.22), Inches(8.0), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = FONT_MAIN
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN

    # Main Title
    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.46), Inches(11.5), Inches(0.5))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_MAIN
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE

    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.92), Inches(12.1), Inches(0.35))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = FONT_MAIN
        p_sub.font.size = Pt(11.5)
        p_sub.font.color.rgb = TEXT_MUTED

    # Footer
    div = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(7.08), Inches(12.133), Inches(0.015))
    div.fill.solid()
    div.fill.fore_color.rgb = BORDER_SLATE
    div.line.fill.background()

    foot_left = slide.shapes.add_textbox(Inches(0.6), Inches(7.14), Inches(8.0), Inches(0.28))
    tf_fl = foot_left.text_frame
    tf_fl.margin_left = tf_fl.margin_top = 0
    p_fl = tf_fl.paragraphs[0]
    p_fl.text = "VOCALIS  |  Multimodal Speech Emotion & Sentiment Intelligence POC  |  Presenters: Vaibhav & Balram"
    p_fl.font.name = FONT_MAIN
    p_fl.font.size = Pt(8.5)
    p_fl.font.color.rgb = TEXT_MUTED

    foot_right = slide.shapes.add_textbox(Inches(10.5), Inches(7.14), Inches(2.233), Inches(0.28))
    tf_fr = foot_right.text_frame
    tf_fr.margin_right = tf_fr.margin_top = 0
    p_fr = tf_fr.paragraphs[0]
    p_fr.alignment = PP_ALIGN.RIGHT
    p_fr.text = f"Slide {page_num} of {total_pages}"
    p_fr.font.name = FONT_MAIN
    p_fr.font.size = Pt(8.5)
    p_fr.font.bold = True
    p_fr.font.color.rgb = TEXT_MUTED


def add_card(slide, left, top, width, height, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=None, accent_left=None):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg_color
    if border_color:
        card.line.color.rgb = border_color
        card.line.width = Pt(1)
    else:
        card.line.fill.background()

    if accent_top:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.06))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_top
        bar.line.fill.background()

    if accent_left:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.07), Inches(height))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_left
        bar.line.fill.background()

    return card


def add_pill(slide, left, top, width, height, text, bg_color, text_color=TEXT_WHITE, font_size=9.5):
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.rgb = bg_color
    pill.line.fill.background()
    tf = pill.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = text
    p.font.name = FONT_MAIN
    p.font.size = Pt(font_size)
    p.font.bold = True
    p.font.color.rgb = text_color
    return pill


def add_text_block(slide, left, top, width, height, lines, default_size=11, default_color=TEXT_LIGHT, line_spacing=1.15):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for idx, item in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        if isinstance(item, str):
            p.text = item
            p.font.name = FONT_MAIN
            p.font.size = Pt(default_size)
            p.font.color.rgb = default_color
        elif isinstance(item, dict):
            p.space_before = Pt(item.get("space_before", 2))
            p.space_after = Pt(item.get("space_after", 3))
            if "align" in item:
                p.alignment = item["align"]
            runs = item.get("runs")
            if runs:
                for r_spec in runs:
                    r = p.add_run()
                    r.text = r_spec.get("text", "")
                    r.font.name = r_spec.get("font", FONT_MAIN)
                    r.font.size = Pt(r_spec.get("size", default_size))
                    r.font.bold = r_spec.get("bold", False)
                    r.font.color.rgb = r_spec.get("color", default_color)
            else:
                p.text = item.get("text", "")
                p.font.name = item.get("font", FONT_MAIN)
                p.font.size = Pt(item.get("size", default_size))
                p.font.bold = item.get("bold", False)
                p.font.color.rgb = item.get("color", default_color)
    return tb


def style_table_cell(cell, text, font_size=10, bold=False, color=TEXT_LIGHT, bg_color=None, align=PP_ALIGN.LEFT, font_name=FONT_MAIN):
    if bg_color:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg_color
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.1)
    cell.margin_right = Inches(0.1)
    cell.margin_top = Inches(0.05)
    cell.margin_bottom = Inches(0.05)
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    total_slides = 11

    # ==========================================================================
    # SLIDE 1: EXECUTIVE COVER SLIDE
    # ==========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)

    # Top & Left decorative accent bars
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = ACCENT_INDIGO
    top_bar.line.fill.background()

    add_pill(s1, 0.8, 0.9, 4.2, 0.36, "EXECUTIVE POC REPORT & ARCHITECTURE REVIEW", BG_CARD_ALT, ACCENT_CYAN, 10)

    add_text_block(s1, 0.8, 1.45, 11.7, 2.2, [
        {"text": "VOCALIS: Tri-Modal Speech Emotion", "size": 38, "bold": True, "color": TEXT_WHITE, "space_after": 2},
        {"text": "& Sentiment Intelligence Platform", "size": 36, "bold": True, "color": ACCENT_CYAN, "space_after": 14},
        {"text": "Bridging Biomechanical Prosody, Speech Foundation Models (Emotion2Vec+), and Single-Pass LLM Arbitration (Kev) for Multilingual Enterprise Audio within a 4 GB Edge VRAM Budget.", "size": 14, "color": TEXT_LIGHT},
    ])

    # Presenter & Scope Banner Card
    add_card(s1, 0.8, 3.65, 11.733, 1.05, bg_color=BG_CARD, border_color=ACCENT_INDIGO, accent_left=ACCENT_INDIGO)
    add_text_block(s1, 1.05, 3.80, 5.5, 0.75, [
        {"text": "PRESENTED BY", "size": 9.5, "bold": True, "color": ACCENT_CYAN, "space_after": 4},
        {"text": "Vaibhav & Balram", "size": 20, "bold": True, "color": TEXT_WHITE},
    ])
    add_text_block(s1, 6.2, 3.80, 6.1, 0.75, [
        {"text": "POC FOCUS AREAS", "size": 9.5, "bold": True, "color": ACCENT_AMBER, "space_after": 4},
        {"text": "1. Research Summary, Shortlisted Options, Trade-offs & Final Recommendations\n2. Concise Tech Stack Setup, Engineering Challenges & 2,000-Clip + SER Benchmarks", "size": 10.5, "color": TEXT_LIGHT},
    ])

    # 4 Executive Highlight Metric Cards at Bottom
    kpis = [
        ("83.3% / 96.7%", "3-Way Sentiment Accuracy", "83.3% overall (Kev-4B) & 96.7% EN (Kev-0.8B) on 60-clip audited benchmark", ACCENT_GREEN),
        ("100% Sentiment", "Japanese JVNV SER Validation", "96.0% 8-emotion & 100% 3-sentiment accuracy on Japanese emotional speech (+32% vs SER)", ACCENT_CYAN),
        ("3.7x Faster", "C++ CUDA GGUF Engine", "Single forward-pass PointerHead (0 generated tokens) via llama-server.exe", ACCENT_INDIGO),
        ("< 4.0 GB VRAM", "Zero-OOM Edge Orchestration", "Staged load-infer-evict pipeline running ~5.2 GB of models on a 4 GB GTX 1650", ACCENT_AMBER),
    ]
    for idx, (val, lbl, desc, col) in enumerate(kpis):
        x = 0.8 + idx * 2.98
        add_card(s1, x, 5.05, 2.80, 1.85, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=col)
        add_text_block(s1, x + 0.18, 5.22, 2.44, 1.55, [
            {"text": val, "size": 21, "bold": True, "color": col, "space_after": 4},
            {"text": lbl, "size": 11.5, "bold": True, "color": TEXT_WHITE, "space_after": 6},
            {"text": desc, "size": 9.5, "color": TEXT_MUTED},
        ])

    # ==========================================================================
    # SLIDE 2: LAYMAN GUIDE — HOW AUDIO RELATES TO SPEECH & WHY TEXT ALONE FAILS
    # ==========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(
        s2,
        "Foundations  |  Speech vs. Text",
        "Why Words Alone Lie: How Audio Encodes True Human Emotion",
        "Standard ASR transcripts strip away 70% of paralinguistic signal—making identical words mean opposite things.",
        page_num=2, total_pages=total_slides
    )

    # Top Diagram: 3 Pillars of Spoken Communication
    pillars = [
        ("1. LEXICAL CONTENT (WHAT IS SAID)", "The literal transcribed words", "Captured by ASR (Faster-Whisper).\nExample: \"Thank you so much for the help today.\"\nRisk: Polite words frequently mask frustration, sadness, or sarcasm.", ACCENT_CYAN),
        ("2. BIOMECHANICAL PROSODY (HOW IT IS SAID)", "Physical vocal dynamics (Librosa)", "Pitch contour (Hz & slope), syllable speed (onsets/sec), vocal loudness (dB), and voice tremor.\nReveals physical tension, sighing, or shouting.", ACCENT_AMBER),
        ("3. DEEP ACOUSTIC TIMBRE (SER PRIOR)", "Sub-lexical emotional resonance", "Captured by Speech Foundation Models (Emotion2Vec+).\nDetects micro-modulations in vocal tract resonance across 1024-dim audio embeddings.", ACCENT_INDIGO),
    ]
    for i, (ptitle, psub, pbody, pcol) in enumerate(pillars):
        x = 0.6 + i * 4.1
        add_card(s2, x, 1.38, 3.93, 2.05, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=pcol)
        add_text_block(s2, x + 0.18, 1.52, 3.57, 1.8, [
            {"text": ptitle, "size": 10.5, "bold": True, "color": pcol, "space_after": 2},
            {"text": psub, "size": 11.5, "bold": True, "color": TEXT_WHITE, "space_after": 6},
            {"text": pbody, "size": 9.8, "color": TEXT_LIGHT},
        ])

    # Bottom Section: Same Sentence Spoken in 3 Different Intonations
    add_card(s2, 0.6, 3.62, 12.133, 3.30, bg_color=BG_CARD_ALT, border_color=ACCENT_INDIGO)
    add_text_block(s2, 0.85, 3.75, 11.6, 0.45, [
        {"runs": [
            {"text": "REAL POC CASE STUDY: ", "size": 11, "bold": True, "color": ACCENT_AMBER},
            {"text": "One Exact Sentence — ", "size": 11, "bold": True, "color": TEXT_WHITE},
            {"text": "“Thank you so much for the help today.”", "size": 12, "bold": True, "color": ACCENT_CYAN},
            {"text": " — Spoken in 3 Different Intonations", "size": 11, "bold": True, "color": TEXT_WHITE},
        ]}
    ])

    tones = [
        (
            "INTONATION A: WARM & APPRECIATIVE",
            "POSITIVE SENTIMENT",
            ACCENT_GREEN,
            [
                "• Pitch: 225 Hz (rising terminal contour)",
                "• Tempo: 5.1 onsets/sec (lively, fluid pace)",
                "• Energy: 75 dB (warm resonant projection)",
                "• SER Prior: happy (94% confidence)",
                "→ True State: satisfaction_gratitude",
            ]
        ),
        (
            "INTONATION B: FLAT ROUTINE CLOSING",
            "NEUTRAL SENTIMENT",
            ACCENT_CYAN,
            [
                "• Pitch: 160 Hz (narrow std: 14 Hz, flat)",
                "• Tempo: 4.5 onsets/sec (standard business)",
                "• Energy: 70 dB (nominal conversational)",
                "• SER Prior: neutral (89% confidence)",
                "→ True State: neutral_inquiry",
            ]
        ),
        (
            "INTONATION C: SUBDUED / SARCASTIC (POC CLIP)",
            "NEGATIVE SENTIMENT (FLIP)",
            ACCENT_ROSE,
            [
                "• Pitch: 193 Hz (downward falling contour)",
                "• Tempo: 3.4 onsets/sec (slow, subdued pace)",
                "• SER Prior: sad (82% confidence)",
                "• Kev-4B Arbitration: Sarcasm = 57.2% (True)",
                "→ True State: Negative (62.3% polarity)",
            ]
        ),
    ]
    for i, (ttitle, badge, col, bullets) in enumerate(tones):
        x = 0.85 + i * 3.92
        add_card(s2, x, 4.25, 3.76, 2.48, bg_color=BG_DARK, border_color=col, accent_left=col)
        add_pill(s2, x + 0.18, 4.38, 2.2, 0.26, badge, col, TEXT_WHITE, 8.5)
        add_text_block(s2, x + 0.18, 4.70, 3.45, 1.95, [
            {"text": ttitle, "size": 10, "bold": True, "color": TEXT_WHITE, "space_after": 5},
            *[{"text": b, "size": 9.5, "color": TEXT_LIGHT, "space_after": 2} for b in bullets]
        ])

    # ==========================================================================
    # SLIDE 3: HOW SER & ACOUSTIC PROSODY WORK (LAYMAN VISUAL DIAGRAM)
    # ==========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(
        s3,
        "Foundations  |  Acoustics & Prosody",
        "How SER & Biomechanical Prosody Decode Vocal Emotion",
        "Vocalis pairs a neural Speech Foundation Model (Emotion2Vec+) with explicit physical DSP biomarkers (Librosa).",
        page_num=3, total_pages=total_slides
    )

    # Top Visual Flow Diagram (5 Connected Blocks)
    flow_steps = [
        ("Raw Audio (.wav)", "16kHz Mono Signal\nUn-normalized & Gain-Normalized Streams", BORDER_SLATE),
        ("Silero VAD + Whisper", "Spoken Segment Isolation\nExact Multilingual Text Transcript", ACCENT_CYAN),
        ("Librosa Prosody DSP", "Physical Vocal Biomarkers\nF0 Pitch, Tempo, dB Energy, Jitter", ACCENT_AMBER),
        ("Emotion2Vec+ Large", "Deep Acoustic SER Prior\n1024-Dim Utterance Representation", ACCENT_VIOLET),
        ("Kev Arbitrator", "Tri-Modal Synthesis\n8 Emotions + Sarcasm + 3 Sentiments", ACCENT_GREEN),
    ]
    for idx, (ftitle, fdesc, fcol) in enumerate(flow_steps):
        x = 0.6 + idx * 2.48
        add_card(s3, x, 1.38, 2.22, 1.25, bg_color=BG_CARD, border_color=fcol, accent_top=fcol)
        add_text_block(s3, x + 0.12, 1.50, 1.98, 1.05, [
            {"text": f"STAGE {idx+1}", "size": 8, "bold": True, "color": fcol, "space_after": 1},
            {"text": ftitle, "size": 10.5, "bold": True, "color": TEXT_WHITE, "space_after": 4},
            {"text": fdesc, "size": 8.8, "color": TEXT_MUTED},
        ])
        if idx < 4:
            arr = s3.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.26), Inches(1.88), Inches(0.18), Inches(0.24))
            arr.fill.solid()
            arr.fill.fore_color.rgb = ACCENT_INDIGO
            arr.line.fill.background()

    # Bottom Matrix: How Acoustic Prosody Maps to Human Emotions
    tbl_shape = s3.shapes.add_table(5, 5, Inches(0.6), Inches(2.85), Inches(12.133), Inches(4.0))
    tbl = tbl_shape.table
    widths = [Inches(2.1), Inches(2.3), Inches(2.6), Inches(2.6), Inches(2.533)]
    for i, w in enumerate(widths):
        tbl.columns[i].width = w

    headers = [
        "Acoustic Feature (Librosa)",
        "What It Measures Physically",
        "High / Elevated Signal Means",
        "Low / Falling Signal Means",
        "Why Neural SER Alone Isn't Enough"
    ]
    for c_idx, h in enumerate(headers):
        style_table_cell(tbl.cell(0, c_idx), h, font_size=10, bold=True, color=TEXT_WHITE, bg_color=BG_CARD_ALT)

    rows = [
        (
            "Pitch Contour (F0 Hz & Slope)",
            "Vocal cord vibration frequency (YIN algorithm) & terminal trajectory",
            "High F0 (>210 Hz) / Wide Std (>45 Hz):\nUrgency, Panic, Outrage, or Animated Joy",
            "Falling Contour / Monotone (<20 Hz std):\nDisappointed Sadness, Dejection, Sarcasm",
            "Japanese broadcast pitch-accent naturally modulates F0, tricking SER into false 'happy' predictions."
        ),
        (
            "Speech Tempo (Onsets / sec)",
            "Syllable nuclei rate per second on silence-trimmed active speech",
            "Rapid Pace (>5.5 onsets/sec):\nFrantic Panic, Agitated Frustration, Hostility",
            "Slow / Measured Pace (<3.5 onsets/sec):\nSubdued Sadness, Reluctance, Ironic Drag",
            "Leading/trailing mic silence artificially lowers tempo unless active speech is trimmed first."
        ),
        (
            "Vocal Energy (RMS dB)",
            "Root-Mean-Square acoustic pressure relative to baseline floor",
            "High Intensity (>78 dB):\nShouting, Outrage, Escalated Confrontation",
            "Subdued Delivery (<65 dB):\nDefeated Sorrow, Quiet Resignation, Fatigue",
            "Auto-gain normalization (300x boost) erases whisper/quiet cues unless DSP reads raw audio."
        ),
        (
            "Voice Quality (Jitter & HNR)",
            "Cycle-to-cycle pitch perturbation & spectral centroid brightness",
            "High Jitter / Vocal Tremor:\nNervous Anxiety, Fear, Emotional Distress",
            "Clean Modal Phonation:\nCalm Routine Inquiry or Controlled Sarcasm",
            "Gives the LLM explicit physical descriptors ('tense', 'trembling') alongside numeric anchors."
        ),
    ]
    for r_idx, r_data in enumerate(rows, start=1):
        row_bg = BG_CARD if r_idx % 2 == 1 else BG_DARK
        style_table_cell(tbl.cell(r_idx, 0), r_data[0], font_size=9.8, bold=True, color=ACCENT_AMBER, bg_color=row_bg)
        style_table_cell(tbl.cell(r_idx, 1), r_data[1], font_size=9.5, color=TEXT_LIGHT, bg_color=row_bg)
        style_table_cell(tbl.cell(r_idx, 2), r_data[2], font_size=9.5, color=ACCENT_ROSE, bg_color=row_bg)
        style_table_cell(tbl.cell(r_idx, 3), r_data[3], font_size=9.5, color=ACCENT_CYAN, bg_color=row_bg)
        style_table_cell(tbl.cell(r_idx, 4), r_data[4], font_size=9.5, color=TEXT_MUTED, bg_color=row_bg)

    # ==========================================================================
    # SLIDE 4: 8 CANONICAL EMOTIONS & 3-WAY SENTIMENT TAXONOMY
    # ==========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(
        s4,
        "Taxonomy & Business Polarity",
        "8 Canonical Emotions Mapped to 3-Way Actionable Sentiment",
        "Granular 8-way emotion detection rolls up into Positive, Neutral, and Negative business polarity with automatic sarcasm inversion.",
        page_num=4, total_pages=total_slides
    )

    # Column 1: POSITIVE
    add_card(s4, 0.6, 1.40, 3.65, 5.45, bg_color=BG_CARD, border_color=ACCENT_GREEN, accent_top=ACCENT_GREEN)
    add_pill(s4, 0.85, 1.62, 1.9, 0.30, "POSITIVE SENTIMENT", ACCENT_GREEN, TEXT_WHITE, 9.5)
    add_text_block(s4, 0.85, 2.05, 3.15, 2.1, [
        {"text": "1. satisfaction_gratitude", "size": 14, "bold": True, "color": TEXT_WHITE, "space_after": 6},
        {"text": "• Genuine client delight, relief, warm appreciation, and heartfelt gratitude.", "size": 10, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Acoustic Signature: Warm resonant energy, fluid tempo, rising/animated pitch contour.", "size": 10, "color": TEXT_MUTED, "space_after": 12},
    ])
    # Sarcasm Flip Callout inside Positive column
    add_card(s4, 0.82, 4.15, 3.21, 2.45, bg_color=BG_DARK, border_color=ACCENT_ROSE, accent_left=ACCENT_ROSE)
    add_text_block(s4, 1.0, 4.28, 2.9, 2.2, [
        {"text": "⚡ AUTOMATIC SARCASM FLIP RULE", "size": 10, "bold": True, "color": ACCENT_ROSE, "space_after": 6},
        {"text": "When Kev detects polite gratitude words paired with negative vocal acoustics (sarcasm_detected ≥ 0.45):", "size": 9.5, "color": TEXT_LIGHT, "space_after": 6},
        {"text": "satisfaction_gratitude + sarcasm_flag=True\n➔ Flips Polarity to NEGATIVE", "size": 10, "bold": True, "color": ACCENT_AMBER},
    ])

    # Column 2: NEUTRAL
    add_card(s4, 4.48, 1.40, 3.45, 5.45, bg_color=BG_CARD, border_color=ACCENT_CYAN, accent_top=ACCENT_CYAN)
    add_pill(s4, 4.73, 1.62, 1.85, 0.30, "NEUTRAL SENTIMENT", ACCENT_CYAN, BG_DARK, 9.5)
    add_text_block(s4, 4.73, 2.05, 2.95, 4.5, [
        {"text": "2. neutral_inquiry", "size": 14, "bold": True, "color": TEXT_WHITE, "space_after": 6},
        {"text": "• Calm routine business inquiries, objective operational questions, and matter-of-fact reporting.", "size": 10, "color": TEXT_LIGHT, "space_after": 6},
        {"text": "• Acoustic Signature: Steady baseline F0 contour, standard conversational pace (4–5.5 onsets/sec), balanced modal phonation.", "size": 10, "color": TEXT_MUTED, "space_after": 14},
        {"text": "• Business Action: Standard SLA routing; no compliance escalation required.", "size": 10, "bold": True, "color": ACCENT_CYAN},
    ])

    # Column 3: NEGATIVE (6 Emotions)
    add_card(s4, 8.16, 1.40, 4.57, 5.45, bg_color=BG_CARD, border_color=ACCENT_ROSE, accent_top=ACCENT_ROSE)
    add_pill(s4, 8.41, 1.62, 2.0, 0.30, "NEGATIVE SENTIMENT (6 CLASSES)", ACCENT_ROSE, TEXT_WHITE, 9.5)

    neg_emos = [
        ("3. frustration_irritation", "Impatient complaints, repeated friction, vocal exasperation & tense tone."),
        ("4. outrage_hostility", "Extreme fury, shouting (>78 dB), aggressive confrontation, legal threats."),
        ("5. disappointed_sad", "Deep sorrow, emotional dejection, heavy sighing, falling pitch contour."),
        ("6. sarcasm_passive_aggression", "Ironic praise & mocking cynicism concealing sharp anger or contempt."),
        ("7. anxiety_confusion", "Nervous vocal tremor (high jitter), worried uncertainty, helpless distress."),
        ("8. urgency_panic", "Critical crisis, frantic rapid speech (>6 onsets/s), immediate freeze demand."),
    ]
    for idx, (ename, edesc) in enumerate(neg_emos):
        y = 2.05 + idx * 0.78
        add_card(s4, 8.38, y, 4.13, 0.70, bg_color=BG_DARK, border_color=BORDER_SLATE, accent_left=ACCENT_ROSE)
        add_text_block(s4, 8.55, y + 0.06, 3.85, 0.58, [
            {"text": ename, "size": 10.5, "bold": True, "color": TEXT_WHITE, "space_after": 1},
            {"text": edesc, "size": 8.8, "color": TEXT_MUTED},
        ])

    # ==========================================================================
    # SLIDE 5: RESEARCH SUMMARY — SHORTLISTED OPTIONS & TRADE-OFFS (MAJOR FOCUS #1)
    # ==========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(
        s5,
        "Major Focus #1  |  Research & Architecture Trade-Offs",
        "Shortlisted Options Evaluated, Trade-Offs & Selection Matrix",
        "We benchmarked 5 distinct paradigms across accuracy, cross-lingual robustness, sarcasm resolution, and 4 GB VRAM feasibility.",
        page_num=5, total_pages=total_slides
    )

    tbl5_shape = s5.shapes.add_table(6, 6, Inches(0.6), Inches(1.40), Inches(12.133), Inches(5.45))
    t5 = tbl5_shape.table
    w5 = [Inches(2.25), Inches(1.95), Inches(1.45), Inches(1.55), Inches(2.58), Inches(2.353)]
    for i, w in enumerate(w5):
        t5.columns[i].width = w

    h5 = [
        "Shortlisted Option",
        "Models Evaluated",
        "Accuracy (EN / JA)",
        "Latency & VRAM",
        "Key Trade-Offs Observed in POC",
        "Final Verdict"
    ]
    for c_idx, h in enumerate(h5):
        style_table_cell(t5.cell(0, c_idx), h, font_size=10, bold=True, color=TEXT_WHITE, bg_color=BG_CARD_ALT)

    rows5 = [
        (
            "1. Standalone Acoustic SER",
            "• Emotion2Vec+ Large\n• ehcalabres XLSR-EN\n• Bagus XLSR-JA",
            "EN: 75.7%–83.3%\nJA: 26.1%–26.7%\n(ehcalabres: 16.7%)",
            "203 ms / clip\n~1.24 GB VRAM",
            "Pros: Fast; 86.1% on acted SER (TESS/RAVDESS).\nCons: Zero lexical understanding; collapses to 26.1% on Japanese due to pitch-accent bias.",
            "❌ Reject as Standalone\n(Keep Emotion2Vec+ as Stage-2 Acoustic Prior)",
            ACCENT_ROSE
        ),
        (
            "2. Encoder Text+Audio Router",
            "• Laya mmBERT\n  (Multilingual Encoder)",
            "EN: 66.7%\nJA: 66.7%\nOverall: 66.7%",
            "112 ms / clip\n~0.95 GB VRAM",
            "Pros: Ultra-low latency; balanced EN/JA.\nCons: Shallow cross-attention; misses nuanced sarcasm and complex banking domain phrasing.",
            "⚠️ Superseded\n(Outperformed by Kev-0.8B & Kev-4B)",
            ACCENT_AMBER
        ),
        (
            "3. Native Audio Omni LLM",
            "• Qwen2.5-Omni-3B\n  (Q4_K_M GGUF +\n  mmproj-Q8_0 GGUF)",
            "Fails on subtle tone\n(Predicts 'happy' on sad/sarcastic clip)",
            "23,670 ms / clip\n~3.64 GB Disk/VRAM",
            "Pros: Direct audio-to-text architecture.\nCons: Experimental llama.cpp mtmd audio projector loses prosody, hallucinates ASR, 10x slower.",
            "❌ Reject for 4 GB Edge\n(Too slow & quantized mmproj degrades tone)",
            ACCENT_ROSE
        ),
        (
            "4. Post-LLM Score Fusion",
            "• Linear / Confidence\n  Weighted Fusion of\n  SER + Kev Probs",
            "60-Clip: 60.0%\n(Drops -18.3% vs\nPrompt Priming)",
            "650–3,000 ms\n~1.2–2.4 GB VRAM",
            "Pros: Boosts low-confidence acoustic clips.\nCons: Double-counts SER and biases heavily toward Emotion2Vec+'s 95%-confident Japanese errors.",
            "❌ Reject Late Fusion\n(Over-weights noisy SER priors on JA)",
            ACCENT_ROSE
        ),
        (
            "5. Tri-Modal Prompt Priming + Kev (WINNER)",
            "• Whisper Turbo INT8\n• Emotion2Vec+ Large\n• Librosa DSP\n• Kev-4B / 0.8B GGUF",
            "60-Clip: 78.3% Emo\n83.3% Sentiment\nJVNV-JA: 96% Emo /\n100% Sentiment",
            "Kev-0.8B: 592 ms\nKev-4B: 2,950 ms\nFits <4 GB via Eviction",
            "Pros: Single forward-pass PointerHead (0 tokens generated); resolves vocal-lexical sarcasm; +32% over SER on Japanese emotional speech.",
            "✅ ADOPTED FOR PROD\n(Kev-4B for Accuracy;\nKev-0.8B for Speed)",
            ACCENT_GREEN
        ),
    ]
    for r_idx, r in enumerate(rows5, start=1):
        is_winner = (r_idx == 5)
        bg = BG_CARD_ALT if is_winner else (BG_CARD if r_idx % 2 == 1 else BG_DARK)
        style_table_cell(t5.cell(r_idx, 0), r[0], font_size=9.5, bold=True, color=ACCENT_GREEN if is_winner else TEXT_WHITE, bg_color=bg)
        style_table_cell(t5.cell(r_idx, 1), r[1], font_size=9.0, color=TEXT_LIGHT, bg_color=bg)
        style_table_cell(t5.cell(r_idx, 2), r[2], font_size=9.0, bold=is_winner, color=ACCENT_CYAN if is_winner else TEXT_LIGHT, bg_color=bg)
        style_table_cell(t5.cell(r_idx, 3), r[3], font_size=9.0, color=TEXT_LIGHT, bg_color=bg)
        style_table_cell(t5.cell(r_idx, 4), r[4], font_size=8.8, color=TEXT_LIGHT, bg_color=bg)
        style_table_cell(t5.cell(r_idx, 5), r[5], font_size=9.0, bold=True, color=r[6], bg_color=bg)

    # ==========================================================================
    # SLIDE 6: HIGH-VALUE DEEP DIVE — TRI-MODAL ARBITRAGE & THE KEV PROMPT
    # ==========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(
        s6,
        "Core Innovation  |  Tri-Modal Arbitrage & Prompt Engineering",
        "Inside the Kev SystemOne Prompt & Single-Pass PointerHead",
        "Instead of generating slow text tokens, Kev reads the tri-modal state in one forward pass and outputs calibrated probabilities.",
        page_num=6, total_pages=total_slides
    )

    # Left Card: Exact Kev Prompt Structure
    add_card(s6, 0.6, 1.40, 6.2, 5.45, bg_color=BG_CODE, border_color=ACCENT_INDIGO, accent_top=ACCENT_INDIGO)
    add_text_block(s6, 0.82, 1.55, 5.8, 0.35, [
        {"text": "EXACT TRI-MODAL PROMPT (`state`) SENT TO `/v1/systemone`", "size": 10.5, "bold": True, "color": ACCENT_CYAN}
    ])

    prompt_lines = [
        {"text": "Speaker transcript: 「Thank you so much for the help today.」", "font": FONT_MONO, "size": 9.5, "bold": True, "color": TEXT_WHITE, "space_after": 4},
        {"text": "Acoustic speech emotion prior: sad (confidence: 82%)", "font": FONT_MONO, "size": 9.5, "bold": True, "color": ACCENT_ROSE, "space_after": 4},
        {"text": "Prosodic acoustic features:", "font": FONT_MONO, "size": 9.5, "color": ACCENT_AMBER, "space_after": 2},
        {"text": "- Pitch contour: 193.0 Hz (std: 49.0 Hz, downward falling terminal contour)", "font": FONT_MONO, "size": 8.8, "color": TEXT_LIGHT, "space_after": 2},
        {"text": "- Speech tempo: 3.4 onsets/sec (slow / measured / subdued pace)", "font": FONT_MONO, "size": 8.8, "color": TEXT_LIGHT, "space_after": 2},
        {"text": "- Energy & vocal intensity: 73.5 dB (moderate baseline business volume)", "font": FONT_MONO, "size": 8.8, "color": TEXT_LIGHT, "space_after": 2},
        {"text": "- Voice quality: standard modal phonation, clean harmonic structure", "font": FONT_MONO, "size": 8.8, "color": TEXT_LIGHT, "space_after": 10},
        {"text": "QUESTIONS SCHEMA (`IB_QUESTIONS`):", "size": 10, "bold": True, "color": ACCENT_INDIGO, "space_after": 4},
        {"text": "1. detected_emotion (type: 'choice' -> 8 canonical criteria):\n   \"Analyze the caller's spoken statement in the full context of their vocal acoustics and delivery. Resolve any tension between the words and vocal tone (e.g. sarcasm, polite complaints, subdued relief).\"\n2. sarcasm_detected (type: 'noul' -> binary probability [0,1]):\n   \"Is the speaker using sarcasm, irony, or passive-aggressive politeness that contradicts their vocal tone?\"", "font": FONT_MONO, "size": 8.5, "color": TEXT_MUTED},
    ]
    add_text_block(s6, 0.82, 1.95, 5.75, 4.75, prompt_lines)

    # Right Top Card: Why Both Numbers & Natural Language Labels Are Included
    add_card(s6, 7.0, 1.40, 5.733, 2.55, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=ACCENT_AMBER)
    add_text_block(s6, 7.22, 1.55, 5.3, 2.3, [
        {"text": "WHY BOTH NUMBERS & NATURAL LANGUAGE LABELS?", "size": 11, "bold": True, "color": ACCENT_AMBER, "space_after": 6},
        {"text": "• Natural Language Labels Drive Semantic Attention: Pre-trained Qwen3.5 weights natively understand phrases like \"downward falling terminal contour\" and \"slow / measured / subdued pace\".", "size": 9.8, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Numeric Anchors Act as Relative Magnitude Scalars: While LLMs aren't calibrated DSP calculators on raw Hz alone, seeing `confidence: 82%` vs `45%` or `85 dB` vs `60 dB` provides relative intensity scaling.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Zero Inference Penalty: Because Kev's PointerHead performs 1 prefill pass (0 autoregressive tokens), extra descriptive tokens cost <1 ms.", "size": 9.8, "color": ACCENT_CYAN},
    ])

    # Right Bottom Card: Live Arbitrage Result on rec_1791807581.wav
    add_card(s6, 7.0, 4.12, 5.733, 2.73, bg_color=BG_CARD, border_color=ACCENT_GREEN, accent_top=ACCENT_GREEN)
    add_text_block(s6, 7.22, 4.27, 5.3, 2.45, [
        {"text": "HOW KEV-4B ARBITRATES THIS EXACT RECORDING", "size": 11, "bold": True, "color": ACCENT_GREEN, "space_after": 6},
        {"text": "1. Detects Lexical-Acoustic Conflict: Polite surface words (\"Thank you so much\") clash with `sad (82%)` + `downward falling contour`.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "2. Suppresses Overconfidence: Drops `satisfaction_gratitude` confidence from 0.912 (unprimed) down to 0.368, shifting 26.8% mass to `disappointed_sad` (19.0%) + `sarcasm` (7.8%).", "size": 9.8, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "3. Fires Sarcasm Head (`noul = 0.5715`): Sets `sarcasm_flag = True`, flipping final 3-way sentiment polarity to NEGATIVE (62.25%).", "size": 9.8, "bold": True, "color": ACCENT_ROSE},
    ])

    # ==========================================================================
    # SLIDE 7: ABLATION PROOF — WHY PROMPT PRIMING BEATS LATE SCORE FUSION
    # ==========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7)
    add_header(
        s7,
        "Research Findings  |  Priming vs. Fusion Mathematics",
        "Why Prompt Priming Beats Post-LLM Mathematical Score Fusion",
        "Controlled ablation experiments prove Kev needs the SER prior inside its prompt—and why post-LLM fusion biases toward SER errors.",
        page_num=7, total_pages=total_slides
    )

    # Top Table: 4-Way Controlled Ablation on rec_1791807581.wav
    t7_shape = s7.shapes.add_table(5, 6, Inches(0.6), Inches(1.40), Inches(12.133), Inches(3.15))
    t7 = t7_shape.table
    w7 = [Inches(2.9), Inches(1.4), Inches(1.6), Inches(1.7), Inches(1.7), Inches(2.833)]
    for i, w in enumerate(w7):
        t7.columns[i].width = w

    h7 = [
        "Ablation Configuration (Kev-4B)",
        "Kev Confidence",
        "Sarcasm (`noul`)",
        "Top Emotion Dist.",
        "3-Way Sentiment",
        "Mathematical / Attention Behavior"
    ]
    for c_idx, h in enumerate(h7):
        style_table_cell(t7.cell(0, c_idx), h, font_size=9.5, bold=True, color=TEXT_WHITE, bg_color=BG_CARD_ALT)

    rows7 = [
        (
            "A. Prosody Labels Only\n(No SER Prior, No Fusion)",
            "0.9123\n(Overconfident)",
            "15.9% (False) ❌",
            "92.3% Gratitude\n1.4% Sad",
            "Positive (92.3%) ❌",
            "Without the 'sad (82%)' SER prior, polite text overwhelms soft prosody adjectives."
        ),
        (
            "B. Prosody Labels Only\n+ Late Math Fusion w/ SER",
            "0.9123 vs\n0.8204 (SER)",
            "15.9% (False) ❌",
            "65.3% Gratitude\n28.1% Sad",
            "Positive (65.3%) ❌",
            "Because unprimed Kev is 91.2% confident, 70/30 late fusion still lets 'Gratitude' win."
        ),
        (
            "C. SER Priming + Late Math Fusion\n(Double-Counting SER)",
            "0.3683 vs\n0.8204 (SER)",
            "57.2% (True) ✅",
            "62.5% Sad\n13.9% Gratitude",
            "Negative ✅\n(But Biased!)",
            "Double-counts SER! Drops 60-clip benchmark from 78.3% to 60.0% due to JA SER errors."
        ),
        (
            "D. Tri-Modal Prompt Priming\n+ 3-Sentiment Rollup (PROD)",
            "0.3683\n(Calibrated)",
            "57.2% (True) ✅",
            "44.7% Grat (Sarc)\n19.0% Sad / 7.8% Sarc",
            "Negative (62.3%) ✅\n(Unbiased!)",
            "Sarcasm flag flips sentiment to Negative while preserving 78.3% / 83.3% benchmark accuracy!"
        ),
    ]
    for r_idx, r in enumerate(rows7, start=1):
        is_best = (r_idx == 4)
        bg = BG_CARD_ALT if is_best else (BG_CARD if r_idx % 2 == 1 else BG_DARK)
        style_table_cell(t7.cell(r_idx, 0), r[0], font_size=9.2, bold=True, color=ACCENT_GREEN if is_best else TEXT_WHITE, bg_color=bg)
        style_table_cell(t7.cell(r_idx, 1), r[1], font_size=9.0, color=TEXT_LIGHT, bg_color=bg)
        style_table_cell(t7.cell(r_idx, 2), r[2], font_size=9.0, bold=True, color=ACCENT_GREEN if "True" in r[2] else ACCENT_ROSE, bg_color=bg)
        style_table_cell(t7.cell(r_idx, 3), r[3], font_size=9.0, color=TEXT_LIGHT, bg_color=bg)
        style_table_cell(t7.cell(r_idx, 4), r[4], font_size=9.0, bold=True, color=ACCENT_GREEN if "✅" in r[4] else ACCENT_ROSE, bg_color=bg)
        style_table_cell(t7.cell(r_idx, 5), r[5], font_size=8.8, color=TEXT_LIGHT, bg_color=bg)

    # Bottom 2 Cards explaining the structural insight
    add_card(s7, 0.6, 4.75, 5.95, 2.10, bg_color=BG_CARD, border_color=ACCENT_ROSE, accent_left=ACCENT_ROSE)
    add_text_block(s7, 0.82, 4.88, 5.55, 1.85, [
        {"text": "WHY LATE MATHEMATICAL FUSION IS DANGEROUS", "size": 11, "bold": True, "color": ACCENT_ROSE, "space_after": 6},
        {"text": "• In Japanese financial broadcasts, speakers use formal politeness with wide pitch-accent modulation.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 3},
        {"text": "• Emotion2Vec+ hallucinates `happy` with 85%–99% confidence on 73.9% of Japanese clips.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 3},
        {"text": "• If we fuse probabilities after Kev, a 95%-confident wrong SER score overrides Kev's correct neutral/frustrated arbitration!", "size": 9.8, "bold": True, "color": ACCENT_AMBER},
    ])

    add_card(s7, 6.78, 4.75, 5.95, 2.10, bg_color=BG_CARD, border_color=ACCENT_GREEN, accent_left=ACCENT_GREEN)
    add_text_block(s7, 7.0, 4.88, 5.55, 1.85, [
        {"text": "WHY EARLY PROMPT PRIMING + SENTIMENT ROLLUP WINS", "size": 11, "bold": True, "color": ACCENT_GREEN, "space_after": 6},
        {"text": "• Inside Kev's transformer layers, self-attention weighs the SER prior against the transcript and Librosa prosody.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 3},
        {"text": "• When Japanese text is purely informational, Kev ignores false `happy` SER priors.", "size": 9.8, "color": TEXT_LIGHT, "space_after": 3},
        {"text": "• When English text is polite (`Thank you`) but acoustics are `sad (82%)` + `falling pitch`, Kev-4B triggers `sarcasm_flag = True` and flips sentiment to Negative!", "size": 9.8, "bold": True, "color": ACCENT_CYAN},
    ])

    # ==========================================================================
    # SLIDE 8: SETUP STEPS, TECH STACK & 4 GB VRAM ORCHESTRATION (MAJOR FOCUS #2)
    # ==========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8)
    add_header(
        s8,
        "Major Focus #2  |  Tech Stack & Edge Hardware Setup",
        "Concise Setup, Tech Stack & 4 GB VRAM Sequential Eviction",
        "Running ~5.2 GB of neural weights on a 4.0 GB NVIDIA GTX 1650 with zero OOMs via deterministic 4-stage GPU eviction.",
        page_num=8, total_pages=total_slides
    )

    # Top Section: 4-Stage Sequential VRAM Pipeline Cards
    stages = [
        ("STAGE 1: ASR & VAD", "Faster-Whisper Large-v3 Turbo", "• Compute: INT8-FP16 (CTranslate2)\n• VRAM Peak: 1,120 MB\n• Speed: 0.35x RTF (2.9x real-time)\n• Action: Transcribe + Silero VAD\n➔ del model + torch.cuda.empty_cache()", ACCENT_CYAN),
        ("STAGE 2: ACOUSTIC + DSP", "Emotion2Vec+ Large + Librosa", "• Compute: PyTorch FP16 + CPU DSP\n• VRAM Peak: 1,240 MB\n• Speed: 203 ms SER + 55 ms DSP\n• Action: Dual-stream raw/norm audio\n➔ del model + torch.cuda.empty_cache()", ACCENT_AMBER),
        ("STAGE 3: LLM ARBITRATION", "Kev-4B INT4 / Kev-0.8B Q8_0", "• Compute: C++ CUDA llama-server.exe\n• VRAM Peak: 2,410 MB (4B) / 803 MB\n• Speed: Single forward pass (/v1/systemone)\n• Action: 8-Emotion + Sarcasm readout\n➔ Terminate subprocess / Evict VRAM", ACCENT_INDIGO),
        ("STAGE 4: TEMPORAL UI", "Time-Weighted Synthesis", "• Compute: Host CPU (0 MB VRAM)\n• Backend: FastAPI Async REST API\n• Frontend: React 19 + TypeScript + Vite\n• Action: Duration-weighted emotion & 3-way sentiment timeline", ACCENT_GREEN),
    ]
    for i, (st_lbl, st_mod, st_body, st_col) in enumerate(stages):
        x = 0.6 + i * 3.08
        add_card(s8, x, 1.40, 2.90, 2.65, bg_color=BG_CARD, border_color=st_col, accent_top=st_col)
        add_text_block(s8, x + 0.16, 1.54, 2.58, 2.4, [
            {"text": st_lbl, "size": 9.5, "bold": True, "color": st_col, "space_after": 2},
            {"text": st_mod, "size": 11.5, "bold": True, "color": TEXT_WHITE, "space_after": 6},
            {"text": st_body, "size": 9.2, "color": TEXT_LIGHT},
        ])

    # Bottom Left: Concise Setup Steps (4 Steps)
    add_card(s8, 0.6, 4.25, 6.0, 2.60, bg_color=BG_CODE, border_color=BORDER_SLATE, accent_left=ACCENT_CYAN)
    add_text_block(s8, 0.82, 4.38, 5.6, 2.35, [
        {"text": "CONCISE 4-STEP SETUP & DEPLOYMENT WORKFLOW", "size": 11, "bold": True, "color": ACCENT_CYAN, "space_after": 6},
        {"text": "1. Environment & CUDA Runtime: Python 3.11 venv + PyTorch 2.6 (CUDA 12.4) + CTranslate2 (`int8_float16`) + FunASR + Librosa.", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "2. Pre-Quantized GGUF Checkpoints (`data/models/`): Place `Kev-4B-Q4_K_M.gguf` (2.4 GB) & `Kev-0.8B-Q8_0.gguf` (803 MB) paired with custom `tools/llama/llama-server.exe` (`/v1/systemone` PointerHead).", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "3. Launch Backend API: `python -m uvicorn backend.app:app --port 8000` (auto-manages GPU staging & audio normalization).", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "4. Launch Studio UI: `cd web && npm run dev -- --port 5173` (Live Mic Studio, Clip Inspector, 2,000-Clip Explorer).", "size": 9.5, "color": TEXT_LIGHT},
    ])

    # Bottom Right: Tech Stack Summary Table
    add_card(s8, 6.8, 4.25, 5.933, 2.60, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_left=ACCENT_INDIGO)
    add_text_block(s8, 7.02, 4.38, 5.5, 2.35, [
        {"text": "PRODUCTION TECH STACK MATRIX", "size": 11, "bold": True, "color": ACCENT_INDIGO, "space_after": 6},
        {"text": "• Speech-to-Text (ASR): `deepdml/faster-whisper-large-v3-turbo-ct2` (INT8-FP16) + Silero VAD", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Acoustic SER Foundation: `iic/emotion2vec_plus_large` (ModelScope / FunASR 1024-dim)", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Biomechanical DSP: `Librosa` + `SoundFile` (YIN F0 pitch slope, RMS dB, Onset tempo, Jitter)", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Decision Arbitrator: `Kev-4B` (Q4_K_M GGUF) & `Kev-0.8B` (Q8_0 GGUF) via C++ CUDA `llama.cpp`", "size": 9.5, "color": TEXT_LIGHT, "space_after": 4},
        {"text": "• Full-Stack Application: FastAPI + Uvicorn (Backend) | React 19 + TypeScript + Tailwind CSS (UI)", "size": 9.5, "color": TEXT_LIGHT},
    ])

    # ==========================================================================
    # SLIDE 9: BENCHMARK RESULTS (60-CLIP, 2,000-CLIP & 4-CORPUS SER VALIDATION)
    # ==========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9)
    add_header(
        s9,
        "Quantitative Evaluation  |  Benchmarks & Validation",
        "Comprehensive Benchmark Suite: Enterprise & SER Validation",
        "Evaluated across 2,000 multilingual business clips (Earnings22 + ReazonSpeech) and 122 gold-standard SER validation clips.",
        page_num=9, total_pages=total_slides
    )

    # Left Table: 60-Clip Audited & 2,000-Clip Enterprise Benchmarks
    add_card(s9, 0.6, 1.38, 6.1, 5.50, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=ACCENT_INDIGO)
    add_text_block(s9, 0.8, 1.50, 5.7, 0.35, [
        {"text": "A. MULTILINGUAL ENTERPRISE BENCHMARKS (60 & 2,000 CLIPS)", "size": 10.5, "bold": True, "color": ACCENT_CYAN}
    ])

    t9a_shape = s9.shapes.add_table(7, 4, Inches(0.75), Inches(1.90), Inches(5.8), Inches(3.85))
    t9a = t9a_shape.table
    for i, w in enumerate([Inches(2.3), Inches(1.1), Inches(1.2), Inches(1.2)]):
        t9a.columns[i].width = w
    for c_idx, h in enumerate(["Dataset / Model", "8-Emotion", "3-Sentiment", "EN / JA Sent."]):
        style_table_cell(t9a.cell(0, c_idx), h, font_size=9.0, bold=True, color=TEXT_WHITE, bg_color=BG_CARD_ALT)

    rows9a = [
        ("60-Clip: Emotion2Vec+ (SER Only)", "55.0%", "63.3%", "93.3% / 33.3%"),
        ("60-Clip: Vocalis + Kev-0.8B INT4", "70.0%", "75.0%", "96.7% / 53.3%"),
        ("60-Clip: Vocalis + Kev-4B INT4", "78.3%", "83.3%", "90.0% / 76.7%"),
        ("60-Clip: Best Per-Language", "80.0%", "86.7%", "96.7% / 76.7%"),
        ("2,000-Clip: Emotion2Vec+ (SER)", "50.90%", "55.15%", "82.7% / 27.6%"),
        ("2,000-Clip: Vocalis Best (4B/0.8B)", "63.05%", "67.60%", "84.3% / 50.9%"),
    ]
    for r_idx, r in enumerate(rows9a, start=1):
        hl = r_idx in (3, 4, 6)
        bg = BG_CARD_ALT if hl else BG_DARK
        style_table_cell(t9a.cell(r_idx, 0), r[0], font_size=8.8, bold=hl, color=ACCENT_GREEN if hl else TEXT_LIGHT, bg_color=bg)
        style_table_cell(t9a.cell(r_idx, 1), r[1], font_size=8.8, bold=hl, color=TEXT_WHITE, bg_color=bg, align=PP_ALIGN.CENTER)
        style_table_cell(t9a.cell(r_idx, 2), r[2], font_size=8.8, bold=True, color=ACCENT_CYAN if hl else TEXT_LIGHT, bg_color=bg, align=PP_ALIGN.CENTER)
        style_table_cell(t9a.cell(r_idx, 3), r[3], font_size=8.8, color=TEXT_LIGHT, bg_color=bg, align=PP_ALIGN.CENTER)

    add_text_block(s9, 0.8, 5.90, 5.7, 0.85, [
        {"text": "• On Japanese ReazonSpeech (1,000 clips), Vocalis boosts accuracy from 26.1% (SER) to 49.1% Emotion / 50.9% Sentiment (+23.0% gain).", "size": 9.0, "color": TEXT_LIGHT, "space_after": 2},
        {"text": "• On English Earnings22 (1,000 clips), Kev-4B reaches 84.30% 3-Way Sentiment accuracy.", "size": 9.0, "color": ACCENT_GREEN},
    ])

    # Right Table: 122-Clip Multi-Corpus SER Validation Benchmark (TESS, RAVDESS, SAVEE, JVNV-JA)
    add_card(s9, 6.9, 1.38, 5.833, 5.50, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_top=ACCENT_GREEN)
    add_text_block(s9, 7.1, 1.50, 5.4, 0.35, [
        {"text": "B. ACTUAL SER VALIDATION SETS (122 HUMAN-ACTED CLIPS)", "size": 10.5, "bold": True, "color": ACCENT_GREEN}
    ])

    t9b_shape = s9.shapes.add_table(6, 4, Inches(7.05), Inches(1.90), Inches(5.533), Inches(3.45))
    t9b = t9b_shape.table
    for i, w in enumerate([Inches(1.85), Inches(1.22), Inches(1.22), Inches(1.243)]):
        t9b.columns[i].width = w
    for c_idx, h in enumerate(["SER Corpus (Clips)", "Emotion2Vec+", "Kev-0.8B Q8_0", "Kev-4B INT4"]):
        style_table_cell(t9b.cell(0, c_idx), h, font_size=9.0, bold=True, color=TEXT_WHITE, bg_color=BG_CARD_ALT)

    rows9b = [
        ("JVNV-JA (25 Japanese)\nFree-Form Emotional", "64.0% Emo\n88.0% Sent", "92.0% Emo\n96.0% Sent", "96.0% Emo ✅\n100.0% Sent ✅"),
        ("TESS (36 English)\n'Say the word <X>'", "100.0% Emo\n100.0% Sent", "36.1% Emo\n41.7% Sent", "91.7% Emo ✅\n86.1% Sent ✅"),
        ("SAVEE (25 English)\nNeutral Carrier Text", "84.0% Emo\n88.0% Sent", "20.0% Emo\n48.0% Sent", "48.0% Emo\n72.0% Sent"),
        ("RAVDESS (36 English)\n'Dogs are sitting...'", "88.9% Emo\n97.2% Sent", "30.6% Emo\n30.6% Sent", "52.8% Emo\n55.6% Sent"),
        ("OVERALL (122 Clips)", "86.1% Emo\n94.3% Sent", "42.6% Emo\n50.8% Sent", "72.1% Emo\n77.1% Sent"),
    ]
    for r_idx, r in enumerate(rows9b, start=1):
        hl = r_idx in (1, 2, 5)
        bg = BG_CARD_ALT if hl else BG_DARK
        style_table_cell(t9b.cell(r_idx, 0), r[0], font_size=8.6, bold=True, color=TEXT_WHITE, bg_color=bg)
        style_table_cell(t9b.cell(r_idx, 1), r[1], font_size=8.6, color=TEXT_LIGHT, bg_color=bg, align=PP_ALIGN.CENTER)
        style_table_cell(t9b.cell(r_idx, 2), r[2], font_size=8.6, color=TEXT_MUTED, bg_color=bg, align=PP_ALIGN.CENTER)
        style_table_cell(t9b.cell(r_idx, 3), r[3], font_size=8.6, bold=True, color=ACCENT_GREEN if hl else TEXT_LIGHT, bg_color=bg, align=PP_ALIGN.CENTER)

    add_text_block(s9, 7.1, 5.50, 5.45, 1.25, [
        {"text": "• JVNV-JA Breakthrough: On natural Japanese emotional speech, Kev-4B reaches 96.0% Emotion & 100.0% Sentiment (vs 64.0% / 88.0% for standalone SER).", "size": 9.0, "bold": True, "color": ACCENT_GREEN, "space_after": 3},
        {"text": "• TESS Carrier Phrase Test: Even when text is 100% neutral ('Say the word ripe'), Kev-4B achieves 91.7% accuracy (+55.6% over Kev-0.8B) by trusting high-confidence acoustic priors.", "size": 9.0, "color": TEXT_LIGHT},
    ])

    # ==========================================================================
    # SLIDE 10: CHALLENGES FACED & POC OBSERVATIONS (MAJOR FOCUS #2)
    # ==========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10)
    add_header(
        s10,
        "Major Focus #2  |  Engineering Challenges & POC Observations",
        "Key Challenges Faced & Technical Observations from the POC",
        "Five critical failure modes uncovered during real-audio testing and how each was systematically resolved.",
        page_num=10, total_pages=total_slides
    )

    challenges = [
        (
            "1. 300x RMS Gain Normalization Destroyed Prosody",
            "Live mic recordings had low raw amplitude (RMS ~0.0002). Applying a 300x gain boost before feature extraction inflated quiet sad speech to 75.8 dB and boosted ambient room noise, causing Librosa to hallucinate '6.5 onsets/sec (rapid pace)' and 'loud volume'.",
            "Implemented Dual-Stream Audio Routing: `Librosa` extracts physical prosody from the raw un-normalized waveform (+ silence trimming), while `Whisper` & `Emotion2Vec+` receive the gain-normalized waveform.",
            ACCENT_ROSE
        ),
        (
            "2. Bilingual SER Label Leakage ('难过/sad')",
            "ModelScope's `Emotion2Vec+ Large` outputs bilingual Chinese/English labels (`'难过/sad'`, `'生气/angry'`). Passing raw unsplit strings into the English Kev prompt weakened cross-modal conditioning in Live Studio.",
            "Added `clean_e2v_label()` in `vocalis_segmented_pipeline.py` to strip Chinese prefixes and map `<unk>`/`other` to canonical English emotion tokens before prompt formatting.",
            ACCENT_AMBER
        ),
        (
            "3. 4 GB VRAM Ceiling & PyTorch CUDA Fragmentation",
            "Simultaneously loading Whisper + Emotion2Vec+ + Kev-4B requires ~5.2 GB VRAM (crashing a 4 GB GTX 1650). Moreover, PyTorch retains ~500 MB of reserved CUDA context even after `empty_cache()`, and Windows Triton absence slowed NF4 by 3.7x.",
            "Built deterministic `evict_vram()` staging between models and offloaded Stage-3 Kev inference to an isolated C++ `llama-server.exe` process using pre-quantized GGUF checkpoints (`3.7x` speedup).",
            ACCENT_INDIGO
        ),
        (
            "4. Japanese Pitch-Accent Acoustic Hallucinations",
            "Standalone `Emotion2Vec+` failed on 73.9% of Japanese broadcasts (`26.1%` accuracy), predicting `happy` with 85%–99% confidence because standard Japanese pitch-accent modulation resembles Western happy intonation.",
            "Proved that Early Prompt Priming allows Kev to override false `happy` SER priors using the Japanese transcript, whereas Late Mathematical Fusion gets overpowered by SER's false 95% confidence.",
            ACCENT_CYAN
        ),
        (
            "5. Model Capacity Threshold for Sarcasm (0.8B vs 4B)",
            "On polite sarcastic speech ('Thank you so much' spoken sadly), `Kev-0.8B` anchored to the positive words (`83.1%` confidence, `19.6%` sarcasm), and `Qwen2.5-Omni-3B` GGUF (`mtmd`) lost prosodic nuance (`23.6s` latency).",
            "Adopted `Kev-4B INT4 (Q4_K_M GGUF)` as the default Live Studio arbitrator—its 4B backbone resolves vocal-lexical conflict (`sarcasm = 57.2%`) and flips sentiment polarity to Negative (`62.3%`).",
            ACCENT_GREEN
        ),
    ]

    for idx, (ctitle, cobs, csol, ccol) in enumerate(challenges):
        y = 1.38 + idx * 1.11
        add_card(s10, 0.6, y, 12.133, 1.02, bg_color=BG_CARD, border_color=BORDER_SLATE, accent_left=ccol)
        add_text_block(s10, 0.82, y + 0.08, 3.1, 0.86, [
            {"text": f"CHALLENGE #{idx+1}", "size": 8.5, "bold": True, "color": ccol, "space_after": 2},
            {"text": ctitle, "size": 10.5, "bold": True, "color": TEXT_WHITE},
        ])
        add_text_block(s10, 4.05, y + 0.08, 4.35, 0.86, [
            {"text": "POC OBSERVATION / ROOT CAUSE", "size": 8.0, "bold": True, "color": TEXT_MUTED, "space_after": 2},
            {"text": cobs, "size": 8.8, "color": TEXT_LIGHT},
        ])
        add_text_block(s10, 8.55, y + 0.08, 4.05, 0.86, [
            {"text": "ENGINEERING RESOLUTION", "size": 8.0, "bold": True, "color": ACCENT_GREEN, "space_after": 2},
            {"text": csol, "size": 8.8, "color": TEXT_LIGHT},
        ])

    # ==========================================================================
    # SLIDE 11: FINAL RECOMMENDATIONS & PRODUCTION ROADMAP (MAJOR FOCUS #1)
    # ==========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s11)
    add_header(
        s11,
        "Major Focus #1  |  Executive Recommendations",
        "Final Recommendations & Production Deployment Strategy",
        "Actionable architectural decisions validated by our 2,000-clip enterprise benchmark and 122-clip SER validation suite.",
        page_num=11, total_pages=total_slides
    )

    recs = [
        (
            "RECOMMENDATION 1",
            "Standardize on Tri-Modal Prompt Priming with Kev-4B INT4 (GGUF) for Live & High-Stakes Calls",
            "• Deploy `Kev-4B-Q4_K_M.gguf` via C++ `llama-server.exe` as the primary production arbitrator.\n"
            "• Delivers 83.3% 3-way sentiment accuracy on audited EN/JA calls, 100% sentiment / 96% emotion on Japanese `JVNV-JA`, and 91.7% on `TESS`.\n"
            "• Crucial for catching subtle sarcasm and vocal-lexical contradictions where 0.8B models anchor too heavily on polite words.",
            ACCENT_GREEN
        ),
        (
            "RECOMMENDATION 2",
            "Use Kev-0.8B (Q8_0 GGUF) as a High-Throughput Tier for Bulk Historical Archival",
            "• For batch processing millions of historical call recordings where sub-second throughput is prioritized, deploy `Kev-0.8B-Q8_0.gguf`.\n"
            "• Achieves 1.51 clips/sec (592 ms/clip) in only 803 MB VRAM while maintaining 83.4% English sentiment accuracy across 1,000 Earnings22 clips.",
            ACCENT_CYAN
        ),
        (
            "RECOMMENDATION 3",
            "Adopt Dual-Tier Reporting: 3-Way Sentiment Polarity + 8-Class Emotion Drill-Down",
            "• Surface 3-Way Sentiment (`Positive`, `Neutral`, `Negative`) with the automatic `sarcasm_flag` flip as the primary executive KPI in dashboards.\n"
            "• Retain the 8 canonical emotions (`outrage_hostility`, `urgency_panic`, `disappointed_sad`, etc.) for downstream compliance & routing workflows.",
            ACCENT_INDIGO
        ),
        (
            "RECOMMENDATION 4",
            "Avoid Post-LLM Mathematical Score Fusion & Brittle Keyword Rules",
            "• Keep fusion strictly inside Kev's prompt attention (Early Priming) rather than post-hoc probability averaging, preventing Japanese SER pitch-accent errors from corrupting predictions.\n"
            "• Maintain Dual-Stream Audio Routing (un-normalized audio to Librosa DSP; normalized audio to Whisper/Emotion2Vec+) as a mandatory preprocessing standard.",
            ACCENT_AMBER
        ),
    ]

    for idx, (rtag, rtitle, rbody, rcol) in enumerate(recs):
        col_idx = idx % 2
        row_idx = idx // 2
        x = 0.6 + col_idx * 6.16
        y = 1.42 + row_idx * 2.45
        add_card(s11, x, y, 5.96, 2.25, bg_color=BG_CARD, border_color=rcol, accent_top=rcol)
        add_pill(s11, x + 0.22, y + 0.18, 1.85, 0.26, rtag, rcol, TEXT_WHITE, 8.5)
        add_text_block(s11, x + 0.22, y + 0.50, 5.52, 1.65, [
            {"text": rtitle, "size": 11.5, "bold": True, "color": TEXT_WHITE, "space_after": 6},
            {"text": rbody, "size": 9.3, "color": TEXT_LIGHT},
        ])

    # Bottom Executive Sign-Off Banner
    add_card(s11, 0.6, 6.32, 12.133, 0.62, bg_color=BG_CARD_ALT, border_color=ACCENT_INDIGO)
    add_text_block(s11, 0.85, 6.45, 11.6, 0.38, [
        {"runs": [
            {"text": "POC STATUS: ", "size": 10.5, "bold": True, "color": ACCENT_GREEN},
            {"text": "Production-Ready Prototype Live  |  ", "size": 10.5, "bold": True, "color": TEXT_WHITE},
            {"text": "2,000 Multilingual Clips + 122 SER Validation Clips Benchmarked  |  ", "size": 10, "color": TEXT_LIGHT},
            {"text": "Presenters: Vaibhav & Balram", "size": 10.5, "bold": True, "color": ACCENT_CYAN},
        ]}
    ])

    out_pptx = ROOT / "Vocalis_Executive_Presentation_Vaibhav_Balram.pptx"
    prs.save(str(out_pptx))
    print(f"[✓] Saved business-grade PowerPoint presentation to: {out_pptx}")


if __name__ == "__main__":
    build_presentation()
