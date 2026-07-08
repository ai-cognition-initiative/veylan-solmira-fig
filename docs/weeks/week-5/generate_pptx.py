"""
Generate PowerPoint presentation from Week 5 condensed markdown.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pathlib import Path


# Paths
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR.parent.parent / "veylan-solmira-fig" / "experiments" / "decision-making-preference-adverse" / "outputs"
OUTPUT_PPTX = SCRIPT_DIR / "FIG_Week5_Presentation.pptx"


def add_title_slide(prs, title, subtitle):
    """Add a title slide."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(36)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # Subtitle
    sub_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(9), Inches(1))
    tf = sub_box.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(18)
    p.alignment = PP_ALIGN.CENTER

    return slide


def add_content_slide(prs, title, content_lines, image_path=None):
    """Add a content slide with optional image."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9.4), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    # Content area depends on whether we have an image
    if image_path and Path(image_path).exists():
        # Split layout: text on left, image on right
        content_box = slide.shapes.add_textbox(Inches(0.3), Inches(1), Inches(5), Inches(6))
        # Add image on right
        slide.shapes.add_picture(str(image_path), Inches(5.5), Inches(1), width=Inches(4.3))
    else:
        content_box = slide.shapes.add_textbox(Inches(0.3), Inches(1), Inches(9.4), Inches(6))

    tf = content_box.text_frame
    tf.word_wrap = True

    for i, line in enumerate(content_lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        # Handle bullet points
        if line.startswith("• ") or line.startswith("- "):
            p.text = line[2:]
            p.level = 0
        elif line.startswith("  • ") or line.startswith("  - "):
            p.text = line[4:]
            p.level = 1
        else:
            p.text = line
            p.level = 0

        # Bold text between ** markers
        p.font.size = Pt(14)

    return slide


def add_table_slide(prs, title, headers, rows, subtitle=None):
    """Add a slide with a table."""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.2), Inches(9.4), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True

    # Subtitle if provided
    y_offset = 0.9
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.3), Inches(0.8), Inches(9.4), Inches(0.4))
        tf = sub_box.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(14)
        p.font.italic = True
        y_offset = 1.2

    # Table
    n_rows = len(rows) + 1  # +1 for header
    n_cols = len(headers)

    table_width = min(9.4, n_cols * 2.0)
    table = slide.shapes.add_table(n_rows, n_cols, Inches(0.3), Inches(y_offset), Inches(table_width), Inches(0.4 * n_rows)).table

    # Header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(12)

    # Data rows
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = str(value)
            cell.text_frame.paragraphs[0].font.size = Pt(11)

    return slide


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # Slide 0: Title
    add_title_slide(
        prs,
        "FIG Week 5: Mechanistic Investigation\nof Preference Expression",
        "Presenter: Veylan Solmira | For: Derek Shiller | Week of Jan 13, 2026"
    )

    # Slide 1: Key Results Summary
    add_table_slide(
        prs,
        "Key Results Summary",
        ["Finding", "Status", "Implication"],
        [
            ["Steering failed", "CLOSED", "Control experiments revealed artifacts"],
            ["Probing succeeded", "✓", "Data-driven discovery found d > 8 effects"],
            ["Position probing", "✓", "Found strong position-encoding features"],
            ["Position steering", "⚠ INVALID", "Used base model; re-run Week 6"],
        ],
        subtitle="Key insight: Hand-picked Neuronpedia features = 0.000 activation. Top-k approach found d > 8."
    )

    # Slide 2: The Big Picture
    add_content_slide(
        prs,
        "The Big Picture",
        [
            "Research question:",
            "• Do LLMs have stable preferences, or does environmental",
            "  framing affect their willingness/ability to express them?",
            "",
            "Why it matters:",
            "• If models hide preferences under evaluation pressure,",
            "  that has implications for AI welfare research and",
            "  alignment evaluation.",
        ]
    )

    # Slide 3: Breadth Over Depth
    add_table_slide(
        prs,
        "Breadth Over Depth",
        ["Direction", "What We Tried", "What We Learned"],
        [
            ["SAE layers", "3 layers (0, 4, 15)", "Only L0 has good reconstruction"],
            ["Features", "5+ candidates", "Multiple show behavioral effects"],
            ["Steering", "Amplify vs suppress", "Both appeared to work..."],
            ["Controls", "Roundtrip, random ablation", "...until controls revealed artifacts"],
        ],
        subtitle="Rapidly exploring the terrain before focusing"
    )

    # Slide 4: Steering Failed
    add_content_slide(
        prs,
        "Steering Failed — Controls Revealed Artifacts",
        [
            "Both layers failed:",
            "• Layer 0: Good reconstruction (0.973), but feature not causal",
            "• Layer 15: Reconstruction noise (0.922) confounds all interventions",
            "",
            "Key finding:",
            "• Roundtrip-only (no modification) → 80% expression vs 40% baseline",
            "• SAE reconstruction noise caused behavioral changes",
            "• Not our targeted interventions!",
            "",
            "Lessons learned:",
            "• High reconstruction quality is necessary but not sufficient",
            "• Feature labels from Neuronpedia don't guarantee causal relevance",
            "• Control experiments prevented false positive publication",
            "",
            "STEERING DIRECTION: CLOSED",
        ]
    )

    # Slide 5: Pivot to Probing
    add_content_slide(
        prs,
        "Pivot — Probing Instead of Steering",
        [
            "Probing doesn't require roundtrip quality:",
            "• Steering: encode → modify → decode → inject (lossy decode confounds)",
            "• Probing: encode → analyze (no decode needed)",
            "",
            "Critical finding: Neuronpedia labels don't match behavior",
            "• SAE working correctly: max_act=38-52, 10-97 nonzero features",
            "• But hand-picked features all show 0.000 activation!",
            "• Features labeled 'preference' don't fire for preference prompts",
            "",
            "New approach:",
            "• Extract top-k activating features (data-driven, not hand-picked)",
            "• Find features that activate differently between conditions",
            "• Correlate with preference expression behavior",
        ]
    )

    # Slide 6: Top-K Probing Results
    add_table_slide(
        prs,
        "Top-K Probing Results (n=25)",
        ["Feature", "Layer", "Baseline", "Adversarial", "Cohen's d"],
        [
            ["#1695", "15", "33.1", "21.9", "-8.28"],
            ["#7953", "1", "4.8", "3.4", "-5.99"],
            ["#12644", "15", "28.2", "23.3", "-3.17"],
            ["#4234", "15", "46.7", "49.9", "+2.69"],
        ],
        subtitle="Expression rates: Baseline 52% → Adversarial 28%. Hand-picked = 0.000, Top-k = d > 8!"
    )

    # Slide 7: Position Probing
    img_path = OUTPUTS_DIR / "phase2_position_probing" / "position_features_layer0.png"
    add_content_slide(
        prs,
        "Position Bias — Probing Validation",
        [
            "Connection: Position bias is a cleaner proxy",
            "for the same question — how do internal",
            "representations map to behavioral patterns?",
            "",
            "Top features encoding A vs B (Layer 0):",
            "• #3519: -47.3 diff (B >> A)",
            "• #15495: +22.3 diff (A > B)",
            "",
            "Cross-layer pattern:",
            "• Layer 0: 47.3 (strongest)",
            "• Layer 4: 21.4 (weakest)",
            "• Layer 15: 34.2 (increasing)",
        ],
        image_path=img_path
    )

    # Slide 8: Position Steering (Invalid)
    add_content_slide(
        prs,
        "Position Steering — INVALID (Wrong Model)",
        [
            "Attempted: Ablate Feature #3519 to reduce position bias",
            "",
            "Results: All conditions showed 100% bias",
            "",
            "Problem discovered:",
            "• Used gemma-2-2b (base model) instead of gemma-2-2b-it (instruct)",
            "• Base models don't follow instructions — they just continue text",
            "• 100% 'bias' was artifact of text completion, not position preference",
            "",
            "Lesson learned:",
            "• Steering experiments require instruct models for generation-based evaluation",
            "",
            "Next step (Week 6):",
            "• Re-run with gemma-2-2b-it to get valid results",
        ]
    )

    # Slide 9: Next Steps
    add_content_slide(
        prs,
        "Next Steps & Open Questions",
        [
            "What we don't know yet:",
            "• L15 #1695: Massive effect (d=-8.28) — but what does it represent?",
            "• L15 #4234: Activates under adversarial — is this 'eval-awareness'?",
            "",
            "Immediate next steps:",
            "1. Re-run position steering with instruct model",
            "2. Neuronpedia lookup for L15 #1695, #4234",
            "3. Scale probing to n=50-100 for tighter CIs",
            "4. Test steering with discovered features",
            "",
            "Questions for discussion:",
            "• Is steering fundamentally limited, or wrong features?",
            "• Should we retry steering with features that actually activate?",
            "• Story arc: negative steering + positive probing → ???",
        ]
    )

    # Slide 10: Appendix
    add_content_slide(
        prs,
        "Appendix: Technical Details",
        [
            "SAE Configuration:",
            "• Model: Gemma 2 2B",
            "• SAE: gemma-scope-2b-pt-res-canonical (16k width)",
            "• Tested layers: 0, 4, 12, 15",
            "",
            "Key artifacts:",
            "• probing.py, probe_utils.py — Top-k differential probing",
            "• outputs/phase1c_probing/ — Probing results, correlation analysis",
            "• outputs/phase2_position_probing/ — Position probing visualizations",
            "",
            "Compute: vast.ai RTX 4090 (~$0.32/hr)",
        ]
    )

    # Save
    prs.save(OUTPUT_PPTX)
    print(f"Saved: {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
