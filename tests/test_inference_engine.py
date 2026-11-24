"""
Unit tests for the sentence segmentation and inference engine in ProductLens 3.0.
"""

from productlens.inference.engine import ReviewAnalysisEngine

def test_vertical_detection():
    # Electronics keywords
    assert ReviewAnalysisEngine.detect_vertical("The OLED screen and battery are great.") == "electronics_tech"
    # Apparel keywords
    assert ReviewAnalysisEngine.detect_vertical("The pure cotton fabric is soft and the sizing fits well.") == "apparel_fashion"
    assert ReviewAnalysisEngine.detect_vertical("Kurta stitching was loose after first wash.") == "apparel_fashion"


def test_segment_text_offsets():
    text = (
        "Good Camera specially selfie and main... But no ultra wide lens.\n\n"
        "Good gaming performance: Bgmi 90fps, brawl stars 144fps.\n\n"
        "Outstanding display and battery."
    )
    segments = ReviewAnalysisEngine.segment_text(text)
    assert len(segments) == 3

    for seg_text, start, end in segments:
        assert text[start:end] == seg_text
