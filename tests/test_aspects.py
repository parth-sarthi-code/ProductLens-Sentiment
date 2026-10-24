import pytest
from productlens.aspects.extractor import AspectExtractor


def test_aspect_extractor_tech():
    extractor = AspectExtractor()
    text = "The laptop screen is bright and vivid, but the battery life is terrible."
    mentions = extractor.extract_aspects(text)
    aspect_names = [m.aspect.lower() for m in mentions]

    assert "screen" in aspect_names
    assert any("battery" in a for a in aspect_names)


def test_aspect_extractor_apparel():
    extractor = AspectExtractor()
    text = "The cotton fabric feels soft and premium, but the sizing runs small."
    mentions = extractor.extract_aspects(text)
    aspect_names = [m.aspect.lower() for m in mentions]

    assert any(a in ["fabric", "cotton fabric"] for a in aspect_names)
    assert any(a in ["sizing", "size"] for a in aspect_names)


def test_aspect_extractor_empty():
    extractor = AspectExtractor()
    assert extractor.extract_aspects("") == []
    assert extractor.extract_aspects("   ") == []


def test_aspect_extractor_custom_aspects():
    extractor = AspectExtractor(known_aspects={"ergonomic handle", "price point"})
    text = "The ergonomic handle was great, but the price point is steep."
    mentions = extractor.extract_aspects(text)
    aspect_names = [m.aspect.lower() for m in mentions]
    assert "ergonomic handle" in aspect_names
    assert "price point" in aspect_names


def test_find_aspect_span():
    from productlens.utils import find_aspect_span

    text = "The battery life on this laptop is incredible."
    start, end = find_aspect_span(text, "battery life")
    assert text[start:end] == "battery life"

    # Substring fallback
    start, end = find_aspect_span(text, "nonexistent")
    assert (start, end) == (0, len(text))

