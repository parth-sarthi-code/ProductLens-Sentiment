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
