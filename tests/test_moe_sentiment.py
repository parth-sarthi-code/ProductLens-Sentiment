import pytest
from productlens.sentiment.router import MoERouter
from productlens.sentiment.pipeline import ProductLensMoEPipeline


def test_moe_router_levels():
    router = MoERouter()

    # Level 1: Category Metadata
    assert router.route(category="Electronics", text="good")[0] == "tech"
    assert router.route(category="Clothing_Shoes_and_Jewelry", text="good")[0] == "apparel"
    assert router.route(category="Home_and_Kitchen", text="good")[0] == "home"

    # Level 2: Lexical Keywords (no category provided)
    assert router.route(category=None, text="The laptop battery is fast", aspect="battery")[0] == "tech"
    assert router.route(category=None, text="The cotton t-shirt shrank", aspect="fabric")[0] == "apparel"
    assert router.route(category=None, text="The television speakers are loud", aspect="sound")[0] == "home"

    # Level 3: Universal Fallback
    assert router.route(category=None, text="The service was fast but ambiance poor", aspect="ambiance")[0] == "fallback"


def test_moe_pipeline_e2e():
    pipeline = ProductLensMoEPipeline(lazy_load=True)

    # 1. Tech review prediction
    tech_pred = pipeline.predict(
        text="The screen resolution on this laptop is incredible, but the fan is loud.",
        aspect="screen",
        category="Computers",
    )
    assert tech_pred.sentiment == "positive"
    assert tech_pred.routed_expert == "tech"
    assert tech_pred.confidence > 0.5

    # 2. Apparel review prediction
    apparel_pred = pipeline.predict(
        text="The shirt shrank in the wash and the material is scratchy.",
        aspect="material",
        category="Clothing",
    )
    assert apparel_pred.sentiment == "negative"
    assert apparel_pred.routed_expert == "apparel"
