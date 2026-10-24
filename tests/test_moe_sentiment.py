import pytest
from productlens.sentiment.router import MoERouter
from productlens.sentiment.pipeline import ProductLensMoEPipeline, checkpoints_exist


# ---------------------------------------------------------------------------
# Router tests (Fix 3)
# ---------------------------------------------------------------------------

class TestMoERouter:
    def setup_method(self):
        self.router = MoERouter()

    # --- Level 1: Metadata routing ---
    def test_level1_electronics(self):
        assert self.router.route(category="Electronics", text="good")[0] == "tech"

    def test_level1_clothing(self):
        assert self.router.route(category="Clothing_Shoes_and_Jewelry", text="good")[0] == "apparel"

    def test_level1_home(self):
        assert self.router.route(category="Home_and_Kitchen", text="good")[0] == "home"

    # --- Level 2: Lexical routing (no category) ---
    def test_level2_tech_battery(self):
        result = self.router.route(category=None, text="The laptop battery is fast", aspect="battery")
        assert result[0] == "tech"

    def test_level2_apparel_fabric(self):
        result = self.router.route(category=None, text="The cotton t-shirt shrank", aspect="fabric")
        assert result[0] == "apparel"

    def test_level2_home_sound(self):
        result = self.router.route(category=None, text="The television speakers are loud", aspect="sound")
        assert result[0] == "home"

    # --- Level 3: Fallback ---
    def test_level3_fallback(self):
        result = self.router.route(
            category=None,
            text="The service was fast but ambiance poor",
            aspect="ambiance",
        )
        assert result[0] == "fallback"

    # --- Misrouting regressions (Fix 3) ---
    def test_no_misroute_cost_to_tech(self):
        """'cost' contains 'os' as substring — must NOT route to tech."""
        result = self.router.route(category=None, text="the cost is high", aspect="cost")
        assert result[0] != "tech", (
            f"'cost' was misrouted to tech via 'os' substring match (got: {result[0]})"
        )

    def test_no_misroute_packaging_to_home(self):
        """'packaging' contains 'ac' as substring — must NOT route to home."""
        result = self.router.route(category=None, text="packaging was poor", aspect="packaging")
        assert result[0] != "home", (
            f"'packaging' was misrouted to home via 'ac' substring match (got: {result[0]})"
        )

    def test_no_misroute_short_category(self):
        """A 2-char category like 'pc' must not wildcard-match arbitrary synonyms."""
        # 'pc' is 2 chars; min length guard should prevent it matching 'apparel' synonyms
        domain, _ = self.router.route(category="pc", text="shirt was fine")
        # 'pc' < 4 chars, so substring match is blocked; result should NOT be apparel
        # (it would route to tech via exact synonym check if 'pc' were in taxonomy, which it's not now)
        # Mainly checking no crash and no spurious apparel hit
        assert domain != "apparel"


# ---------------------------------------------------------------------------
# Max sequence length tests (Fix 8)
# ---------------------------------------------------------------------------

class TestSequenceLength:
    def test_max_seq_len_constant(self):
        from productlens.constants import MAX_SEQ_LEN
        assert MAX_SEQ_LEN == 128


# ---------------------------------------------------------------------------
# Pipeline tests (Fix 2)
# ---------------------------------------------------------------------------

def test_missing_checkpoint_raises():
    """Pipeline must raise FileNotFoundError, never run with random weights."""
    pipeline = ProductLensMoEPipeline(
        checkpoint_paths={
            "tech":     "/nonexistent/tech.pt",
            "home":     "/nonexistent/home.pt",
            "apparel":  "/nonexistent/apparel.pt",
            "beauty":   "/nonexistent/beauty.pt",
            "fallback": "/nonexistent/fallback.pt",
        },
        lazy_load=True,
    )
    with pytest.raises(FileNotFoundError):
        pipeline.predict(text="The laptop screen is great", aspect="screen", category="Electronics")


@pytest.mark.skipif(
    not checkpoints_exist(),
    reason="Model checkpoints not downloaded — run scripts/download_weights.py first",
)
class TestMoEPipelineE2E:
    def setup_method(self):
        self.pipeline = ProductLensMoEPipeline(lazy_load=True)

    def test_tech_review_positive_screen(self):
        pred = self.pipeline.predict(
            text="The screen resolution on this laptop is incredible, but the fan is loud.",
            aspect="screen",
            category="Computers",
        )
        assert pred.sentiment == "positive"
        assert pred.routed_expert == "tech"
        assert pred.confidence > 0.5

    def test_apparel_review_negative_material(self):
        pred = self.pipeline.predict(
            text="The shirt shrank in the wash and the material is scratchy.",
            aspect="material",
            category="Clothing",
        )
        assert pred.sentiment == "negative"
        assert pred.routed_expert == "apparel"

    def test_routed_expert_reflects_actual_model(self):
        """routed_expert must name the model that actually ran, not the router's selection."""
        pred = self.pipeline.predict(
            text="The sound quality on this television is superb.",
            aspect="sound",
            category="Home_and_Kitchen",
        )
        # routed_expert should be 'home', not some other key
        assert pred.routed_expert in ("home", "fallback"), (
            f"Unexpected routed_expert: {pred.routed_expert}"
        )
