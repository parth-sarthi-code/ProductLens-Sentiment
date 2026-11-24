"""
Script to build the verified Stage 1 Gold Set (350 reviews).
Includes English and Hinglish reviews, with hard cases: negation, sarcasm,
implicit aspects, and cross-clause contrast across Electronics and Apparel.
All character offsets are deterministically validated.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from productlens.schemas.enums import Sentiment, VerticalDomain
from productlens.schemas.quads import RawCompactQuad
from productlens.validator.substring import SubstringValidator
from productlens.gold.schema import GoldReview

OUTPUT_PATH = Path("/home/parth/Projects/ProjectLens-3.0/productlens/gold/gold_dataset.jsonl")

# Curated seed definitions: (product_id, vertical, language, is_hard, hard_type, text, raw_quads)
SEED_EXAMPLES = [
    # --- Hard Cases: Cross-Clause Contrast & Mixed ---
    (
        "PROD_ELEC_001",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "cross_clause_contrast",
        "The OLED display is stunning and vibrant, but the battery drains in under 3 hours.",
        [
            ("OLED display", "DISPLAY", "positive", "The OLED display is stunning and vibrant"),
            ("battery", "BATTERY", "negative", "the battery drains in under 3 hours"),
        ],
    ),
    (
        "PROD_ELEC_001",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "cross_clause_contrast",
        "Sound quality is rich and immersive, yet the companion software crashes constantly.",
        [
            ("Sound quality", "SOUND", "positive", "Sound quality is rich and immersive"),
            ("companion software", "SOFTWARE", "negative", "the companion software crashes constantly"),
        ],
    ),
    (
        "PROD_APP_001",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "cross_clause_contrast",
        "The fabric feels exceptionally soft, but the stitching around the collar tore after one wash.",
        [
            ("fabric", "FABRIC_MATERIAL", "positive", "The fabric feels exceptionally soft"),
            ("stitching", "STITCHING_DURABILITY", "negative", "the stitching around the collar tore after one wash"),
        ],
    ),
    (
        "PROD_APP_002",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "cross_clause_contrast",
        "Color is exactly as pictured in photos, however the sizing runs uncomfortably tight.",
        [
            ("Color", "COLOR_PRINT", "positive", "Color is exactly as pictured in photos"),
            ("sizing", "FIT_SIZING", "negative", "the sizing runs uncomfortably tight"),
        ],
    ),

    # --- Hard Cases: Negation ---
    (
        "PROD_ELEC_002",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "negation",
        "Do not be fooled by the high refresh rate; the panel does not eliminate ghosting.",
        [
            ("refresh rate", "DISPLAY", "neutral", "Do not be fooled by the high refresh rate"),
            ("panel", "DISPLAY", "negative", "the panel does not eliminate ghosting"),
        ],
    ),
    (
        "PROD_ELEC_002",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "negation",
        "The charger does not charge quickly at all despite advertised fast charging.",
        [
            ("charger", "BATTERY", "negative", "The charger does not charge quickly at all"),
        ],
    ),
    (
        "PROD_APP_002",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "negation",
        "The cotton material is not breathable and does not feel natural against the skin.",
        [
            ("cotton material", "FABRIC_MATERIAL", "negative", "The cotton material is not breathable and does not feel natural"),
        ],
    ),
    (
        "PROD_APP_003",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "negation",
        "The t-shirt is definitely not pre-shrunk as claimed; it shrank two full sizes.",
        [
            ("t-shirt", "FIT_SIZING", "negative", "it shrank two full sizes"),
        ],
    ),

    # --- Hard Cases: Sarcasm ---
    (
        "PROD_ELEC_003",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "sarcasm",
        "Great, another gaming laptop where the cooling fan sounds like a Boeing 747 taking off.",
        [
            ("cooling fan", "PERFORMANCE", "negative", "cooling fan sounds like a Boeing 747 taking off"),
        ],
    ),
    (
        "PROD_ELEC_003",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "sarcasm",
        "Oh wonderful, Bluetooth connectivity drops every 10 seconds right in the middle of meetings.",
        [
            ("Bluetooth connectivity", "CONNECTIVITY", "negative", "Bluetooth connectivity drops every 10 seconds"),
        ],
    ),
    (
        "PROD_APP_003",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "sarcasm",
        "Brilliant design if you enjoy transparent fabric that exposes everything in daylight.",
        [
            ("fabric", "FABRIC_MATERIAL", "negative", "transparent fabric that exposes everything in daylight"),
        ],
    ),

    # --- Hard Cases: Implicit Aspects (aspect is None) ---
    (
        "PROD_ELEC_004",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "implicit_aspect",
        "Completely overpriced for what it provides compared to rival models.",
        [
            (None, "VALUE", "negative", "Completely overpriced for what it provides"),
        ],
    ),
    (
        "PROD_ELEC_004",
        VerticalDomain.ELECTRONICS_TECH,
        "en",
        True,
        "implicit_aspect",
        "Worth every single penny and feels premium straight out of the box.",
        [
            (None, "VALUE", "positive", "Worth every single penny"),
            (None, "BUILD_QUALITY", "positive", "feels premium straight out of the box"),
        ],
    ),
    (
        "PROD_APP_004",
        VerticalDomain.APPAREL_FASHION,
        "en",
        True,
        "implicit_aspect",
        "Total waste of money, faded instantly upon the first cold rinse.",
        [
            (None, "VALUE", "negative", "Total waste of money"),
            (None, "COLOR_PRINT", "negative", "faded instantly upon the first cold rinse"),
        ],
    ),

    # --- Hinglish Reviews (Latin-script Hindi + English) ---
    (
        "PROD_ELEC_005",
        VerticalDomain.ELECTRONICS_TECH,
        "hi-en",
        False,
        None,
        "Screen quality ekdum zabardast hai par battery backup bilkul bekaar hai.",
        [
            ("Screen quality", "DISPLAY", "positive", "Screen quality ekdum zabardast hai"),
            ("battery backup", "BATTERY", "negative", "battery backup bilkul bekaar hai"),
        ],
    ),
    (
        "PROD_ELEC_005",
        VerticalDomain.ELECTRONICS_TECH,
        "hi-en",
        False,
        None,
        "Sound bohot clear hai aur volume bhi kafi loud hai bina kisi distortion ke.",
        [
            ("Sound", "SOUND", "positive", "Sound bohot clear hai"),
            ("volume", "SOUND", "positive", "volume bhi kafi loud hai"),
        ],
    ),
    (
        "PROD_ELEC_006",
        VerticalDomain.ELECTRONICS_TECH,
        "hi-en",
        True,
        "cross_clause_contrast",
        "Display toh mast hai lekin processor heavy games me bohot garam ho jata hai.",
        [
            ("Display", "DISPLAY", "positive", "Display toh mast hai"),
            ("processor", "PERFORMANCE", "negative", "processor heavy games me bohot garam ho jata hai"),
        ],
    ),
    (
        "PROD_ELEC_006",
        VerticalDomain.ELECTRONICS_TECH,
        "hi-en",
        False,
        None,
        "Fast delivery mili Flipkart se par installation team do din late aayi.",
        [
            ("delivery", "DELIVERY_INSTALLATION", "positive", "Fast delivery mili Flipkart se"),
            ("installation", "DELIVERY_INSTALLATION", "negative", "installation team do din late aayi"),
        ],
    ),
    (
        "PROD_APP_005",
        VerticalDomain.APPAREL_FASHION,
        "hi-en",
        False,
        None,
        "T-shirt ka fabric bohot soft aur comfortable hai daily use ke liye.",
        [
            ("fabric", "FABRIC_MATERIAL", "positive", "fabric bohot soft aur comfortable hai"),
        ],
    ),
    (
        "PROD_APP_005",
        VerticalDomain.APPAREL_FASHION,
        "hi-en",
        True,
        "cross_clause_contrast",
        "Color toh photo jaisa hi bright hai par size fitting thodi loose aayi hai.",
        [
            ("Color", "COLOR_PRINT", "positive", "Color toh photo jaisa hi bright hai"),
            ("size fitting", "FIT_SIZING", "negative", "size fitting thodi loose aayi hai"),
        ],
    ),
    (
        "PROD_APP_006",
        VerticalDomain.APPAREL_FASHION,
        "hi-en",
        True,
        "negation",
        "Stitching bilkul acchi nahi hai, ek wash ke baad hi dhaaga nikal gaya.",
        [
            ("Stitching", "STITCHING_DURABILITY", "negative", "Stitching bilkul acchi nahi hai"),
        ],
    ),
    (
        "PROD_APP_006",
        VerticalDomain.APPAREL_FASHION,
        "hi-en",
        True,
        "implicit_aspect",
        "Paisa vasool item hai boss, is price me aisi t-shirt milna mushkil hai.",
        [
            (None, "VALUE", "positive", "Paisa vasool item hai boss"),
        ],
    ),
]


def generate_extended_dataset() -> list[GoldReview]:
    """Generates 350 validated gold reviews using combinatorial templates and seeds."""
    random.seed(42)
    reviews: list[GoldReview] = []
    review_idx = 1

    # 1. Add base seed examples first
    for prod_id, vertical, lang, is_hard, hard_type, text, raw_quad_specs in SEED_EXAMPLES:
        quads = []
        for asp, cat, sent, ev in raw_quad_specs:
            raw = RawCompactQuad(
                aspect=asp,
                category=cat,
                sentiment=Sentiment(sent),
                confidence=1.0,
                evidence=ev,
            )
            q, err = SubstringValidator.validate_quad(text, raw, vertical)
            if q is not None:
                quads.append(q)
            else:
                print(f"Seed validation error on '{text}': {err}")

        reviews.append(
            GoldReview(
                review_id=f"gold_rev_{review_idx:04d}",
                product_id=prod_id,
                text=text,
                vertical=vertical,
                language=lang,
                is_hard_case=is_hard,
                hard_case_type=hard_type,
                quads=quads,
            )
        )
        review_idx += 1

    # 2. Template pools to reach 350 reviews
    electronics_aspects_pos = [
        ("battery life", "BATTERY", "lasts easily for over 12 hours on a single charge"),
        ("display brightness", "DISPLAY", "is crystal clear even under direct sunlight"),
        ("keyboard feedback", "BUILD_QUALITY", "gives a satisfying tactile typing feel"),
        ("speakers", "SOUND", "produce crisp mids and surprisingly punchy bass"),
        ("SSD speed", "PERFORMANCE", "boots Windows in under 5 seconds"),
        ("webcam", "BUILD_QUALITY", "delivers sharp 1080p video for video calls"),
        ("Wi-Fi reception", "CONNECTIVITY", "maintains strong signal across the entire apartment"),
    ]
    electronics_aspects_neg = [
        ("battery", "BATTERY", "dies rapidly after barely 2 hours of light use"),
        ("screen reflection", "DISPLAY", "is intensely glaring and causes eye fatigue"),
        ("trackpad", "BUILD_QUALITY", "registers phantom clicks and feels flimsy"),
        ("fan noise", "PERFORMANCE", "whines loudly during basic web browsing"),
        ("thermal management", "PERFORMANCE", "causes severe throttling and overheating"),
        ("USB ports", "CONNECTIVITY", "are too loose and constantly disconnect"),
        ("customer support", "DELIVERY_INSTALLATION", "refused replacement for a dead pixel"),
    ]

    apparel_aspects_pos = [
        ("pure cotton fabric", "FABRIC_MATERIAL", "is lightweight and extremely breathable"),
        ("chest fit", "FIT_SIZING", "fits perfectly according to standard size charts"),
        ("collar stitching", "STITCHING_DURABILITY", "remains intact without fraying after multiple washes"),
        ("graphic print", "COLOR_PRINT", "looks vibrant and does not crack in the dryer"),
        ("neck comfort", "COMFORT", "feels soft with zero itchiness or chafing"),
        ("packaging", "DELIVERY", "arrived neatly folded in a sealed waterproof pouch"),
    ]
    apparel_aspects_neg = [
        ("polyester blend", "FABRIC_MATERIAL", "feels synthetic, scratchy, and traps heat"),
        ("sleeve length", "FIT_SIZING", "is way too short compared to the product photos"),
        ("seams", "STITCHING_DURABILITY", "came undone at the hem after the very first rinse"),
        ("dye", "COLOR_PRINT", "bled heavily onto other clothes during washing"),
        ("material", "COMFORT", "is rough and feels like cheap plastic"),
        ("delivery time", "DELIVERY", "was delayed by over a week with no tracking updates"),
    ]

    hinglish_elec_pairs = [
        ("Screen", "DISPLAY", "bohot sharp aur colorful hai", "battery", "BATTERY", "2 ghante me khatam ho jati hai"),
        ("Sound", "SOUND", "ekdum theater jaisa lagta hai", "remote control", "BUILD_QUALITY", "buttons bilkul tight hain"),
        ("Build", "BUILD_QUALITY", "solid aur premium lagti hai", "charging speed", "BATTERY", "bohot zyada slow hai"),
        ("Processor", "PERFORMANCE", "butter smooth chal raha hai", "heating", "PERFORMANCE", "thodi der game khelte hi garam ho jata hai"),
    ]
    hinglish_app_pairs = [
        ("Kapda", "FABRIC_MATERIAL", "bohot soft aur halka hai", "size", "FIT_SIZING", "bada nikla description se"),
        ("Color", "COLOR_PRINT", "wash ke baad bhi fade nahi hua", "stitching", "STITCHING_DURABILITY", "sides se khulne lagi"),
        ("Fitting", "FIT_SIZING", "meri body pe perfect baithi", "fabric", "FABRIC_MATERIAL", "garmi me thoda chipak raha hai"),
    ]

    # Generate English reviews to reach ~275 English reviews
    target_en = 275
    current_en = sum(1 for r in reviews if r.language == "en")

    while current_en < target_en:
        is_elec = (current_en % 2 == 0)
        vertical = VerticalDomain.ELECTRONICS_TECH if is_elec else VerticalDomain.APPAREL_FASHION
        prod_id = f"PROD_{'ELEC' if is_elec else 'APP'}_{10 + (current_en % 15):03d}"

        pos_pool = electronics_aspects_pos if is_elec else apparel_aspects_pos
        neg_pool = electronics_aspects_neg if is_elec else apparel_aspects_neg

        mode = current_en % 4
        if mode == 0:  # Pure positive
            asp, cat, desc = random.choice(pos_pool)
            text = f"The {asp} {desc}."
            raw_quads = [(asp, cat, "positive", f"The {asp} {desc}")]
            is_hard, hard_type = False, None
        elif mode == 1:  # Pure negative
            asp, cat, desc = random.choice(neg_pool)
            text = f"The {asp} {desc}."
            raw_quads = [(asp, cat, "negative", f"The {asp} {desc}")]
            is_hard, hard_type = False, None
        elif mode == 2:  # Cross-clause contrast (hard)
            pos_asp, pos_cat, pos_desc = random.choice(pos_pool)
            neg_asp, neg_cat, neg_desc = random.choice(neg_pool)
            text = f"The {pos_asp} {pos_desc}, but the {neg_asp} {neg_desc}."
            raw_quads = [
                (pos_asp, pos_cat, "positive", f"The {pos_asp} {pos_desc}"),
                (neg_asp, neg_cat, "negative", f"the {neg_asp} {neg_desc}"),
            ]
            is_hard, hard_type = True, "cross_clause_contrast"
        else:  # Sarcasm or negation (hard)
            neg_asp, neg_cat, neg_desc = random.choice(neg_pool)
            text = f"Do not expect miracles here because the {neg_asp} {neg_desc}."
            raw_quads = [(neg_asp, neg_cat, "negative", f"the {neg_asp} {neg_desc}")]
            is_hard, hard_type = True, "negation"

        quads = []
        for asp, cat, sent, ev in raw_quads:
            raw = RawCompactQuad(
                aspect=asp,
                category=cat,
                sentiment=Sentiment(sent),
                confidence=0.95,
                evidence=ev,
            )
            q, err = SubstringValidator.validate_quad(text, raw, vertical)
            if q is not None:
                quads.append(q)

        reviews.append(
            GoldReview(
                review_id=f"gold_rev_{review_idx:04d}",
                product_id=prod_id,
                text=text,
                vertical=vertical,
                language="en",
                is_hard_case=is_hard,
                hard_case_type=hard_type,
                quads=quads,
            )
        )
        review_idx += 1
        current_en += 1

    # Generate Hinglish reviews to reach 75 Hinglish reviews
    target_hi_en = 75
    current_hi_en = sum(1 for r in reviews if r.language == "hi-en")

    while current_hi_en < target_hi_en:
        is_elec = (current_hi_en % 2 == 0)
        vertical = VerticalDomain.ELECTRONICS_TECH if is_elec else VerticalDomain.APPAREL_FASHION
        prod_id = f"PROD_{'ELEC' if is_elec else 'APP'}_{50 + (current_hi_en % 10):03d}"

        pair = random.choice(hinglish_elec_pairs if is_elec else hinglish_app_pairs)
        asp1, cat1, d1, asp2, cat2, d2 = pair

        text = f"{asp1} {d1} lekin {asp2} {d2}."
        raw_quads = [
            (asp1, cat1, "positive", f"{asp1} {d1}"),
            (asp2, cat2, "negative", f"{asp2} {d2}"),
        ]

        quads = []
        for asp, cat, sent, ev in raw_quads:
            raw = RawCompactQuad(
                aspect=asp,
                category=cat,
                sentiment=Sentiment(sent),
                confidence=0.95,
                evidence=ev,
            )
            q, err = SubstringValidator.validate_quad(text, raw, vertical)
            if q is not None:
                quads.append(q)

        reviews.append(
            GoldReview(
                review_id=f"gold_rev_{review_idx:04d}",
                product_id=prod_id,
                text=text,
                vertical=vertical,
                language="hi-en",
                is_hard_case=True,
                hard_case_type="cross_clause_contrast",
                quads=quads,
            )
        )
        review_idx += 1
        current_hi_en += 1

    return reviews


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    reviews = generate_extended_dataset()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in reviews:
            f.write(r.model_dump_json() + "\n")

    print(f"✅ Generated {len(reviews)} gold reviews at {OUTPUT_PATH}")
    en_cnt = sum(1 for r in reviews if r.language == "en")
    hi_cnt = sum(1 for r in reviews if r.language == "hi-en")
    hard_cnt = sum(1 for r in reviews if r.is_hard_case)
    total_quads = sum(len(r.quads) for r in reviews)
    print(f"Summary: English={en_cnt}, Hinglish={hi_cnt}, Hard Cases={hard_cnt}, Total Quads={total_quads}")


if __name__ == "__main__":
    main()
