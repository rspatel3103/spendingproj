"""Synthetic data generator.

Produces 13 months of realistic, messy transaction history for one demo
account plus a seed vendor knowledge base, so the categorizer/forecaster
agents and the eval harness have something non-trivial to work against
before any real financial data exists.

Generates:
  - ~150-250 transactions/month across 13 months for one demo account,
    mixing 8 recurring vendors (rent, subscriptions, utilities, payroll,
    phone, gym) with noisy one-off spending across ten category pools.
  - Seasonal bumps: heavier shopping in Nov/Dec, heavier transport in
    Jun-Aug.
  - 103 VendorKB rows whose descriptions LEAD with the merchant's literal
    statement descriptor -- the single biggest lever on retrieval quality
    in this project (see the comment above VENDOR_KB_SEED for the
    measurements).
  - Two deliberately distinct tiers of hard-to-read descriptions:
      * AMBIGUOUS_DESCRIPTIONS -- POS-obscured strings a keyword matcher
        would miscategorize, but which the vendor KB CAN resolve. These
        test whether retrieval works, and belong in the eval set.
      * UNRESOLVABLE_DESCRIPTIONS -- strings identifying no merchant at
        all, with deliberately no KB entry. These are what populate the
        human review queue, and are excluded from the eval set because
        they have no defensible ground truth.

Transactions are written with vendor_name/category/subcategory/
confidence_score left NULL -- populating those is the categorizer
agent's job, not this script's.

Run with:
    python -m scripts.generate_synthetic_data            # append
    python -m scripts.generate_synthetic_data --reset    # wipe first
"""

import argparse
import asyncio
import calendar
import random
from datetime import date
from decimal import Decimal
from typing import Optional

from sqlalchemy import delete

from app.db.models import (
    AgentMetric,
    Base,
    DecisionLog,
    EvalResult,
    Transaction,
    VendorKB,
)
from app.db.session import async_session_factory, engine

random.seed(42)

ACCOUNT_ID = "acct_demo_001"

# --- Recurring vendors -------------------------------------------------
# Each: (description, day_of_month, base_amount, variance_pct, summer_bump_pct)
RECURRING_VENDORS = [
    ("ACH DEBIT WESTVIEW APARTMENTS", 1, Decimal("-1800.00"), 0.05, 0.0),
    ("NETFLIX.COM", 5, Decimal("-15.99"), 0.0, 0.0),
    ("SPOTIFY USA", 8, Decimal("-10.99"), 0.0, 0.0),
    ("COMCAST XFINITY", 10, Decimal("-85.00"), 0.05, 0.0),
    ("PACIFIC GAS AND ELECTRIC", 12, Decimal("-110.00"), 0.05, 0.35),
    ("VERIZON WIRELESS", 14, Decimal("-95.00"), 0.05, 0.0),
    ("24 HOUR FITNESS", 3, Decimal("-49.99"), 0.0, 0.0),
]
PAYROLL_DESCRIPTION = "DIRECT DEP PAYROLL ACME CORP"
PAYROLL_DAYS = (1, 15)
# Twice-monthly, so gross monthly income is 2x this. Sized against the
# generated spending pools to leave the household modestly cash-flow
# positive -- if spending outruns income the forecaster just draws a
# straight line into an ever-deeper deficit, which demonstrates nothing.
PAYROLL_BASE_AMOUNT = Decimal("4100.00")
PAYROLL_VARIANCE_PCT = 0.02

SUMMER_MONTHS = {6, 7, 8}
HOLIDAY_MONTHS = {11, 12}

# --- One-off category pools --------------------------------------------
# category -> (subcategory, [description pool], (min_amount, max_amount))
CATEGORY_POOLS = {
    "groceries": {
        "subcategory": "supermarket",
        "descriptions": [
            "TRADER JOE'S #142",
            "WHOLE FOODS MARKET",
            "SAFEWAY STORE 0451",
            "WM SUPERCENTER #4521",
            "SPROUTS FARMERS MKT",
            "COSTCO WHSE #0455",
            "ALDI 42019",
            "H MART #221",
            "GROCERY OUTLET 118",
        ],
        "amount_range": (Decimal("8.00"), Decimal("40.00")),
    },
    "dining": {
        "subcategory": "restaurant",
        "descriptions": [
            "CHIPOTLE ONLINE",
            "MCDONALD'S F1234",
            "SHAKE SHACK #221",
            "STARBUCKS STORE #556",
            "DOORDASH*THAI PLACE",
            "SWEETGREEN 445",
            "PANERA BREAD #601",
            "PEET'S COFFEE 0912",
            "SUBWAY 3391",
            "UBER EATS*SUSHI GO",
            "GRUBHUB*NOODLE HSE",
            "BLUE BOTTLE 0032",
        ],
        "amount_range": (Decimal("5.00"), Decimal("25.00")),
    },
    "transport": {
        "subcategory": "rideshare",
        "descriptions": [
            "UBER TRIP HELP.UBER.COM",
            "LYFT *RIDE THU",
            "SHELL OIL 57443210",
            "CHEVRON 0091823",
            "BART TICKET",
            "ARCO 42881",
            "SFMTA PARKING METER",
            "CLIPPER CARD RELOAD",
            "JIFFY LUBE #1201",
        ],
        "amount_range": (Decimal("6.00"), Decimal("28.00")),
    },
    "shopping": {
        "subcategory": "retail",
        "descriptions": [
            "AMAZON.COM*A1B2C3D4",
            "TARGET T-1122",
            "BEST BUY #00234",
            "HOME DEPOT #1092",
            "IKEA CCA #1234",
            "ETSY.COM - ORDER",
            "REI #0032",
            "NORDSTROM #0410",
            "UNIQLO USA 0221",
            "LOWE'S #2214",
            "WAYFAIR.COM ORDER",
        ],
        "amount_range": (Decimal("10.00"), Decimal("60.00")),
    },
    "entertainment": {
        "subcategory": "leisure",
        "descriptions": [
            "AMC THEATRES 0341",
            "STEAM GAMES",
            "TICKETMASTER EVENT",
            "HULU 877-8244658",
            "NINTENDO CO0221",
            "ALAMO DRAFTHOUSE SF",
            "PRESIDIO BOWL SF",
        ],
        "amount_range": (Decimal("8.00"), Decimal("35.00")),
    },
    "healthcare": {
        "subcategory": "pharmacy",
        "descriptions": [
            "CVS PHARMACY #6621",
            "WALGREENS #4471",
            "KAISER PERMANENTE COPAY",
            "QUEST DIAGNOSTICS",
            "ONE MEDICAL MEMBER",
            "WARBY PARKER 0091",
            "DELTA DENTAL PREM",
        ],
        "amount_range": (Decimal("10.00"), Decimal("75.00")),
    },
    "personal_care": {
        "subcategory": "grooming",
        "descriptions": [
            "SQ *YOGA STUDIO",
            "GREAT CLIPS #3021",
            "SEPHORA #0455",
            "MASSAGE ENVY 0210",
            "SQ *NAIL SALON",
            "SQ *PILATES CO",
        ],
        "amount_range": (Decimal("12.00"), Decimal("70.00")),
    },
    "transfers": {
        "subcategory": "p2p",
        "descriptions": [
            "VENMO PAYMENT",
            "ZELLE TRANSFER TO J DOE",
            "CASH APP*TRANSFER",
            "WISE US INC",
            "VANGUARD BUY INVESTMENT",
        ],
        "amount_range": (Decimal("15.00"), Decimal("110.00")),
    },
    "travel": {
        "subcategory": "booking",
        "descriptions": [
            "UNITED AIRLINES 0162",
            "MARRIOTT HOTELS 4471",
            "HERTZ RENT-A-CAR",
            "EXPEDIA 7412298",
        ],
        # Deliberately modest. An earlier version used 60-400 here, which
        # -- at ~5 travel transactions a month -- produced $1,200/mo of
        # travel spend and pushed the whole dataset into a permanent
        # deficit, making every forecast a runaway negative. The forecaster
        # is only interesting if the underlying household is roughly
        # solvent.
        "amount_range": (Decimal("25.00"), Decimal("140.00")),
    },
    "subscriptions": {
        "subcategory": "software",
        "descriptions": [
            "ADOBE *CREATIVE CLD",
            "APPLE.COM/BILL",
            "NYTIMES*NYTIMES",
            "OPENAI *CHATGPT SUBSCR",
            "DROPBOX*monthly",
        ],
        "amount_range": (Decimal("3.00"), Decimal("55.00")),
    },
}

BASE_CATEGORY_WEIGHTS = {
    "groceries": 20,
    "dining": 20,
    "transport": 14,
    "shopping": 13,
    "entertainment": 8,
    "healthcare": 6,
    "personal_care": 6,
    "transfers": 7,
    "travel": 3,
    "subscriptions": 3,
}

# --- Tier 1: ambiguous but RESOLVABLE via the vendor KB -----------------
# POS-obscured strings a naive keyword matcher would miscategorize, but
# which every have a matching VENDOR_KB_SEED entry quoting the descriptor.
# These test whether retrieval actually works, so they belong in the eval
# set and should categorize with high confidence.
AMBIGUOUS_DESCRIPTIONS = [
    "SQ *JOES COFFEE",
    "PAYPAL *MISCSVC",
    "TST* TAQUERIA",
    "SQ *THE CORNER MKT",
    "PAYPAL *INST XFER",
    "VENMO PAYMENT",
    "CASH APP*TRANSFER",
    "AMZN MKTP US*A1B2C3",
    "SQ *BARBERSHOP",
    "PP*GOOGLE STORAGE",
    "TST*RAMEN BAR",
    "SQ *YOGA STUDIO",
    "PAYPAL *UBER",
    "SQ *DOG GROOMER",
    "APLPAY MISC RETAIL",
    "PAYPAL *AIRBNB",
    "SQ *FARMERS MARKET",
    "SQ *CORNER BAKERY",
    "SQ *BIKE REPAIR",
    "TST*PIZZA KITCHEN",
    "TST*BRUNCH CAFE",
]

# --- Tier 2: genuinely UNRESOLVABLE -------------------------------------
# Statement strings that identify no merchant at all -- a payment
# processor reference and nothing else. There is deliberately NO
# VENDOR_KB_SEED entry for any of these, so retrieval finds nothing
# useful, the model reports low confidence, and the apply layer queues
# them for a human. This is what makes the review queue demonstrable
# rather than empty.
#
# They are excluded from the eval set (see scripts/build_eval_set.py):
# scoring a row whose correct answer nobody can determine from the
# description would measure guessing, not accuracy.
UNRESOLVABLE_DESCRIPTIONS = [
    "SQ *MERCHANT 88213",
    "POS DEBIT 4471",
    "PAYPAL *TRANSFER",
    "APLPAY PURCHASE",
    "TST* 0092841",
    "SQ *VENDOR 5521",
    "CHECKCARD 77120",
    "PP*MERCHANT SVCS",
]

# Each unresolvable descriptor appears this many times across the dataset,
# so the review queue holds a few dozen rows (~2-3% of the total) rather
# than a token handful.
UNRESOLVABLE_REPEATS = 8

# --- VendorKB seed (103 entries) ----------------------------------------
# Every description LEADS with the literal statement descriptor the
# merchant produces, then a short gloss -- e.g.
# "'WM SUPERCENTER #4521' - big-box retailer, mainly grocery and household".
#
# The ordering and the brevity are both deliberate, and both were measured
# rather than guessed. Retrieval embeds the raw transaction string against
# these documents, and cosine similarity against a long paragraph gets
# diluted by everything that isn't the descriptor. Measured on
# 'WM SUPERCENTER #4521' against three phrasings of the same entry:
#
#   no descriptor at all ................ 0.408
#   descriptor buried in trailing prose . 0.576
#   descriptor first, short gloss ....... 0.641
#
# and on 'AMAZON.COM*A1B2C3D4', 0.426 -> 0.668 for the same change. That
# similarity is quoted into the prompt, and the categorizer is instructed
# to lower its confidence when retrieval looks weak -- so a buried alias
# shows up downstream as a transaction sitting in the review queue.
#
# The description text never reaches the LLM (see _format_vendor_hits in
# app/agents/categorizer.py, which passes only vendor name, category, and
# similarity). It exists purely as embedding material, which is why
# optimizing it for retrieval rather than for readability is the right
# trade here.
#
# Deliberately absent: any entry covering UNRESOLVABLE_DESCRIPTIONS.
# Those are supposed to fail retrieval and land in the review queue --
# tests/test_generate_synthetic_data.py enforces that.
VENDOR_KB_SEED = [
    # --- recurring / household -----------------------------------------
    ("Westview Apartments", "housing", "rent", "'ACH DEBIT WESTVIEW APARTMENTS' - monthly apartment rent, auto-drafted on the 1st.", "seed"),
    ("Netflix", "entertainment", "streaming", "'NETFLIX.COM' - monthly video streaming subscription.", "seed"),
    ("Spotify", "entertainment", "streaming", "'SPOTIFY USA' - monthly music streaming subscription.", "seed"),
    ("Comcast Xfinity", "utilities", "internet", "'COMCAST XFINITY' - home internet and cable, billed monthly.", "seed"),
    ("Pacific Gas and Electric", "utilities", "electricity", "'PACIFIC GAS AND ELECTRIC' - regional electric utility, higher in summer.", "seed"),
    ("Verizon Wireless", "utilities", "phone", "'VERIZON WIRELESS' - mobile phone carrier, billed monthly.", "seed"),
    ("24 Hour Fitness", "personal_care", "gym", "'24 HOUR FITNESS' - monthly gym membership. Fitness/wellness, categorised with yoga and pilates rather than with medical care.", "seed"),
    ("Acme Corp Payroll", "income", "payroll", "'DIRECT DEP PAYROLL ACME CORP' - salary direct deposit, twice monthly on the 1st and 15th.", "seed"),
    ("State Farm Insurance", "housing", "insurance", "'STATE FARM INSURANCE' - renters insurance premium, billed monthly.", "seed"),
    ("Recology", "utilities", "waste", "'RECOLOGY SF' - municipal waste and recycling service.", "seed"),
    ("SF Water Department", "utilities", "water", "'SFPUC WATER DEPT' - municipal water utility.", "seed"),
    # --- POS-obscured merchants (Square / Toast / PayPal passthrough) ---
    ("Joe's Coffee", "dining", "coffee", "'SQ *JOES COFFEE' - independent coffee shop billing through Square.", "seed"),
    ("Miscellaneous PayPal Merchant", "shopping", "online_retail", "'PAYPAL *MISCSVC' - PayPal-processed online purchase, merchant name not disclosed.", "seed"),
    ("Local Taqueria", "dining", "restaurant", "'TST* TAQUERIA' - neighborhood Mexican restaurant on the Toast POS system.", "seed"),
    ("The Corner Market", "groceries", "convenience_store", "'SQ *THE CORNER MKT' - small corner grocery and convenience store using Square.", "seed"),
    ("Instant Transfer", "transfers", "p2p", "'PAYPAL *INST XFER' - instant PayPal-to-bank transfer.", "seed"),
    ("The Neighborhood Barbershop", "personal_care", "grooming", "'SQ *BARBERSHOP' - local barbershop billing through Square.", "seed"),
    ("Google Storage", "subscriptions", "cloud_storage", "'PP*GOOGLE STORAGE' - Google One cloud storage subscription billed via PayPal.", "seed"),
    ("Ramen Bar", "dining", "restaurant", "'TST*RAMEN BAR' - casual ramen restaurant on the Toast POS system.", "seed"),
    ("Riverside Yoga Studio", "personal_care", "fitness_class", "'SQ *YOGA STUDIO' - yoga class packages billed through Square. Wellness, not entertainment.", "seed"),
    ("Dog Grooming Salon", "personal_care", "pet_care", "'SQ *DOG GROOMER' - pet grooming business billing through Square.", "seed"),
    ("Apple Pay Misc Retail", "shopping", "retail", "'APLPAY MISC RETAIL' - tap-to-pay purchase at a small retailer, no merchant name passed through.", "seed"),
    ("Farmers Market Vendor", "groceries", "farmers_market", "'SQ *FARMERS MARKET' - local farmers market stall accepting Square.", "seed"),
    ("Nail Salon", "personal_care", "grooming", "'SQ *NAIL SALON' - nail salon billing through Square.", "seed"),
    ("Corner Bakery", "dining", "bakery", "'SQ *CORNER BAKERY' - neighborhood bakery using Square.", "seed"),
    ("Pilates Studio", "personal_care", "fitness_class", "'SQ *PILATES CO' - pilates class packs billed through Square.", "seed"),
    ("Bike Repair Shop", "transport", "maintenance", "'SQ *BIKE REPAIR' - bicycle repair and tune-up shop using Square.", "seed"),
    ("Toast Pizza Kitchen", "dining", "restaurant", "'TST*PIZZA KITCHEN' - pizza restaurant on the Toast POS system.", "seed"),
    ("Toast Brunch Cafe", "dining", "restaurant", "'TST*BRUNCH CAFE' - brunch cafe on the Toast POS system.", "seed"),
    # --- groceries -------------------------------------------------------
    ("Walmart Supercenter", "groceries", "supermarket", "'WM SUPERCENTER #4521' - big-box retailer, mainly grocery and household shopping.", "seed"),
    ("Trader Joe's", "groceries", "supermarket", '"TRADER JOE\'S #142" - grocery chain known for private-label products.', "seed"),
    ("Whole Foods Market", "groceries", "supermarket", "'WHOLE FOODS MARKET' - upscale grocery chain.", "seed"),
    ("Safeway", "groceries", "supermarket", "'SAFEWAY STORE 0451' - mainstream regional supermarket chain.", "seed"),
    ("Sprouts Farmers Market", "groceries", "supermarket", "'SPROUTS FARMERS MKT' - grocery chain focused on fresh and organic foods.", "seed"),
    ("Costco Wholesale", "groceries", "warehouse_club", "'COSTCO WHSE #0455' - membership warehouse club, bulk grocery and household.", "seed"),
    ("Aldi", "groceries", "supermarket", "'ALDI 42019' - discount grocery chain.", "seed"),
    ("H Mart", "groceries", "supermarket", "'H MART #221' - Asian grocery supermarket chain.", "seed"),
    ("Grocery Outlet", "groceries", "supermarket", "'GROCERY OUTLET 118' - discount grocery chain.", "seed"),
    # --- dining ----------------------------------------------------------
    ("Chipotle Mexican Grill", "dining", "fast_casual", "'CHIPOTLE ONLINE' - fast-casual Mexican chain, app orders.", "seed"),
    ("McDonald's", "dining", "fast_food", '"MCDONALD\'S F1234" - fast food chain.', "seed"),
    ("Shake Shack", "dining", "fast_casual", "'SHAKE SHACK #221' - fast-casual burger chain.", "seed"),
    ("Starbucks", "dining", "coffee", "'STARBUCKS STORE #556' - coffee shop chain.", "seed"),
    ("DoorDash", "dining", "food_delivery", "'DOORDASH*THAI PLACE' - food delivery app, restaurant name appended.", "seed"),
    ("Sweetgreen", "dining", "fast_casual", "'SWEETGREEN 445' - fast-casual salad chain.", "seed"),
    ("Panera Bread", "dining", "fast_casual", "'PANERA BREAD #601' - bakery-cafe chain.", "seed"),
    ("Peet's Coffee", "dining", "coffee", '"PEET\'S COFFEE 0912" - coffee shop chain.', "seed"),
    ("Subway", "dining", "fast_food", "'SUBWAY 3391' - sandwich chain.", "seed"),
    ("Uber Eats", "dining", "food_delivery", "'UBER EATS*SUSHI GO' - food delivery, restaurant name appended.", "seed"),
    ("Grubhub", "dining", "food_delivery", "'GRUBHUB*NOODLE HSE' - food delivery, restaurant name appended.", "seed"),
    ("Blue Bottle Coffee", "dining", "coffee", "'BLUE BOTTLE 0032' - specialty coffee chain.", "seed"),
    # --- transport -------------------------------------------------------
    ("Uber", "transport", "rideshare", "'UBER TRIP HELP.UBER.COM' or 'PAYPAL *UBER' - rideshare trip.", "seed"),
    ("Lyft", "transport", "rideshare", "'LYFT *RIDE THU' - rideshare trip, day abbreviation appended.", "seed"),
    ("Shell", "transport", "gas_station", "'SHELL OIL 57443210' - gas station chain.", "seed"),
    ("Chevron", "transport", "gas_station", "'CHEVRON 0091823' - gas station chain.", "seed"),
    ("BART", "transport", "public_transit", "'BART TICKET' - regional public rail transit.", "seed"),
    ("Arco", "transport", "gas_station", "'ARCO 42881' - discount gas station chain.", "seed"),
    ("SFMTA Parking", "transport", "parking", "'SFMTA PARKING METER' - municipal parking meters and garages.", "seed"),
    ("Clipper Card", "transport", "public_transit", "'CLIPPER CARD RELOAD' - regional transit fare card reload.", "seed"),
    ("Jiffy Lube", "transport", "maintenance", "'JIFFY LUBE #1201' - oil change and vehicle maintenance.", "seed"),
    # --- shopping --------------------------------------------------------
    ("Amazon.com", "shopping", "online_retail", "'AMAZON.COM*A1B2C3D4' - online retail purchase from Amazon's main storefront.", "seed"),
    ("Amazon Marketplace", "shopping", "online_retail", "'AMZN MKTP US*A1B2C3' - third-party seller purchase via Amazon marketplace.", "seed"),
    ("Target", "shopping", "retail", "'TARGET T-1122' - big-box general merchandise retailer.", "seed"),
    ("Best Buy", "shopping", "electronics", "'BEST BUY #00234' - electronics and appliance retailer.", "seed"),
    ("Home Depot", "shopping", "home_improvement", "'HOME DEPOT #1092' - home improvement and hardware retailer.", "seed"),
    ("IKEA", "shopping", "furniture", "'IKEA CCA #1234' - furniture and home goods retailer.", "seed"),
    ("Etsy", "shopping", "online_retail", "'ETSY.COM - ORDER' - online marketplace for handmade and vintage goods.", "seed"),
    ("REI", "shopping", "sporting_goods", "'REI #0032' - outdoor gear and apparel co-op.", "seed"),
    ("Nordstrom", "shopping", "clothing", "'NORDSTROM #0410' - department store chain.", "seed"),
    ("Uniqlo", "shopping", "clothing", "'UNIQLO USA 0221' - clothing retailer.", "seed"),
    ("Lowe's", "shopping", "home_improvement", '"LOWE\'S #2214" - home improvement retailer.', "seed"),
    ("Wayfair", "shopping", "furniture", "'WAYFAIR.COM ORDER' - online furniture and home goods retailer.", "seed"),
    # --- entertainment ---------------------------------------------------
    ("AMC Theatres", "entertainment", "movies", "'AMC THEATRES 0341' - movie theater chain.", "seed"),
    ("Steam", "entertainment", "video_games", "'STEAM GAMES' - digital video game storefront.", "seed"),
    ("Ticketmaster", "entertainment", "events", "'TICKETMASTER EVENT' - event and concert ticketing platform.", "seed"),
    ("Hulu", "entertainment", "streaming", "'HULU 877-8244658' - video streaming subscription.", "seed"),
    ("Nintendo eShop", "entertainment", "video_games", "'NINTENDO CO0221' - digital game storefront for Nintendo consoles.", "seed"),
    ("Alamo Drafthouse", "entertainment", "movies", "'ALAMO DRAFTHOUSE SF' - cinema chain with in-theater dining.", "seed"),
    ("Presidio Bowl", "entertainment", "leisure", "'PRESIDIO BOWL SF' - local bowling alley.", "seed"),
    # --- healthcare ------------------------------------------------------
    ("CVS Pharmacy", "healthcare", "pharmacy", "'CVS PHARMACY #6621' - retail pharmacy and convenience chain.", "seed"),
    ("Walgreens", "healthcare", "pharmacy", "'WALGREENS #4471' - retail pharmacy and convenience chain.", "seed"),
    ("Kaiser Permanente", "healthcare", "medical", "'KAISER PERMANENTE COPAY' - healthcare provider visit copay.", "seed"),
    ("Quest Diagnostics", "healthcare", "lab_work", "'QUEST DIAGNOSTICS' - medical laboratory testing service.", "seed"),
    ("One Medical", "healthcare", "medical", "'ONE MEDICAL MEMBER' - primary care practice membership.", "seed"),
    ("Warby Parker", "healthcare", "vision", "'WARBY PARKER 0091' - eyewear retailer and optometry provider.", "seed"),
    ("Delta Dental", "healthcare", "dental", "'DELTA DENTAL PREM' - dental insurance premium.", "seed"),
    # --- personal care ---------------------------------------------------
    ("Great Clips", "personal_care", "grooming", "'GREAT CLIPS #3021' - hair salon chain.", "seed"),
    ("Sephora", "personal_care", "cosmetics", "'SEPHORA #0455' - cosmetics and beauty retailer.", "seed"),
    ("Massage Envy", "personal_care", "spa", "'MASSAGE ENVY 0210' - massage and spa franchise.", "seed"),
    # --- transfers -------------------------------------------------------
    ("Venmo", "transfers", "p2p", "'VENMO PAYMENT' - peer-to-peer payment app, splitting bills or paying friends.", "seed"),
    ("Cash App", "transfers", "p2p", "'CASH APP*TRANSFER' - peer-to-peer payment app transfer.", "seed"),
    ("Zelle", "transfers", "p2p", "'ZELLE TRANSFER TO J DOE' - bank-to-bank peer-to-peer transfer, recipient name appended.", "seed"),
    ("Wise", "transfers", "international", "'WISE US INC' - international money transfer service.", "seed"),
    ("Vanguard Brokerage", "transfers", "investment", "'VANGUARD BUY INVESTMENT' - brokerage account contribution.", "seed"),
    # --- subscriptions ---------------------------------------------------
    ("Adobe Creative Cloud", "subscriptions", "software", "'ADOBE *CREATIVE CLD' - monthly creative software subscription.", "seed"),
    ("iCloud Storage", "subscriptions", "cloud_storage", "'APPLE.COM/BILL' - Apple iCloud storage subscription.", "seed"),
    ("New York Times", "subscriptions", "news", "'NYTIMES*NYTIMES' - digital news subscription.", "seed"),
    ("ChatGPT Plus", "subscriptions", "software", "'OPENAI *CHATGPT SUBSCR' - monthly AI assistant subscription.", "seed"),
    ("Dropbox", "subscriptions", "cloud_storage", "'DROPBOX*monthly' - cloud file storage subscription.", "seed"),
    # --- travel ----------------------------------------------------------
    ("Airbnb", "travel", "lodging", "'PAYPAL *AIRBNB' - short-term lodging booking, routed through PayPal.", "seed"),
    ("United Airlines", "travel", "airfare", "'UNITED AIRLINES 0162' - airline ticket purchase.", "seed"),
    ("Marriott Hotels", "travel", "lodging", "'MARRIOTT HOTELS 4471' - hotel stay.", "seed"),
    ("Hertz", "travel", "car_rental", "'HERTZ RENT-A-CAR' - rental car booking.", "seed"),
    ("Expedia", "travel", "booking", "'EXPEDIA 7412298' - online travel booking platform.", "seed"),
]


# --- Ground truth for eval -----------------------------------------------
# The pools/vendors above already "know" each description's category; these
# tables expose that as a lookup instead of discarding it, so an eval
# harness can grab true labels without re-guessing or touching the LLM.

CATEGORY_NAME_MAP = {
    "housing": "Housing",
    "utilities": "Utilities",
    "groceries": "Groceries",
    "dining": "Dining",
    "transport": "Transport",
    "shopping": "Shopping",
    "entertainment": "Entertainment",
    "healthcare": "Healthcare",
    "subscriptions": "Subscriptions",
    "income": "Income",
    "transfers": "Transfer",
    # Both of these used to map to "Other" because the categorizer had no
    # matching category -- which is what made PAYPAL *AIRBNB a guaranteed
    # eval miss (the KB correctly said travel/lodging; the taxonomy had
    # nowhere to put it). Travel and Personal Care are now real categories
    # in DEFAULT_CATEGORIES, so ground truth agrees with the KB again.
    "personal_care": "Personal Care",
    "travel": "Travel",
}

RECURRING_TRUE_CATEGORIES = {
    "ACH DEBIT WESTVIEW APARTMENTS": "Housing",
    "NETFLIX.COM": "Entertainment",
    "SPOTIFY USA": "Entertainment",
    "COMCAST XFINITY": "Utilities",
    "PACIFIC GAS AND ELECTRIC": "Utilities",
    "VERIZON WIRELESS": "Utilities",
    "24 HOUR FITNESS": "Personal Care",
    PAYROLL_DESCRIPTION: "Income",
}

# Only the ambiguous strings NOT already covered by CATEGORY_POOLS need an
# explicit entry here (VENMO PAYMENT, CASH APP*TRANSFER, and SQ *YOGA STUDIO
# double as regular pool descriptions and resolve via the pool lookup).
AMBIGUOUS_TRUE_CATEGORIES = {
    "SQ *JOES COFFEE": "Dining",
    "PAYPAL *MISCSVC": "Shopping",
    "TST* TAQUERIA": "Dining",
    "SQ *THE CORNER MKT": "Groceries",
    "PAYPAL *INST XFER": "Transfer",
    "AMZN MKTP US*A1B2C3": "Shopping",
    "SQ *BARBERSHOP": "Personal Care",
    "PP*GOOGLE STORAGE": "Subscriptions",
    "TST*RAMEN BAR": "Dining",
    "PAYPAL *UBER": "Transport",
    "SQ *DOG GROOMER": "Personal Care",
    "APLPAY MISC RETAIL": "Shopping",
    "PAYPAL *AIRBNB": "Travel",
    "SQ *FARMERS MARKET": "Groceries",
    "SQ *CORNER BAKERY": "Dining",
    "SQ *BIKE REPAIR": "Transport",
    "TST*PIZZA KITCHEN": "Dining",
    "TST*BRUNCH CAFE": "Dining",
}


def true_category_for_description(description: str) -> str:
    """The ground-truth category for a generator-produced raw_description.

    Raises for UNRESOLVABLE_DESCRIPTIONS: by construction those identify
    no merchant, so there is no defensible ground truth to return.
    Callers that sample the database (scripts/build_eval_set.py) must
    filter them out rather than score them.
    """
    if description in UNRESOLVABLE_DESCRIPTIONS:
        raise ValueError(
            f"{description!r} is deliberately unresolvable -- it has no ground-truth "
            f"category and must be excluded from the eval set, not scored."
        )

    if description in RECURRING_TRUE_CATEGORIES:
        return RECURRING_TRUE_CATEGORIES[description]

    for category_key, pool in CATEGORY_POOLS.items():
        if description in pool["descriptions"]:
            return CATEGORY_NAME_MAP[category_key]

    if description in AMBIGUOUS_TRUE_CATEGORIES:
        return AMBIGUOUS_TRUE_CATEGORIES[description]

    raise ValueError(f"no true category mapping for description: {description!r}")


def month_range(n_months: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    """Return the n_months (year, month) pairs ending at (end_year, end_month), oldest first."""
    months = []
    year, month = end_year, end_month
    for _ in range(n_months):
        months.append((year, month))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def jittered_day(base_day: int, days_in_month: int) -> int:
    jitter = random.randint(-1, 1)
    return min(max(base_day + jitter, 1), days_in_month)


def build_recurring_transactions(year: int, month: int) -> list[Transaction]:
    days_in_month = calendar.monthrange(year, month)[1]
    rows = []

    for description, base_day, base_amount, variance_pct, summer_bump_pct in RECURRING_VENDORS:
        amount = base_amount
        if variance_pct:
            amount *= Decimal(str(random.uniform(1 - variance_pct, 1 + variance_pct)))
        if summer_bump_pct and month in SUMMER_MONTHS:
            amount *= Decimal(str(1 + summer_bump_pct))
        rows.append(
            Transaction(
                account_id=ACCOUNT_ID,
                posted_date=date(year, month, jittered_day(base_day, days_in_month)),
                amount=amount.quantize(Decimal("0.01")),
                raw_description=description,
                is_recurring=True,
            )
        )

    for payroll_day in PAYROLL_DAYS:
        amount = PAYROLL_BASE_AMOUNT * Decimal(
            str(random.uniform(1 - PAYROLL_VARIANCE_PCT, 1 + PAYROLL_VARIANCE_PCT))
        )
        rows.append(
            Transaction(
                account_id=ACCOUNT_ID,
                posted_date=date(year, month, min(payroll_day, days_in_month)),
                amount=amount.quantize(Decimal("0.01")),
                raw_description=PAYROLL_DESCRIPTION,
                is_recurring=True,
            )
        )

    return rows


def seasonal_category_weights(month: int) -> dict[str, int]:
    weights = dict(BASE_CATEGORY_WEIGHTS)
    if month in HOLIDAY_MONTHS:
        weights["shopping"] = int(weights["shopping"] * 2.2)
    if month in SUMMER_MONTHS:
        weights["transport"] = int(weights["transport"] * 1.8)
    return weights


def _partition_across_months(descriptions: list[str], months: list[tuple[int, int]]) -> list[list[str]]:
    """Partition a shuffled copy of `descriptions` into per-month chunks.

    Coverage of every entry is guaranteed by construction rather than by
    chance -- the original modular-index scheme could collide across
    months and silently drop entries (see tests/test_generate_synthetic_data.py).
    Handles both more and fewer descriptions than months.
    """
    quotas = [0] * len(months)
    for i in range(len(descriptions)):
        quotas[i % len(months)] += 1

    shuffled = list(descriptions)
    random.shuffle(shuffled)

    assignments = []
    cursor = 0
    for quota in quotas:
        assignments.append(shuffled[cursor : cursor + quota])
        cursor += quota
    return assignments


def ambiguous_assignments_for_months(months: list[tuple[int, int]]) -> list[list[str]]:
    """Assign each AMBIGUOUS_DESCRIPTIONS string to exactly one month."""
    return _partition_across_months(AMBIGUOUS_DESCRIPTIONS, months)


def unresolvable_assignments_for_months(months: list[tuple[int, int]]) -> list[list[str]]:
    """Assign each UNRESOLVABLE_DESCRIPTIONS string to months, repeated.

    Repeated REPEATS_PER_DESCRIPTION times so the review queue has enough
    volume to be worth looking at -- a queue with 8 rows in it doesn't
    demonstrate much.
    """
    repeated = UNRESOLVABLE_DESCRIPTIONS * UNRESOLVABLE_REPEATS
    return _partition_across_months(repeated, months)


def build_oneoff_transactions(
    year: int,
    month: int,
    count: int,
    ambiguous_descriptions: list[str],
    unresolvable_descriptions: Optional[list[str]] = None,
) -> list[Transaction]:
    days_in_month = calendar.monthrange(year, month)[1]
    weights = seasonal_category_weights(month)
    categories = list(weights.keys())
    category_weights = list(weights.values())

    # Ambiguous first, then unresolvable, then the ordinary weighted pools.
    seeded = list(ambiguous_descriptions) + list(unresolvable_descriptions or [])

    rows = []
    for i in range(count):
        posted_date = date(year, month, random.randint(1, days_in_month))

        if i < len(seeded):
            description = seeded[i]
            low, high = Decimal("5.00"), Decimal("40.00")
        else:
            category = random.choices(categories, weights=category_weights, k=1)[0]
            pool = CATEGORY_POOLS[category]
            description = random.choice(pool["descriptions"])
            low, high = pool["amount_range"]
            if category == "shopping" and month in HOLIDAY_MONTHS:
                high *= Decimal("1.6")
            if category == "transport" and month in SUMMER_MONTHS:
                high *= Decimal("1.5")

        amount = Decimal(str(round(random.uniform(float(low), float(high)), 2)))

        rows.append(
            Transaction(
                account_id=ACCOUNT_ID,
                posted_date=posted_date,
                amount=-amount,
                raw_description=description,
                is_recurring=False,
            )
        )

    return rows


async def seed_vendor_kb(session) -> int:
    rows = [
        VendorKB(
            vendor_name=vendor_name,
            canonical_category=category,
            canonical_subcategory=subcategory,
            description=description,
            source=source,
        )
        for vendor_name, category, subcategory, description, source in VENDOR_KB_SEED
    ]
    session.add_all(rows)
    await session.commit()
    return len(rows)


async def reset_tables() -> None:
    """Delete every row this dataset owns, plus everything derived from it.

    Transactions are the root: decision_log, agent_metrics, and
    eval_results all reference transaction ids, so regenerating
    transactions without clearing those leaves orphaned rows pointing at
    ids that no longer exist (or, worse, at *different* rows that reused
    the ids).
    """
    async with async_session_factory() as session:
        for model in (EvalResult, DecisionLog, AgentMetric, Transaction, VendorKB):
            await session.execute(delete(model))
        await session.commit()


async def main(reset: bool = False) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    if reset:
        await reset_tables()
        print("Reset: cleared transactions, vendor_kb, decision_log, agent_metrics, eval_results.")

    total_transactions = 0

    async with async_session_factory() as session:
        vendor_count = await seed_vendor_kb(session)

    today = date.today()
    months = month_range(13, today.year, today.month)
    # Spread both tiers of special descriptions across the year rather than
    # dumping them in one month -- guarantees every one gets placed
    # somewhere (see _partition_across_months).
    ambiguous_assignments = ambiguous_assignments_for_months(months)
    unresolvable_assignments = unresolvable_assignments_for_months(months)

    n_ambiguous = 0
    n_unresolvable = 0

    for (year, month), ambiguous_descriptions, unresolvable_descriptions in zip(
        months, ambiguous_assignments, unresolvable_assignments
    ):
        recurring_rows = build_recurring_transactions(year, month)
        target_total = random.randint(150, 250)
        oneoff_count = max(target_total - len(recurring_rows), 0)
        oneoff_rows = build_oneoff_transactions(
            year, month, oneoff_count, ambiguous_descriptions, unresolvable_descriptions
        )

        async with async_session_factory() as session:
            session.add_all(recurring_rows + oneoff_rows)
            await session.commit()

        n_ambiguous += len(ambiguous_descriptions)
        n_unresolvable += len(unresolvable_descriptions)
        total_transactions += len(recurring_rows) + len(oneoff_rows)
        print(f"{year}-{month:02d}: wrote {len(recurring_rows) + len(oneoff_rows)} transactions")

    print(
        f"\nDone. VendorKB rows: {vendor_count}. Total transactions: {total_transactions}.\n"
        f"  resolvable-ambiguous rows placed: {n_ambiguous} "
        f"({len(AMBIGUOUS_DESCRIPTIONS)} distinct)\n"
        f"  unresolvable rows placed:         {n_unresolvable} "
        f"({len(UNRESOLVABLE_DESCRIPTIONS)} distinct -- these should land in the review queue)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear transactions/vendor_kb and all derived tables before seeding.",
    )
    args = parser.parse_args()
    asyncio.run(main(reset=args.reset))
