"""
config/settings.py
==================
Single source of truth for the entire project.

All other modules import from here. Nothing in this file has side effects:
no directories are created, no files are read, no network calls are made.

Dissertation: "Agentic AI for Sovereign Risk Assessment under
               Climate-Related Fiscal Stress"
Experimental matrix: 3 economy types × 3 climate risk levels (3×3)
"""

from pathlib import Path

# ── 1. PATHS ──────────────────────────────────────────────────────────────────
# All paths are resolved relative to this file so the project can be moved
# or run from any working directory without breaking.

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR:      Path = PROJECT_ROOT / "data"
RAW_DATA_DIR:  Path = DATA_DIR / "raw"
EXTRACTED_DIR: Path = DATA_DIR / "extracted"
INTERIM_DIR:   Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR:  Path = PROJECT_ROOT / "results"
FIGURES_DIR:  Path = RESULTS_DIR / "figures"
MODELS_DIR:   Path = RESULTS_DIR / "models"
LOGS_DIR:     Path = RESULTS_DIR / "logs"

# Raw source files 
RAW_FILES: dict[str, Path] = {
    "world_bank":       RAW_DATA_DIR / "world_bank_data.csv",
    "emdat":            RAW_DATA_DIR / "emdat_disaster_data.xlsx",
    "imf_debt":         RAW_DATA_DIR / "imf-central_gov_debt_data.xlsx",
    "imf_weo":          RAW_DATA_DIR / "world_economic_outlook_imf.xls",
    "inform_risk":      RAW_DATA_DIR / "European_commission-INFORM_Risk_Mid_2025_v071.xlsx",
    "wb_income_class":  RAW_DATA_DIR / "wbg_income_class_2025_10_07.xlsx",
    # ND-GAIN annual time series (preferred over INFORM — has data back to 1995)
    "nd_gain":          RAW_DATA_DIR / "notre_dame-gain_countryindex_2026" / "resources" / "gain" / "gain.csv",
    "nd_vulnerability": RAW_DATA_DIR / "notre_dame-gain_countryindex_2026" / "resources" / "vulnerability" / "vulnerability.csv",
    "nd_readiness":     RAW_DATA_DIR / "notre_dame-gain_countryindex_2026" / "resources" / "readiness" / "readiness.csv",
}


# ── 2. TIME PERIOD ────────────────────────────────────────────────────────────

START_YEAR: int = 1990
END_YEAR:   int = 2024


# ── 3. ECONOMY TAXONOMY ───────────────────────────────────────────────────────
# The dissertation's 3×3 matrix requires exactly three economy types.
# Classification follows IMF World Economic Outlook groupings, operationalised
# via World Bank income categories.
#
# Mapping rule:
#   High income       ->        Advanced
#   Upper middle income    ->   Emerging Market
#   Lower middle income    ->   Developing
#   Low income     ->     Developing  

ECONOMY_TYPES: list[str] = ["Advanced", "Emerging Market", "Developing"]

# Country list — to be finalised after data exploration in wbg_data.ipynb.
# Once you have reviewed coverage across all WB economies, uncomment and
# populate COUNTRIES_BY_TYPE, then re-run the extraction pipeline.
#
# COUNTRIES_BY_TYPE: dict[str, list[str]] = {
#     "Advanced": [
#         "USA", "JPN", "DEU", "GBR", "FRA",
#         "CAN", "AUS", "CHE", "NLD", "SWE",
#     ],
#     "Emerging Market": [
#         "CHN", "BRA", "IND", "MEX", "IDN",
#         "TUR", "ZAF", "THA", "MYS", "PHL",
#     ],
#     "Developing": [
#         "NGA", "BGD", "KEN", "VNM", "GHA",
#         "PAK", "EGY", "MAR", "LKA", "BOL",
#         "ETH", "MWI", "NER", "HTI", "SLE",
#     ],
# }

# Placeholder — both will be populated once COUNTRIES_BY_TYPE is uncommented above.
COUNTRIES_BY_TYPE: dict[str, list[str]] = {}
COUNTRIES: list[str] = []
COUNTRY_TO_ECONOMY_TYPE: dict[str, str] = {}

# World Bank income group label → dissertation economy type.
# Used when classifying countries that are not in COUNTRIES_BY_TYPE above.
WB_INCOME_TO_ECONOMY_TYPE: dict[str, str] = {
    "High income":          "Advanced",
    "Upper middle income":  "Emerging Market",
    "Lower middle income":  "Developing",
    "Low income":           "Developing",
}


# ── 4. WORLD BANK INDICATORS ──────────────────────────────────────────────────
# Format: { WB_indicator_code: target_column_name }
#
# Columns prefixed with their economic concept for readability in downstream
# code. 

WB_INDICATORS: dict[str, str] = {
    # Output and growth
    "NY.GDP.MKTP.CD":    "gdp_current_usd",        # GDP (current US$)
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",             # GDP growth (annual %)
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",      # GDP per capita (current US$)
    "SP.POP.TOTL":       "population",              # Total population

    # Fiscal accounts (required for MDP state space and reward function)
    "GC.DOD.TOTL.GD.ZS": "debt_to_gdp",            # Central gov debt (% GDP)
    "GC.BAL.CASH.GD.ZS": "fiscal_balance_gdp",     # Cash surplus/deficit (% GDP)
    "GC.TAX.TOTL.GD.ZS": "tax_revenue_gdp",        # Tax revenue (% GDP)
    "GC.XPN.TOTL.GD.ZS": "gov_expenditure_gdp",    # Government expenditure (% GDP)
    "GC.XPN.INTP.GD.ZS": "interest_payments_gdp",  # Interest payments (% GDP)
                                                    # → used to derive primary balance

    # Prices and monetary
    "FP.CPI.TOTL.ZG":    "inflation_cpi",           # CPI inflation (annual %)
    "NY.GDP.DEFL.KD.ZG": "inflation_deflator",      # GDP deflator inflation (annual %)
                                                    # → preferred for real interest rate calc
    "FR.INR.LNDP.ZS":    "real_interest_rate",      # Real interest rate (%) — sparse

    # External sector
    "NE.EXP.GNFS.ZS":    "exports_gdp",             # Exports of goods & services (% GDP)
    "NE.IMP.GNFS.ZS":    "imports_gdp",             # Imports of goods & services (% GDP)
    "BN.CAB.XOKA.GD.ZS": "current_account_gdp",    # Current account balance (% GDP)
    "FI.RES.TOTL.CD":    "foreign_reserves_usd",    # Total reserves (current US$)
}


# ── 5. DATA QUALITY THRESHOLDS ────────────────────────────────────────────────

MIN_YEARS_FOR_RL: int = 20

# Maximum allowable missingness (fraction) per variable before a variable
# is flagged for review in the cleaning pipeline.
MAX_MISSINGNESS_FRACTION: float = 0.30

# IQR multiplier for outlier detection (standard = 1.5; 3.0 = extreme outliers only)
# Using 3.0 to avoid flagging genuine macroeconomic extremes (e.g. Zimbabwe inflation)
# as errors.
IQR_OUTLIER_MULTIPLIER: float = 3.0


# ── 6. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
# Temporal split — the model cannot observe any data after TRAIN_END_YEAR
# during training. This prevents data leakage from test period.
#
# Train: START_YEAR … TRAIN_END_YEAR  (inclusive)  → ~29 years
# Test:  TEST_START_YEAR … END_YEAR   (inclusive)  → ~6 years
#
# Dissertation note: The test window (2019–2024) includes COVID-19 (2020),
# which is an out-of-distribution shock — important to discuss in evaluation.

TRAIN_END_YEAR:   int = 2018
TEST_START_YEAR:  int = 2019


# ── 7. NORMALISATION ──────────────────────────────────────────────────────────
# Normalisation statistics are computed on the training set only and stored
# in PROCESSED_DIR / "normalisation_params.csv" for reproducibility and
# for denormalising model outputs at inference time.

# Columns that must never be normalised (identifiers, binary flags, categoricals)
NO_NORMALISE_COLS: list[str] = [
    "country",
    "year",
    "economy_type",
    "income_group",
    "recession",       # binary
    "high_debt",       # binary
    "high_inflation",  # binary
]


# ── 8. LOGGING ────────────────────────────────────────────────────────────────
# All pipeline scripts use Python's logging module (not print).
# Log files are written to LOGS_DIR / "<script_name>.log".

LOG_LEVEL:       str = "INFO"
LOG_FORMAT:      str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
