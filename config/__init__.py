"""
config/__init__.py
==================
Re-exports all public names from settings.py.

Usage anywhere in the project:
    from config import RAW_DATA_DIR, COUNTRIES, WB_INDICATORS
    from config import START_YEAR, END_YEAR, ECONOMY_TYPES
"""

from config.settings import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    EXTRACTED_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    LOGS_DIR,
    RAW_FILES,

    # Time period
    START_YEAR,
    END_YEAR,

    # Economy taxonomy
    ECONOMY_TYPES,
    COUNTRIES_BY_TYPE,
    COUNTRIES,
    COUNTRY_TO_ECONOMY_TYPE,
    WB_INCOME_TO_ECONOMY_TYPE,

    # World Bank indicators
    WB_INDICATORS,

    # Data quality thresholds
    MIN_YEARS_FOR_RL,
    MAX_MISSINGNESS_FRACTION,
    IQR_OUTLIER_MULTIPLIER,

    # Train/test split
    TRAIN_END_YEAR,
    TEST_START_YEAR,

    # Normalisation
    NO_NORMALISE_COLS,

    # Logging
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "RAW_DATA_DIR", "EXTRACTED_DIR",
    "INTERIM_DIR", "PROCESSED_DIR", "RESULTS_DIR", "FIGURES_DIR",
    "MODELS_DIR", "LOGS_DIR", "RAW_FILES",
    "START_YEAR", "END_YEAR",
    "ECONOMY_TYPES", "COUNTRIES_BY_TYPE", "COUNTRIES",
    "COUNTRY_TO_ECONOMY_TYPE", "WB_INCOME_TO_ECONOMY_TYPE",
    "WB_INDICATORS",
    "MIN_YEARS_FOR_RL", "MAX_MISSINGNESS_FRACTION", "IQR_OUTLIER_MULTIPLIER",
    "TRAIN_END_YEAR", "TEST_START_YEAR",
    "NO_NORMALISE_COLS",
    "LOG_LEVEL", "LOG_FORMAT", "LOG_DATE_FORMAT",
]
