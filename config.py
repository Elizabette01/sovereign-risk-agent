"""
Configuration file for Sovereign Climate Risk AI Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
# List of countries to include in the analysis, categorized by development status
COUNTRIES = [
    'USA', 'JPN', 'DEU', 'GBR', 'FRA', 'CAN', 'AUS', 'CHE', 'NLD', 'SWE',  # Advanced
    'MEX', 'CHN','TUR', 'ZAF', 'BRA', 'IND', 'IDN', 'THA', 'MYS','PHL',  # Emerging
    'NGA','BGD', 'KEN', 'VNM', 'GHA', 'PAK', 'EGY', 'MAR', 'LKA', 'BOL', # Developing
    'AFG', 'CAF', 'NER', 'ETH', 'SSD', 'SOM', 'TCD', 'HTI', 'MWI', 'SLE',  # Underdeveloped
]

START_YEAR = 1990
END_YEAR = 2024
