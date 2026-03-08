"""
PRECISE DATA PREPROCESSING PIPELINE
Tailored to your exact dataset structure

Datasets:
1. World Bank WDI (world_bank_data.csv) - already downloaded
2. EM-DAT Disasters (emdat_disaster_data.xlsx) - event-level data
3. INFORM Risk (European_commission-INFORM_Risk_Mid_2025_v071.xlsx)
4. World Bank Income Classification (wbg_income_class_2025_10_07.xlsx)
5. IMF Government Debt (imf-central_gov_debt_data.xls)
6. IMF WEO (world_economic_outlook_imf.xls)

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, COUNTRIES, START_YEAR, END_YEAR

# Create extraction directory
EXTRACTED_DIR = Path('data/extracted')
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DATA EXTRACTION PIPELINE")
print("="*80)
print(f"Reading from: {RAW_DATA_DIR}")
print(f"Exporting to: {EXTRACTED_DIR}")
print("="*80)


# ============================================================================
# COUNTRY CODE MAPPING (Simple automatic approach)
# ============================================================================

def get_country_mapping():
    """
    Simple approach: Just use World Bank's built-in country list.
    Returns a dictionary mapping country names to ISO3 codes.
    """
    
    print("\n[Fetching country mappings from World Bank...]")
    
    try:
        import wbgapi as wb
        
        # Get all economies and create simple mapping
        mapping = {}
        
        for economy in wb.economy.list():
            code = economy['id']
            name = economy['value']
            
            # Add the mapping
            mapping[name] = code
            mapping[code] = code  # Map code to itself
        
        print(f"  ✓ Loaded {len(mapping)} country mappings")
        return mapping
        
    except Exception as e:
        print(f"  ⚠️  Could not fetch from World Bank: {e}")
        print("  Using empty mapping - will keep country names as-is")
        return {}


# Load mapping once at startup
COUNTRY_MAPPING = get_country_mapping()



# ============================================================================
# 1. WORLD BANK WDI DATA (Already Downloaded)
# ============================================================================

def extract_world_bank_data():
    """Extract World Bank macroeconomic data."""
    
    print("\n[1/6] Loading World Bank WDI data...")
    
    input_file = RAW_DATA_DIR / 'world_bank_data.csv'
    
    if not input_file.exists():
        print(f"  ✗ File not found: {input_file}")
        print("  Please run: python src/data/download_data.py")
        return False
    
    df = pd.read_csv(input_file)
    
    # Select relevant columns
    required_cols = [
        'country', 'year', 'gdp_growth', 'debt_to_gdp', 'inflation',
        'exports_gdp', 'imports_gdp', 'gdp_per_capita', 
        'gdp_current_usd', 'population', 'foreign_reserves', 'current_account_balance'
    ]
    
    # Filter to time period only (keep ALL countries)
    # df_clean = df_clean[
    #     (df_clean['year'] >= START_YEAR) & 
    #     (df_clean['year'] <= END_YEAR)
    # ]
    
    # Sort
    df_clean = df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # Export
    output_file = EXTRACTED_DIR / 'world_bank_data.csv'
    df_clean.to_csv(output_file, index=False)
    
    print(f"  ✓ Extracted: {df_clean.shape}")
    print(f"    Countries: {df_clean['country'].nunique()}")
    print(f"    Years: {df_clean['year'].min()}-{df_clean['year'].max()}")
    print(f"    Saved to: {output_file}")
    
    return True


# ============================================================================
# 2. EM-DAT DISASTER DATA (Event-level → Country-year aggregation)
# ============================================================================

def extract_emdat_data():
    """
    Load EM-DAT disaster data and aggregate to country-year level.
    
    Input: Event-level data (one row per disaster)
    Output: Country-year level (one row per country-year)
    """
    
    print("\n[2/6] Loading EM-DAT disaster data...")
    
    input_file = RAW_DATA_DIR / 'emdat_disaster_data.xlsx'
    
    if not input_file.exists():
        print(f"  ⚠️  File not found: {input_file}")
        print("  Skipping - will use synthetic data in preprocessing")
        return False
    
    try:
        # Read Excel file
        df = pd.read_excel(input_file)
        
        print(f"  Raw data: {df.shape}")

        
        # Standardize column names
        df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
        
        # Find relevant columns
        country_col = None
        year_col = None
        damage_type_col = None
        disaster_type_col = None

        for col in df.columns:
            if 'iso' in col and len(col) <= 5:
                country_col = col
            elif 'start' in col and 'year' in col:
                year_col = col
            elif 'year' in col and 'start' not in col and year_col is None:
                year_col = col
            elif 'damage' in col and ('adjusted' in col or 'total' in col):
                damage_type_col = col
            elif 'disaster' in col and 'type' in col:
                disaster_type_col = col
        
        if not country_col or not year_col:
            print(f"  ✗ Could not find required columns")
            print(f"    Available: {list(df.columns)[:10]}")
            return False
        
        print(f"  Detected columns:")
        print(f"    Country: {country_col}")
        print(f"    Year: {year_col}")
        print(f"    Damage: {damage_type_col if damage_type_col else 'Not found'}")
        
        # Select and rename
        df_clean = df[[country_col, year_col]].copy()
        if damage_type_col:
            df_clean[damage_type_col] = df[damage_type_col]
        
        if disaster_type_col:
            df_clean[disaster_type_col] = df[disaster_type_col]
        
        df_clean.columns = ['country', 'year'] + (['damage_usd'] if damage_type_col else []) + (['disaster_type'] if disaster_type_col else [])
        
        # Clean year
        df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce')
        df_clean = df_clean.dropna(subset=['year'])
        df_clean['year'] = df_clean['year'].astype(int)
        
        # Filter to our period
        df_clean = df_clean[
            (df_clean['year'] >= START_YEAR) & 
            (df_clean['year'] <= END_YEAR)
        ]
        
         # Standardize country codes
        df_clean['country'] = df_clean['country'].map(COUNTRY_MAPPING).fillna(df_clean['country'])
        
        # Filter to our countries
        df_clean = df_clean[df_clean['country'].isin(COUNTRIES)]
        
        # Clean damage values
        if 'damage_usd' in df_clean.columns:
            # Remove "no data" strings
            df_clean['damage_usd'] = df_clean['damage_usd'].replace(['no data', 'No data', '--', '...', ''], np.nan)
            df_clean['damage_usd'] = pd.to_numeric(df_clean['damage_usd'], errors='coerce')
            # Convert from thousands to actual USD
            df_clean['damage_usd'] = df_clean['damage_usd'] * 1000
            df_clean['damage_usd'] = df_clean['damage_usd'].fillna(0)
        
        # Aggregate to country-year
        agg_dict = {'disaster_count': ('year', 'size')}
        if 'damage_usd' in df_clean.columns:
            agg_dict['total_damage_usd'] = ('damage_usd', 'sum')
        
        df_agg = df_clean.groupby(['country', 'year']).agg(**agg_dict).reset_index()
        
        # Sort
        df_agg = df_agg.sort_values(['country', 'year']).reset_index(drop=True)
        
        # Export
        output_file = EXTRACTED_DIR / 'emdat_disasters.csv'
        df_agg.to_csv(output_file, index=False)
        
        print(f"  ✓ Extracted: {df_agg.shape}")
        print(f"    Saved to: {output_file}")
        print(f"    Total disasters: {df_agg['disaster_count'].sum()}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

# ============================================================================
# 3. INFORM CLIMATE RISK DATA
# ============================================================================

def extract_inform_risk_data():
    """Extract INFORM climate risk indicators."""
    
    print("\n[3/6] Loading INFORM climate risk data...")
    
    input_file = RAW_DATA_DIR / 'European_commission-INFORM_Risk_Mid_2025_v071.xlsx'
    
    if not input_file.exists():
        print(f"  ⚠️  File not found: {input_file}")
        print("  Skipping - will use synthetic data in preprocessing")
        return False()
    
    try:
        # Try to find the data sheet
        excel_file = pd.ExcelFile(input_file)
        print(f"  Available sheets: {excel_file.sheet_names}")
        
        # Common sheet names for INFORM data
        data_sheet = None
        for sheet in excel_file.sheet_names:
            if any(word in sheet.lower() for word in ['data', 'inform', 'risk', 'country']):
                data_sheet = sheet
                break
        
        if not data_sheet:
            data_sheet = excel_file.sheet_names[1]  # Use first sheet
        
        print(f"  Reading sheet: {data_sheet}")
        df = pd.read_excel(input_file, sheet_name=data_sheet)
        
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}...")
        
        # Standardize column names
        df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
        
        # Find relevant columns
        country_col = None
        iso_col = None
        risk_col = None
        risk_class_col = None
        hazard_col = None
        vuln_col = None
        
        for col in df.columns:
            if 'country' in col:
                country_col = col
            elif 'iso' in col and len(col) <= 5:
                iso_col = col
            elif 'inform' in col and 'risk' in col:
                risk_col = col
            elif 'class' in col:
                risk_class_col = col
            elif 'hazard' in col:
                hazard_col = col
            elif 'vulnerab' in col:
                vuln_col = col
        
        # if not country_col:
        #     print("  ⚠️  Could not find country column")
        #     return create_synthetic_risk_data()
        
        # Select relevant columns
        keep_cols = [country_col]
        if iso_col:
            keep_cols.append(iso_col)
        if risk_col:
            keep_cols.append(risk_col)
        if hazard_col:
            keep_cols.append(hazard_col)
        if vuln_col:
            keep_cols.append(vuln_col)
        if risk_class_col:
            keep_cols.append(risk_class_col)
        
        df_clean = df[keep_cols].copy()
        df_clean.rename(columns={country_col: 'country'}, inplace=True)
    
        # Standardize country codes
        df_clean['country'] = df_clean['country'].map(COUNTRY_MAPPING).fillna(df_clean['country'])
        
        # Export
        output_file = EXTRACTED_DIR / 'inform_risk.csv'
        df_clean.to_csv(output_file, index=False)
        
        print(f"  ✓ Extracted: {df_clean.shape}")
        print(f"    Countries: {df_clean['country'].nunique()}")
        print(f"    Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading INFORM: {e}")
        return False


# ============================================================================
# 4. WORLD BANK INCOME CLASSIFICATION
# ============================================================================

def extract_income_classification():
    """Extract World Bank income group classification."""
    
    print("\n[4/6] Loading income classification...")
    
    input_file = RAW_DATA_DIR / 'wbg_income_class_2025_10_07.xlsx'
    
    if not input_file.exists():
        print(f"  ⚠️  File not found: {input_file}")
        print("  Skipping - will use default classification in preprocessing")
        return False
    try:
        df = pd.read_excel(input_file)
        
        print(f"  Shape: {df.shape}")
        
        # Standardize column names
        df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
        
        # Find relevant columns
        country_col = None
        code_col = None
        income_col = None
        
        for col in df.columns:
            if 'economy' in col:
                country_col = col
            if 'code' in col:
                code_col = col
            if 'income' in col and 'group' in col:
                income_col = col
        
        if not code_col or not income_col:
            print("  ⚠️  Could not find required columns")
            return False
        
        # Select and rename
        df_clean = df[[code_col, income_col]].copy()
        df_clean.columns = ['country', 'income_group']
        
        # Map to economy type
        economy_type_map = {
            'High income': 'Advanced',
            'Upper middle income': 'Emerging',
            'Lower middle income': 'Developing',
            'Low income': 'Least developed'
        }
        
        df_clean['economy_type'] = df_clean['income_group'].map(economy_type_map)
        
        # Export
        output_file = EXTRACTED_DIR / 'income_classification.csv'
        df_clean.to_csv(output_file, index=False)
        
        print(f"  ✓ Extracted: {df_clean.shape}")
        print(f"    Countries: {df_clean['country'].nunique()}")
        print(f"    Saved to: {output_file}")
        print(f"    Income groups: {df_clean['income_group'].unique().tolist()}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading classification: {e}")
        return False

# ============================================================================
# 5. IMF GOVERNMENT DEBT DATA
# ============================================================================

def extract_imf_debt_data():
    """
    Extract IMF government debt data (supplement World Bank).
    
    Input format:
    - Column 1: "Central Government Debt (Percent of GDP)" - contains country names
    - Columns 2+: Years (1950, 1951, ..., 2024)
    - Missing data: "no data" string
    
    Output format:
    - country: Country name
    - year: Year
    - imf_debt_gdp: Debt as % of GDP
    """
    
    print("\n[5/6] Loading IMF debt data...")
    
    input_file = RAW_DATA_DIR / 'imf-central_gov_debt_data.xlsx'
    
    if not input_file.exists():
        print(f"  ℹ️  File not found: {input_file}")
        print("  Will use World Bank debt data only")
        return None
    
    try:
        # Read Excel file
        df = pd.read_excel(input_file)
        
        print(f"  Shape: {df.shape}")
        
        country_col = df.columns[0]
        df.rename(columns={country_col: 'country'}, inplace=True)
        
        # Get year columns (all columns except first)
        year_columns = [col for col in df.columns if col != 'country']
        
        # Convert to long format
        df_long = df.melt(
            id_vars=['country'],
            value_vars=year_columns,
            var_name='year',
            value_name='imf_debt_gdp'
        )
        print(f"  After melt: {df_long.shape}")
         
        # Filter to our time period
        df_long = df_long[
            (df_long['year'] >= START_YEAR) & 
            (df_long['year'] <= END_YEAR)
        ]
        
        # Clean year column - convert to integer
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
        df_long = df_long.dropna(subset=['year'])
        df_long['year'] = df_long['year'].astype(int)
        
        # Clean debt values
        df_long['imf_debt_gdp'] = df_long['imf_debt_gdp'].replace(
            ['no data', 'No data', 'NO DATA', '--', '...', ''], 
            np.nan
        )
        # Convert to numeric
        df_long['imf_debt_gdp'] = pd.to_numeric(df_long['imf_debt_gdp'], errors='coerce')
       
        # Standardize country names to ISO codes (but don't filter)
        df_long['country'] = df_long['country'].map(COUNTRY_MAPPING).fillna(df_long['country'])
        
        # Drop rows where debt value is NaN
        df_long = df_long.dropna(subset=['imf_debt_gdp'])
        
        # Sort
        df_long = df_long.sort_values(['country', 'year']).reset_index(drop=True)

        # Select final columns
        df_clean = df_long[['country', 'year', 'imf_debt_gdp']].copy()
        
        # Export
        output_file = EXTRACTED_DIR / 'imf_debt.csv'
        df_clean.to_csv(output_file, index=False)
        
        print(f"  ✓ Extracted: {df_clean.shape}")
        print(f"    Saved to: {output_file}")
        print(f"    Countries: {df_clean['country'].unique().tolist()}")
        print(f"    Year range: {df_clean['year'].min()}-{df_clean['year'].max()}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading IMF debt: {e}")
        return None



# ============================================================================
# 6. IMF WORLD ECONOMIC OUTLOOK DATA
# ============================================================================

# def extract_imf_weo_data():
#     """Load IMF WEO data for validation."""
    
#     print("\n[6/6] Loading IMF WEO data...")
    
#     file_path = RAW_DATA_DIR / 'world_economic_outlook_imf.xls'
    
#     if not file_path.exists():
#         print(f"  ℹ️  File not found: {file_path}")
#         print("  Will use World Bank data only")
#         return None
    
#     try:
#         df = pd.read_excel(file_path)
        
        # Process IMF WEO format
        # This will need customization based on actual file structure
        
    #     return None  # Placeholder
        
    # except Exception as e:
    #     print(f"  ⚠️  Error: {e}")
    #     return None


def main():
    """Run all extraction tasks."""
    
    print("\nStarting data extraction...\n")
    
    results = {}
    
    results['world_bank'] = extract_world_bank_data()
    results['emdat'] = extract_emdat_data()
    results['inform'] = extract_inform_risk_data()
    results['income'] = extract_income_classification()
    results['imf_debt'] = extract_imf_debt_data()
    # results['imf_weo'] = extract_imf_weo_data()
    
    # Summary
    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print("="*80)
    
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {dataset}")
    
    successful = sum(1 for v in results.values() if v)
    print(f"\nExtracted {successful}/{len(results)} datasets")
    
    print(f"\nExtracted files saved to: {EXTRACTED_DIR}")
    
    print("\nNext step: Run preprocessing pipeline")
    print("  python src/data/preprocess_pipeline.py")
    
    return True


if __name__ == "__main__":
    main()