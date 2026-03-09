"""
STEP 1: LOAD AND MERGE DATASETS

This script:
1. Loads all extracted CSV files from data/extracted/
2. Merges them on [country, year]
3. Saves merged dataset to data/interim/

Input: data/extracted/*.csv
Output: data/interim/merged_data.csv

"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import START_YEAR, END_YEAR

# Directories
EXTRACTED_DIR = Path('data/extracted')
INTERIM_DIR = Path('data/interim')
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STEP 1: LOAD AND MERGE DATASETS")
print("="*80)
print(f"Reading from: {EXTRACTED_DIR}")
print(f"Saving to: {INTERIM_DIR}")
print("="*80)


# ============================================================================
# 1. LOAD ALL EXTRACTED DATA
# ============================================================================

def load_extracted_data():
    """Load all extracted CSV files."""
    
    print("\n" + "="*80)
    print("LOADING EXTRACTED DATA")
    print("="*80)
    
    datasets = {}
    
    # World Bank data (required)
    print("\n[1/5] Loading World Bank data...")
    wb_file = EXTRACTED_DIR / 'world_bank_data.csv'
    if wb_file.exists():
        datasets['world_bank'] = pd.read_csv(wb_file)
        print(f"    Countries: {datasets['world_bank']['country'].nunique()}")
        print(f"    Years: {datasets['world_bank']['year'].min()}-{datasets['world_bank']['year'].max()}")
    else:
        print(f"  ✗ File not found: {wb_file}")
        print("  ERROR: World Bank data is required!")
        return None
    
    # EM-DAT disasters
    print("\n[2/5] Loading EM-DAT disaster data...")
    emdat_file = EXTRACTED_DIR / 'emdat_disasters.csv'
    if emdat_file.exists():
        datasets['emdat'] = pd.read_csv(emdat_file)
        print(f"  ✓ Loaded: {datasets['emdat'].shape}")
        print(f"    Total disasters: {datasets['emdat']['disaster_count'].sum()}")
    else:
        print(f"  ⚠️  File not found: {emdat_file}")
        datasets['emdat'] = None
    
    # INFORM risk
    print("\n[3/5] Loading INFORM climate risk data...")
    inform_file = EXTRACTED_DIR / 'inform_risk.csv'
    if inform_file.exists():
        datasets['inform'] = pd.read_csv(inform_file)
        print(f"  ✓ Loaded: {datasets['inform'].shape}")
        print(f"    Countries: {datasets['inform']['country'].nunique()}")
    else:
        print(f"  ⚠️  File not found: {inform_file}")
        datasets['inform'] = None
    
    # Income classification
    print("\n[4/5] Loading income classification...")
    income_file = EXTRACTED_DIR / 'income_classification.csv'
    if income_file.exists():
        datasets['income'] = pd.read_csv(income_file)
        print(f"  ✓ Loaded: {datasets['income'].shape}")
        print(f"    Income groups: {datasets['income']['income_group'].unique().tolist()}")
    else:
        print(f"  ⚠️  File not found: {income_file}")
        datasets['income'] = None
    
    # IMF debt
    print("\n[5/5] Loading IMF debt data...")
    imf_debt_file = EXTRACTED_DIR / 'imf_debt.csv'
    if imf_debt_file.exists():
        datasets['imf_debt'] = pd.read_csv(imf_debt_file)
        print(f"  ✓ Loaded: {datasets['imf_debt'].shape}")
        print(f"    Countries: {datasets['imf_debt']['country'].nunique()}")
    else:
        print(f"  ℹ️  File not found: {imf_debt_file}")
        print("     Will use World Bank debt data only")
        datasets['imf_debt'] = None
    
    return datasets


# ============================================================================
# 2. MERGE ALL DATASETS
# ============================================================================

def merge_datasets(datasets):
    """Merge all datasets on [country, year]."""
    
    print("\n" + "="*80)
    print("MERGING DATASETS")
    print("="*80)
    
    # Start with World Bank (most complete)
    master = datasets['world_bank'].copy()
    print(f"\nBase dataset (World Bank): {master.shape}")
    
    # Track merge statistics
    merge_stats = {
        'base_records': len(master),
        'base_countries': master['country'].nunique()
    }
    
    # === MERGE 1: IMF Debt (to fill gaps in WB debt) ===
    if datasets['imf_debt'] is not None:
        print("\n[Merge 1/4] Adding IMF debt data...")
        
        initial_shape = master.shape
        master = master.merge(
            datasets['imf_debt'],
            on=['country', 'year'],
            how='left'
        )
        
        # Use IMF debt to fill missing World Bank debt
        if 'debt_to_gdp' in master.columns and 'imf_debt_gdp' in master.columns:
            missing_before = master['debt_to_gdp'].isna().sum()
            master['debt_to_gdp'] = master['debt_to_gdp'].fillna(master['imf_debt_gdp'])
            missing_after = master['debt_to_gdp'].isna().sum()
            filled = missing_before - missing_after
            
            print(f"  Shape before: {initial_shape}")
            print(f"  Shape after: {master.shape}")
            print(f"  ✓ Filled {filled} missing debt values using IMF data")
            
            # Drop IMF column (now merged)
            master = master.drop(columns=['imf_debt_gdp'])
        
        merge_stats['imf_debt_filled'] = filled if 'filled' in locals() else 0
    else:
        print("\n[Merge 1/4] Skipping IMF debt (not available)")
    
    # === MERGE 2: EM-DAT Disasters ===
    if datasets['emdat'] is not None:
        print("\n[Merge 2/4] Adding disaster data...")
        
        initial_shape = master.shape
        master = master.merge(
            datasets['emdat'],
            on=['country', 'year'],
            how='left'
        )
        
        # Fill missing disaster counts with 0 (no disaster)
        if 'disaster_count' in master.columns:
            master['disaster_count'] = master['disaster_count'].fillna(0)
        if 'total_damage_usd' in master.columns:
            master['total_damage_usd'] = master['total_damage_usd'].fillna(0)
        
        print(f"  Shape before: {initial_shape}")
        print(f"  Shape after: {master.shape}")
        print(f"  ✓ Added disaster data")
        
        # Calculate damage as % of GDP
        if 'total_damage_usd' in master.columns and 'gdp_current_usd' in master.columns:
            master['damage_pct_gdp'] = (
                master['total_damage_usd'] / master['gdp_current_usd']
            ) * 100
            master['damage_pct_gdp'] = master['damage_pct_gdp'].fillna(0)
            print(f"  ✓ Created damage_pct_gdp")
        
        merge_stats['disasters_added'] = True
    else:
        print("\n[Merge 2/4] Skipping disaster data (not available)")
        merge_stats['disasters_added'] = False
    
    # === MERGE 3: INFORM Climate Risk ===
    if datasets['inform'] is not None:
        print("\n[Merge 3/4] Adding climate risk scores...")
        
        initial_shape = master.shape
        # Note: INFORM is country-level only (no year)
        master = master.merge(
            datasets['inform'],
            on='country',
            how='left'
        )
        
        print(f"  Shape before: {initial_shape}")
        print(f"  Shape after: {master.shape}")
        print(f"  ✓ Added climate risk scores")
        
        merge_stats['climate_risk_added'] = True
    else:
        print("\n[Merge 3/4] Skipping climate risk (not available)")
        merge_stats['climate_risk_added'] = False
    
    # === MERGE 4: Income Classification ===
    if datasets['income'] is not None:
        print("\n[Merge 4/4] Adding income classification...")
        
        initial_shape = master.shape
        master = master.merge(
            datasets['income'],
            on='country',
            how='left'
        )
        
        # Create economy type from income group
        if 'income_group' in master.columns:
            economy_map = {
                'High income': 'Advanced',
                'Upper middle income': 'Emerging',
                'Lower middle income': 'Developing',
                'Low income': 'Least developed'
            }
            master['economy_type'] = master['income_group'].map(economy_map)
            print(f"  ✓ Created economy_type variable")
        
        print(f"  Shape before: {initial_shape}")
        print(f"  Shape after: {master.shape}")
        print(f"  ✓ Added income classification")
        
        merge_stats['income_added'] = True
    else:
        print("\n[Merge 4/4] Skipping income classification (not available)")
        merge_stats['income_added'] = False
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("MERGE SUMMARY")
    print("="*80)
    
    print(f"\nFinal merged dataset:")
    print(f"  Records: {len(master)}")
    print(f"  Countries: {master['country'].nunique()}")
    print(f"  Years: {master['year'].min()}-{master['year'].max()}")
    print(f"  Features: {len(master.columns)}")
    
    print(f"\nColumns in merged dataset:")
    for i, col in enumerate(master.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check data coverage
    print(f"\nData coverage by country (top 10):")
    coverage = master.groupby('country').size().sort_values(ascending=False).head(10)
    for country, count in coverage.items():
        expected = END_YEAR - START_YEAR + 1
        pct = (count / expected) * 100
        print(f"  {country}: {count}/{expected} years ({pct:.0f}%)")
    
    return master, merge_stats


# ============================================================================
# 3. SAVE MERGED DATA
# ============================================================================

def save_merged_data(df, stats):
    """Save merged dataset and statistics."""
    
    print("\n" + "="*80)
    print("SAVING MERGED DATA")
    print("="*80)
    
    # Save merged dataset
    output_file = INTERIM_DIR / 'merged_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved merged data: {output_file}")
    print(f"  Shape: {df.shape}")
    
    # Save merge statistics
    stats_df = pd.DataFrame([stats])
    stats_file = INTERIM_DIR / 'merge_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"✓ Saved merge statistics: {stats_file}")
    
    print(f"\nAll files saved to: {INTERIM_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute merge pipeline."""
    
    try:
        # Load data
        datasets = load_extracted_data()
        
        if datasets is None:
            print("\n✗ Cannot proceed without World Bank data")
            return False
        
        # Merge datasets
        merged_data, merge_stats = merge_datasets(datasets)
        
        # Save
        save_merged_data(merged_data, merge_stats)
        
        print("\n" + "="*80)
        print("✓✓✓ STEP 1 COMPLETE! ✓✓✓")
        print("="*80)
        print("\nNext step: Feature engineering and quality checks")
        print("  python src/data/02_feature_engineering.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()