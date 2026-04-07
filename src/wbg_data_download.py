"""
Data Download Script 

This script downloads economic data using the World Bank API 

"""

import pandas as pd
import numpy as np
import wbgapi as wb
import requests
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import project configuration
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR, COUNTRIES, START_YEAR, END_YEAR, WB_INDICATORS


print("="*60)
print("SOVEREIGN CLIMATE RISK AI - DATA DOWNLOAD")
print("="*60)
print(f"Downloading data for all countries")
print(f"Time period: {START_YEAR}-{END_YEAR}")
print("="*60)


# ============================================================================
# SECTION 1: WORLD BANK DATA
# ============================================================================

def download_world_bank_data():
    """
    Download economic and fiscal indicators from World Bank.
    
    World Bank uses indicator codes. Here are the key ones we need:
    - NY.GDP.MKTP.CD: GDP (current US$)
    - NY.GDP.MKTP.KD.ZG: GDP growth (annual %)
    - GC.DOD.TOTL.GD.ZS: Central government debt (% of GDP)
    - GC.BAL.CASH.GD.ZS: Cash surplus/deficit (% of GDP)
    - FP.CPI.TOTL.ZG: Inflation (consumer prices, annual %)
    - NE.EXP.GNFS.ZS: Exports (% of GDP)
    - NE.IMP.GNFS.ZS: Imports (% of GDP)
    """
    
    print("\n[1/4] Downloading World Bank data...")
    
    # Define the indicators we want
    indicators = {
        'NY.GDP.MKTP.CD': 'gdp_current_usd',
        'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
        'GC.DOD.TOTL.GD.ZS': 'debt_to_gdp',
        'GC.BAL.CASH.GD.ZS': 'fiscal_balance',
        'FP.CPI.TOTL.ZG': 'inflation',
        'NE.EXP.GNFS.ZS': 'exports_gdp',
        'NE.IMP.GNFS.ZS': 'imports_gdp',
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'SP.POP.TOTL': 'population',
        'FI.RES.TOTL.CD': 'foreign_reserves',
        'BN.CAB.XOKA.GD.ZS': 'current_account_balance'

    }
    
    # Convert our country codes to World Bank codes
    # World Bank uses ISO 3-letter codes (USA, JPN, DEU, etc.)
    wb_countries = 'all'     
    all_data = []
    
    for indicator_code, indicator_name in indicators.items():
        print(f"  Fetching {indicator_name}...")
        
        try:
            # Method that works: Use wb.data.fetch (returns generator)
            data_generator = wb.data.fetch(
                indicator_code,
                wb_countries,
                time=range(START_YEAR, END_YEAR + 1)
            )
            
            # Convert generator to list of records
            records = []
            for item in data_generator:
                # Each item is a dict with: value, series, economy, aggregate, time
                records.append({
                    'country': item.get('economy', ''),
                    'year': item.get('time', ''),
                    indicator_name: item.get('value', None)
                })
            
            if not records:
                print(f"    ⚠️  No data returned for {indicator_name}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Clean year column (remove 'YR' prefix if present)
            df['year'] = df['year'].astype(str).str.replace('YR', '', regex=False)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            
            # Drop rows with invalid data
            df = df.dropna(subset=['year', 'country'])
            df['year'] = df['year'].astype(int)
            
            print(f"    ✓ Downloaded: {len(df)} records")
            print(f"    Sample values: {df[indicator_name].dropna().head(3).tolist()}")
            
            all_data.append(df)
            time.sleep(0.5)  
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            # Try fallback method with DataFrame
            try:
                print(f"    Trying alternative method...")
                df = wb.data.DataFrame(
                    indicator_code,
                    wb_countries,
                    time=range(START_YEAR, END_YEAR + 1),
                    labels=False  # Use codes instead of names
                )
                
                # Reset index and melt to long format
                df = df.reset_index()
                
                # Find year columns (start with 'YR')
                year_cols = [col for col in df.columns if str(col).startswith('YR')]
                
                if year_cols:
                    # Melt to long format
                    df_long = df.melt(
                        id_vars=['economy'],
                        value_vars=year_cols,
                        var_name='year',
                        value_name=indicator_name
                    )
                    
                    df_long.rename(columns={'economy': 'country'}, inplace=True)
                    df_long['year'] = df_long['year'].str.replace('YR', '', regex=False)
                    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
                    df_long = df_long.dropna(subset=['year'])
                    df_long['year'] = df_long['year'].astype(int)
                    
                    print(f"    ✓ Alternative method worked: {len(df_long)} records")
                    all_data.append(df_long)
                else:
                    print(f"    ✗ Alternative method also failed")
                    
            except Exception as e2:
                print(f"    ✗ Both methods failed: {e2}")
                continue
    
    # Merge all indicators
    if all_data:
        print(f"\n  Merging {len(all_data)} indicators...")
        wb_data = all_data[0]
        
        for i, df in enumerate(all_data[1:], 1):
            print(f"    Merging {i+1}/{len(all_data)}...")
            wb_data = wb_data.merge(df, on=['country', 'year'], how='outer')
        
        # Sort and clean
        wb_data = wb_data.sort_values(['country', 'year']).reset_index(drop=True)
        
        # Save
        output_path = RAW_DATA_DIR / 'world_bank_data.csv'
        wb_data.to_csv(output_path, index=False)
        
        print(f"\n  ✓ Saved World Bank data: {output_path}")
        print(f"    Shape: {wb_data.shape[0]} rows, {wb_data.shape[1]} columns")
        print(f"    Countries: {wb_data['country'].nunique()}")
        print(f"    Year range: {wb_data['year'].min()}-{wb_data['year'].max()}")
        
        # Show data quality
        print(f"\n  Data Quality:")
        for col in wb_data.columns:
            if col not in ['country', 'year']:
                non_null = wb_data[col].notna().sum()
                pct = (non_null / len(wb_data)) * 100
                print(f"    {col}: {pct:.1f}% complete")
        
        return wb_data
    else:
        print("\n  ✗ No data downloaded successfully")
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to download all data."""
    
    start_time = time.time()
    
    try:
        # Download all datasets
        wb_data = download_world_bank_data()
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE!")
        print("="*60)
        
        # Summary
        print("\nFiles created in data/raw/:")
        for file in RAW_DATA_DIR.glob("*.csv"):
            size_kb = file.stat().st_size / 1024
            print(f"  • {file.name} ({size_kb:.1f} KB)")
        
        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.1f} seconds")
        
        print("\n✓ Ready for preprocessing!")
        print("  Next step: Run src/data/preprocess.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        return False


if __name__ == "__main__":
    main()