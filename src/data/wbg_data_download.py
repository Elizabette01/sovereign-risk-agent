"""
Data Download Script 

This script downloads fiscal, economic, and climate data from various sources.
Sources:
- World Bank API for economic indicators
- IMF WEO for fiscal data (csv imported)
- EM-DAT for climate disaster data (csv imported from:https://public.emdat.be/data)
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
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, COUNTRIES, START_YEAR, END_YEAR


print("="*60)
print("SOVEREIGN CLIMATE RISK AI - DATA DOWNLOAD")
print("="*60)
print(f"Downloading data for {len(COUNTRIES)} countries")
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
    }
    
    # Convert our country codes to World Bank codes
    # World Bank uses ISO 3-letter codes (USA, JPN, DEU, etc.)
    wb_countries = COUNTRIES  # Our config already uses correct codes
    
    all_data = []
    
    for indicator_code, indicator_name in indicators.items():
        print(f"  Fetching {indicator_name}...")
        
        try:
            # Fetch data from World Bank API
            # wb.data.DataFrame fetches data for multiple countries at once
            df = wb.data.DataFrame(
                indicator_code,
                wb_countries,
                time=range(START_YEAR, END_YEAR + 1),
                labels=True  # Use country names instead of codes
            )
            
            # Reshape from wide to long format
            df = df.reset_index()
            df = df.melt(
                id_vars=['economy'], 
                var_name='year', 
                value_name=indicator_name
            )
            df.rename(columns={'economy': 'country'}, inplace=True)
            
            all_data.append(df)
            
            time.sleep(0.5)  # Be polite to the API
            
        except Exception as e:
            print(f"    ⚠️  Warning: Could not fetch {indicator_name}: {e}")
            continue
    
    # Merge all indicators together
    if all_data:
        wb_data = all_data[0]
        for df in all_data[1:]:
            wb_data = wb_data.merge(df, on=['country', 'year'], how='outer')
        
        # Convert year to integer
        wb_data['year'] = wb_data['year'].astype(int)
        
        # Save to CSV
        output_path = RAW_DATA_DIR / 'world_bank_data.csv'
        wb_data.to_csv(output_path, index=False)
        print(f"  ✓ Saved World Bank data: {output_path}")
        print(f"    Shape: {wb_data.shape[0]} rows, {wb_data.shape[1]} columns")
        
        return wb_data
    else:
        print("  ✗ Failed to download World Bank data")
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
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()