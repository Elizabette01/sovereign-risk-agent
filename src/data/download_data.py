"""
Data Download Script for Sovereign Climate Risk AI Project

This script downloads fiscal, economic, and climate data from various sources.
Run this first to get all the raw data you need.

Author: [Your Name]
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

# Create raw data directory if it doesn't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
# SECTION 2: IMF FISCAL DATA
# ============================================================================

def download_imf_fiscal_data():
    """
    Download fiscal data from IMF.
    
    Note: IMF's official API is complex. For this project, we'll use
    a simplified approach with their bulk download data.
    
    Alternative: You can manually download CSV from:
    https://www.imf.org/external/datamapper/datasets/WEO
    """
    
    print("\n[2/4] Downloading IMF fiscal data...")
    
    # IMF WEO data - we'll create a simplified version
    # In practice, you might download their Excel file manually
    # For now, let's create a function that could work with their data
    
    print("  ℹ️  IMF data requires manual download")
    print("     Visit: https://www.imf.org/en/Publications/WEO/weo-database/2024/October")
    print("     Download: Government debt, Primary balance, Revenue, Expenditure")
    print("     Save to: data/raw/imf_weo_manual.csv")
    
    # Check if manual file exists
    manual_file = RAW_DATA_DIR / 'imf_weo_manual.csv'
    if manual_file.exists():
        print(f"  ✓ Found manually downloaded IMF data")
        imf_data = pd.read_csv(manual_file)
        return imf_data
    else:
        print("  ⚠️  No manual IMF file found - will proceed with World Bank fiscal data")
        return None


# ============================================================================
# SECTION 3: CLIMATE DISASTER DATA (EM-DAT)
# ============================================================================

def download_climate_data():
    """
    Download climate disaster data.
    
    EM-DAT requires registration. For this tutorial, we'll create
    synthetic data based on typical disaster patterns.
    
    Real data: Register at https://www.emdat.be/
    """
    
    print("\n[3/4] Creating climate disaster dataset...")
    
    # Check if manually downloaded EM-DAT file exists
    emdat_file = RAW_DATA_DIR / 'emdat_data.csv'
    
    if emdat_file.exists():
        print(f"  ✓ Found EM-DAT data file")
        climate_data = pd.read_csv(emdat_file)
        return climate_data
    
    # If not, create synthetic climate data for demonstration
    print("  ℹ️  Creating synthetic climate data for demonstration")
    print("     (Replace with real EM-DAT data for actual research)")
    
    np.random.seed(42)
    
    climate_data = []
    
    # Climate vulnerability scores (higher = more vulnerable)
    vulnerability = {
        'USA': 0.3, 'JPN': 0.4, 'DEU': 0.3, 'GBR': 0.3, 'FRA': 0.3,
        'MEX': 0.6, 'TUR': 0.5, 'ZAF': 0.7, 'BRA': 0.6, 'IND': 0.7,
        'BGD': 0.9, 'KEN': 0.8, 'VNM': 0.8, 'PHL': 0.9
    }
    
    for country in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            vuln = vulnerability.get(country, 0.5)
            
            # Probability of disaster increases with time (climate change effect)
            time_factor = 1 + (year - START_YEAR) * 0.02
            
            # Number of disasters (Poisson distribution)
            n_disasters = np.random.poisson(vuln * time_factor * 2)
            
            if n_disasters > 0:
                # Disaster severity (1=mild, 2=moderate, 3=severe)
                severities = np.random.choice([1, 2, 3], size=n_disasters, p=[0.6, 0.3, 0.1])
                
                # Economic damage (% of GDP)
                damages = []
                for severity in severities:
                    if severity == 1:
                        damage = np.random.uniform(0.1, 0.5)
                    elif severity == 2:
                        damage = np.random.uniform(0.5, 2.0)
                    else:
                        damage = np.random.uniform(2.0, 5.0)
                    damages.append(damage)
                
                total_damage = sum(damages)
            else:
                total_damage = 0
                severities = []
            
            climate_data.append({
                'country': country,
                'year': year,
                'n_disasters': n_disasters,
                'total_damage_pct_gdp': total_damage,
                'max_severity': max(severities) if severities else 0,
                'vulnerability_index': vuln
            })
    
    climate_df = pd.DataFrame(climate_data)
    
    # Save synthetic data
    output_path = RAW_DATA_DIR / 'climate_disasters_synthetic.csv'
    climate_df.to_csv(output_path, index=False)
    print(f"  ✓ Created synthetic climate data: {output_path}")
    print(f"    Shape: {climate_df.shape[0]} rows, {climate_df.shape[1]} columns")
    
    return climate_df


# ============================================================================
# SECTION 4: GOVERNANCE AND INSTITUTIONAL DATA
# ============================================================================

def download_governance_data():
    """
    Download governance and institutional quality indicators.
    
    World Bank Worldwide Governance Indicators:
    - CC: Control of Corruption
    - GE: Government Effectiveness
    - PV: Political Stability
    - RQ: Regulatory Quality
    - RL: Rule of Law
    - VA: Voice and Accountability
    """
    
    print("\n[4/4] Downloading governance indicators...")
    
    # For this demo, we'll create a simplified governance score
    # Real data: http://info.worldbank.org/governance/wgi/
    
    print("  ℹ️  Creating simplified governance scores")
    print("     (Replace with WGI data for actual research)")
    
    # Simplified governance scores (0-1, higher is better)
    governance_scores = {
        'USA': 0.85, 'JPN': 0.90, 'DEU': 0.95, 'GBR': 0.90, 'FRA': 0.85,
        'MEX': 0.50, 'TUR': 0.45, 'ZAF': 0.60, 'BRA': 0.55, 'IND': 0.60,
        'BGD': 0.35, 'KEN': 0.40, 'VNM': 0.50, 'PHL': 0.45
    }
    
    gov_data = []
    for country in COUNTRIES:
        for year in range(START_YEAR, END_YEAR + 1):
            # Add small random variation over time
            score = governance_scores.get(country, 0.5)
            score += np.random.normal(0, 0.02)
            score = np.clip(score, 0, 1)
            
            gov_data.append({
                'country': country,
                'year': year,
                'governance_score': score
            })
    
    gov_df = pd.DataFrame(gov_data)
    
    output_path = RAW_DATA_DIR / 'governance_indicators.csv'
    gov_df.to_csv(output_path, index=False)
    print(f"  ✓ Created governance data: {output_path}")
    print(f"    Shape: {gov_df.shape[0]} rows, {gov_df.shape[1]} columns")
    
    return gov_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to download all data."""
    
    start_time = time.time()
    
    try:
        # Download all datasets
        wb_data = download_world_bank_data()
        imf_data = download_imf_fiscal_data()
        climate_data = download_climate_data()
        gov_data = download_governance_data()
        
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