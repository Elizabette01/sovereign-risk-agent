"""
World Bank API Diagnostic Script
This will help us understand exactly what the API is returning

Run this first to diagnose the issue!
"""

import wbgapi as wb
import pandas as pd

print("="*60)
print("WORLD BANK API DIAGNOSTIC")
print("="*60)

# Test 1: Check if wbgapi is installed correctly
print("\n[Test 1] Checking wbgapi installation...")
try:
    print(f"  ✓ wbgapi version: {wb.__version__}")
except:
    print("  ✗ Could not get version")

# Test 2: Try to fetch data for just ONE country and ONE indicator
print("\n[Test 2] Testing single country, single indicator...")
print("  Fetching: USA, GDP growth, years 2020-2023")

try:
    # Method 1: Using wb.data.DataFrame
    print("\n  Method 1: wb.data.DataFrame")
    df = wb.data.DataFrame(
        'NY.GDP.MKTP.KD.ZG',  # GDP growth
        'USA',
        time=range(2020, 2024)
    )
    print(f"    Type: {type(df)}")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Index: {df.index.tolist()}")
    print("\n    Raw DataFrame:")
    print(df)
    print("\n    Data types:")
    print(df.dtypes)
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Try alternative method
print("\n[Test 3] Testing alternative data fetch method...")
try:
    # Method 2: Using wb.data.fetch
    print("\n  Method 2: wb.data.fetch")
    data = wb.data.fetch(
        'NY.GDP.MKTP.KD.ZG',
        'USA',
        time=range(2020, 2024)
    )
    print(f"    Type: {type(data)}")
    print(f"    Content: {data}")
    
    # Convert to DataFrame
    if data:
        records = []
        for item in data:
            print(f"    Item: {item}")
            records.append(item)
        
        df_alt = pd.DataFrame(records)
        print(f"\n    DataFrame shape: {df_alt.shape}")
        print(df_alt)
    
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check available countries
print("\n[Test 4] Checking available country codes...")
try:
    # Get list of countries
    countries = wb.economy.list()
    print(f"    Total countries available: {len(countries)}")
    
    # Check if our countries exist
    test_countries = ['USA', 'JPN', 'DEU', 'GBR']
    for country in test_countries:
        try:
            info = wb.economy.info(country)
            print(f"    ✓ {country}: {info.get('value', 'Unknown')}")
        except:
            print(f"    ✗ {country}: Not found")
            
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 5: Check indicator
print("\n[Test 5] Checking indicator code...")
try:
    indicator_info = wb.series.info('NY.GDP.MKTP.KD.ZG')
    print(f"    ✓ Indicator found: {indicator_info.get('value', 'Unknown')}")
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 6: Try with labels=False
print("\n[Test 6] Testing with labels=False...")
try:
    df = wb.data.DataFrame(
        'NY.GDP.MKTP.KD.ZG',
        'USA',
        time=range(2020, 2024),
        labels=False  # Use codes instead of names
    )
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print("\n    Data:")
    print(df)
    
    # Check if there's actual data
    if df.empty:
        print("    ⚠️  DataFrame is empty!")
    else:
        print(f"    ✓ Got data! Non-null values: {df.notna().sum().sum()}")
        
except Exception as e:
    print(f"    ✗ Error: {e}")

# Test 7: Try different time specification
print("\n[Test 7] Testing different time specifications...")
try:
    # Try with string time period
    print("  Trying time='2020:2023'...")
    df = wb.data.DataFrame(
        'NY.GDP.MKTP.KD.ZG',
        'USA',
        time='2020:2023'
    )
    print(f"    Shape: {df.shape}")
    print(df)
    
except Exception as e:
    print(f"    ✗ Error: {e}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print("\nPlease share the output above so I can fix the download script!")