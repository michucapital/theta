"""
Download SPY spot trades for full US cash session (9:30 AM - 4:00 PM)
with filtering applied.

CHANGE DATE ON LINE 11 ONLY
"""
from downloader import ThetaDownloader
from processor import DataProcessor
from datetime import datetime

# ============================================================================
# CONFIGURATION - CHANGE THIS DATE ONLY
# ============================================================================
TARGET_DATE = "20260513"  # Format: YYYYMMDD
# ============================================================================


def main():
    """Download full US cash session of SPY spot data with filtering"""

    # Initialize
    downloader = ThetaDownloader()
    processor = DataProcessor()

    print("\n" + "="*80)
    print("SPY SPOT DOWNLOAD - FULL US CASH SESSION")
    print("="*80)
    print(f"Target Date: {TARGET_DATE}")
    print("="*80)

    # Download spot data
    print("\nStep 1: Downloading SPY spot trades...")
    spot_df = downloader.download_spot_trades(TARGET_DATE)
    print(f"✓ Downloaded {len(spot_df):,} total rows")

    # Filter for US cash market session
    # 9:30 AM = 34200000 ms
    # 4:00 PM = 57600000 ms
    market_open_ms = 34200000
    market_close_ms = 57600000

    print(f"\nStep 2: Filtering for US cash session (9:30 AM - 4:00 PM)...")
    session_df = spot_df[
        (spot_df['ms_of_day'] >= market_open_ms) &
        (spot_df['ms_of_day'] <= market_close_ms)
    ]
    print(f"✓ Filtered to {len(session_df):,} rows in cash session")

    # Apply condition filters using processor
    print("\nStep 3: Applying condition filters...")
    print("-" * 60)
    print("Filters (ALL must match):")
    print("  - ext_condition1 = 255")
    print("  - ext_condition2 = 255")
    print("  - ext_condition3 = 255")
    print("  - ext_condition4 = 115")
    print("  - condition = 115")
    print("  - exchange != 57")
    print("-" * 60)

    # Processor filters and saves automatically to data/raw/spy_spot_YYYYMMDD_raw.parquet
    filtered_df = processor.filter_spot_trades(session_df, TARGET_DATE)

    # Show statistics
    print("\n" + "="*80)
    print("SESSION STATISTICS")
    print("="*80)
    print(f"Original rows (full day):       {len(spot_df):,}")
    print(f"After time filter (9:30-4:00):  {len(session_df):,}")
    print(f"After condition filters:        {len(filtered_df):,}")
    print(f"Total reduction:                {(1 - len(filtered_df)/len(spot_df))*100:.2f}%")

    print(f"\nTime range: {filtered_df['ms_of_day'].min():,} - {filtered_df['ms_of_day'].max():,} ms")

    # Verify filters were applied correctly
    print("\n" + "="*80)
    print("FILTER VERIFICATION")
    print("="*80)

    filter_checks = {
        'ext_condition1 == 255': (filtered_df['ext_condition1'] == 255).all(),
        'ext_condition2 == 255': (filtered_df['ext_condition2'] == 255).all(),
        'ext_condition3 == 255': (filtered_df['ext_condition3'] == 255).all(),
        'ext_condition4 == 115': (filtered_df['ext_condition4'] == 115).all(),
        'condition == 115': (filtered_df['condition'] == 115).all(),
        'exchange != 57': (filtered_df['exchange'] != 57).all(),
    }

    for filter_name, result in filter_checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {filter_name:25s} {status}")

    print("\n" + "="*80)
    print("SUCCESS! Data downloaded, filtered, and saved")
    print("="*80)
    print(f"Output: data/raw/spy_spot_{TARGET_DATE}_raw.parquet")
    print(f"Rows:   {len(filtered_df):,}")
    print("="*80)


if __name__ == "__main__":
    main()
