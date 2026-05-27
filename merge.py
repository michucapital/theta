"""
Merge SPY spot and options data - Match each option trade to nearest spot trade
"""
import pandas as pd
from pathlib import Path
import sys

def merge_spy_data(date_str: str):
    """
    Match each option trade to the nearest spot trade by TIME

    Args:
        date_str: Date in YYYYMMDD format
    """
    print("="*80)
    print(f"MERGING SPY DATA FOR {date_str}")
    print("="*80)

    raw_dir = Path("data/raw")
    merged_dir = Path("data/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Load spot data
    spot_file = raw_dir / f"spy_spot_{date_str}_raw.parquet"
    print(f"\nLoading spot data from: {spot_file}")
    spot_df = pd.read_parquet(spot_file)
    print(f"  ✓ Loaded {len(spot_df):,} spot trades")

    # Load options data
    options_file = raw_dir / f"spy_options_{date_str}_raw.parquet"
    print(f"\nLoading options data from: {options_file}")
    options_df = pd.read_parquet(options_file)
    print(f"  ✓ Loaded {len(options_df):,} options trades")

    # Prepare spot data - rename columns with SPOT_ prefix
    print(f"\nPreparing spot data for merge...")
    spot_for_merge = pd.DataFrame()
    spot_for_merge['SPOT_TIME'] = spot_df['ms_of_day']  # Keep as SPOT_TIME (not TIME)
    spot_for_merge['SPOT_PRICE'] = spot_df['price']
    spot_for_merge['SPOT_SIZE'] = spot_df['size']
    spot_for_merge['SPOT_EXCHANGE'] = spot_df['exchange']
    spot_for_merge['SPOT_SEQUENCE'] = spot_df['sequence']
    spot_for_merge['SPOT_FLAGS'] = spot_df['condition']

    # Sort both by time for merge_asof (options by TIME+SEQUENCE for proper chronological order)
    spot_for_merge = spot_for_merge.sort_values('SPOT_TIME').reset_index(drop=True)
    options_df = options_df.sort_values(['TIME', 'SEQUENCE']).reset_index(drop=True)

    print(f"  ✓ Spot time range: {spot_for_merge['SPOT_TIME'].min()} to {spot_for_merge['SPOT_TIME'].max()}")
    print(f"  ✓ Options time range: {options_df['TIME'].min()} to {options_df['TIME'].max()}")

    # Merge: match each option trade to nearest spot trade
    print(f"\nMatching each option trade to nearest spot trade...")
    merged_df = pd.merge_asof(
        options_df,
        spot_for_merge,
        left_on='TIME',
        right_on='SPOT_TIME',
        direction='nearest'
    )

    print(f"  ✓ Merged {len(merged_df):,} option trades with spot data")

    # Check for any missing matches
    missing_spot = merged_df['SPOT_PRICE'].isna().sum()
    if missing_spot > 0:
        print(f"  ⚠ Warning: {missing_spot} option trades have no spot match")

    # Calculate time difference between option and matched spot
    if 'TIME' in merged_df.columns and 'SPOT_TIME' in merged_df.columns:
        time_diff = (merged_df['TIME'] - merged_df['SPOT_TIME']).abs()
        print(f"\n  Time matching statistics:")
        print(f"    Mean difference: {time_diff.mean():.2f} ms")
        print(f"    Median difference: {time_diff.median():.0f} ms")
        print(f"    Max difference: {time_diff.max()} ms")
        print(f"    Exact matches: {(time_diff == 0).sum():,} ({(time_diff == 0).sum()/len(merged_df)*100:.1f}%)")

    # Sort final merged data by TIME, then SEQUENCE for chronological order
    print(f"\nSorting merged data by TIME, then SEQUENCE...")
    merged_df = merged_df.sort_values(['TIME', 'SEQUENCE']).reset_index(drop=True)

    # Save
    output_file = merged_dir / f"spy_merged_{date_str}.parquet"
    merged_df.to_parquet(output_file, index=False)
    print(f"\n✓ Saved merged data: {output_file}")

    # Summary
    print(f"\n{'='*80}")
    print("MERGE SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows: {len(merged_df):,}")
    print(f"\nColumns in merged data:")
    for col in merged_df.columns:
        print(f"  • {col}")

    print(f"\nSample merged data (first 3 rows):")
    print(merged_df[['TIME', 'SPOT_TIME', 'STRIKE', 'RIGHT', 'PRICE', 'SPOT_PRICE', 'SIZE', 'SPOT_SIZE']].head(3))

    print(f"\nSpot price statistics:")
    print(f"  Min: ${merged_df['SPOT_PRICE'].min():.2f}")
    print(f"  Max: ${merged_df['SPOT_PRICE'].max():.2f}")
    print(f"  Mean: ${merged_df['SPOT_PRICE'].mean():.2f}")

    # Count how many option trades per unique spot time
    unique_spot_times = merged_df.groupby('SPOT_TIME').size()
    print(f"\nOption trades per spot timestamp:")
    print(f"  Mean: {unique_spot_times.mean():.1f}")
    print(f"  Median: {unique_spot_times.median():.0f}")
    print(f"  Max: {unique_spot_times.max()}")

    return merged_df


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python merge.py YYYYMMDD")
        print("Example: python merge.py 20251111")
        sys.exit(1)

    date_str = sys.argv[1]

    try:
        merge_spy_data(date_str)
        print(f"\n✓ Merge complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
