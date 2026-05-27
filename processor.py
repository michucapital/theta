"""
Data processor for SPY spot and options trades
"""
from config import CONFIG
import pandas as pd

class DataProcessor:
    """Filters SPY spot and options data based on requirements"""

    def __init__(self):
        # Ensure directories exist
        CONFIG.raw_data_path.mkdir(parents=True, exist_ok=True)

    def filter_spot_trades(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """
        Filter SPY spot trades based on condition requirements.

        Filters applied (ALL must match):
        - ext_condition1 = 255
        - ext_condition2 = 255
        - ext_condition3 = 255
        - ext_condition4 = 115
        - condition = 115
        - exchange != 57 (exclude exchange 57)

        Args:
            df: Raw spot trades DataFrame
            date_str: Date string for file naming (YYYYMMDD)

        Returns:
            Filtered DataFrame
        """
        print("=" * 60)
        print(f"Filtering SPY Spot Trades for {date_str}")
        print("=" * 60)

        original_count = len(df)
        print(f"Original rows: {original_count:,}")

        # Apply filters one by one showing impact of each
        df_filtered = df.copy()

        # Filter 1: ext_condition1 = 255
        df_filtered = df_filtered[df_filtered['ext_condition1'] == CONFIG.spot_ext_condition1]
        print(f"After ext_condition1 == {CONFIG.spot_ext_condition1}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        # Filter 2: ext_condition2 = 255
        df_filtered = df_filtered[df_filtered['ext_condition2'] == CONFIG.spot_ext_condition2]
        print(f"After ext_condition2 == {CONFIG.spot_ext_condition2}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        # Filter 3: ext_condition3 = 255
        df_filtered = df_filtered[df_filtered['ext_condition3'] == CONFIG.spot_ext_condition3]
        print(f"After ext_condition3 == {CONFIG.spot_ext_condition3}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        # Filter 4: ext_condition4 = 115
        df_filtered = df_filtered[df_filtered['ext_condition4'] == CONFIG.spot_ext_condition4]
        print(f"After ext_condition4 == {CONFIG.spot_ext_condition4}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        # Filter 5: condition = 115
        df_filtered = df_filtered[df_filtered['condition'] == CONFIG.spot_condition]
        print(f"After condition == {CONFIG.spot_condition}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        # Filter 6: exchange != 57
        df_filtered = df_filtered[df_filtered['exchange'] != CONFIG.spot_excluded_exchange]
        print(f"After excluding exchange {CONFIG.spot_excluded_exchange}: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")

        rows_removed = original_count - len(df_filtered)
        print(f"\n✓ Filtering complete!")
        print(f"  Kept: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")
        print(f"  Removed: {rows_removed:,} rows ({rows_removed/original_count*100:.1f}%)")

        # Save filtered data to data/raw/
        output_path = CONFIG.raw_data_path / f"spy_spot_{date_str}_raw.parquet"
        df_filtered.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")

        return df_filtered

    def filter_options_trades(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        """
        Filter SPY options trades based on FLAGS requirements.

        Filters applied:
        - FLAGS must be one of: 18, 125, 95

        Args:
            df: Raw options trades DataFrame
            date_str: Date string for file naming (YYYYMMDD)

        Returns:
            Filtered DataFrame
        """
        print("=" * 60)
        print(f"Filtering SPY Options Trades for {date_str}")
        print("=" * 60)

        original_count = len(df)
        print(f"Original rows: {original_count:,}")

        # Check if flags column exists (might be named differently)
        flag_column = None
        for col in df.columns:
            if 'flag' in col.lower() or col == 'condition':
                flag_column = col
                break

        if flag_column is None:
            print("WARNING: No 'flags' or 'condition' column found in options data!")
            print(f"Available columns: {list(df.columns)}")
            print("Returning unfiltered data for inspection.")
            return df

        print(f"Using column: '{flag_column}'")
        print(f"Valid flags: {CONFIG.option_valid_flags}")

        # Show distribution before filtering
        print(f"\n{flag_column} distribution (top 10):")
        print(df[flag_column].value_counts().head(10).to_string())

        # Apply filter
        df_filtered = df[df[flag_column].isin(CONFIG.option_valid_flags)]

        rows_removed = original_count - len(df_filtered)
        print(f"\n✓ Filtering complete!")
        print(f"  Kept: {len(df_filtered):,} rows ({len(df_filtered)/original_count*100:.1f}%)")
        print(f"  Removed: {rows_removed:,} rows ({rows_removed/original_count*100:.1f}%)")

        # Save filtered data to data/raw/
        output_path = CONFIG.raw_data_path / f"spy_options_{date_str}_raw.parquet"
        df_filtered.to_parquet(output_path, index=False)
        print(f"\n✓ Saved to: {output_path}")

        return df_filtered
