"""
OPTIMIZED: Download SPY 0DTE Options - RAW DATA (No Filtering)
Downloads all trades without any FLAGS filtering
"""
from thetadata_http import ThetaHttpClient
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ===== TARGET DATE =====
TARGET_DATE = '20260513'  # Change this date as needed (YYYYMMDD format)
# =======================

class OptionsDownloaderRaw:
    """Fast parallel downloader for SPY 0DTE options - NO FILTERING"""

    def __init__(self, max_workers=8):
        """
        Args:
            max_workers: Number of parallel download threads (default: 8)
        """
        self.client = ThetaHttpClient(
            base_url="http://127.0.0.1:25510",
            timeout_seconds=120
        )

        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_workers = max_workers
        self.lock = Lock()
        self.progress = {'completed': 0, 'total': 0, 'trades': 0}

    def get_strikes_for_expiration(self, root: str, expiration: str) -> list:
        """Get all available strikes."""
        params = {'root': root, 'exp': expiration}
        try:
            response_bytes = self.client.get_bytes('/v2/list/strikes', params)
            response_json = json.loads(response_bytes.decode('utf-8'))

            if response_json['header'].get('error_type'):
                return list(range(620000, 703000, 1000))

            return response_json['response']
        except:
            return list(range(620000, 703000, 1000))

    def download_strike_right(self, root: str, expiration: str, strike: int, right: str,
                             date_str: str, start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Download trades for one strike/right combination."""
        params = {
            'root': root,
            'exp': expiration,
            'strike': str(strike),
            'right': right,
            'start_date': date_str,
            'end_date': date_str,
            'start_time': str(start_time_ms),
            'end_time': str(end_time_ms)
        }

        try:
            response_bytes = self.client.get_bytes('/v2/hist/option/trade', params)
            response_json = json.loads(response_bytes.decode('utf-8'))

            if response_json['header'].get('error_type'):
                return pd.DataFrame()

            data = response_json['response']
            if not data:
                return pd.DataFrame()

            format_info = response_json['header'].get('format', [
                'ms_of_day', 'sequence', 'ext_condition1', 'ext_condition2',
                'ext_condition3', 'ext_condition4', 'condition', 'size',
                'exchange', 'price', 'condition_flags', 'price_flags',
                'volume_type', 'records_back', 'date'
            ])

            df = pd.DataFrame(data, columns=format_info)
            df['right'] = right
            df['strike'] = strike

            # Update progress
            with self.lock:
                self.progress['completed'] += 1
                self.progress['trades'] += len(df)

                # Print progress every 20 calls
                if self.progress['completed'] % 20 == 0:
                    pct = (self.progress['completed'] / self.progress['total']) * 100
                    print(f"  Progress: {self.progress['completed']}/{self.progress['total']} "
                          f"({pct:.0f}%) | {self.progress['trades']:,} trades")

            return df

        except Exception as e:
            with self.lock:
                self.progress['completed'] += 1
            return pd.DataFrame()

    def download_strike(self, args):
        """
        Download both calls and puts for one strike.
        Args is tuple: (root, expiration, strike, date_str, start_time_ms, end_time_ms)
        """
        root, expiration, strike, date_str, start_time_ms, end_time_ms = args

        # Download calls and puts
        df_calls = self.download_strike_right(
            root, expiration, strike, 'C', date_str, start_time_ms, end_time_ms
        )

        df_puts = self.download_strike_right(
            root, expiration, strike, 'P', date_str, start_time_ms, end_time_ms
        )

        # Combine
        dfs = []
        if not df_calls.empty:
            dfs.append(df_calls)
        if not df_puts.empty:
            dfs.append(df_puts)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def download_spy_0dte_options_parallel(self, date_str: str,
                                          start_time_ms: int, end_time_ms: int) -> pd.DataFrame:
        """Download SPY 0DTE options using parallel requests."""
        print(f"\n{'='*80}")
        print(f"RAW DATA DOWNLOAD - SPY 0DTE Options (No Filtering)")
        print(f"{'='*80}")
        print(f"Date: {date_str}")
        print(f"Time: {start_time_ms/1000/60/60:.2f}h to {end_time_ms/1000/60/60:.2f}h")
        print(f"Workers: {self.max_workers} parallel threads")

        expiration = date_str
        strikes = self.get_strikes_for_expiration('SPY', expiration)

        if not strikes:
            print("\n✗ No strikes found!")
            return pd.DataFrame()

        print(f"\nStrikes to download: {len(strikes)}")
        print(f"Total API calls: {len(strikes) * 2} (calls + puts)")
        print(f"\nDownloading...")

        # Initialize progress
        self.progress = {'completed': 0, 'total': len(strikes) * 2, 'trades': 0}

        # Prepare arguments for all strikes
        tasks = [
            ('SPY', expiration, strike, date_str, start_time_ms, end_time_ms)
            for strike in strikes
        ]

        # Download in parallel
        all_trades = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.download_strike, task) for task in tasks]

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if not df.empty:
                        all_trades.append(df)
                except Exception as e:
                    print(f"  ⚠ Error in download: {e}")

        if not all_trades:
            print("\n✗ No trades found!")
            return pd.DataFrame()

        # Combine all
        df_combined = pd.concat(all_trades, ignore_index=True)

        print(f"\n{'='*80}")
        print(f"✓ Downloaded {len(df_combined):,} total trades")
        print(f"  Active strikes: {len(all_trades)}/{len(strikes)}")

        return df_combined

    def prepare_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format columns - KEEPS ALL DATA, NO FILTERING"""
        if df.empty:
            return df

        df_out = pd.DataFrame()

        # TIME - Keep in milliseconds format
        if 'ms_of_day' in df.columns:
            df_out['TIME'] = df['ms_of_day']

        # Direct mappings
        column_map = {
            'sequence': 'SEQUENCE',
            'strike': 'STRIKE',
            'right': 'RIGHT',
            'condition': 'FLAGS',
            'size': 'SIZE',
            'exchange': 'EXCHANGE',
            'price': 'PRICE',
            'condition_flags': 'UNKNOWN'
        }

        for src, dst in column_map.items():
            if src in df.columns:
                df_out[dst] = df[src]

        # UNKNOWN - use default if not present
        if 'UNKNOWN' not in df_out.columns:
            df_out['UNKNOWN'] = 7

        return df_out

    def download_and_process(self, date_str: str = '20250407', sort_by_time: bool = True):
        """
        Main function - NO FILTERING APPLIED

        Args:
            date_str: Date in YYYYMMDD format
            sort_by_time: If True, sorts final output by TIME column
        """
        start_time_ms = 34_200_000
        end_time_ms = 53_100_000

        import time
        start = time.time()

        try:
            # Download
            df_raw = self.download_spy_0dte_options_parallel(date_str, start_time_ms, end_time_ms)

            if df_raw.empty:
                print("\n✗ No data!")
                return None

            # Show FLAGS distribution (but don't filter)
            print(f"\n{'='*80}")
            print("FLAGS Distribution (NO FILTERING APPLIED)")
            print(f"{'='*80}")

            if 'condition' in df_raw.columns:
                print(f"\nCondition (FLAGS) distribution:")
                print(df_raw['condition'].value_counts().sort_index())

            # Format (no filtering)
            df_final = self.prepare_output_columns(df_raw)

            # Sort by TIME, then SEQUENCE (chronological order)
            if sort_by_time and 'TIME' in df_final.columns and 'SEQUENCE' in df_final.columns:
                print(f"\nSorting by TIME, then SEQUENCE (chronological order)...")
                df_final = df_final.sort_values(['TIME', 'SEQUENCE']).reset_index(drop=True)
                print(f"  Time range: {df_final['TIME'].min()} to {df_final['TIME'].max()}")

            # Save as Parquet
            final_path = self.output_dir / f"spy_options_{date_str}_raw.parquet"
            df_final.to_parquet(final_path, index=False)
            print(f"✓ Saved: {final_path}")

            elapsed = time.time() - start

            # Summary
            print(f"\n{'='*80}")
            print("FINAL SUMMARY")
            print(f"{'='*80}")
            print(f"Downloaded: {len(df_final):,} trades (ALL FLAGS INCLUDED)")
            print(f"Time: {elapsed:.1f} seconds")
            print(f"Speed: {len(df_final)/elapsed:.0f} trades/second")

            if 'FLAGS' in df_final.columns:
                print(f"\nFLAGS breakdown:")
                flags_dist = df_final['FLAGS'].value_counts().sort_index()
                for flag, count in flags_dist.items():
                    pct = count / len(df_final) * 100
                    print(f"  FLAG {flag}: {count:,} trades ({pct:.1f}%)")

            print(f"\nNote: This is RAW data with NO filtering applied")
            print(f"      TIME column is in milliseconds format")

            return df_final

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    # Create downloader with 8 parallel workers
    downloader = OptionsDownloaderRaw(max_workers=8)

    # Download RAW data (no filtering)
    df = downloader.download_and_process(TARGET_DATE, sort_by_time=True)

    if df is not None:
        print(f"\n✓ Complete! RAW unfiltered data saved in data/raw/")
        print(f"\nFile: spy_options_{TARGET_DATE}_raw.parquet")


if __name__ == "__main__":
    main()
