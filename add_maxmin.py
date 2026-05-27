from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay         = "20260513"   # Format: YYYYMMDD
ProcessAllFiles   = False         # Set to False to process only the day above
N_WORKERS         = 7            # Set to 7 per request

# Indicator Parameters
WINDOW_SIZE       = 12_000       # Size of the lookback window to find min/max
SHIFT_TICKS       = 2_000        # How many ticks backward to check for the peak
THRESHOLD         = 0.3          # Absolute threshold for the extreme to be valid
# ============================================================================

def compute_maxmin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the MaxMin column using vectorized pandas operations.
    Replicates the exact Excel formula:
    =IF(AND(B10001 < -0.3, B10001=MIN(B2:B12001)), -1, 0)
    """
    eq_signal = df["EqSignal"]

    # 1. Calculate the rolling max and min of the 12,000 tick window ending at the current tick
    roll_max = eq_signal.rolling(window=WINDOW_SIZE).max()
    roll_min = eq_signal.rolling(window=WINDOW_SIZE).min()

    # 2. The target value we are evaluating is the one 2,000 ticks ago
    target_val = eq_signal.shift(SHIFT_TICKS)

    # 3. Evaluate the condition at current index [i]:
    # Is the value from 2,000 ticks ago the absolute max/min of the entire 12,000 tick window ending now?
    is_local_max = (target_val == roll_max) & (target_val > THRESHOLD)
    is_local_min = (target_val == roll_min) & (target_val < -THRESHOLD)

    # 4. Create the array and assign 1 and -1 based on conditions
    temp_maxmin = np.zeros(len(df), dtype=np.int8)
    temp_maxmin[is_local_max] = 1
    temp_maxmin[is_local_min] = -1

    df["MaxMin"] = temp_maxmin

    return df

def _process_file(file_path_str: str) -> str:
    """Worker function to process a single parquet file."""
    file_path = Path(file_path_str)
    try:
        df = pd.read_parquet(file_path)

        if "EqSignal" not in df.columns:
            return f"  ERROR: missing 'EqSignal' column in {file_path.name}"

        df = compute_maxmin(df)

        # Count how many peaks and troughs we found to report back
        peaks = int((df["MaxMin"] == 1).sum())
        troughs = int((df["MaxMin"] == -1).sum())

        df.to_parquet(file_path, compression="snappy", engine="pyarrow", index=False)

        return f"  OK   : {file_path.name} | Peaks (+1): {peaks:,} | Troughs (-1): {troughs:,}"

    except Exception as exc:
        import traceback
        return f"  ERROR: {file_path.name} — {exc}\n{traceback.format_exc()}"

def main():
    print("=" * 80)
    print("ADD COLUMN: MaxMin (Delayed Local Extremes of EqSignal)")
    print("=" * 80)

    workers = N_WORKERS if N_WORKERS > 0 else max(1, (os.cpu_count() or 2) - 1)

    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback   : {WINDOW_SIZE:,} ticks")
    print(f"Target     : {SHIFT_TICKS:,} ticks ago")
    print(f"Threshold  : > {THRESHOLD} / < -{THRESHOLD}")
    print(f"Workers    : {workers}")
    print()

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "data" / "merged"

    if not data_folder.exists():
        print(f"ERROR: Directory not found -> {data_folder}")
        return

    if ProcessAllFiles:
        files = sorted(data_folder.glob("spy_*_merged.parquet"))
    else:
        target_file = data_folder / f"spy_{ChooseDay}_merged.parquet"
        if not target_file.exists():
            print(f"ERROR: File not found -> {target_file}")
            return
        files = [target_file]

    if not files:
        print("ERROR: No parquet files found to process.")
        return

    print(f"Files to process: {len(files)}")
    print()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, str(fp)): fp for fp in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            tqdm.write(future.result())

    print()
    print(f"Completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
