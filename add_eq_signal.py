# add_eq_signal.py
# Computes EqSignal — the simple difference between Eq12 and Eq25.
#
#   EqSignal = Eq12 - Eq25
#
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay         = "20260513"   # YYYYMMDD — used only if ProcessAllFiles = False
ProcessAllFiles   = False
OverwriteIfExists = True
N_WORKERS         = 7            # 0 = auto (cpu_count - 1)
# ============================================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

TARGET_COLS   = ["EqSignal"]
REQUIRED_COLS = ["Eq12", "Eq25"]

def compute_eq_signal(df: pd.DataFrame) -> pd.DataFrame:
    eq12_np = df["Eq12"].to_numpy(dtype=np.float64)
    eq25_np = df["Eq25"].to_numpy(dtype=np.float64)

    df["EqSignal"] = eq12_np - eq25_np
    return df

# ============================================================================
# WORKER
# ============================================================================
def _process_file(file_path_str: str) -> str:
    file_path = Path(file_path_str)
    try:
        df = pd.read_parquet(file_path)

        if all(c in df.columns for c in TARGET_COLS) and not OverwriteIfExists:
            return f"  SKIP : {file_path.name}"

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return f"  ERROR: missing cols {missing} in {file_path.name}"

        df = compute_eq_signal(df)

        nan_n   = int(np.isnan(df["EqSignal"].to_numpy()).sum())
        valid_n = int((~np.isnan(df["EqSignal"].to_numpy())).sum())

        df.to_parquet(file_path, compression="snappy", engine="pyarrow", index=False)

        return (f"  OK   : {file_path.name} | rows={len(df):,} | "
                f"EqSignal NaNs={nan_n:,}  valid={valid_n:,}")

    except Exception as exc:
        import traceback
        return f"  ERROR: {file_path.name} — {exc}\n{traceback.format_exc()}"

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("ADD COLUMN: EqSignal  (Eq12 - Eq25)")
    print("=" * 80)

    workers = N_WORKERS if N_WORKERS > 0 else max(1, (os.cpu_count() or 2) - 1)

    print(f"Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Overwrite  : {OverwriteIfExists}")
    print(f"Workers    : {workers}")
    print()

    script_dir  = Path(__file__).resolve().parent
    data_folder = script_dir / "data" / "merged"

    if not data_folder.exists():
        print(f"ERROR: {data_folder} not found"); return

    if ProcessAllFiles:
        files = sorted(data_folder.glob("spy_*_merged.parquet"))
    else:
        t = data_folder / f"spy_{ChooseDay}_merged.parquet"
        if not t.exists():
            print(f"ERROR: {t} not found"); return
        files = [t]

    if not files:
        print("ERROR: no files found"); return

    print(f"Files to process: {len(files)}")
    print()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, str(fp)): fp for fp in files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Processing", unit="file"):
            tqdm.write(future.result())

    print()
    print(f"Completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()