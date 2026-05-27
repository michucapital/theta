# add_eq30.py
# Computes Eq30.
# Eq30 is the same as Equilibrium from add_strike_columns.py,
# but uses a 25,000-tick rolling window instead of 12,500.
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay         = "20260513"
ProcessAllFiles   = False
OverwriteIfExists = True
WEIGHT_MULTIPLIER = 1    # newest-bucket weight (oldest is always 1)
                         # set to 1 for uniform weights
N_WORKERS         = 7    # 0 = auto (cpu_count - 1)

WINDOW_EQ30 = 25_000
PERIOD_EQ30 = 500
# ============================================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

VALID_FLAGS = {18, 95, 125}
TARGET_COLS = ["Eq30"]
REQUIRED_COLS = ["RIGHT", "FLAGS", "SIZE", "PRICE", "STRIKE", "spot_price"]

assert WINDOW_EQ30 % PERIOD_EQ30 == 0, (
    f"WINDOW_EQ30 ({WINDOW_EQ30}) must be divisible by PERIOD_EQ30 ({PERIOD_EQ30})"
)
assert WINDOW_EQ30 > 0 and PERIOD_EQ30 > 0, "WINDOW_EQ30 and PERIOD_EQ30 must be positive"


def weighted_rolling_sum_batched(arr2d: np.ndarray,
                                 window: int,
                                 period: int,
                                 w_max: float,
                                 w_min: float = 1.0) -> np.ndarray:
    """
    Bucket-weighted rolling sum for every column in arr2d (shape N x K).
    Weights ramp linearly from w_min (oldest bucket) to w_max (newest).
    Returns (N, K); first (window-1) rows are NaN.
    """
    N, K = arr2d.shape
    n_buckets = window // period

    m_arr = np.arange(n_buckets, dtype=np.float64)
    weights = (
        w_max - m_arr * (w_max - w_min) / (n_buckets - 1)
        if n_buckets > 1 else np.array([w_max])
    )

    cs = np.empty((N + 1, K), dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr2d, axis=0, out=cs[1:])

    result = np.zeros((N, K), dtype=np.float64)
    valid_start = window - 1

    for m in range(n_buckets):
        upper_s = valid_start - m * period + 1
        upper_e = N           - m * period + 1
        lower_s = upper_s - period
        lower_e = upper_e - period
        result[valid_start:] += weights[m] * (cs[upper_s:upper_e] - cs[lower_s:lower_e])

    result[:valid_start] = np.nan
    return result


def add_eq30(df: pd.DataFrame) -> pd.DataFrame:
    flag_mask = df["FLAGS"].isin(VALID_FLAGS).to_numpy()
    right_np  = df["RIGHT"].to_numpy()
    strike_np = df["STRIKE"].to_numpy(dtype=np.float64)
    spot_np   = df["spot_price"].to_numpy(dtype=np.float64)
    price_np  = df["PRICE"].to_numpy(dtype=np.float64)
    size_np   = df["SIZE"].to_numpy(dtype=np.float64)

    call_mask = flag_mask & (right_np == "C")
    put_mask  = flag_mask & (right_np == "P")

    call_base = np.where(
        strike_np >= spot_np,
        price_np * size_np,
        (price_np - spot_np + strike_np) * size_np
    )
    call_weight = np.where(call_mask, np.maximum(call_base, 0.0), 0.0)

    put_base = np.where(
        strike_np <= spot_np,
        price_np * size_np,
        (price_np + spot_np - strike_np) * size_np
    )
    put_weight = np.where(put_mask, np.maximum(put_base, 0.0), 0.0)

    call_prod = strike_np * call_weight
    put_prod  = strike_np * put_weight

    ws = weighted_rolling_sum_batched(
        np.column_stack([call_weight, call_prod, put_weight, put_prod]),
        WINDOW_EQ30,
        PERIOD_EQ30,
        w_max=float(WEIGHT_MULTIPLIER)
    )
    cw, cpw, pw, ppw = ws[:, 0], ws[:, 1], ws[:, 2], ws[:, 3]

    with np.errstate(invalid="ignore", divide="ignore"):
        cs_eq = np.where((cw == 0) | np.isnan(cw), 0.0, cpw / cw)
        ps_eq = np.where((pw == 0) | np.isnan(pw), 0.0, ppw / pw)
        denom = cpw + ppw
        eq30 = np.where(
            (denom == 0) | np.isnan(denom),
            np.nan,
            (cpw * cs_eq + ppw * ps_eq) / denom
        )

    df["Eq30"] = eq30
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

        df = add_eq30(df)

        nan_eq30 = int(np.isnan(df["Eq30"].to_numpy()).sum())

        df.to_parquet(file_path, compression="snappy", engine="pyarrow", index=False)

        return f"  OK   : {file_path.name} | rows={len(df):,} | Eq30 NaNs={nan_eq30:,}"

    except Exception as exc:
        import traceback
        return f"  ERROR: {file_path.name} — {exc}\n{traceback.format_exc()}"


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("ADD COLUMN: Eq30")
    print("            (same as Equilibrium, but with 25,000-tick window)")
    print("=" * 80)

    workers = N_WORKERS if N_WORKERS > 0 else max(1, (os.cpu_count() or 2) - 1)

    print(f"Started          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Weight multiplier: 1 (oldest) -> {WEIGHT_MULTIPLIER} (newest)")
    print(f"Flags            : {sorted(VALID_FLAGS)}")
    print(f"Overwrite        : {OverwriteIfExists}")
    print(f"Workers          : {workers}")
    print()
    print(f"  {'Column':<14}  {'Window':>8}  {'Period':>8}  {'Buckets':>8}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Eq30':<14}  {WINDOW_EQ30:>8,}  {PERIOD_EQ30:>8,}  {WINDOW_EQ30 // PERIOD_EQ30:>8,}")
    print()

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "data" / "merged"

    if not data_folder.exists():
        print(f"ERROR: {data_folder} not found")
        return

    if ProcessAllFiles:
        files = sorted(data_folder.glob("spy_*_merged.parquet"))
    else:
        t = data_folder / f"spy_{ChooseDay}_merged.parquet"
        if not t.exists():
            print(f"ERROR: {t} not found")
            return
        files = [t]

    if not files:
        print("ERROR: no files found")
        return

    print(f"Files to process: {len(files)}")
    print()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_file, str(fp)): fp for fp in files}
        for future in tqdm(as_completed(futures), total=len(files), desc="Processing", unit="file"):
            tqdm.write(future.result())

    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()