# add_strike_columns.py
# Computes CallStrike, PutStrike, Equilibrium, and CPavg.
# Each column has its own independently configurable rolling window.
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay         = "20260513"
ProcessAllFiles   = False
OverwriteIfExists = True
WEIGHT_MULTIPLIER = 1    # newest-bucket weight (oldest is always 1)
                          # set to 1 for uniform weights
N_WORKERS         = 7    # 0 = auto (cpu_count - 1)

# ── Per-column rolling window  (window must be divisible by period) ───────────
#                window    period    → buckets
WINDOW_CS  = 60_000 ;  PERIOD_CS  = 2000   # CallStrike   → 25 buckets
WINDOW_PS  = 60_000 ;  PERIOD_PS  = 2000   # PutStrike    → 25 buckets
WINDOW_EQ  = 12_500 ;  PERIOD_EQ  = 500   # Equilibrium  → 25 buckets
# CPavg = (CallStrike + PutStrike) / 2 — no independent window needed
# ============================================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime


VALID_FLAGS = {18, 95, 125}
TARGET_COLS = ["CallStrike", "PutStrike", "Eq12", "CPavg"]

# Validate config at import time
for _name, _w, _p in [("CS", WINDOW_CS, PERIOD_CS),
                       ("PS", WINDOW_PS, PERIOD_PS),
                       ("EQ", WINDOW_EQ, PERIOD_EQ)]:
    assert _w % _p == 0, f"WINDOW_{_name} ({_w}) must be divisible by PERIOD_{_name} ({_p})"
    assert _w > 0 and _p > 0, f"WINDOW_{_name} and PERIOD_{_name} must be positive"


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
    N, K      = arr2d.shape
    n_buckets = window // period

    m_arr   = np.arange(n_buckets, dtype=np.float64)
    weights = w_max - m_arr * (w_max - w_min) / (n_buckets - 1) \
              if n_buckets > 1 else np.array([w_max])

    cs    = np.empty((N + 1, K), dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr2d, axis=0, out=cs[1:])

    result      = np.zeros((N, K), dtype=np.float64)
    valid_start = window - 1

    for m in range(n_buckets):
        upper_s = valid_start - m * period + 1
        upper_e = N           - m * period + 1
        lower_s = upper_s - period
        lower_e = upper_e - period
        result[valid_start:] += weights[m] * (cs[upper_s:upper_e]
                                              - cs[lower_s:lower_e])
    result[:valid_start] = np.nan
    return result


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    flag_mask = df["FLAGS"].isin(VALID_FLAGS).to_numpy()
    right_np  = df["RIGHT"].to_numpy()
    strike_np = df["STRIKE"].to_numpy(dtype=np.float64)
    spot_np   = df["spot_price"].to_numpy(dtype=np.float64)
    price_np  = df["PRICE"].to_numpy(dtype=np.float64)
    size_np   = df["SIZE"].to_numpy(dtype=np.float64)

    call_mask = flag_mask & (right_np == "C")
    put_mask  = flag_mask & (right_np == "P")

    # Per-tick extrinsic weights (≥ 0): OTM = full premium, ITM = time value only
    call_base   = np.where(strike_np >= spot_np,
                           price_np * size_np,
                           (price_np - spot_np + strike_np) * size_np)
    call_weight = np.where(call_mask, np.maximum(call_base, 0.0), 0.0)

    put_base    = np.where(strike_np <= spot_np,
                           price_np * size_np,
                           (price_np + spot_np - strike_np) * size_np)
    put_weight  = np.where(put_mask, np.maximum(put_base, 0.0), 0.0)

    call_prod = strike_np * call_weight
    put_prod  = strike_np * put_weight

    # ── Rolling sums, cached by (window, period) to avoid duplicate work ──────
    _cache: dict = {}

    def get_rolling(window: int, period: int):
        key = (window, period)
        if key not in _cache:
            arr2d = np.column_stack([call_weight, call_prod,
                                     put_weight,  put_prod])
            ws = weighted_rolling_sum_batched(arr2d, window, period,
                                              w_max=float(WEIGHT_MULTIPLIER))
            _cache[key] = (ws[:, 0], ws[:, 1], ws[:, 2], ws[:, 3])
        return _cache[key]

    # ── CallStrike ────────────────────────────────────────────────────────────
    cw_cs, cpw_cs, _, _ = get_rolling(WINDOW_CS, PERIOD_CS)
    with np.errstate(invalid="ignore", divide="ignore"):
        call_strike = np.where((cw_cs == 0) | np.isnan(cw_cs), 0.0,
                               cpw_cs / cw_cs)

    # ── PutStrike ─────────────────────────────────────────────────────────────
    _, _, pw_ps, ppw_ps = get_rolling(WINDOW_PS, PERIOD_PS)
    with np.errstate(invalid="ignore", divide="ignore"):
        put_strike = np.where((pw_ps == 0) | np.isnan(pw_ps), 0.0,
                              ppw_ps / pw_ps)

    # ── Equilibrium ───────────────────────────────────────────────────────────
    cw_eq, cpw_eq, pw_eq, ppw_eq = get_rolling(WINDOW_EQ, PERIOD_EQ)
    with np.errstate(invalid="ignore", divide="ignore"):
        cs_eq   = np.where((cw_eq == 0) | np.isnan(cw_eq), 0.0, cpw_eq / cw_eq)
        ps_eq   = np.where((pw_eq == 0) | np.isnan(pw_eq), 0.0, ppw_eq / pw_eq)
        denom   = cpw_eq + ppw_eq
        equilibrium = np.where(
            (denom == 0) | np.isnan(denom),
            np.nan,
            (cpw_eq * cs_eq + ppw_eq * ps_eq) / denom
        )

    # ── CPavg = (CallStrike + PutStrike) / 2 ─────────────────────────────────
    with np.errstate(invalid="ignore"):
        cp_avg = np.where(
            (call_strike == 0) | (put_strike == 0),
            np.nan,
            (call_strike + put_strike) * 0.5
        )
    # Explicit warmup guard: NaN until both CS and PS windows are fully warm
    cp_avg[:max(WINDOW_CS, WINDOW_PS) - 1] = np.nan

    df["CallStrike"]  = call_strike
    df["PutStrike"]   = put_strike
    df["Eq12"] = equilibrium
    df["CPavg"]       = cp_avg
    return df


# ============================================================================
# WORKER  (module-level so it is picklable for multiprocessing)
# ============================================================================
def _process_file(file_path_str: str) -> str:
    file_path = Path(file_path_str)
    try:
        df = pd.read_parquet(file_path)

        if all(c in df.columns for c in TARGET_COLS) and not OverwriteIfExists:
            return f"  SKIP : {file_path.name}"

        missing = [c for c in ["RIGHT", "FLAGS", "SIZE", "PRICE", "STRIKE", "spot_price"]
                   if c not in df.columns]
        if missing:
            return f"  ERROR: missing cols {missing} in {file_path.name}"

        df = add_columns(df)

        nan_cs = int(np.isnan(df["CallStrike"].replace(0, np.nan).to_numpy()).sum())
        nan_ps = int(np.isnan(df["PutStrike"].replace(0, np.nan).to_numpy()).sum())
        nan_eq = int(np.isnan(df["Eq12"].to_numpy()).sum())
        nan_cp = int(np.isnan(df["CPavg"].to_numpy()).sum())

        df.to_parquet(file_path, compression="snappy", engine="pyarrow", index=False)

        return (f"  OK   : {file_path.name} | rows={len(df):,} | "
                f"CS NaNs={nan_cs:,}  PS NaNs={nan_ps:,}  "
                f"EQ NaNs={nan_eq:,}  CPavg NaNs={nan_cp:,}")

    except Exception as exc:
        import traceback
        return f"  ERROR: {file_path.name} — {exc}\n{traceback.format_exc()}"


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("ADD COLUMNS: CallStrike, PutStrike, Eq12, CPavg")
    print("             (per-column configurable rolling windows)")
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
    for name, w, p in [("CallStrike",  WINDOW_CS, PERIOD_CS),
                        ("PutStrike",   WINDOW_PS, PERIOD_PS),
                        ("Eq12", WINDOW_EQ, PERIOD_EQ),
                        ("CPavg",       max(WINDOW_CS,WINDOW_PS), "-", )]:
        if name == "CPavg":
            print(f"  {name:<14}  {'(CS+PS)/2':>8}  {'':>8}  {'':>8}")
        else:
            print(f"  {name:<14}  {w:>8,}  {p:>8,}  {w//p:>8,}")
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
        for future in tqdm(as_completed(futures), total=len(files),
                           desc="Processing", unit="file"):
            tqdm.write(future.result())

    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
