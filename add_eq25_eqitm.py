# add_eq25_eqitm.py
# Computes Eq25 and EqITM — uniform (unweighted) rolling Equilibriums over WINDOW ticks.
#
#   Eq25 uses all options (stripping intrinsic value from ITM options).
#   EqITM uses ONLY ITM options (stripping intrinsic value, treating OTM as 0 weight).
#
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay         = "20260513"   # YYYYMMDD — used only if ProcessAllFiles = False
ProcessAllFiles   = False
OverwriteIfExists = True
N_WORKERS         = 7            # Default to 7 workers

WINDOW_EQ25       = 25_000       # rolling lookback in raw ticks for Eq25
WINDOW_EQITM      = 12_500       # rolling lookback in raw ticks for EqITM
# ============================================================================

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

VALID_FLAGS   = {18, 95, 125}
TARGET_COLS   = ["Eq25", "EqITM"]
REQUIRED_COLS = ["RIGHT", "FLAGS", "SIZE", "PRICE", "STRIKE", "spot_price"]


def uniform_rolling_sum(arr2d: np.ndarray, window: int) -> np.ndarray:
    """
    Plain O(N) rolling sum for every column in arr2d (shape N x K).
    Uses a single 2-D prefix-sum pass — no bucket loop needed.
    Returns (N, K); first (window-1) rows are NaN.
    """
    N, K  = arr2d.shape
    cs    = np.empty((N + 1, K), dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(arr2d, axis=0, out=cs[1:])

    result               = np.full((N, K), np.nan, dtype=np.float64)
    result[window - 1:]  = cs[window:] - cs[:N - window + 1]
    return result


def compute_eq25_and_eqitm(df: pd.DataFrame) -> pd.DataFrame:
    flag_mask = df["FLAGS"].isin(VALID_FLAGS).to_numpy()
    right_np  = df["RIGHT"].to_numpy()
    strike_np = df["STRIKE"].to_numpy(dtype=np.float64)
    spot_np   = df["spot_price"].to_numpy(dtype=np.float64)
    price_np  = df["PRICE"].to_numpy(dtype=np.float64)
    size_np   = df["SIZE"].to_numpy(dtype=np.float64)

    call_mask = flag_mask & (right_np == "C")
    put_mask  = flag_mask & (right_np == "P")

    # --------------------------------------------------------------------------
    # 1. Eq25 Weights (All Options - Extrinsic Value)
    # --------------------------------------------------------------------------
    call_base_std = np.where(strike_np >= spot_np,
                             price_np * size_np,
                             (price_np - spot_np + strike_np) * size_np)
    call_weight_std = np.where(call_mask, np.maximum(call_base_std, 0.0), 0.0)

    put_base_std = np.where(strike_np <= spot_np,
                            price_np * size_np,
                            (price_np + spot_np - strike_np) * size_np)
    put_weight_std = np.where(put_mask, np.maximum(put_base_std, 0.0), 0.0)

    call_prod_std = strike_np * call_weight_std
    put_prod_std  = strike_np * put_weight_std

    # --------------------------------------------------------------------------
    # 2. EqITM Weights (ITM Options ONLY - Extrinsic Value)
    # --------------------------------------------------------------------------
    call_base_itm = np.where(strike_np < spot_np,
                             (price_np - spot_np + strike_np) * size_np,
                             0.0)
    call_weight_itm = np.where(call_mask, np.maximum(call_base_itm, 0.0), 0.0)

    put_base_itm = np.where(strike_np > spot_np,
                            (price_np + spot_np - strike_np) * size_np,
                            0.0)
    put_weight_itm = np.where(put_mask, np.maximum(put_base_itm, 0.0), 0.0)

    call_prod_itm = strike_np * call_weight_itm
    put_prod_itm  = strike_np * put_weight_itm

    # ── Prefix-sum passes (split to allow independent window lengths) ─────────
    ws_std = uniform_rolling_sum(
        np.column_stack([call_weight_std, call_prod_std, put_weight_std, put_prod_std]),
        WINDOW_EQ25
    )
    cw_std, cpw_std, pw_std, ppw_std = ws_std[:, 0], ws_std[:, 1], ws_std[:, 2], ws_std[:, 3]

    ws_itm = uniform_rolling_sum(
        np.column_stack([call_weight_itm, call_prod_itm, put_weight_itm, put_prod_itm]),
        WINDOW_EQITM
    )
    cw_itm, cpw_itm, pw_itm, ppw_itm = ws_itm[:, 0], ws_itm[:, 1], ws_itm[:, 2], ws_itm[:, 3]

    # ── Calculate Formulas ────────────────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        # Eq25
        cs_eq_std = np.where((cw_std == 0) | np.isnan(cw_std), 0.0, cpw_std / cw_std)
        ps_eq_std = np.where((pw_std == 0) | np.isnan(pw_std), 0.0, ppw_std / pw_std)
        denom_std = cpw_std + ppw_std
        eq25 = np.where(
            (denom_std == 0) | np.isnan(denom_std),
            np.nan,
            (cpw_std * cs_eq_std + ppw_std * ps_eq_std) / denom_std
        )

        # EqITM
        cs_eq_itm = np.where((cw_itm == 0) | np.isnan(cw_itm), 0.0, cpw_itm / cw_itm)
        ps_eq_itm = np.where((pw_itm == 0) | np.isnan(pw_itm), 0.0, ppw_itm / pw_itm)
        denom_itm = cpw_itm + ppw_itm
        eqitm = np.where(
            (denom_itm == 0) | np.isnan(denom_itm),
            np.nan,
            (cpw_itm * cs_eq_itm + ppw_itm * ps_eq_itm) / denom_itm
        )

    # Assign columns directly, dropping any old eq30 legacy references if they exist
    if "Eq30" in df.columns:
        df.drop(columns=["Eq30"], inplace=True)

    df["Eq25"] = eq25
    df["EqITM"] = eqitm
    
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

        df = compute_eq25_and_eqitm(df)

        valid_n_25  = int((~np.isnan(df["Eq25"].to_numpy())).sum())
        valid_n_itm = int((~np.isnan(df["EqITM"].to_numpy())).sum())

        df.to_parquet(file_path, compression="snappy", engine="pyarrow", index=False)

        return (f"  OK   : {file_path.name} | rows={len(df):,} | "
                f"Eq25 valid={valid_n_25:,} | EqITM valid={valid_n_itm:,}")

    except Exception as exc:
        import traceback
        return f"  ERROR: {file_path.name} — {exc}\n{traceback.format_exc()}"


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("ADD COLUMNS: Eq25 & EqITM (Uniform rolling Equilibriums)")
    print("=" * 80)

    workers = N_WORKERS if N_WORKERS > 0 else max(1, (os.cpu_count() or 2) - 1)

    print(f"Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eq25 Window  : {WINDOW_EQ25:,} ticks")
    print(f"EqITM Window : {WINDOW_EQITM:,} ticks")
    print(f"Flags        : {sorted(VALID_FLAGS)}")
    print(f"Overwrite    : {OverwriteIfExists}")
    print(f"Workers      : {workers}")
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
    print(f"Completed    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
