# filter_dataset.py
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay = "20260513"  # Format: YYYYMMDD
# ============================================================================

from pathlib import Path
import pandas as pd
import numpy as np

TEMPLATE_COLUMNS = [
    "TIME", "SEQUENCE", "STRIKE", "RIGHT", "FLAGS", "SIZE", "EXCHANGE", "PRICE",
    "spot_time", "spot_price",
    "CallStrike", "PutStrike", "Equilibrium", "CPavg", "EqSignal", "Eq25", "EqITM", "Eq12", "MaxMin"
]

RENAME_MAP = {
    "SPOT_TIME": "spot_time",
    "SPOT_PRICE": "spot_price",
}

FLOAT_TEMPLATE_DEFAULTS = {
    "CallStrike": 0.0,
    "PutStrike": 0.0,
    "Equilibrium": np.nan,
    "CPavg": np.nan,
    "EqSignal": np.nan,
    "Eq25": np.nan,
    "EqITM": np.nan,
    "Eq12": np.nan,
}

INT_TEMPLATE_DEFAULTS = {
    "MaxMin": 0,
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns and v not in df.columns}
    if existing_map:
        df = df.rename(columns=existing_map)
    return df


def add_missing_template_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col, default in FLOAT_TEMPLATE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    for col, default in INT_TEMPLATE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    return df


def main():
    print("=" * 80)
    print(f"FILTERING + FORMATTING DATASET: {ChooseDay}")
    print("=" * 80)

    script_dir = Path(__file__).resolve().parent
    data_folder = script_dir / "data" / "merged"
    file_path = data_folder / f"spy_{ChooseDay}_merged.parquet"
    out_path = data_folder / f"spy_{ChooseDay}_merged.parquet"

    if not file_path.exists():
        print(f"ERROR: File not found -> {file_path}")
        return

    print(f"Reading: {file_path.name}")
    df = pd.read_parquet(file_path)
    initial_rows = len(df)
    initial_cols = list(df.columns)
    print(f"Initial rows: {initial_rows:,}")
    print(f"Initial columns ({len(initial_cols)}): {initial_cols}")

    df = standardize_columns(df)

    required_source_cols = ["TIME", "SEQUENCE", "STRIKE", "RIGHT", "FLAGS", "SIZE", "EXCHANGE", "PRICE", "spot_time", "spot_price"]
    missing = [c for c in required_source_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Dataset is missing required columns: {missing}")
        return

    df = df[df["spot_price"].notna()]
    df = df[df["spot_price"] >= 600.0].copy()

    rows_after_base_filter = len(df)
    print(
        f"Rows after dropping < 600 & NaNs: {rows_after_base_filter:,} "
        f"(Dropped: {initial_rows - rows_after_base_filter:,})"
    )

    if df.empty:
        print("Dataset is empty after base filters. Exiting.")
        return

    df["spot_price"] = pd.to_numeric(df["spot_price"], errors="coerce").round(2)
    df = df[df["spot_price"].notna()].copy()

    df["STRIKE"] = pd.to_numeric(df["STRIKE"], errors="coerce") / 1000.0
    df = df[df["STRIKE"].notna()].copy()

    prices = df["spot_price"].to_numpy(dtype=np.float64)
    valid_mask = np.ones(len(prices), dtype=bool)
    last_valid_price = prices[0]

    for i in range(1, len(prices)):
        if abs(prices[i] - last_valid_price) > 1.0:
            valid_mask[i] = False
        else:
            last_valid_price = prices[i]

    df = df[valid_mask].copy()

    rows_after_jumps = len(df)
    print(
        f"Rows after removing > 1.0 price jumps: {rows_after_jumps:,} "
        f"(Dropped: {rows_after_base_filter - rows_after_jumps:,})"
    )

    if df.empty:
        print("Dataset is empty after jump filter. Exiting.")
        return

    df["RIGHT"] = df["RIGHT"].astype(str).str.strip().str.upper()

    int_columns = ["TIME", "SEQUENCE", "FLAGS", "SIZE", "EXCHANGE", "spot_time"]
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=int_columns).copy()

    for col in int_columns:
        df[col] = df[col].astype(np.int64)

    df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")
    df = df[df["PRICE"].notna()].copy()

    df["spot_price"] = df["spot_price"].astype(np.float64)
    df["STRIKE"] = df["STRIKE"].astype(np.float64)
    df["PRICE"] = df["PRICE"].astype(np.float64)

    df = add_missing_template_columns(df)

    extra_cols = [c for c in df.columns if c not in TEMPLATE_COLUMNS]
    if extra_cols:
        print(f"Dropping extra columns ({len(extra_cols)}): {extra_cols}")

    df = df[TEMPLATE_COLUMNS].copy()
    df["MaxMin"] = df["MaxMin"].fillna(0).astype(np.int8)

    print(f"Final columns ({len(df.columns)}): {list(df.columns)}")
    print(f"Saving to: {out_path.name}")
    df.to_parquet(out_path, compression="snappy", engine="pyarrow", index=False)

    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
