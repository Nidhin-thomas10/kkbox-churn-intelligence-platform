"""
data_loader.py
==============
KKBox Churn Intelligence Platform — Layer 1: Data Pipeline

Responsibilities:
- Load all four raw KKBox CSV files with correct dtypes
- Validate data quality (null rates, shape, expected columns)
- Aggregate user_logs.csv (training period) + user_logs_v2.csv (March 2017) to per-user features
- Join members + transactions + log aggregates + train labels
- Write a clean processed snapshot to data/processed/master.parquet

Usage:
    from src.data_loader import load_all, build_master_dataset
    df = build_master_dataset()
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Expected columns — used for validation
# ---------------------------------------------------------------------------
EXPECTED_COLS = {
    "train":        {"msno", "is_churn"},
    "members":      {"msno", "city", "bd", "gender", "registered_via",
                     "registration_init_time"},
    "transactions": {"msno", "payment_method_id", "payment_plan_days",
                     "plan_list_price", "actual_amount_paid", "is_auto_renew",
                     "transaction_date", "membership_expire_date", "is_cancel"},
    "user_logs":    {"msno", "date", "num_25", "num_50", "num_75",
                     "num_985", "num_100", "num_unq", "total_secs"},
}


# ---------------------------------------------------------------------------
# 1. Loaders
# ---------------------------------------------------------------------------

def load_train(path: Path = RAW / "train_v2.csv") -> pd.DataFrame:
    """Load churn labels. Returns msno + is_churn."""
    log.info("Loading train labels ...")
    df = pd.read_csv(path, dtype={"msno": str, "is_churn": np.int8})
    _validate(df, "train")
    log.info(f"  Train shape: {df.shape} | churn rate: {df.is_churn.mean():.3%}")
    return df


def load_members(path: Path = RAW / "members_v3.csv") -> pd.DataFrame:
    """Load member demographics with cleaned age and dates."""
    log.info("Loading members ...")
    df = pd.read_csv(
        path,
        dtype={"msno": str, "city": np.int8, "gender": str,
               "registered_via": np.int8},
        parse_dates=False,
    )
    _validate(df, "members")

    # --- Age cleaning: outliers range from -7000 to 2015, treat as missing ---
    df["bd"] = df["bd"].where((df["bd"] >= 10) & (df["bd"] <= 80), np.nan)
    df.rename(columns={"bd": "age"}, inplace=True)

    # --- Registration date to datetime ---
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"], format="%Y%m%d", errors="coerce"
    )

    # --- Days since registration to cutoff (Feb 2017) — proxy for tenure ---
    cutoff = pd.Timestamp("2017-03-31")
    df["days_since_registration"] = (
        cutoff - df["registration_init_time"]
    ).dt.days.clip(lower=0)

    # --- Registration month and year (seasonality signal) ---
    df["reg_year"]  = df["registration_init_time"].dt.year
    df["reg_month"] = df["registration_init_time"].dt.month

    log.info(f"  Members shape: {df.shape}")
    return df


def load_transactions(path: Path = RAW / "transactions_v2.csv") -> pd.DataFrame:
    """
    Load transaction history.
    We aggregate to per-user summary features — last transaction
    state plus rolling statistics across all transactions.
    """
    log.info("Loading transactions ...")
    df = pd.read_csv(
        path,
        dtype={
            "msno": str,
            "payment_method_id": np.int8,
            "payment_plan_days": np.int16,
            "plan_list_price": np.float32,
            "actual_amount_paid": np.float32,
            "is_auto_renew": np.int8,
            "is_cancel": np.int8,
        },
        parse_dates=False,
    )
    _validate(df, "transactions")

    df["transaction_date"]      = pd.to_datetime(df["transaction_date"],      format="%Y%m%d", errors="coerce")
    df["membership_expire_date"] = pd.to_datetime(df["membership_expire_date"], format="%Y%m%d", errors="coerce")

    # Discount amount
    df["discount"] = df["plan_list_price"] - df["actual_amount_paid"]
    df["discount_pct"] = np.where(
        df["plan_list_price"] > 0,
        df["discount"] / df["plan_list_price"],
        0.0,
    ).astype(np.float32)

    log.info(f"  Transactions shape: {df.shape}")
    return df


def _parse_and_engineer_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and compute per-row derived columns on a log chunk/frame."""
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["total_songs"] = (
        df["num_25"] + df["num_50"] + df["num_75"] +
        df["num_985"] + df["num_100"]
    )
    df["completion_rate"] = np.where(
        df["total_songs"] > 0,
        (df["num_985"] + df["num_100"]) / df["total_songs"],
        0.0,
    ).astype(np.float32)
    return df


def _agg_log_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a log dataframe to per-user-per-day summary (reduces size ~100x)."""
    return df.groupby(["msno", "date"], sort=False).agg(
        total_songs    = ("total_songs", "sum"),
        num_unq        = ("num_unq", "sum"),
        total_secs     = ("total_secs", "sum"),
        completion_rate= ("completion_rate", "mean"),
    ).reset_index()


def load_user_logs(
    path_train: Path = RAW / "user_logs.csv",
    path_v2:    Path = RAW / "user_logs_v2.csv",
    chunksize:  int  = 200_000,
) -> pd.DataFrame:
    """
    Stream both log files and aggregate to per-user features without
    ever materialising more than one chunk in RAM at a time.

    Strategy:
      - Maintain a running per-user accumulator dict.
      - For each chunk: compute derived cols, then update accumulator.
      - No concat of raw chunks. No final groupby on a large frame.
      - Peak RAM: ~1.5-2 GB regardless of input file size.
    """
    import gc

    dtype_map = {
        "msno":       str,
        "num_25":     np.int32,
        "num_50":     np.int32,
        "num_75":     np.int32,
        "num_985":    np.int32,
        "num_100":    np.int32,
        "num_unq":    np.int32,
        "total_secs": np.float32,
    }

    # Accumulator: msno -> [active_days, total_songs, num_unq, total_secs,
    #                        completion_sum, completion_count, dates_set]
    # We track dates as a set to count unique active days correctly.
    acc = {}

    def process_file(path, label):
        total_rows = 0
        chunk_count = 0
        log.info(f"Streaming {label} ...")
        for chunk in pd.read_csv(path, dtype=dtype_map, chunksize=chunksize):
            # Parse date
            chunk["date"] = pd.to_datetime(
                chunk["date"], format="%Y%m%d", errors="coerce"
            )
            chunk = chunk.dropna(subset=["date"])

            # Derived cols
            chunk["total_songs"] = (
                chunk["num_25"] + chunk["num_50"] + chunk["num_75"] +
                chunk["num_985"] + chunk["num_100"]
            )
            chunk["completion_rate"] = np.where(
                chunk["total_songs"] > 0,
                (chunk["num_985"] + chunk["num_100"]) / chunk["total_songs"],
                0.0,
            ).astype(np.float32)

            # Update accumulator — one pass, no groupby on large frame
            for row in chunk.itertuples(index=False):
                msno = row.msno
                if msno not in acc:
                    acc[msno] = {
                        "dates":            set(),
                        "total_songs":      0.0,
                        "num_unq":          0.0,
                        "total_secs":       0.0,
                        "completion_sum":   0.0,
                        "completion_count": 0,
                        "last_date":        row.date,
                    }
                a = acc[msno]
                a["dates"].add(row.date)
                a["total_songs"]      += row.total_songs
                a["num_unq"]          += row.num_unq
                a["total_secs"]       += row.total_secs
                a["completion_sum"]   += row.completion_rate
                a["completion_count"] += 1
                if row.date > a["last_date"]:
                    a["last_date"] = row.date

            total_rows += len(chunk)
            chunk_count += 1
            if chunk_count % 20 == 0:
                log.info(f"  ... {label}: {total_rows:,} rows processed")

        log.info(f"  {label} done — {total_rows:,} rows, {len(acc):,} unique users")

    process_file(path_train, "user_logs.csv")
    process_file(path_v2,    "user_logs_v2.csv")
    gc.collect()

    log.info("Building per-user feature frame from accumulator ...")
    records = []
    for msno, a in acc.items():
        dates = sorted(a["dates"])
        records.append({
            "msno":             msno,
            "total_active_days": len(dates),
            "total_songs_ever": a["total_songs"],
            "total_unq_ever":   a["num_unq"],
            "total_secs_ever":  a["total_secs"],
            "avg_completion":   a["completion_sum"] / a["completion_count"] if a["completion_count"] > 0 else 0.0,
            "last_active_date": a["last_date"],
            "first_active_date": dates[0] if dates else pd.NaT,
        })

    df_full = pd.DataFrame(records)
    log.info(f"  Full history frame: {df_full.shape}")
    log.info(f"  Date range: {df_full['last_active_date'].min()} → {df_full['last_active_date'].max()}")
    return df_full, acc


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, name: str) -> None:
    """Check expected columns exist and log null rates for key fields."""
    expected = EXPECTED_COLS[name]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}")

    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > 0.2]
    if not high_null.empty:
        log.warning(f"[{name}] High null rate columns (>20%):\n{high_null.to_string()}")


# ---------------------------------------------------------------------------
# 3. Transaction aggregation
# ---------------------------------------------------------------------------

def aggregate_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse transaction history to one row per user.

    Features produced:
    - Last transaction state: payment method, plan days, price, auto-renew, cancel
    - Transaction count
    - Auto-renew ratio across all transactions
    - Cancel ratio across all transactions
    - Mean discount percentage
    - Days between last transaction and membership expiry
    """
    log.info("Aggregating transactions to per-user features ...")

    # Sort so last() gives us the most recent transaction
    tx = tx.sort_values(["msno", "transaction_date"])

    # Last transaction features
    last = tx.groupby("msno").last().reset_index()
    last = last.rename(columns={
        "payment_method_id":    "last_payment_method",
        "payment_plan_days":    "last_plan_days",
        "plan_list_price":      "last_plan_price",
        "actual_amount_paid":   "last_amount_paid",
        "is_auto_renew":        "last_auto_renew",
        "is_cancel":            "last_is_cancel",
        "transaction_date":     "last_transaction_date",
        "membership_expire_date": "last_expire_date",
        "discount":             "last_discount",
        "discount_pct":         "last_discount_pct",
    })

    # Aggregate statistics across all transactions
    agg = tx.groupby("msno").agg(
        transaction_count       = ("transaction_date", "count"),
        auto_renew_ratio        = ("is_auto_renew", "mean"),
        cancel_ratio            = ("is_cancel", "mean"),
        mean_discount_pct       = ("discount_pct", "mean"),
        mean_plan_days          = ("payment_plan_days", "mean"),
        total_amount_paid       = ("actual_amount_paid", "sum"),
    ).reset_index()

    result = last.merge(agg, on="msno", how="left")

    # Days from last transaction to expiry — negative = expired before renewing
    result["days_to_expiry"] = (
        result["last_expire_date"] - result["last_transaction_date"]
    ).dt.days

    log.info(f"  Transaction features shape: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# 4. User log aggregation
# ---------------------------------------------------------------------------

def aggregate_user_logs(
    log_output: tuple,
    cutoff_date: str = "2017-03-31"
) -> pd.DataFrame:
    """
    Build rolling window features from the accumulator produced by load_user_logs.

    Receives (df_full, acc) where:
      - df_full: per-user summary frame (full history)
      - acc: dict of msno -> {dates: set, ...} for window calculations

    Window features (7d, 30d, 90d before cutoff):
      - active_days, total_songs, unique_songs, total_secs, avg_completion

    Trend signals:
      - trend_7d_30d: recent vs monthly activity ratio (< 1 = declining)
      - trend_30d_90d: monthly vs quarterly activity ratio
      - days_since_last: recency signal
    """
    import gc
    df_full, acc = log_output
    cutoff = pd.Timestamp(cutoff_date)
    log.info("Computing rolling window features from accumulator ...")

    windows = {"7d": 7, "30d": 30, "90d": 90}
    records = []

    for msno, a in acc.items():
        dates = a["dates"]
        row = {"msno": msno}

        for label, days in windows.items():
            window_start = cutoff - pd.Timedelta(days=days)
            window_dates = [d for d in dates if window_start < d <= cutoff]
            row[f"active_days_{label}"] = len(window_dates)

        # Days since last activity
        last = a["last_date"]
        row["days_since_last"] = (cutoff - last).days if pd.notna(last) else 999

        records.append(row)

    log.info("  Window records built, creating DataFrame ...")
    window_df = pd.DataFrame(records)
    gc.collect()

    # Merge with full history features
    log_features = df_full.merge(window_df, on="msno", how="outer")

    # Trend signals — declining activity is a strong churn predictor
    log_features["trend_7d_30d"] = np.where(
        log_features["active_days_30d"] > 0,
        log_features["active_days_7d"] / (log_features["active_days_30d"] / 4.3),
        0.0,
    ).astype(np.float32)

    log_features["trend_30d_90d"] = np.where(
        log_features["active_days_90d"] > 0,
        log_features["active_days_30d"] / (log_features["active_days_90d"] / 3.0),
        0.0,
    ).astype(np.float32)

    log_features = log_features.fillna(0)
    log.info(f"  Log features shape: {log_features.shape}")
    return log_features


# ---------------------------------------------------------------------------
# 5. Master join
# ---------------------------------------------------------------------------

def build_master_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build the full analytical dataset by joining all sources.

    If data/processed/master.parquet already exists, loads from cache
    unless force_rebuild=True. This avoids re-processing the 30GB+1.4GB
    log files on every run.

    Returns:
        pd.DataFrame: One row per user, all features + is_churn label.
    """
    cache_path = PROCESSED / "master.parquet"

    if cache_path.exists() and not force_rebuild:
        log.info(f"Loading cached master dataset from {cache_path}")
        df = pd.read_parquet(cache_path)
        log.info(f"  Loaded shape: {df.shape}")
        return df

    log.info("Building master dataset from scratch ...")

    # Load all sources
    train    = load_train()
    members  = load_members()
    tx       = load_transactions()
    logs     = load_user_logs()  # loads user_logs.csv + user_logs_v2.csv

    # Aggregate
    tx_agg   = aggregate_transactions(tx)
    log_agg  = aggregate_user_logs(logs)  # logs is (df_full, acc) tuple

    # Free memory — we no longer need the raw frames
    del tx, logs
    import gc; gc.collect()

    # Join everything onto train labels (left join — keep all labelled users)
    log.info("Joining all sources ...")
    df = train.merge(members,  on="msno", how="left")
    df = df.merge(tx_agg,      on="msno", how="left")
    df = df.merge(log_agg,     on="msno", how="left")

    # Final shape and churn rate check
    log.info(f"  Master dataset shape: {df.shape}")
    log.info(f"  Churn rate: {df.is_churn.mean():.3%}")
    log.info(f"  Null rates (top 10):\n{df.isnull().mean().sort_values(ascending=False).head(10).to_string()}")

    # Cache to parquet for fast reloads
    df.to_parquet(cache_path, index=False)
    log.info(f"  Saved to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# 6. Quick sanity check — run this file directly to test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_master_dataset(force_rebuild=True)
    print("\n=== Master Dataset ===")
    print(f"Shape:      {df.shape}")
    print(f"Churn rate: {df.is_churn.mean():.3%}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nSample:\n{df.head(3).to_string()}")
