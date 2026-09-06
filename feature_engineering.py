"""
feature_engineering.py
======================
KKBox Churn Intelligence Platform — Layer 2: Feature Engineering

Transforms the raw master dataset into a model-ready feature matrix.

Features produced:
  - Cleaned demographics (age, city, registration channel)
  - Behavioural ratios (completion rate, listening intensity)
  - Temporal decay signals (activity trend across windows)
  - Transaction risk signals (cancel ratio, auto-renew, discount dependency)
  - Interaction features (engagement × contract risk)
  - Days-based recency features

Usage:
    from src.feature_engineering import build_features
    X, y, feature_names = build_features(df)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# 1. Main entry point
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame):
    """
    Transform master dataset into model-ready features.

    Args:
        df: Master dataset from data_loader.build_master_dataset()

    Returns:
        X:             Feature DataFrame (no target column)
        y:             Target Series (is_churn)
        feature_names: List of feature column names
    """
    log.info("Building feature matrix ...")
    df = df.copy()

    df = _clean_demographics(df)
    df = _build_behavioural_features(df)
    df = _build_transaction_features(df)
    df = _build_interaction_features(df)
    df = _build_temporal_features(df)

    # Drop columns not used in modelling
    drop_cols = [
        "msno", "is_churn",
        "registration_init_time", "last_transaction_date",
        "last_expire_date", "gender",  # gender has 65% null — dropped
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df["is_churn"].astype(np.int8)
    X = df.drop(columns=drop_cols)

    # Final null fill — any remaining NaN becomes 0
    null_counts = X.isnull().sum()
    if null_counts.any():
        log.warning(f"Filling {null_counts[null_counts > 0].to_dict()} nulls with 0")
        X = X.fillna(0)

    feature_names = list(X.columns)
    log.info(f"  Feature matrix shape: {X.shape}")
    log.info(f"  Target churn rate:    {y.mean():.3%}")
    log.info(f"  Features: {feature_names}")

    return X, y, feature_names


# ---------------------------------------------------------------------------
# 2. Demographics
# ---------------------------------------------------------------------------

def _clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean member demographic fields.

    - Age: cap outliers (valid range 10-80), fill missing with median
    - City: treat as categorical integer (already encoded)
    - registered_via: registration channel (categorical)
    - days_since_registration: tenure proxy
    """
    # Age cleaning
    if "bd" in df.columns:
        df["age"] = df["bd"].where((df["bd"] >= 10) & (df["bd"] <= 80), np.nan)
        df["age"] = df["age"].fillna(df["age"].median())
        df.drop(columns=["bd"], inplace=True)

    # Registration channel — fill missing with mode
    if "registered_via" in df.columns:
        df["registered_via"] = df["registered_via"].fillna(
            df["registered_via"].mode()[0]
        ).astype(np.int8)

    # City — fill missing with 0 (unknown)
    if "city" in df.columns:
        df["city"] = df["city"].fillna(0).astype(np.int8)

    # Tenure from registration date
    if "registration_init_time" in df.columns:
        cutoff = pd.Timestamp("2017-03-31")
        reg = pd.to_datetime(df["registration_init_time"], errors="coerce")
        df["days_since_registration"] = (cutoff - reg).dt.days.clip(lower=0).fillna(0)

    return df


# ---------------------------------------------------------------------------
# 3. Behavioural features
# ---------------------------------------------------------------------------

def _build_behavioural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer listening behaviour features from log aggregates.

    Key signals:
    - listening_intensity: average seconds per active day
    - completion_rate already in master (avg_completion)
    - activity_consistency: ratio of active days to total possible days
    - dropout_risk: days_since_last > 14 (binary flag)
    """
    # Listening intensity — how engaged is the user on days they do listen
    df["listening_intensity"] = np.where(
        df["total_active_days"] > 0,
        df["total_secs_ever"] / df["total_active_days"],
        0.0,
    ).astype(np.float32)

    # 30-day listening intensity
    df["listening_intensity_30d"] = np.where(
        df["active_days_30d"] > 0,
        df["total_secs_30d"] / df["active_days_30d"],
        0.0,
    ).astype(np.float32)

    # Activity consistency — out of 90 possible days, how many were active?
    df["activity_consistency_90d"] = (
        df["active_days_90d"] / 90.0
    ).clip(0, 1).astype(np.float32)

    # Activity consistency — 30 day
    df["activity_consistency_30d"] = (
        df["active_days_30d"] / 30.0
    ).clip(0, 1).astype(np.float32)

    # Dropout risk flag — hasn't listened in 2+ weeks
    df["dropout_risk"] = (df["days_since_last"] > 14).astype(np.int8)

    # Silent user flag — zero activity in March 2017 (strongest churn signal)
    df["silent_march"] = (df["active_days_30d"] == 0).astype(np.int8)

    # Songs per second — proxy for song length / skip behaviour
    df["songs_per_hour"] = np.where(
        df["total_secs_ever"] > 0,
        df["total_songs_ever"] / (df["total_secs_ever"] / 3600),
        0.0,
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# 4. Transaction features
# ---------------------------------------------------------------------------

def _build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer risk signals from transaction history.

    Key signals:
    - is_auto_renew: strongest single predictor of non-churn
    - cancel_ratio: proportion of transactions that were cancellations
    - discount_dependency: relies on discounts to stay subscribed
    - plan_price_tier: binned price tier (low/mid/high)
    - short_plan_risk: on monthly plan (30 days) — higher churn risk
    """
    # Auto-renew is off — strong churn signal
    if "last_auto_renew" in df.columns:
        df["last_auto_renew"] = df["last_auto_renew"].fillna(0).astype(np.int8)

    # Last cancel flag
    if "last_is_cancel" in df.columns:
        df["last_is_cancel"] = df["last_is_cancel"].fillna(0).astype(np.int8)

    # Discount dependency — paying less than list price consistently
    if "mean_discount_pct" in df.columns:
        df["discount_dependent"] = (df["mean_discount_pct"] > 0.1).astype(np.int8)

    # Short plan risk — monthly plans churn more than annual
    if "last_plan_days" in df.columns:
        df["last_plan_days"] = df["last_plan_days"].fillna(30)
        df["short_plan_risk"] = (df["last_plan_days"] <= 30).astype(np.int8)
        df["long_plan_flag"]  = (df["last_plan_days"] >= 365).astype(np.int8)

    # Price tier (binned)
    if "last_plan_price" in df.columns:
        df["last_plan_price"] = df["last_plan_price"].fillna(
            df["last_plan_price"].median()
        )
        df["price_tier"] = pd.cut(
            df["last_plan_price"],
            bins=[0, 100, 200, 500, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float).fillna(0).astype(np.int8)

    # Transaction frequency — how many times have they transacted?
    if "transaction_count" in df.columns:
        df["transaction_count"] = df["transaction_count"].fillna(1)
        df["high_transaction_count"] = (df["transaction_count"] > 5).astype(np.int8)

    # Fill remaining transaction nulls with safe defaults
    tx_cols = [
        "cancel_ratio", "auto_renew_ratio", "mean_discount_pct",
        "total_amount_paid", "mean_plan_days", "days_to_expiry",
        "last_discount_pct", "last_discount", "last_amount_paid",
        "last_plan_price", "last_payment_method",
    ]
    for col in tx_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 5. Interaction features
# ---------------------------------------------------------------------------

def _build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features capture combined risk signals that are more
    predictive than either feature alone.

    Key interactions:
    - silent × cancel: didn't listen AND cancelled = very high risk
    - short_plan × dropout: monthly plan + recent inactivity
    - completion × activity: engaged listeners on active days
    - auto_renew × activity: auto-renew users who stopped listening
    """
    # High risk combo: silent in March + last transaction was a cancel
    if "silent_march" in df.columns and "last_is_cancel" in df.columns:
        df["silent_and_cancelled"] = (
            df["silent_march"] & df["last_is_cancel"]
        ).astype(np.int8)

    # Short plan + dropout risk
    if "short_plan_risk" in df.columns and "dropout_risk" in df.columns:
        df["short_plan_dropout"] = (
            df["short_plan_risk"] & df["dropout_risk"]
        ).astype(np.int8)

    # Engagement score: completion × listening intensity (normalised)
    if "avg_completion" in df.columns and "listening_intensity" in df.columns:
        df["engagement_score"] = (
            df["avg_completion"] *
            df["listening_intensity"].clip(upper=df["listening_intensity"].quantile(0.99))
        ).astype(np.float32)

    # Auto-renew but declining usage — at risk despite auto-renew
    if "last_auto_renew" in df.columns and "trend_7d_30d" in df.columns:
        df["autorenew_declining"] = (
            (df["last_auto_renew"] == 1) & (df["trend_7d_30d"] < 0.5)
        ).astype(np.int8)

    return df


# ---------------------------------------------------------------------------
# 6. Temporal features
# ---------------------------------------------------------------------------

def _build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features capture time-based patterns.

    - days_to_expiry: how close is the subscription to expiring?
    - expiry_imminent: expires within 7 days (binary)
    - long_tenure: registered more than 2 years ago (loyal user)
    - new_user: registered less than 90 days ago (high churn risk)
    """
    # Expiry imminence
    if "days_to_expiry" in df.columns:
        df["expiry_imminent"] = (
            (df["days_to_expiry"] >= 0) & (df["days_to_expiry"] <= 7)
        ).astype(np.int8)
        df["already_expired"] = (df["days_to_expiry"] < 0).astype(np.int8)

    # Tenure segments
    if "days_since_registration" in df.columns:
        df["long_tenure"]  = (df["days_since_registration"] > 730).astype(np.int8)
        df["new_user"]     = (df["days_since_registration"] < 90).astype(np.int8)

    # Clip extreme trend values — ratios > 5 are noise
    if "trend_7d_30d" in df.columns:
        df["trend_7d_30d"]  = df["trend_7d_30d"].clip(0, 5).fillna(0)
    if "trend_30d_90d" in df.columns:
        df["trend_30d_90d"] = df["trend_30d_90d"].clip(0, 5).fillna(0)

    return df


# ---------------------------------------------------------------------------
# 7. Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data_loader import build_master_dataset
    df = build_master_dataset()
    X, y, features = build_features(df)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Churn rate:     {y.mean():.3%}")
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        print(f"  {f}")
    print(f"\nNull check: {X.isnull().sum().sum()} total nulls")
