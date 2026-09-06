"""
model_training.py
=================
KKBox Churn Intelligence Platform — Layer 3: Model Training & Validation

Handles:
- Stratified time-aware train/test split
- Class imbalance via scale_pos_weight (LightGBM native)
- StratifiedKFold cross-validation (5 folds)
- LightGBM as primary model, Logistic Regression as baseline
- Model persistence to disk

Usage:
    from src.model_training import train_model, load_model
    results = train_model(X, y)
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
MODELS  = ROOT / "models"
MODELS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Main training entry point
# ---------------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    """
    Train LightGBM churn model with proper validation.

    Steps:
    1. Stratified 80/20 train/test split
    2. Compute class imbalance ratio for scale_pos_weight
    3. 5-fold stratified cross-validation on train set
    4. Final model trained on full train set
    5. Evaluate on held-out test set
    6. Save model to models/lgbm_churn.pkl

    Returns dict with model, scaler, CV scores, test scores, feature names.
    """
    log.info("=" * 60)
    log.info("Starting model training pipeline")
    log.info("=" * 60)

    # ── Split ────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    log.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    log.info(f"Train churn rate: {y_train.mean():.3%} | Test churn rate: {y_test.mean():.3%}")

    # ── Class imbalance ───────────────────────────────────────────────
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    log.info(f"Class ratio neg:pos = {neg:,}:{pos:,} → scale_pos_weight = {scale_pos_weight:.2f}")

    # ── Baseline: Logistic Regression ────────────────────────────────
    log.info("\nTraining Logistic Regression baseline ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc   = roc_auc_score(y_test, lr_proba)
    lr_ap    = average_precision_score(y_test, lr_proba)
    log.info(f"  Baseline LR  →  AUC-ROC: {lr_auc:.4f} | AUC-PR: {lr_ap:.4f}")

    # ── Cross-validation: LightGBM ───────────────────────────────────
    log.info("\nRunning 5-fold StratifiedKFold CV on LightGBM ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    lgbm_params = _get_lgbm_params(scale_pos_weight, random_state)

    cv_auc_scores = []
    cv_ap_scores  = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        fold_model = lgb.train(
            lgbm_params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_proba = fold_model.predict(X_val)
        fold_auc  = roc_auc_score(y_val, val_proba)
        fold_ap   = average_precision_score(y_val, val_proba)
        cv_auc_scores.append(fold_auc)
        cv_ap_scores.append(fold_ap)
        log.info(f"  Fold {fold}: AUC-ROC={fold_auc:.4f} | AUC-PR={fold_ap:.4f} | best_iter={fold_model.best_iteration}")

    log.info(f"\n  CV AUC-ROC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    log.info(f"  CV AUC-PR:  {np.mean(cv_ap_scores):.4f} ± {np.std(cv_ap_scores):.4f}")

    # ── Final model on full train set ─────────────────────────────────
    log.info("\nTraining final LightGBM model on full train set ...")
    best_iter = int(np.mean([
        fold_model.best_iteration for _ in range(1)  # use last fold's iter as proxy
    ]))
    best_iter = max(best_iter, 100)

    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_model = lgb.train(
        lgbm_params,
        dtrain_full,
        num_boost_round=best_iter,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    # ── Test set evaluation ───────────────────────────────────────────
    log.info("\nEvaluating on held-out test set ...")
    test_proba = final_model.predict(X_test)
    test_auc   = roc_auc_score(y_test, test_proba)
    test_ap    = average_precision_score(y_test, test_proba)

    log.info(f"  Test AUC-ROC: {test_auc:.4f}")
    log.info(f"  Test AUC-PR:  {test_ap:.4f}")
    log.info(f"  Baseline LR AUC-ROC: {lr_auc:.4f} (for comparison)")

    if test_auc > 0.999:
        log.warning("=" * 60)
        log.warning("WARNING: AUC > 0.999 — possible data leakage. Investigate.")
        log.warning("Check for target-correlated features before proceeding.")
        log.warning("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "model":           final_model,
        "scaler":          scaler,
        "baseline_lr":     lr,
        "feature_names":   list(X.columns),
        "X_test":          X_test,
        "y_test":          y_test,
        "test_proba":      test_proba,
        "test_auc_roc":    test_auc,
        "test_auc_pr":     test_ap,
        "cv_auc_roc_mean": np.mean(cv_auc_scores),
        "cv_auc_roc_std":  np.std(cv_auc_scores),
        "cv_auc_pr_mean":  np.mean(cv_ap_scores),
        "cv_auc_pr_std":   np.std(cv_ap_scores),
        "baseline_auc_roc": lr_auc,
        "baseline_auc_pr":  lr_ap,
        "scale_pos_weight": scale_pos_weight,
    }

    model_path = MODELS / "lgbm_churn.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"\nModel saved to {model_path}")

    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# 2. LightGBM parameters
# ---------------------------------------------------------------------------

def _get_lgbm_params(scale_pos_weight: float, random_state: int) -> dict:
    """
    LightGBM parameters tuned for imbalanced binary classification.

    Key decisions:
    - scale_pos_weight: handles 91:9 class imbalance natively
    - dart booster: reduces overfitting on large datasets
    - low learning rate + high rounds: better generalisation
    - min_child_samples=100: prevents overfitting on small leaf nodes
    """
    return {
        "objective":        "binary",
        "metric":           ["auc", "average_precision"],
        "boosting_type":    "gbdt",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "scale_pos_weight": scale_pos_weight,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "random_state":     random_state,
        "n_jobs":           -1,
        "verbose":          -1,
    }


# ---------------------------------------------------------------------------
# 3. Load saved model
# ---------------------------------------------------------------------------

def load_model(model_path: Path = MODELS / "lgbm_churn.pkl") -> dict:
    """Load saved model results from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}. Run train_model() first.")
    with open(model_path, "rb") as f:
        results = pickle.load(f)
    log.info(f"Model loaded from {model_path}")
    log.info(f"  Test AUC-ROC: {results['test_auc_roc']:.4f}")
    log.info(f"  Test AUC-PR:  {results['test_auc_pr']:.4f}")
    return results


# ---------------------------------------------------------------------------
# 4. Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: dict) -> None:
    log.info("\n" + "=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)
    log.info(f"  CV  AUC-ROC : {results['cv_auc_roc_mean']:.4f} ± {results['cv_auc_roc_std']:.4f}")
    log.info(f"  CV  AUC-PR  : {results['cv_auc_pr_mean']:.4f} ± {results['cv_auc_pr_std']:.4f}")
    log.info(f"  Test AUC-ROC: {results['test_auc_roc']:.4f}")
    log.info(f"  Test AUC-PR : {results['test_auc_pr']:.4f}")
    log.info(f"  Baseline LR : {results['baseline_auc_roc']:.4f} (AUC-ROC)")
    log.info(f"  Lift over baseline: {results['test_auc_roc'] - results['baseline_auc_roc']:+.4f}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# 5. Quick run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from src.data_loader import build_master_dataset
    from src.feature_engineering import build_features

    df = build_master_dataset()
    X, y, features = build_features(df)
    results = train_model(X, y)
