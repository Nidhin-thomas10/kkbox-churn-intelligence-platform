"""
shap_analysis.py
================
KKBox Churn Intelligence Platform — Layer 4: Interpretability

Produces SHAP-based explanations at both global and local level.

Global:
- SHAP summary plot (beeswarm)
- SHAP bar plot (mean absolute impact)
- SHAP dependence plots for top features

Local:
- SHAP waterfall plot for any individual customer
- Force plot for individual prediction

Usage:
    from src.shap_analysis import run_shap_analysis, explain_customer
    shap_results = run_shap_analysis(results)
    explain_customer(shap_results, customer_idx=42)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS  = ROOT / "models"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Main SHAP analysis entry point
# ---------------------------------------------------------------------------

def run_shap_analysis(
    results: dict,
    sample_size: int = 5000,
    save_plots: bool = True,
) -> dict:
    """
    Run full SHAP analysis on trained LightGBM model.

    Uses a random sample for global plots (SHAP is O(n) but plots
    get cluttered above 5k points). Full test set used for local
    explanations.

    Args:
        results:     Output from model_training.train_model()
        sample_size: Number of test samples to use for global SHAP plots
        save_plots:  Save figures to reports/

    Returns:
        shap_results dict with explainer, shap_values, sample DataFrame
    """
    model    = results["model"]
    X_test   = results["X_test"]
    features = results["feature_names"]

    log.info("Initialising SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    # Sample for global plots
    sample_size = min(sample_size, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42).reset_index(drop=True)

    log.info(f"Computing SHAP values on {sample_size:,} samples ...")
    shap_values = explainer.shap_values(X_sample)

    # LightGBM binary returns list [neg_class, pos_class] or single array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class (churn)

    log.info(f"  SHAP values shape: {shap_values.shape}")

    # Mean absolute SHAP per feature
    mean_shap = pd.DataFrame({
        "feature":    features,
        "mean_shap":  np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)

    log.info("\nTop 15 features by mean |SHAP|:")
    log.info(mean_shap.head(15).to_string(index=False))

    if save_plots:
        _plot_shap_summary(shap_values, X_sample, features)
        _plot_shap_bar(mean_shap)
        _plot_shap_dependence(shap_values, X_sample, mean_shap, features)

    shap_results = {
        "explainer":    explainer,
        "shap_values":  shap_values,
        "X_sample":     X_sample,
        "X_test":       X_test,
        "features":     features,
        "mean_shap":    mean_shap,
        "model":        model,
    }

    # Save SHAP results for dashboard use
    shap_path = MODELS / "shap_results.pkl"
    with open(shap_path, "wb") as f:
        pickle.dump({
            "explainer": explainer,
            "mean_shap": mean_shap,
            "features":  features,
        }, f)
    log.info(f"\nSHAP explainer saved to {shap_path}")

    return shap_results


# ---------------------------------------------------------------------------
# 2. Individual customer explanation
# ---------------------------------------------------------------------------

def explain_customer(
    shap_results: dict,
    customer_idx: int = 0,
    X_custom: pd.DataFrame = None,
    save: bool = True,
) -> dict:
    """
    Generate SHAP waterfall explanation for a single customer.

    Args:
        shap_results: Output from run_shap_analysis()
        customer_idx: Row index in X_test (or X_custom if provided)
        X_custom:     Optional custom DataFrame row to explain
        save:         Save waterfall plot to reports/

    Returns:
        dict with customer features, shap values, churn probability
    """
    explainer = shap_results["explainer"]
    features  = shap_results["features"]
    model     = shap_results["model"]

    # Get the customer row
    if X_custom is not None:
        X_row = X_custom
    else:
        X_row = shap_results["X_test"].iloc[[customer_idx]]

    # Compute SHAP for this customer
    sv = explainer.shap_values(X_row)
    if isinstance(sv, list):
        sv = sv[1]

    churn_prob  = float(model.predict(X_row)[0])
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[1]

    # Build explanation DataFrame
    explanation = pd.DataFrame({
        "feature": features,
        "value":   X_row.values[0],
        "shap":    sv[0],
    }).sort_values("shap", key=abs, ascending=False)

    log.info(f"\nCustomer {customer_idx} — Churn probability: {churn_prob:.3f}")
    log.info(f"Base rate (expected value): {expected_val:.3f}")
    log.info("\nTop drivers:")
    log.info(explanation.head(10).to_string(index=False))

    # Waterfall plot
    shap_exp = shap.Explanation(
        values       = sv[0],
        base_values  = expected_val,
        data         = X_row.values[0],
        feature_names= features,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_exp, max_display=15, show=False)
    plt.title(f"Customer Churn Explanation — P(churn) = {churn_prob:.3f}", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / f"waterfall_customer_{customer_idx}.png",
                    dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    return {
        "customer_idx": customer_idx,
        "churn_prob":   churn_prob,
        "explanation":  explanation,
        "shap_values":  sv[0],
        "expected_val": expected_val,
    }


# ---------------------------------------------------------------------------
# 3. Global SHAP plots
# ---------------------------------------------------------------------------

def _plot_shap_summary(shap_values, X_sample, features, save: bool = True):
    """Beeswarm summary plot — shows direction and magnitude of each feature."""
    log.info("Generating SHAP summary (beeswarm) plot ...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=features,
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary Plot — Feature Impact on Churn Probability", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_shap_bar(mean_shap: pd.DataFrame, top_n: int = 20, save: bool = True):
    """Bar plot of mean absolute SHAP values."""
    log.info("Generating SHAP bar plot ...")
    top = mean_shap.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top["feature"][::-1], top["mean_shap"][::-1], color="#7c3aed")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features — Global SHAP Importance", fontsize=13)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_shap_dependence(shap_values, X_sample, mean_shap, features, top_n: int = 4, save: bool = True):
    """
    Partial dependence plots for top N features.
    Shows how SHAP value changes as feature value changes.
    """
    log.info(f"Generating SHAP dependence plots for top {top_n} features ...")
    top_features = mean_shap["feature"].head(top_n).tolist()
    feature_idx  = {f: i for i, f in enumerate(features)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        idx = feature_idx[feat]
        ax  = axes[i]
        ax.scatter(
            X_sample[feat],
            shap_values[:, idx],
            alpha=0.3, s=8, color="#2563eb",
        )
        ax.axhline(0, color="gray", linestyle="--", lw=1)
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=12)

    plt.suptitle("SHAP Dependence Plots — Top 4 Features", fontsize=13, y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 4. Load saved SHAP results
# ---------------------------------------------------------------------------

def load_shap_results(path: Path = MODELS / "shap_results.pkl") -> dict:
    """Load saved SHAP explainer from disk."""
    if not path.exists():
        raise FileNotFoundError(f"No SHAP results at {path}. Run run_shap_analysis() first.")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 5. Quick run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.model_training import load_model
    results     = load_model()
    shap_results = run_shap_analysis(results)
    explain_customer(shap_results, customer_idx=0)
    explain_customer(shap_results, customer_idx=1)
