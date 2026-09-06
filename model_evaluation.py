"""
model_evaluation.py
===================
KKBox Churn Intelligence Platform — Layer 3b: Model Evaluation

Produces all evaluation artefacts:
- Confusion matrix at business-optimal threshold
- Precision-Recall curve (primary metric for imbalanced data)
- ROC curve
- Threshold analysis — find optimal F1 / precision / recall tradeoff
- Classification report
- Feature importance (LightGBM native gain)

Usage:
    from src.model_evaluation import evaluate_model, find_optimal_threshold
    evaluate_model(results)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_model(results: dict, save_plots: bool = True) -> dict:
    """
    Run full evaluation suite on trained model results.

    Args:
        results:    Output dict from model_training.train_model()
        save_plots: Save figures to reports/ directory

    Returns:
        eval_metrics dict with all computed metrics and optimal threshold
    """
    y_test     = results["y_test"]
    test_proba = results["test_proba"]
    features   = results["feature_names"]
    model      = results["model"]

    log.info("Running full model evaluation ...")

    # Find business-optimal threshold
    threshold, threshold_metrics = find_optimal_threshold(y_test, test_proba)
    y_pred = (test_proba >= threshold).astype(int)

    log.info(f"\nOptimal threshold: {threshold:.3f}")
    log.info(f"  Precision: {threshold_metrics['precision']:.4f}")
    log.info(f"  Recall:    {threshold_metrics['recall']:.4f}")
    log.info(f"  F1:        {threshold_metrics['f1']:.4f}")

    # Classification report
    log.info("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    log.info("\n" + report)

    # Confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log.info(f"Confusion Matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

    # Feature importance
    importance_df = _get_feature_importance(model, features)
    log.info(f"\nTop 15 features by gain:")
    log.info(importance_df.head(15).to_string(index=False))

    # Save all plots
    if save_plots:
        _plot_confusion_matrix(cm, threshold, save=True)
        _plot_pr_curve(y_test, test_proba, results["test_auc_pr"], save=True)
        _plot_roc_curve(y_test, test_proba, results["test_auc_roc"], save=True)
        _plot_threshold_analysis(y_test, test_proba, threshold, save=True)
        _plot_feature_importance(importance_df, save=True)
        log.info(f"\nPlots saved to {REPORTS}/")

    eval_metrics = {
        "threshold":        threshold,
        "precision":        threshold_metrics["precision"],
        "recall":           threshold_metrics["recall"],
        "f1":               threshold_metrics["f1"],
        "auc_roc":          results["test_auc_roc"],
        "auc_pr":           results["test_auc_pr"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "feature_importance": importance_df,
        "classification_report": report,
    }

    _print_eval_summary(eval_metrics)
    return eval_metrics


# ---------------------------------------------------------------------------
# 2. Optimal threshold finder
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    strategy: str = "f1",
) -> tuple:
    """
    Find the classification threshold that maximises the chosen strategy.

    Strategy options:
    - 'f1'        : maximise F1 score (balanced precision/recall)
    - 'precision' : maximise precision at recall >= 0.5
    - 'recall'    : maximise recall at precision >= 0.5

    In a churn context, 'f1' is usually right for portfolio purposes.
    In production, the business decides: false negatives (missed churners)
    typically cost more than false positives (unnecessary interventions).

    Returns:
        threshold:        float, optimal cutoff
        threshold_metrics: dict with precision, recall, f1 at that threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    if strategy == "f1":
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )
        best_idx  = np.argmax(f1_scores[:-1])
        threshold = float(thresholds[best_idx])

    elif strategy == "precision":
        valid = recalls[:-1] >= 0.5
        if valid.any():
            best_idx  = np.argmax(precisions[:-1][valid])
            threshold = float(thresholds[valid][best_idx])
        else:
            threshold = 0.5

    elif strategy == "recall":
        valid = precisions[:-1] >= 0.5
        if valid.any():
            best_idx  = np.argmax(recalls[:-1][valid])
            threshold = float(thresholds[valid][best_idx])
        else:
            threshold = 0.5

    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    return threshold, metrics


# ---------------------------------------------------------------------------
# 3. Feature importance
# ---------------------------------------------------------------------------

def _get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract LightGBM feature importance by gain."""
    importance = model.feature_importance(importance_type="gain")
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
    return df


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, threshold: float, save: bool = True):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.3f})", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_pr_curve(y_true, y_proba, auc_pr: float, save: bool = True):
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls, precisions, color="#2563eb", lw=2,
            label=f"LightGBM (AUC-PR = {auc_pr:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1.5,
               label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_roc_curve(y_true, y_proba, auc_roc: float, save: bool = True):
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#7c3aed", lw=2,
            label=f"LightGBM (AUC-ROC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_threshold_analysis(y_true, y_proba, optimal_threshold: float, save: bool = True):
    """
    Plot precision, recall, and F1 across all thresholds.
    Shows the business tradeoff: higher threshold = more precise but fewer caught.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2563eb", lw=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#dc2626", lw=2)
    ax.plot(thresholds, f1s,        label="F1",        color="#16a34a", lw=2)
    ax.axvline(optimal_threshold, color="gray", linestyle="--", lw=1.5,
               label=f"Optimal threshold ({optimal_threshold:.3f})")
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save: bool = True):
    top = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top["feature"][::-1], top["importance_pct"][::-1], color="#2563eb")
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title(f"Top {top_n} Features by Gain (LightGBM)", fontsize=13)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 5. Summary printer
# ---------------------------------------------------------------------------

def _print_eval_summary(metrics: dict):
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    log.info(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    log.info(f"  Threshold: {metrics['threshold']:.3f}")
    log.info(f"  Precision: {metrics['precision']:.4f}")
    log.info(f"  Recall:    {metrics['recall']:.4f}")
    log.info(f"  F1:        {metrics['f1']:.4f}")
    log.info(f"  TP={metrics['tp']:,} FP={metrics['fp']:,} "
             f"TN={metrics['tn']:,} FN={metrics['fn']:,}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# 6. Quick run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.model_training import load_model
    results = load_model()
    eval_metrics = evaluate_model(results)
