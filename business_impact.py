"""
business_impact.py
==================
KKBox Churn Intelligence Platform — Layer 5: Business Translation

Converts model predictions into business-actionable outputs:
- Customer risk segmentation (High / Medium / Low)
- Revenue at risk per segment
- Retention ROI scenarios
- Intervention recommendations per risk tier

Usage:
    from src.business_impact import build_business_report
    report = build_business_report(df, results, eval_metrics)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
from pathlib import Path

log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS  = ROOT / "models"
REPORTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Business constants
# KKBox pricing tiers in TWD (Taiwan Dollar)
# Average plan price ~149 TWD/month ≈ $4.70 USD
# We work in TWD throughout and convert for display
# ---------------------------------------------------------------------------
AVG_MONTHLY_REVENUE_TWD = 149.0
TWD_TO_USD              = 0.031


# ---------------------------------------------------------------------------
# 1. Main entry point
# ---------------------------------------------------------------------------

def build_business_report(
    df: pd.DataFrame,
    results: dict,
    eval_metrics: dict,
    save_plots: bool = True,
) -> dict:
    """
    Build full business impact report from model predictions.

    Args:
        df:           Master dataset (from data_loader)
        results:      Model results dict (from model_training)
        eval_metrics: Evaluation metrics dict (from model_evaluation)
        save_plots:   Save figures to reports/

    Returns:
        report dict with segmented customer table and revenue metrics
    """
    log.info("Building business impact report ...")

    # ── Score all customers ───────────────────────────────────────────
    model    = results["model"]
    features = results["feature_names"]

    from src.feature_engineering import build_features
    X_all, y_all, _ = build_features(df)

    log.info(f"Scoring {len(X_all):,} customers ...")
    churn_proba = model.predict(X_all)

    # ── Build scored DataFrame ────────────────────────────────────────
    scored = df[["msno", "is_churn"]].copy()
    scored["churn_probability"] = churn_proba

    # Monthly revenue — use last_plan_price if available, else average
    if "last_plan_price" in df.columns:
        scored["monthly_revenue_twd"] = df["last_plan_price"].fillna(
            AVG_MONTHLY_REVENUE_TWD
        ).clip(lower=10, upper=2000)
    else:
        scored["monthly_revenue_twd"] = AVG_MONTHLY_REVENUE_TWD

    scored["monthly_revenue_usd"] = scored["monthly_revenue_twd"] * TWD_TO_USD

    # ── Risk segmentation ─────────────────────────────────────────────
    scored["risk_tier"] = pd.cut(
        scored["churn_probability"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # ── Revenue at risk ───────────────────────────────────────────────
    scored["revenue_at_risk_twd"] = (
        scored["churn_probability"] * scored["monthly_revenue_twd"]
    )
    scored["revenue_at_risk_usd"] = (
        scored["churn_probability"] * scored["monthly_revenue_usd"]
    )

    # ── Intervention recommendations ──────────────────────────────────
    scored["intervention"] = scored["risk_tier"].map({
        "High":   "Immediate outreach — discount offer or loyalty reward",
        "Medium": "Targeted email — highlight new features or plan upgrade",
        "Low":    "Standard engagement — newsletter and usage tips",
    })

    # ── Segment summary ───────────────────────────────────────────────
    segment_summary = _build_segment_summary(scored)
    log.info("\nSegment Summary:")
    log.info(segment_summary.to_string(index=False))

    # ── Retention ROI scenarios ───────────────────────────────────────
    roi_scenarios = _build_roi_scenarios(scored)
    log.info("\nRetention ROI Scenarios (monthly, USD):")
    log.info(roi_scenarios.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────
    if save_plots:
        _plot_risk_distribution(scored, save=True)
        _plot_revenue_at_risk(segment_summary, save=True)
        _plot_roi_scenarios(roi_scenarios, save=True)
        _plot_churn_probability_hist(scored, save=True)

    # ── Save scored customer table ────────────────────────────────────
    scored_path = REPORTS / "scored_customers.parquet"
    scored.to_parquet(scored_path, index=False)
    log.info(f"\nScored customer table saved to {scored_path}")

    report = {
        "scored":           scored,
        "segment_summary":  segment_summary,
        "roi_scenarios":    roi_scenarios,
        "total_revenue_at_risk_usd": scored["revenue_at_risk_usd"].sum(),
        "high_risk_count":  (scored["risk_tier"] == "High").sum(),
        "medium_risk_count":(scored["risk_tier"] == "Medium").sum(),
        "low_risk_count":   (scored["risk_tier"] == "Low").sum(),
    }

    _print_executive_summary(report)
    return report


# ---------------------------------------------------------------------------
# 2. Segment summary
# ---------------------------------------------------------------------------

def _build_segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-tier summary: customer count, avg churn prob, total revenue at risk."""
    summary = scored.groupby("risk_tier", observed=True).agg(
        customer_count      = ("msno", "count"),
        avg_churn_prob      = ("churn_probability", "mean"),
        total_rev_at_risk_usd = ("revenue_at_risk_usd", "sum"),
        avg_monthly_rev_usd = ("monthly_revenue_usd", "mean"),
        actual_churners     = ("is_churn", "sum"),
    ).reset_index()

    summary["capture_rate"] = (
        summary["actual_churners"] / summary["customer_count"] * 100
    )

    summary["total_rev_at_risk_usd"] = summary["total_rev_at_risk_usd"].round(0)
    summary["avg_churn_prob"]        = summary["avg_churn_prob"].round(4)
    summary["avg_monthly_rev_usd"]   = summary["avg_monthly_rev_usd"].round(2)
    summary["capture_rate"]          = summary["capture_rate"].round(2)

    return summary[["risk_tier", "customer_count", "avg_churn_prob",
                     "actual_churners", "capture_rate",
                     "avg_monthly_rev_usd", "total_rev_at_risk_usd"]]


# ---------------------------------------------------------------------------
# 3. Retention ROI scenarios
# ---------------------------------------------------------------------------

def _build_roi_scenarios(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Model the revenue impact of retaining different percentages
    of high-risk customers.

    Assumes:
    - Intervention cost: $2 USD per customer contacted
    - Revenue saved: retained customers × monthly revenue
    - Net ROI = revenue saved - intervention cost
    """
    high_risk = scored[scored["risk_tier"] == "High"]
    total_high_risk_rev = high_risk["revenue_at_risk_usd"].sum()
    n_high_risk         = len(high_risk)
    intervention_cost   = 2.0  # USD per customer

    scenarios = []
    for pct in [10, 20, 30, 50, 75]:
        retained     = int(n_high_risk * pct / 100)
        rev_saved    = high_risk.nlargest(retained, "revenue_at_risk_usd")[
            "revenue_at_risk_usd"
        ].sum()
        cost         = retained * intervention_cost
        net_roi      = rev_saved - cost
        roi_multiple = rev_saved / cost if cost > 0 else 0

        scenarios.append({
            "retention_pct":      pct,
            "customers_retained": retained,
            "revenue_saved_usd":  round(rev_saved, 0),
            "intervention_cost_usd": round(cost, 0),
            "net_roi_usd":        round(net_roi, 0),
            "roi_multiple":       round(roi_multiple, 1),
        })

    return pd.DataFrame(scenarios)


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def _plot_risk_distribution(scored: pd.DataFrame, save: bool = True):
    tier_counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#16a34a", "#f59e0b", "#dc2626"]
    bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.5)
    ax.bar_label(bars, fmt="{:,.0f}", padding=5, fontsize=11)
    ax.set_xlabel("Risk Tier", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Customer Risk Distribution", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "risk_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_revenue_at_risk(segment_summary: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
    bar_colors = [colors[t] for t in segment_summary["risk_tier"]]

    bars = ax.bar(
        segment_summary["risk_tier"],
        segment_summary["total_rev_at_risk_usd"],
        color=bar_colors, width=0.5,
    )
    ax.bar_label(bars, fmt="${:,.0f}", padding=5, fontsize=10)
    ax.set_xlabel("Risk Tier", fontsize=12)
    ax.set_ylabel("Monthly Revenue at Risk (USD)", fontsize=12)
    ax.set_title("Revenue at Risk by Segment", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "revenue_at_risk.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_roi_scenarios(roi_scenarios: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = roi_scenarios["retention_pct"].astype(str) + "%"
    ax.bar(x, roi_scenarios["revenue_saved_usd"],
           label="Revenue Saved", color="#2563eb", alpha=0.85)
    ax.bar(x, roi_scenarios["intervention_cost_usd"],
           label="Intervention Cost", color="#dc2626", alpha=0.85)
    ax.set_xlabel("% of High-Risk Customers Retained", fontsize=12)
    ax.set_ylabel("USD", fontsize=12)
    ax.set_title("Retention ROI Scenarios — High Risk Segment", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "roi_scenarios.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def _plot_churn_probability_hist(scored: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        scored["churn_probability"],
        bins=50, color="#7c3aed", alpha=0.8, edgecolor="white",
    )
    ax.axvline(0.3, color="#f59e0b", linestyle="--", lw=2, label="Medium threshold (0.3)")
    ax.axvline(0.7, color="#dc2626", linestyle="--", lw=2, label="High threshold (0.7)")
    ax.set_xlabel("Churn Probability", fontsize=12)
    ax.set_ylabel("Number of Customers", fontsize=12)
    ax.set_title("Distribution of Churn Probabilities", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    if save:
        fig.savefig(REPORTS / "churn_prob_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# 5. Executive summary printer
# ---------------------------------------------------------------------------

def _print_executive_summary(report: dict):
    scored = report["scored"]
    total  = len(scored)

    log.info("\n" + "=" * 60)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total customers scored:     {total:,}")
    log.info(f"  High risk  (>70%):          {report['high_risk_count']:,} "
             f"({report['high_risk_count']/total*100:.1f}%)")
    log.info(f"  Medium risk (30-70%):       {report['medium_risk_count']:,} "
             f"({report['medium_risk_count']/total*100:.1f}%)")
    log.info(f"  Low risk   (<30%):          {report['low_risk_count']:,} "
             f"({report['low_risk_count']/total*100:.1f}%)")
    log.info(f"\n  Total monthly revenue at risk: "
             f"${report['total_revenue_at_risk_usd']:,.0f} USD")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# 6. Quick run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from src.data_loader import build_master_dataset
    from src.model_training import load_model
    from src.model_evaluation import evaluate_model

    df      = build_master_dataset()
    results = load_model()
    eval_m  = evaluate_model(results, save_plots=False)
    report  = build_business_report(df, results, eval_m)
