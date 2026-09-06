"""
app.py
======
KKBox Churn Intelligence Platform — Layer 6: Streamlit Dashboard

Sections:
1. Summary — key metrics at a glance
2. Customer Risk Table — filterable, sortable risk register
3. SHAP Explorer — individual customer explanation
4. Revenue at Risk — segment breakdown
5. What-If Simulator — change features, see churn probability update

Run:
    streamlit run app.py
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="KKBox Churn Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #2563eb;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .risk-high   { color: #dc2626; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-low    { color: #16a34a; font-weight: 600; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2563eb;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_model_results():
    path = os.path.join(ROOT, "models", "lgbm_churn.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_explainer():
    path = os.path.join(ROOT, "models", "shap_results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_master():
    path = os.path.join(ROOT, "data", "processed", "master.parquet")
    return pd.read_parquet(path)


@st.cache_data
def build_scored_customers(_results, _df):
    """Score all customers and build risk table."""
    from src.feature_engineering import build_features
    X_all, y_all, features = build_features(_df)
    model = _results["model"]
    proba = model.predict(X_all)

    scored = _df[["msno", "is_churn"]].copy()
    scored["churn_probability"] = proba
    scored["risk_tier"] = pd.cut(
        proba,
        bins=[0, 0.3, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    if "last_plan_price" in _df.columns:
        scored["monthly_rev_usd"] = (_df["last_plan_price"].fillna(149) * 0.031).clip(0.5, 62)
    else:
        scored["monthly_rev_usd"] = 4.62

    scored["revenue_at_risk_usd"] = scored["churn_probability"] * scored["monthly_rev_usd"]

    # Pull in readable columns
    for col in ["last_plan_days", "last_auto_renew", "cancel_ratio",
                "active_days_30d", "days_since_last", "total_active_days"]:
        if col in _df.columns:
            scored[col] = _df[col].values

    return scored, X_all, features


# ── Load everything ───────────────────────────────────────────────────
with st.spinner("Loading model and data ..."):
    try:
        results  = load_model_results()
        shap_pkg = load_shap_explainer()
        df       = load_master()
        scored, X_all, features = build_scored_customers(results, df)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://wikimedia.org", width=160)
    st.markdown("## 🎵 Churn Intelligence")
    st.markdown("---")

    section = st.radio(
        "Navigate",
        ["Summary",
         "Customer Risk Table",
         "SHAP Explorer",
         "Revenue at Risk",
         "What-If Simulator"],
    )

    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("AUC-ROC",  f"{results['test_auc_roc']:.4f}")
    st.metric("AUC-PR",   f"{results['test_auc_pr']:.4f}")
    st.metric("CV Folds", "5-fold Stratified")
    st.markdown("---")
    st.caption("KKBox Churn Intelligence Platform v1.0")


# ==========================================================================
# SECTION 1 — Summary
# ==========================================================================
if section == "Summary":
    st.markdown("# 🎵 KKBox Churn Intelligence Platform")
    st.markdown("*Predicting subscriber churn across 970,960 users using LightGBM + SHAP*")
    st.markdown("---")

    # KPI row
    total        = len(scored)
    high_risk    = (scored["risk_tier"] == "High").sum()
    medium_risk  = (scored["risk_tier"] == "Medium").sum()
    low_risk     = (scored["risk_tier"] == "Low").sum()
    total_rev_risk = scored["revenue_at_risk_usd"].sum()
    actual_churn = scored["is_churn"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers",     f"{total:,}")
    c2.metric("High Risk",           f"{high_risk:,}",   f"{high_risk/total*100:.1f}%")
    c3.metric("Medium Risk",         f"{medium_risk:,}", f"{medium_risk/total*100:.1f}%")
    c4.metric("Revenue at Risk/mo",  f"${total_rev_risk:,.0f}")
    c5.metric("Actual Churn Rate",   f"{actual_churn:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Risk Distribution</p>', unsafe_allow_html=True)
        tier_counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#16a34a", "#f59e0b", "#dc2626"]
        bars = ax.bar(tier_counts.index, tier_counts.values, color=colors, width=0.5)
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=10)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.set_title("Customers by Risk Tier", color="white")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown('<p class="section-header">Churn Probability Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scored["churn_probability"], bins=60, color="#7c3aed", alpha=0.85, edgecolor="none")
        ax.axvline(0.3, color="#f59e0b", linestyle="--", lw=1.5, label="Medium (0.3)")
        ax.axvline(0.7, color="#dc2626", linestyle="--", lw=1.5, label="High (0.7)")
        ax.legend(fontsize=9)
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")
        ax.tick_params(colors="white")
        ax.set_title("Distribution of Churn Probabilities", color="white")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)
    arch_cols = st.columns(6)
    layers = [
        ("", "Data Pipeline", "4 raw files\n970k users"),
        ("", "Feature Eng.", "55 features\n7/30/90d windows"),
        ("", "LightGBM", "5-fold CV\nAUC 0.9942"),
        ("", "SHAP", "Global + local\nexplanations"),
        ("", "Business Impact", "Revenue at risk\nROI scenarios"),
        ("", "Dashboard", "Real-time\nwhat-if sim"),
    ]
    for col, (icon, title, desc) in zip(arch_cols, layers):
        col.markdown(f"**{icon} {title}**")
        col.caption(desc)


# ==========================================================================
# SECTION 2 — Customer Risk Table
# ==========================================================================
elif section == "Customer Risk Table":
    st.markdown("##Customer Risk Register")
    st.markdown("Filter and explore individual customer churn risk scores.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        tier_filter = st.multiselect(
            "Risk Tier", ["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
    with col2:
        min_prob = st.slider("Min Churn Probability", 0.0, 1.0, 0.3, 0.05)
    with col3:
        n_rows = st.selectbox("Rows to display", [50, 100, 250, 500], index=1)

    filtered = scored[
        (scored["risk_tier"].isin(tier_filter)) &
        (scored["churn_probability"] >= min_prob)
    ].sort_values("churn_probability", ascending=False).head(n_rows)

    st.markdown(f"**Showing {len(filtered):,} customers**")

    display_cols = ["msno", "churn_probability", "risk_tier",
                    "revenue_at_risk_usd", "monthly_rev_usd", "is_churn"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    def color_risk(val):
        colors = {"High": "color: #dc2626", "Medium": "color: #f59e0b", "Low": "color: #16a34a"}
        return colors.get(val, "")

    styled = filtered[display_cols].style\
        .format({
            "churn_probability":    "{:.3f}",
            "revenue_at_risk_usd":  "${:.2f}",
            "monthly_rev_usd":      "${:.2f}",
        })\
        .applymap(color_risk, subset=["risk_tier"])\
        .background_gradient(subset=["churn_probability"], cmap="RdYlGn_r")

    st.dataframe(styled, use_container_width=True, height=500)

    total_rev = filtered["revenue_at_risk_usd"].sum()
    st.info(f" Total monthly revenue at risk in this view: **${total_rev:,.2f} USD**")


# ==========================================================================
# SECTION 3 — SHAP Explorer
# ==========================================================================
elif section == "SHAP Explorer":
    st.markdown("##  Individual Customer SHAP Explanation")
    st.markdown("Select a customer to see exactly why they are predicted to churn.")
    st.markdown("---")

    high_risk_customers = scored[scored["risk_tier"] == "High"].sort_values(
        "churn_probability", ascending=False
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_msno = st.selectbox(
            "Select Customer (High Risk)",
            high_risk_customers["msno"].head(200).tolist(),
            format_func=lambda x: f"{x[:20]}... — P(churn)={scored.loc[scored['msno']==x,'churn_probability'].values[0]:.3f}"
        )
    with col2:
        st.markdown("&nbsp;")
        explain_btn = st.button("Generate SHAP Explanation", type="primary")

    if explain_btn and selected_msno:
        customer_row_idx = X_all.index[df["msno"].values == selected_msno][0] \
            if (df["msno"].values == selected_msno).any() else 0

        with st.spinner("Computing SHAP values ..."):
            explainer = shap_pkg["explainer"]
            X_row = X_all.loc[[customer_row_idx]]
            sv = explainer.shap_values(X_row)
            if isinstance(sv, list):
                sv = sv[1]

            churn_prob = float(results["model"].predict(X_row)[0])
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Churn Probability", f"{churn_prob:.3f}",
                    "HIGH RISK" if churn_prob > 0.7 else "MEDIUM RISK")
        col2.metric("Base Rate", f"{1/(1+np.exp(-expected_val)):.3f}")
        col3.metric("Revenue at Risk",
                    f"${scored.loc[scored['msno']==selected_msno,'revenue_at_risk_usd'].values[0]:.2f}/mo")

        st.markdown("### SHAP Waterfall — What's driving this prediction?")
        shap_exp = shap.Explanation(
            values        = sv[0],
            base_values   = expected_val,
            data          = X_row.values[0],
            feature_names = features,
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Top Feature Drivers")
        explanation_df = pd.DataFrame({
            "Feature": features,
            "Value":   X_row.values[0],
            "SHAP":    sv[0],
        }).sort_values("SHAP", key=abs, ascending=False).head(15)

        st.dataframe(
            explanation_df.style.format({"Value": "{:.3f}", "SHAP": "{:.4f}"})\
                .background_gradient(subset=["SHAP"], cmap="RdBu_r"),
            use_container_width=True,
        )


# ==========================================================================
# SECTION 4 — Revenue at Risk
# ==========================================================================
elif section == "Revenue at Risk":
    st.markdown("## Revenue at Risk Analysis")
    st.markdown("Monthly subscription revenue at risk by churn probability segment.")
    st.markdown("---")

    segment_summary = scored.groupby("risk_tier", observed=True).agg(
        Customers        = ("msno", "count"),
        Avg_Churn_Prob   = ("churn_probability", "mean"),
        Actual_Churners  = ("is_churn", "sum"),
        Rev_At_Risk_USD  = ("revenue_at_risk_usd", "sum"),
        Avg_Rev_USD      = ("monthly_rev_usd", "mean"),
    ).reset_index()

    segment_summary["Capture_Rate_%"] = (
        segment_summary["Actual_Churners"] / segment_summary["Customers"] * 100
    ).round(1)

    st.dataframe(
        segment_summary.style.format({
            "Avg_Churn_Prob":  "{:.3f}",
            "Rev_At_Risk_USD": "${:,.0f}",
            "Avg_Rev_USD":     "${:.2f}",
            "Capture_Rate_%":  "{:.1f}%",
        }),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Retention ROI Scenarios")
    st.markdown("*Modelled on High Risk segment. Intervention cost: $2 USD per customer.*")

    high_risk_scored = scored[scored["risk_tier"] == "High"].sort_values(
        "revenue_at_risk_usd", ascending=False
    )
    n_high = len(high_risk_scored)

    scenarios = []
    for pct in [10, 20, 30, 50, 75, 100]:
        n_retained   = int(n_high * pct / 100)
        rev_saved    = high_risk_scored.head(n_retained)["revenue_at_risk_usd"].sum()
        cost         = n_retained * 2.0
        net_roi      = rev_saved - cost
        roi_mult     = rev_saved / cost if cost > 0 else 0
        scenarios.append({
            "Retain %":          f"{pct}%",
            "Customers":         n_retained,
            "Revenue Saved USD":  round(rev_saved, 0),
            "Intervention Cost":  round(cost, 0),
            "Net ROI USD":        round(net_roi, 0),
            "ROI Multiple":       f"{roi_mult:.1f}x",
        })

    roi_df = pd.DataFrame(scenarios)
    st.dataframe(
        roi_df.style.format({
            "Revenue Saved USD":  "${:,.0f}",
            "Intervention Cost":  "${:,.0f}",
            "Net ROI USD":        "${:,.0f}",
        }).background_gradient(subset=["Net ROI USD"], cmap="Greens"),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    x = [s["Retain %"] for s in scenarios]
    rev = [s["Revenue Saved USD"] for s in scenarios]
    cost = [s["Intervention Cost"] for s in scenarios]
    ax.bar(x, rev,  label="Revenue Saved",      color="#2563eb", alpha=0.85)
    ax.bar(x, cost, label="Intervention Cost",   color="#dc2626", alpha=0.85)
    ax.set_xlabel("% High-Risk Customers Retained")
    ax.set_ylabel("USD")
    ax.set_title("Retention ROI by Scenario")
    ax.legend()
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    st.pyplot(fig)
    plt.close()


# ==========================================================================
# SECTION 5 — What-If Simulator
# ==========================================================================
elif section == "What-If Simulator":
    st.markdown("## What-If Churn Simulator")
    st.markdown("Adjust customer attributes and see how churn probability changes in real time.")
    st.markdown("---")

    model = results["model"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Contract & Billing")
        last_auto_renew  = st.selectbox("Auto Renew",        [1, 0], format_func=lambda x: "Yes" if x else "No")
        last_is_cancel   = st.selectbox("Last Action Cancel", [0, 1], format_func=lambda x: "Yes" if x else "No")
        last_plan_days   = st.selectbox("Plan Duration",     [30, 90, 180, 365], index=0)
        last_plan_price  = st.slider("Plan Price (TWD)",     49, 500, 149, 10)
        cancel_ratio     = st.slider("Historical Cancel Ratio", 0.0, 1.0, 0.0, 0.05)
        auto_renew_ratio = st.slider("Historical Auto-Renew Ratio", 0.0, 1.0, 1.0, 0.05)

    with col2:
        st.markdown("### Listening Behaviour")
        active_days_7d   = st.slider("Active Days (Last 7)",  0, 7,  5)
        active_days_30d  = st.slider("Active Days (Last 30)", 0, 30, 20)
        active_days_90d  = st.slider("Active Days (Last 90)", 0, 90, 60)
        days_since_last  = st.slider("Days Since Last Listen", 0, 90, 2)
        avg_completion   = st.slider("Avg Song Completion %", 0.0, 1.0, 0.7, 0.05)
        days_since_reg   = st.slider("Days Since Registration", 30, 5000, 1000, 30)

    
    X_median = X_all.median().to_dict()

    sim_overrides = {
        "last_auto_renew":        last_auto_renew,
        "last_is_cancel":         last_is_cancel,
        "last_plan_days":         last_plan_days,
        "last_plan_price":        last_plan_price,
        "cancel_ratio":           cancel_ratio,
        "auto_renew_ratio":       auto_renew_ratio,
        "active_days_7d":         active_days_7d,
        "active_days_30d":        active_days_30d,
        "active_days_90d":        active_days_90d,
        "days_since_last":        days_since_last,
        "avg_completion":         avg_completion,
        "days_since_registration": days_since_reg,
        "silent_march":           int(active_days_30d == 0),
        "dropout_risk":           int(days_since_last > 14),
        "short_plan_risk":        int(last_plan_days <= 30),
        "long_plan_flag":         int(last_plan_days >= 365),
        "last_amount_paid":       last_plan_price,
        "activity_consistency_30d": active_days_30d / 30,
        "activity_consistency_90d": active_days_90d / 90,
        "trend_7d_30d":           (active_days_7d / (active_days_30d / 4.3)) if active_days_30d > 0 else 0,
        "trend_30d_90d":          (active_days_30d / (active_days_90d / 3.0)) if active_days_90d > 0 else 0,
        "expiry_imminent":        int(0 <= X_median.get("days_to_expiry", 30) <= 7),
        "short_plan_dropout":     int(last_plan_days <= 30 and days_since_last > 14),
        "silent_and_cancelled":   int(active_days_30d == 0 and last_is_cancel == 1),
        "autorenew_declining":    int(last_auto_renew == 1 and active_days_7d < 2),
    }

    sim_row = {**X_median, **sim_overrides}
    X_sim = pd.DataFrame([sim_row])[features]

    churn_prob_sim = float(model.predict(X_sim)[0])

    st.markdown("---")
    st.markdown("### Prediction")

    prob_col, tier_col, rev_col = st.columns(3)
    risk_label = "🔴 HIGH RISK" if churn_prob_sim > 0.7 else \
                 "🟡 MEDIUM RISK" if churn_prob_sim > 0.3 else "🟢 LOW RISK"

    prob_col.metric("Churn Probability", f"{churn_prob_sim:.3f}")
    tier_col.metric("Risk Tier", risk_label)
    rev_col.metric("Monthly Revenue at Risk",
                   f"${churn_prob_sim * last_plan_price * 0.031:.2f} USD")

    # Gauge-style probability bar
    st.markdown("#### Churn Risk Gauge")
    bar_color = "#dc2626" if churn_prob_sim > 0.7 else \
                "#f59e0b" if churn_prob_sim > 0.3 else "#16a34a"
    st.markdown(f"""
    <div style="background:#1e293b;border-radius:8px;height:28px;width:100%;overflow:hidden;">
        <div style="background:{bar_color};height:100%;width:{churn_prob_sim*100:.1f}%;
                    transition:width 0.4s ease;border-radius:8px;
                    display:flex;align-items:center;padding-left:10px;color:white;font-weight:600;">
            {churn_prob_sim*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Recommended Intervention")
    if churn_prob_sim > 0.7:
        st.error("🚨 **Immediate outreach required** — offer a loyalty discount or plan upgrade. "
                 "This customer shows high cancellation history and low recent engagement.")
    elif churn_prob_sim > 0.3:
        st.warning("⚠️ **Targeted engagement** — send a personalised email highlighting "
                   "new features or a curated playlist. Monitor for further decline.")
    else:
        st.success("✅ **Standard engagement** — customer is healthy. "
                   "Include in newsletter and usage tip campaigns.")
