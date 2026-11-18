"""
Customer Churn & Revenue Intelligence Platform
================================================
Main Streamlit Application Entry Point

Run:
    streamlit run app.py

This file initializes the platform: loads data, trains the model (if needed),
sets up the database, and renders the Executive Overview as the home page.
"""

import os
import sys
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_data, get_dataset_info
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.churn_model import train_models, load_model, model_is_trained
from src.prediction import get_predictions_for_all
from src.recommendations import apply_recommendations
from src.exports import generate_all_exports
from database.database import init_database, ingest_customers, ingest_predictions, get_database_stats
import src.analytics as analytics
from src.state import ensure_state_initialized

# ---------------------------------------------------------------------------
# Custom CSS — Professional dark enterprise style
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e3a 0%, #16213e 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.3);
    }

    /* KPI card style */
    .kpi-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(54,153,255,0.1) 100%);
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.65);
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-card.danger .kpi-value  { color: #F64E60; }
    .kpi-card.success .kpi-value { color: #1BC5BD; }
    .kpi-card.warning .kpi-value { color: #FFA800; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: rgba(255,255,255,0.9);
        padding: 12px 0 6px;
        border-bottom: 2px solid rgba(108,99,255,0.5);
        margin-bottom: 16px;
    }

    /* Status badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success { background: rgba(27,197,189,0.2); color: #1BC5BD; border: 1px solid #1BC5BD; }
    .badge-danger  { background: rgba(246,78,96,0.2);  color: #F64E60; border: 1px solid #F64E60; }
    .badge-warning { background: rgba(255,168,0,0.2);  color: #FFA800; border: 1px solid #FFA800; }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }

    /* Metric delta */
    [data-testid="metric-container"] {
        background: rgba(108,99,255,0.1);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Data loading & caching
# ===========================================================================

@st.cache_data(show_spinner=False)
def load_and_process():
    """Load, preprocess, and feature-engineer the dataset. Cached."""
    df_raw  = load_data()
    df_clean = preprocess_data(df_raw)
    df_eng   = engineer_features(df_clean)
    return df_eng


@st.cache_data(show_spinner=False)
def get_or_train_model(_df_eng):
    """Train the model if not already saved, then return predictions. Cached."""
    if model_is_trained():
        pipeline, metadata = load_model()
        all_metrics = metadata.get("all_metrics", [])
    else:
        all_metrics, pipeline, metadata = train_models(_df_eng)

    df_pred = get_predictions_for_all(_df_eng)
    df_pred_full = apply_recommendations(df_pred)
    return all_metrics, pipeline, metadata, df_pred_full


# ===========================================================================
# Session state initialization
# ===========================================================================

def initialize_session():
    """Load all data into session_state once per session."""
    ensure_state_initialized()


# ===========================================================================
# Sidebar
# ===========================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 10px;'>
            <div style='font-size:2.2rem'>📊</div>
            <div style='font-size:1.1rem; font-weight:700; color:#6C63FF; margin-top:4px'>
                Churn Intelligence
            </div>
            <div style='font-size:0.75rem; color:rgba(255,255,255,0.5); margin-top:2px'>
                Revenue & Risk Analytics Platform
            </div>
        </div>
        <hr style='border-color:rgba(108,99,255,0.3); margin:10px 0;'>
        """, unsafe_allow_html=True)

        # Navigation hint
        st.markdown("""
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.4); padding:0 4px 10px'>
            📌 Use the pages in the sidebar above to navigate
        </div>
        """, unsafe_allow_html=True)

        # Platform status
        if st.session_state.get("initialized"):
            df_eng   = st.session_state["df_eng"]
            df_pred  = st.session_state["df_pred"]
            metadata = st.session_state["metadata"]

            st.markdown("<div class='section-header'>⚡ Platform Status</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='font-size:0.82rem; color:rgba(255,255,255,0.7); line-height:2'>
                🗃️ <b>Dataset:</b> {len(df_eng):,} customers<br>
                🤖 <b>Model:</b> {metadata.get('best_model_name','—')}<br>
                📅 <b>Trained:</b> {metadata.get('training_date','—')[:10]}<br>
                🎯 <b>ROC-AUC:</b> {max(m['roc_auc'] for m in metadata.get('all_metrics',[{'roc_auc':0}])):.4f}<br>
                ⚠️ <b>High Risk:</b> {(df_pred['risk_category']=='High Risk').sum():,} customers
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr style='border-color:rgba(108,99,255,0.2);'>", unsafe_allow_html=True)

            # Retrain button
            if st.button("🔄 Retrain Model", use_container_width=True):
                from src.churn_model import BEST_MODEL_PATH, METADATA_PATH
                import os
                for p in [BEST_MODEL_PATH, METADATA_PATH]:
                    if os.path.exists(p):
                        os.remove(p)
                # Clear cache
                st.cache_data.clear()
                for key in ["initialized", "df_eng", "all_metrics", "pipeline", "metadata", "df_pred"]:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("""
        <hr style='border-color:rgba(108,99,255,0.2);'>
        <div style='font-size:0.7rem; color:rgba(255,255,255,0.3); text-align:center; padding-top:4px'>
            Customer Churn Intelligence Platform<br>
            IBM Telco Dataset · Scikit-learn · Plotly
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# Home page (Executive Overview preview)
# ===========================================================================

def render_home():
    st.markdown("""
    <div style='padding: 10px 0 24px'>
        <h1 style='font-size:2rem; font-weight:700; color:white; margin:0'>
            📊 Customer Churn & Revenue Intelligence
        </h1>
        <p style='color:rgba(255,255,255,0.55); font-size:0.95rem; margin-top:6px'>
            End-to-end analytics platform · Churn prediction · Revenue at risk · Retention strategy
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("initialized"):
        st.error("❌ Platform failed to initialize. See error below.")
        if st.session_state.get("load_error"):
            st.code(st.session_state["load_error"])
            st.info(
                "📥 **Manual dataset setup:**\n\n"
                "1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
                "2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`\n"
                "3. Rename it to `telco_churn.csv`\n"
                "4. Place it in the `data/raw/` folder\n"
                "5. Restart the app"
            )
        return

    df_eng  = st.session_state["df_eng"]
    df_pred = st.session_state["df_pred"]
    metadata = st.session_state["metadata"]

    # --- KPI Row ---
    total      = len(df_eng)
    churned    = int(df_eng["churn_flag"].sum())
    active     = total - churned
    churn_rate = churned / total * 100
    avg_charge = df_eng["monthly_charges"].mean()
    high_risk  = int((df_pred["risk_category"] == "High Risk").sum())
    rev_at_risk = df_pred[df_pred["risk_category"] == "High Risk"]["monthly_charges"].sum() if "monthly_charges" in df_pred.columns else 0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    kpi_data = [
        (c1, f"{total:,}",           "Total Customers",    ""),
        (c2, f"{active:,}",          "Active Customers",   "success"),
        (c3, f"{churned:,}",         "Churned Customers",  "danger"),
        (c4, f"{churn_rate:.1f}%",   "Churn Rate",         "danger"),
        (c5, f"${avg_charge:.0f}",   "Avg Monthly",        ""),
        (c6, f"{high_risk:,}",       "High Risk",          "warning"),
        (c7, f"${rev_at_risk:,.0f}", "Revenue at Risk*",   "warning"),
    ]
    for col, value, label, style in kpi_data:
        with col:
            st.markdown(f"""
            <div class='kpi-card {style}'>
                <div class='kpi-value'>{value}</div>
                <div class='kpi-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.72rem;color:rgba(255,255,255,0.3)'>*Revenue at Risk = monthly charges of ML-predicted high-risk customers. This is a predicted risk exposure, not a guaranteed loss.</p>", unsafe_allow_html=True)

    # --- Charts Row ---
    st.markdown("<div class='section-header'>📈 Overview Charts</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(analytics.plot_churn_distribution(df_eng), use_container_width=True)
    with col_b:
        st.plotly_chart(analytics.plot_contract_distribution(df_eng), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(analytics.plot_tenure_distribution(df_eng), use_container_width=True)
    with col_d:
        st.plotly_chart(analytics.plot_monthly_charges_distribution(df_eng), use_container_width=True)

    # --- Unique Feature 1: Executive ROI & Retention Impact Simulator ---
    st.markdown("<div class='section-header'>⚡ Live Executive Retention ROI & Revenue Savings Simulator</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:rgba(255,255,255,0.7); font-size:0.88rem; margin-bottom:12px'>
        Simulate the commercial financial return if customer success intervenes on the ML-identified high-risk pool.
    </p>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1])
    with sim_col1:
        target_save_rate = st.slider(
            "🎯 Targeted Retention Success Rate (%)",
            min_value=5, max_value=60, value=25, step=5,
            help="Percentage of high-risk customers successfully retained via retention offers."
        )
        offer_cost = st.slider(
            "🏷️ Average Cost of Retention Offer per Customer ($)",
            min_value=0, max_value=50, value=15, step=5,
            help="Cost of discount, bill credit, or loyalty gift offered during intervention."
        )

    # Calculations
    saved_customers = int(high_risk * (target_save_rate / 100))
    avg_high_risk_charge = (rev_at_risk / high_risk) if high_risk > 0 else 65.0
    monthly_rev_protected = saved_customers * avg_high_risk_charge
    annual_rev_protected = monthly_rev_protected * 12
    total_campaign_cost = saved_customers * offer_cost
    net_annual_profit = annual_rev_protected - total_campaign_cost
    roi_ratio = (net_annual_profit / total_campaign_cost) if total_campaign_cost > 0 else 0.0

    with sim_col2:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Accounts Saved / Year", f"{saved_customers:,} accounts", delta=f"{target_save_rate}% save rate")
            st.metric("Protected Monthly MRR", f"${monthly_rev_protected:,.2f}")
        with m2:
            st.metric("Net Annual Revenue Saved", f"${net_annual_profit:,.2f}", delta="Net Business Impact")
            st.metric("Campaign ROI Multiple", f"{roi_ratio:.1f}x ROI", delta=f"${total_campaign_cost:,.0f} spend")

    st.markdown("---")

    # --- Unique Feature 2: Customer 360 Risk Inspector & Retention Outreach Generator ---
    st.markdown("<div class='section-header'>🔍 Customer 360 Risk Inspector & Outreach Generator</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:rgba(255,255,255,0.7); font-size:0.88rem; margin-bottom:12px'>
        Inspect individual customer accounts in real-time, view diagnostic risk drivers, and auto-generate personalized retention emails.
    </p>
    """, unsafe_allow_html=True)

    if not df_pred.empty:
        sample_customers = df_pred.sort_values("churn_probability", ascending=False)["customer_id"].head(50).tolist()
        selected_cid = st.selectbox("Select Customer to Inspect (Top 50 At-Risk Accounts):", sample_customers)

        cust_row = df_pred[df_pred["customer_id"] == selected_cid].iloc[0]
        prob = cust_row.get("churn_probability", 0.5)
        risk = cust_row.get("risk_category", "High Risk")
        contract = cust_row.get("contract", "Month-to-month")
        tenure = cust_row.get("tenure", 1)
        charges = cust_row.get("monthly_charges", 70.0)
        action = cust_row.get("retention_action", "Contract Incentive Campaign")

        ins1, ins2 = st.columns([1, 1])
        with ins1:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(108,99,255,0.3);border-radius:10px;padding:16px;'>
                <h4 style='margin:0 0 10px;color:#6C63FF'>Account: {selected_cid}</h4>
                <p style='margin:4px 0'><strong>Contract:</strong> {contract}</p>
                <p style='margin:4px 0'><strong>Tenure:</strong> {tenure} months</p>
                <p style='margin:4px 0'><strong>Monthly Charges:</strong> ${charges:.2f}</p>
                <p style='margin:4px 0'><strong>Predicted Churn Probability:</strong> <span style='color:#F64E60;font-weight:700'>{prob*100:.1f}%</span></p>
                <p style='margin:4px 0'><strong>Assigned Risk Tier:</strong> {risk}</p>
                <p style='margin:4px 0'><strong>Recommended Playbook:</strong> <span style='color:#1BC5BD'>{action}</span></p>
            </div>
            """, unsafe_allow_html=True)

        with ins2:
            st.markdown("##### ✉️ Auto-Generated Retention Outreach Pitch")
            email_draft = f"""Subject: Special loyalty offer tailored for your account #{selected_cid}

Dear Valued Customer,

Thank you for being with us for the past {tenure} months. We noticed you are currently on our {contract} plan.

As a token of our appreciation, we would like to offer you an exclusive 15% discount for upgrading to our annual plan, plus complimentary premium tech support for the next 6 months (reducing your monthly cost to approximately ${charges * 0.85:.2f}).

Would you be open to a quick 5-minute chat this week to apply these credits to your account?

Warm regards,
Customer Success & Retention Team"""
            st.text_area("Personalized Customer Communication Draft", email_draft, height=185)

    # --- Quick navigation ---
    st.markdown("<div class='section-header'>🧭 Navigate the Platform</div>", unsafe_allow_html=True)
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        st.info("**👥 Customer Analytics**\nDemographic & service-level churn breakdowns with interactive filters.")
    with nc2:
        st.info("**💰 Revenue Intelligence**\nRevenue by contract, segment, and ML-predicted risk category.")
    with nc3:
        st.info("**🤖 Churn Prediction**\nEnter customer attributes or upload a CSV for instant churn probability.")


# ===========================================================================
# Main
# ===========================================================================

initialize_session()
render_sidebar()
render_home()
