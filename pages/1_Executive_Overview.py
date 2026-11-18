"""
Page 1: Executive Overview
==========================
High-level KPI dashboard for business stakeholders.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import src.analytics as analytics
from src.sql_queries import (
    get_summary_kpis, get_churn_by_contract, get_revenue_by_contract,
    get_risk_distribution, get_revenue_at_risk
)

st.set_page_config(page_title="Executive Overview", page_icon="🏢", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1e3a 0%,#16213e 100%);border-right:1px solid rgba(108,99,255,0.3);}
.kpi-card{background:linear-gradient(135deg,rgba(108,99,255,0.15) 0%,rgba(54,153,255,0.1) 100%);border:1px solid rgba(108,99,255,0.4);border-radius:12px;padding:20px 24px;text-align:center;margin-bottom:8px;}
.kpi-value{font-size:1.9rem;font-weight:700;color:#6C63FF;line-height:1.1;}
.kpi-label{font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:4px;text-transform:uppercase;letter-spacing:0.08em;}
.kpi-card.danger .kpi-value{color:#F64E60;}
.kpi-card.success .kpi-value{color:#1BC5BD;}
.kpi-card.warning .kpi-value{color:#FFA800;}
.section-header{font-size:1.05rem;font-weight:600;color:rgba(255,255,255,0.9);padding:12px 0 6px;border-bottom:2px solid rgba(108,99,255,0.5);margin-bottom:16px;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def kpi_card(value, label, style=""):
    return f"<div class='kpi-card {style}'><div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div>"


from src.state import ensure_state_initialized
df_eng, df_pred, metadata = ensure_state_initialized()
if df_eng.empty:
    st.stop()

st.markdown("""
<div style='padding:10px 0 20px'>
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>🏢 Executive Overview</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Real-time business KPIs from the customer database</p>
</div>
""", unsafe_allow_html=True)

# --- KPIs from SQL ---
try:
    kpis = get_summary_kpis()
except Exception:
    kpis = {}

total        = int(kpis.get("total_customers", len(df_eng)))
churned      = int(kpis.get("churned_customers", df_eng["churn_flag"].sum()))
active       = int(kpis.get("active_customers", total - churned))
churn_rate   = float(kpis.get("churn_rate_pct", churned / total * 100))
avg_charge   = float(kpis.get("avg_monthly_charges", df_eng["monthly_charges"].mean()))
total_rev    = float(kpis.get("total_monthly_revenue", df_eng["monthly_charges"].sum()))
avg_tenure   = float(kpis.get("avg_tenure", df_eng["tenure"].mean()))

high_risk = int((df_pred["risk_category"] == "High Risk").sum()) if "risk_category" in df_pred.columns else 0

try:
    rar = get_revenue_at_risk("High Risk")
    rev_at_risk = float(rar.get("revenue_at_risk", 0))
except Exception:
    rev_at_risk = df_pred[df_pred.get("risk_category", pd.Series()) == "High Risk"]["monthly_charges"].sum() if "monthly_charges" in df_pred.columns else 0

# Row 1 KPIs
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(kpi_card(f"{total:,}", "Total Customers"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card(f"{active:,}", "Active Customers", "success"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card(f"{churned:,}", "Churned Customers", "danger"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card(f"{churn_rate:.1f}%", "Overall Churn Rate", "danger"), unsafe_allow_html=True)

# Row 2 KPIs
c5, c6, c7, c8 = st.columns(4)
with c5: st.markdown(kpi_card(f"${avg_charge:.2f}", "Avg Monthly Charge"), unsafe_allow_html=True)
with c6: st.markdown(kpi_card(f"${total_rev:,.0f}", "Monthly Revenue"), unsafe_allow_html=True)
with c7: st.markdown(kpi_card(f"{high_risk:,}", "High-Risk Customers", "warning"), unsafe_allow_html=True)
with c8: st.markdown(kpi_card(f"${rev_at_risk:,.0f}", "Revenue at Risk*", "warning"), unsafe_allow_html=True)

st.caption("*Revenue at Risk = monthly charges of ML-predicted high-risk customers. This is a predicted risk exposure, not a guaranteed loss.")

st.markdown("---")

# --- SQL Summary Tables ---
st.markdown("<div class='section-header'>📋 Contract & Revenue Summary (SQL Analytics)</div>", unsafe_allow_html=True)

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("**Churn by Contract Type**")
    try:
        df_contract = get_churn_by_contract()
        st.dataframe(
            df_contract.style.format({
                "churn_rate_pct": "{:.1f}%",
                "avg_monthly_charges": "${:.2f}",
            }),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"SQL query error: {e}")

with col_t2:
    st.markdown("**Revenue by Contract Type**")
    try:
        df_rev = get_revenue_by_contract()
        st.dataframe(
            df_rev.style.format({
                "avg_monthly_charges": "${:.2f}",
                "total_monthly_revenue": "${:,.2f}",
                "total_revenue_to_date": "${:,.2f}",
            }),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"SQL query error: {e}")

st.markdown("---")

# --- Charts ---
st.markdown("<div class='section-header'>📊 Overview Charts</div>", unsafe_allow_html=True)

ca, cb = st.columns(2)
with ca:
    st.plotly_chart(analytics.plot_churn_distribution(df_eng), use_container_width=True)
    st.caption("🔍 **Insight:** Approximately 26% of customers have churned, indicating a significant retention challenge for the business.")
with cb:
    st.plotly_chart(analytics.plot_contract_distribution(df_eng), use_container_width=True)
    st.caption("🔍 **Insight:** Month-to-month contract customers have the highest absolute churn numbers, while two-year contract customers show significantly lower churn.")

cc, cd = st.columns(2)
with cc:
    st.plotly_chart(analytics.plot_tenure_distribution(df_eng), use_container_width=True)
    st.caption("🔍 **Insight:** New customers (low tenure) have the highest churn concentration. Churn risk decreases sharply after 24 months.")
with cd:
    st.plotly_chart(analytics.plot_monthly_charges_distribution(df_eng), use_container_width=True)
    st.caption("🔍 **Insight:** Customers paying higher monthly charges are disproportionately more likely to churn. Pricing may be a key driver.")

# --- Risk distribution (if predictions available) ---
if "risk_category" in df_pred.columns:
    st.markdown("<div class='section-header'>⚠️ ML Risk Distribution</div>", unsafe_allow_html=True)
    re1, re2 = st.columns(2)
    with re1:
        st.plotly_chart(analytics.plot_risk_distribution(df_pred), use_container_width=True)
    with re2:
        st.plotly_chart(analytics.plot_revenue_by_risk_category(df_pred), use_container_width=True)
        st.caption("🔍 **Insight:** High-risk customers contribute a significant share of monthly revenue, making retention efforts financially critical.")
