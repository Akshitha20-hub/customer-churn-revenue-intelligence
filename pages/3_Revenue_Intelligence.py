"""
Page 3: Revenue Intelligence
=============================
Financial exposure analysis, Customer Lifetime Value (CLV) distributions,
high-value account rankings, and revenue-at-risk stress testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import src.analytics as analytics
from src.sql_queries import (
    get_revenue_by_contract, get_revenue_at_risk,
    get_high_value_customer_analysis, get_customer_segmentation_sql
)

st.set_page_config(page_title="Revenue Intelligence", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1e3a 0%,#16213e 100%);border-right:1px solid rgba(108,99,255,0.3);}
.kpi-card{background:linear-gradient(135deg,rgba(108,99,255,0.15) 0%,rgba(54,153,255,0.1) 100%);border:1px solid rgba(108,99,255,0.4);border-radius:12px;padding:18px 22px;text-align:center;margin-bottom:8px;}
.kpi-value{font-size:1.85rem;font-weight:700;color:#6C63FF;line-height:1.1;}
.kpi-label{font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:4px;text-transform:uppercase;letter-spacing:0.08em;}
.kpi-card.danger .kpi-value{color:#F64E60;}
.kpi-card.success .kpi-value{color:#1BC5BD;}
.kpi-card.warning .kpi-value{color:#FFA800;}
.section-header{font-size:1.05rem;font-weight:600;color:rgba(255,255,255,0.9);padding:12px 0 6px;border-bottom:2px solid rgba(108,99,255,0.5);margin-bottom:16px;}
.disclaimer-box{background:rgba(255,168,0,0.1);border-left:4px solid #FFA800;border-radius:0 8px 8px 0;padding:12px 18px;margin:12px 0 20px;font-size:0.85rem;color:rgba(255,255,255,0.85);}
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
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>💰 Revenue Intelligence & Financial Exposure</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Financial impact analysis, revenue concentration, and predictive revenue at risk</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-box'>
    ⚠️ <strong>Executive Financial Governance Notice:</strong> "Revenue at Risk" reflects probabilistic revenue exposure modeled from current churn likelihoods. It represents an addressable intervention pool, not guaranteed or realized accounting losses.
</div>
""", unsafe_allow_html=True)

# ----------------- SQL Revenue-at-Risk KPIs -----------------
try:
    rev_risk = get_revenue_at_risk()
except Exception:
    rev_risk = {}

total_mrr       = rev_risk.get("total_platform_monthly_revenue") or float(df_eng["monthly_charges"].sum())
high_risk_mrr   = rev_risk.get("high_risk_monthly_revenue") or float(df_pred[df_pred["risk_category"] == "High Risk"]["monthly_charges"].sum() if "risk_category" in df_pred.columns else 0.0)
high_risk_count = rev_risk.get("high_risk_customers") or int((df_pred["risk_category"] == "High Risk").sum() if "risk_category" in df_pred.columns else 0)
expected_loss   = rev_risk.get("expected_monthly_loss_exposure") or float((df_pred["monthly_charges"] * df_pred["churn_probability"]).sum() if "churn_probability" in df_pred.columns else 0.0)
rev_at_risk_pct = rev_risk.get("revenue_at_risk_pct") or round((high_risk_mrr / total_mrr * 100) if total_mrr > 0 else 0.0, 1)

r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown(kpi_card(f"${total_mrr:,.2f}", "Total Monthly Recurring Revenue (MRR)", "success"), unsafe_allow_html=True)
with r2:
    st.markdown(kpi_card(f"${high_risk_mrr:,.2f}", f"High-Risk MRR ({rev_at_risk_pct:.1f}%)", "danger"), unsafe_allow_html=True)
with r3:
    st.markdown(kpi_card(f"{high_risk_count:,}", "High-Risk Customer Accounts", "warning"), unsafe_allow_html=True)
with r4:
    st.markdown(kpi_card(f"${expected_loss:,.2f}", "Expected Probabilistic Loss / Mo"), unsafe_allow_html=True)

# ----------------- Row 1: Revenue by Contract & Risk Category -----------------
st.markdown("<div class='section-header'>📊 Revenue Breakdown by Contract & Risk Exposure</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(analytics.plot_revenue_by_contract(df_eng), use_container_width=True)

with col2:
    st.plotly_chart(analytics.plot_revenue_by_risk_category(df_pred), use_container_width=True)

# ----------------- Row 2: Charge Distribution & Customer Value Segments -----------------
st.markdown("<div class='section-header'>📈 Spend Tiers & Customer Lifetime Value (CLV)</div>", unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(analytics.plot_monthly_charges_distribution(df_eng), use_container_width=True)

with col4:
    st.plotly_chart(analytics.plot_clv_by_tenure_group(df_eng), use_container_width=True)

# ----------------- Row 3: SQL Analytical Segmentation -----------------
st.markdown("<div class='section-header'>💼 Strategic Business Cohort Segmentation (SQL CTE Engine)</div>", unsafe_allow_html=True)
try:
    df_segments = get_customer_segmentation_sql()
    if not df_segments.empty:
        s_col1, s_col2 = st.columns([3, 2])
        with s_col1:
            fig_seg = px.bar(
                df_segments,
                x="business_segment",
                y="segment_monthly_revenue",
                color="churn_rate_pct",
                color_continuous_scale="Reds",
                title="Monthly Revenue Contribution & Churn Rate by Strategic Cohort",
                labels={"segment_monthly_revenue": "Monthly Revenue ($)", "churn_rate_pct": "Churn Rate (%)"}
            )
            fig_seg.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_seg, use_container_width=True)
        with s_col2:
            st.dataframe(df_segments, use_container_width=True, height=300)
    else:
        st.info("Run database ingestion to load SQL segmentation metrics.")
except Exception as e:
    st.error(f"Error loading segmentation data: {e}")

# ----------------- Row 4: High-Value Customer Leaderboard -----------------
st.markdown("<div class='section-header'>🏆 Top High-Value Accounts Leaderboard (SQL Window Functions)</div>", unsafe_allow_html=True)
st.caption("Identified using SQLite window functions `RANK() OVER (ORDER BY monthly_charges DESC)` and `NTILE(4)` Quartiles.")

try:
    top_accounts = get_high_value_customer_analysis(top_n=50)
    if not top_accounts.empty:
        def highlight_risk(val):
            if val == "High Risk":
                return "color: #F64E60; font-weight: bold;"
            elif val == "Medium Risk":
                return "color: #FFA800;"
            return "color: #1BC5BD;"

        styled_df = top_accounts.style.applymap(highlight_risk, subset=["risk_category"])
        st.dataframe(styled_df, use_container_width=True, height=350)

        csv_hv = top_accounts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Top 50 High-Value Accounts (CSV)",
            data=csv_hv,
            file_name="top_high_value_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("Populate database to inspect high-value account rankings.")
except Exception as e:
    st.error(f"Error loading high-value account rankings: {e}")
