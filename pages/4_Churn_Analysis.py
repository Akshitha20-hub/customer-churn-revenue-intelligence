"""
Page 4: Churn Analysis
=======================
Detailed churn breakdowns by contract, payment, tenure, internet service, and charges.
Includes business insight explanations for each chart.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import src.analytics as analytics
from src.sql_queries import (
    get_churn_by_contract, get_churn_by_payment_method,
    get_churn_by_tenure_group, get_churn_by_internet_service, get_churn_by_gender
)

st.set_page_config(page_title="Churn Analysis", page_icon="📉", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1e3a 0%,#16213e 100%);border-right:1px solid rgba(108,99,255,0.3);}
.section-header{font-size:1.05rem;font-weight:600;color:rgba(255,255,255,0.9);padding:12px 0 6px;border-bottom:2px solid rgba(108,99,255,0.5);margin-bottom:16px;}
.insight-box{background:rgba(108,99,255,0.1);border-left:4px solid #6C63FF;border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0 16px;font-size:0.88rem;color:rgba(255,255,255,0.8);}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

from src.state import ensure_state_initialized
df_eng, df_pred, metadata = ensure_state_initialized()
if df_eng.empty:
    st.stop()

st.markdown("""
<div style='padding:10px 0 20px'>
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>📉 Churn Analysis</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Detailed churn breakdowns with business explanations — powered by SQL analytics</p>
</div>
""", unsafe_allow_html=True)

# --- Churn by Contract ---
st.markdown("<div class='section-header'>📋 Churn by Contract Type</div>", unsafe_allow_html=True)
cc1, cc2 = st.columns([3, 2])
with cc1:
    st.plotly_chart(analytics.plot_churn_rate_by_contract(df_eng), use_container_width=True)
with cc2:
    try:
        df_c = get_churn_by_contract()
        st.markdown("**SQL Data**")
        st.dataframe(df_c.style.format({"churn_rate_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> Month-to-month contract customers churn at approximately 43% — over 4x the rate of
    one-year customers and 15x the rate of two-year customers. Contract length is the single strongest predictor
    of customer retention in this dataset. Business implication: converting month-to-month customers to annual
    contracts is the highest-ROI retention strategy available.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Churn by Payment Method ---
st.markdown("<div class='section-header'>💳 Churn by Payment Method</div>", unsafe_allow_html=True)
cp1, cp2 = st.columns([3, 2])
with cp1:
    st.plotly_chart(analytics.plot_churn_by_payment_method(df_eng), use_container_width=True)
with cp2:
    try:
        df_p = get_churn_by_payment_method()
        st.markdown("**SQL Data**")
        st.dataframe(df_p.style.format({"churn_rate_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> Electronic check users have the highest churn rate (~45%), nearly double that of
    automatic payment methods (bank transfer ~17%, credit card ~15%). Manual payment friction may correlate with
    lower customer satisfaction or engagement. Encouraging auto-pay enrollment can reduce churn and improve
    payment reliability simultaneously.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Churn by Tenure ---
st.markdown("<div class='section-header'>⏱️ Churn by Tenure Group</div>", unsafe_allow_html=True)
ct1, ct2 = st.columns([3, 2])
with ct1:
    st.plotly_chart(analytics.plot_churn_by_tenure_group(df_eng), use_container_width=True)
with ct2:
    try:
        df_t = get_churn_by_tenure_group()
        st.markdown("**SQL Data**")
        st.dataframe(df_t.style.format({"churn_rate_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> New customers (0-12 months) churn at ~47%, while loyal customers (49-72 months)
    churn at under 10%. This classic "churn curve" indicates that the first year of the customer relationship
    is the most critical. Early-stage onboarding, engagement campaigns, and first-contract incentives
    have the highest potential impact on overall churn rate reduction.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Churn by Internet Service ---
st.markdown("<div class='section-header'>🌐 Churn by Internet Service</div>", unsafe_allow_html=True)
ci1, ci2 = st.columns([3, 2])
with ci1:
    st.plotly_chart(analytics.plot_churn_by_internet_service(df_eng), use_container_width=True)
with ci2:
    try:
        df_i = get_churn_by_internet_service()
        st.markdown("**SQL Data**")
        st.dataframe(df_i.style.format({"churn_rate_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> Fiber optic customers have a ~42% churn rate, significantly higher than DSL (~19%)
    and customers with no internet service (~7%). This paradox — premium service with higher churn —
    suggests possible unmet expectations, pricing sensitivity, or competitive pressure from alternative
    fiber providers. Service quality audits for fiber optic customers are strongly recommended.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Churn by Gender ---
st.markdown("<div class='section-header'>👤 Churn by Demographics</div>", unsafe_allow_html=True)
dm1, dm2 = st.columns(2)
with dm1:
    st.plotly_chart(analytics.plot_churn_by_gender(df_eng), use_container_width=True)
with dm2:
    st.plotly_chart(analytics.plot_churn_by_senior_citizen(df_eng), use_container_width=True)

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> Gender has negligible impact on churn (male ~26%, female ~27%), while senior
    citizen status is a significant differentiator (~42% vs ~24%). Senior customers may face higher
    complexity navigating services or have different value perceptions. Dedicated senior customer
    support and simplified plan offerings could meaningfully reduce this gap.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- Charges vs Tenure Scatter ---
st.markdown("<div class='section-header'>💡 Monthly Charges vs. Tenure — Churn Risk Map</div>", unsafe_allow_html=True)
st.plotly_chart(analytics.plot_charges_vs_tenure_scatter(df_eng), use_container_width=True)

st.markdown("""
<div class='insight-box'>
    <b>📌 Key Finding:</b> The scatter plot reveals a clear high-risk zone in the top-left quadrant:
    customers with high monthly charges AND low tenure. These customers haven't yet built loyalty and
    may feel they aren't getting enough value for their cost. The bottom-right quadrant (high tenure,
    varied charges) represents the company's most loyal base — these customers should be enrolled
    in loyalty programs to prevent late-stage churn.
</div>
""", unsafe_allow_html=True)
