"""
Page 6: Retention Strategy
==========================
Operational workbench for customer success and retention marketing teams.
Displays targeted customer action lists, rule-based retention triggers,
and CSV export facilities for campaign execution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import src.analytics as analytics
from src.recommendations import apply_recommendations
from src.prediction import classify_risk

st.set_page_config(page_title="Retention Strategy", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1e3a 0%,#16213e 100%);border-right:1px solid rgba(108,99,255,0.3);}
.kpi-card{background:linear-gradient(135deg,rgba(108,99,255,0.15) 0%,rgba(54,153,255,0.1) 100%);border:1px solid rgba(108,99,255,0.4);border-radius:12px;padding:16px 20px;text-align:center;margin-bottom:8px;}
.kpi-value{font-size:1.8rem;font-weight:700;color:#6C63FF;line-height:1.1;}
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
if df_pred.empty:
    st.stop()

# Ensure recommendations are attached
if "retention_action" not in df_pred.columns:
    df_pred = apply_recommendations(df_pred)
    st.session_state["df_pred"] = df_pred

st.markdown("""
<div style='padding:10px 0 20px'>
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>🎯 Customer Retention Operations & Action Desk</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Actionable customer queues prioritized by risk tier, monthly revenue, and intervention playbooks</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Sidebar Filters & Threshold Adjuster -----------------
st.sidebar.markdown("### 🎛️ Risk Threshold Tuning")
st.sidebar.caption("Adjust the classification boundaries for operational risk categorisation.")

low_threshold = st.sidebar.slider("Low / Medium Boundary", 0.10, 0.50, 0.35, 0.05)
high_threshold = st.sidebar.slider("Medium / High Boundary", 0.40, 0.85, 0.65, 0.05)

# Dynamically reclassify if user changes thresholds
working_df = df_pred.copy()
working_df["risk_category"] = working_df["churn_probability"].apply(
    lambda p: "Low Risk" if p < low_threshold else ("High Risk" if p >= high_threshold else "Medium Risk")
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Queue Filters")

selected_tiers = st.sidebar.multiselect(
    "Filter by Risk Tier",
    options=["High Risk", "Medium Risk", "Low Risk"],
    default=["High Risk"]
)

available_actions = sorted(working_df["retention_action"].dropna().unique().tolist())
selected_actions = st.sidebar.multiselect(
    "Filter by Recommended Action",
    options=available_actions,
    default=available_actions
)

min_charge = st.sidebar.slider(
    "Minimum Monthly Charge ($)",
    min_value=float(working_df["monthly_charges"].min()),
    max_value=float(working_df["monthly_charges"].max()),
    value=20.0
)

# Apply filters
queue_df = working_df[
    (working_df["risk_category"].isin(selected_tiers)) &
    (working_df["retention_action"].isin(selected_actions)) &
    (working_df["monthly_charges"] >= min_charge)
].sort_values("churn_probability", ascending=False)

# ----------------- Operational Overview KPIs -----------------
total_queue = len(queue_df)
queue_mrr = queue_df["monthly_charges"].sum() if total_queue > 0 else 0.0
avg_prob = queue_df["churn_probability"].mean() * 100 if total_queue > 0 else 0.0
critical_count = (queue_df["retention_urgency"] == "Critical").sum() if "retention_urgency" in queue_df.columns else 0

q1, q2, q3, q4 = st.columns(4)
with q1:
    st.markdown(kpi_card(f"{total_queue:,}", "Active Queue Count"), unsafe_allow_html=True)
with q2:
    st.markdown(kpi_card(f"${queue_mrr:,.2f}", "Queue Monthly Revenue", "warning"), unsafe_allow_html=True)
with q3:
    st.markdown(kpi_card(f"{avg_prob:.1f}%", "Avg Churn Likelihood", "danger"), unsafe_allow_html=True)
with q4:
    st.markdown(kpi_card(f"{critical_count:,}", "Critical Urgency Cases", "danger"), unsafe_allow_html=True)

# ----------------- Strategy Distribution Charts -----------------
st.markdown("<div class='section-header'>📊 Retention Action Portfolio Breakdown</div>", unsafe_allow_html=True)
c_act1, c_act2 = st.columns([3, 2])

with c_act1:
    act_counts = queue_df["retention_action"].value_counts().reset_index()
    act_counts.columns = ["Action", "Count"]
    fig_actions = px.bar(
        act_counts,
        x="Count",
        y="Action",
        orientation="h",
        color="Count",
        color_continuous_scale="Purples",
        title="Intervention Playbook Volume Distribution"
    )
    fig_actions.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig_actions, use_container_width=True)

with c_act2:
    if "retention_urgency" in queue_df.columns:
        urg_counts = queue_df["retention_urgency"].value_counts().reset_index()
        urg_counts.columns = ["Urgency", "Count"]
        fig_urg = px.pie(
            urg_counts,
            names="Urgency",
            values="Count",
            title="Urgency Distribution",
            color="Urgency",
            color_discrete_map={"Critical": "#F64E60", "High": "#FFA800", "Standard": "#1BC5BD"},
            hole=0.45
        )
        fig_urg.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_urg, use_container_width=True)

# ----------------- Customer Outreach Queue Table -----------------
st.markdown("<div class='section-header'>📋 Actionable Customer Outreach Queue</div>", unsafe_allow_html=True)

cols_to_show = [
    "customer_id", "risk_category", "churn_probability", "monthly_charges",
    "tenure", "contract", "payment_method", "retention_action", "retention_urgency", "retention_details"
]
avail_cols = [c for c in cols_to_show if c in queue_df.columns]

display_table = queue_df[avail_cols].copy()
if "churn_probability" in display_table.columns:
    display_table["churn_probability"] = (display_table["churn_probability"] * 100).round(1).astype(str) + "%"

st.dataframe(display_table.head(200), use_container_width=True, height=350)

# Export Facility
csv_queue = queue_df[avail_cols].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export High-Risk Retention Queue (CSV for CRM)",
    data=csv_queue,
    file_name="targeted_retention_queue.csv",
    mime="text/csv"
)
