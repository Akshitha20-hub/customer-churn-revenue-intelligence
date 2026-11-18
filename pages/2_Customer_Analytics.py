"""
Page 2: Customer Analytics
===========================
Interactive exploration of customer cohorts, demographics, services, and behavioral distributions.
Equipped with dynamic multi-variable filters and responsive Plotly analytics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import src.analytics as analytics

st.set_page_config(page_title="Customer Analytics", page_icon="👥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1e1e3a 0%,#16213e 100%);border-right:1px solid rgba(108,99,255,0.3);}
.kpi-card{background:linear-gradient(135deg,rgba(108,99,255,0.15) 0%,rgba(54,153,255,0.1) 100%);border:1px solid rgba(108,99,255,0.4);border-radius:12px;padding:16px 20px;text-align:center;margin-bottom:8px;}
.kpi-value{font-size:1.8rem;font-weight:700;color:#6C63FF;line-height:1.1;}
.kpi-label{font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:4px;text-transform:uppercase;letter-spacing:0.08em;}
.section-header{font-size:1.05rem;font-weight:600;color:rgba(255,255,255,0.9);padding:12px 0 6px;border-bottom:2px solid rgba(108,99,255,0.5);margin-bottom:16px;}
.filter-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px;margin-bottom:20px;}
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
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>👥 Customer Analytics & Demographics</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Interactive slice-and-dice cohort exploration across contracts, tenure, and demographics</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Interactive Sidebar Filters -----------------
st.sidebar.markdown("### 🔍 Customer Cohort Filters")

all_contracts = sorted(df_eng["contract"].dropna().unique().tolist())
selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=all_contracts,
    default=all_contracts
)

all_internets = sorted(df_eng["internet_service"].dropna().unique().tolist())
selected_internet = st.sidebar.multiselect(
    "Internet Service",
    options=all_internets,
    default=all_internets
)

all_payments = sorted(df_eng["payment_method"].dropna().unique().tolist())
selected_payments = st.sidebar.multiselect(
    "Payment Method",
    options=all_payments,
    default=all_payments
)

churn_status = st.sidebar.radio(
    "Customer Status",
    options=["All Customers", "Active Only", "Churned Only"],
    index=0
)

tenure_min, tenure_max = int(df_eng["tenure"].min()), int(df_eng["tenure"].max())
selected_tenure = st.sidebar.slider(
    "Tenure Range (Months)",
    min_value=tenure_min,
    max_value=tenure_max,
    value=(tenure_min, tenure_max)
)

# Apply filters
filtered_df = df_eng.copy()
if selected_contracts:
    filtered_df = filtered_df[filtered_df["contract"].isin(selected_contracts)]
if selected_internet:
    filtered_df = filtered_df[filtered_df["internet_service"].isin(selected_internet)]
if selected_payments:
    filtered_df = filtered_df[filtered_df["payment_method"].isin(selected_payments)]

filtered_df = filtered_df[
    (filtered_df["tenure"] >= selected_tenure[0]) &
    (filtered_df["tenure"] <= selected_tenure[1])
]

if churn_status == "Active Only":
    filtered_df = filtered_df[filtered_df["churn_flag"] == 0]
elif churn_status == "Churned Only":
    filtered_df = filtered_df[filtered_df["churn_flag"] == 1]

# ----------------- Summary Metrics -----------------
k1, k2, k3, k4 = st.columns(4)
total_filtered = len(filtered_df)
churn_filtered = int(filtered_df["churn_flag"].sum()) if total_filtered > 0 else 0
churn_rate_filt = (churn_filtered / total_filtered * 100) if total_filtered > 0 else 0.0
avg_spend_filt = filtered_df["monthly_charges"].mean() if total_filtered > 0 else 0.0
avg_tenure_filt = filtered_df["tenure"].mean() if total_filtered > 0 else 0.0

with k1:
    st.markdown(kpi_card(f"{total_filtered:,}", f"Cohort Size ({total_filtered/len(df_eng)*100:.1f}%)"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(f"{churn_rate_filt:.1f}%", "Cohort Churn Rate", "danger" if churn_rate_filt > 30 else "warning"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(f"${avg_spend_filt:.2f}", "Avg Monthly Spend"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card(f"{avg_tenure_filt:.1f} Mo", "Avg Cohort Tenure"), unsafe_allow_html=True)

if filtered_df.empty:
    st.warning("⚠️ No customers match the selected filter criteria. Please broaden your selection.")
    st.stop()

# ----------------- Row 1: Demographics Breakdown -----------------
st.markdown("<div class='section-header'>👤 Demographic Distribution & Churn Risk</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)

with c1:
    # Senior citizen split
    senior_counts = filtered_df.groupby("senior_citizen")["customer_id"].count().reset_index()
    senior_counts["Label"] = senior_counts["senior_citizen"].map({0: "Non-Senior (Younger)", 1: "Senior Citizen (65+)"})
    fig_senior = px.pie(
        senior_counts, names="Label", values="customer_id",
        title="Senior Citizen Ratio",
        color_discrete_sequence=["#6C63FF", "#3699FF"],
        hole=0.45
    )
    fig_senior.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_senior, use_container_width=True)

with c2:
    # Partner & Dependents split
    part_dep = filtered_df.groupby(["partner", "dependents"]).size().reset_index(name="count")
    part_dep["Category"] = "Partner: " + part_dep["partner"] + " | Dep: " + part_dep["dependents"]
    fig_part = px.bar(
        part_dep, x="Category", y="count",
        title="Family & Household Composition",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_part.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_part, use_container_width=True)

with c3:
    # Gender churn comparison
    gender_agg = filtered_df.groupby(["gender", "churn"]).size().reset_index(name="count")
    fig_gen = px.bar(
        gender_agg, x="gender", y="count", color="churn", barmode="group",
        title="Gender Distribution by Churn",
        color_discrete_map={"No": "#1BC5BD", "Yes": "#F64E60"}
    )
    fig_gen.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gen, use_container_width=True)

# ----------------- Row 2: Service & Subscription Analysis -----------------
st.markdown("<div class='section-header'>🌐 Service & Infrastructure Engagement</div>", unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    # Internet service by contract
    net_contract = filtered_df.groupby(["internet_service", "contract"]).size().reset_index(name="Count")
    fig_net = px.bar(
        net_contract, x="internet_service", y="Count", color="contract",
        title="Internet Technology by Contract Commitment",
        color_discrete_sequence=["#F64E60", "#FFA800", "#1BC5BD"],
        barmode="stack"
    )
    fig_net.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_net, use_container_width=True)

with s2:
    # Value-Added Services adoption
    service_cols = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    existing_services = [c for c in service_cols if c in filtered_df.columns]
    if existing_services:
        service_adoption = {}
        for col in existing_services:
            yes_count = (filtered_df[col] == "Yes").sum()
            service_adoption[col.replace("_", " ").title()] = yes_count
        df_serv = pd.DataFrame(list(service_adoption.items()), columns=["Service", "Subscribers"]).sort_values("Subscribers", ascending=True)
        fig_serv = px.bar(
            df_serv, y="Service", x="Subscribers", orientation="h",
            title="Add-On Service Adoption (Adoption Depth)",
            color="Subscribers",
            color_continuous_scale="Purples"
        )
        fig_serv.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_serv, use_container_width=True)

# ----------------- Row 3: Cohort Data Explorer -----------------
st.markdown("<div class='section-header'>📋 Filtered Customer Cohort Data Table</div>", unsafe_allow_html=True)

display_cols = [
    "customer_id", "gender", "senior_citizen", "partner", "tenure",
    "contract", "payment_method", "internet_service", "monthly_charges",
    "total_charges", "churn", "churn_risk_segment"
]
available_disp = [c for c in display_cols if c in filtered_df.columns]

st.dataframe(
    filtered_df[available_disp].head(250),
    use_container_width=True,
    height=320
)

csv_data = filtered_df[available_disp].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Export Filtered Cohort (CSV)",
    data=csv_data,
    file_name="filtered_customer_cohort.csv",
    mime="text/csv"
)
