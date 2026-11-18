"""
Page 5: Churn Prediction
========================
Machine learning churn inference engine featuring:
- Real-time single customer what-if scenario simulator
- Batch CSV file inference and scoring
- Model evaluation benchmarks (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance analysis and confusion matrix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import src.analytics as analytics
from src.prediction import predict_customer, predict_batch, classify_risk
from src.recommendations import get_recommendation, apply_recommendations
from src.churn_model import load_model, model_is_trained

st.set_page_config(page_title="Churn Prediction", page_icon="🔮", layout="wide")

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
.rec-box{background:linear-gradient(135deg,rgba(108,99,255,0.15) 0%,rgba(27,197,189,0.1) 100%);border:1px solid #6C63FF;border-radius:10px;padding:18px 22px;margin:12px 0;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def kpi_card(value, label, style=""):
    return f"<div class='kpi-card {style}'><div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div>"


from src.state import ensure_state_initialized
df_eng, df_pred, metadata = ensure_state_initialized()
if df_eng.empty:
    st.stop()

all_metrics = metadata.get("all_metrics", [])
best_model_name = metadata.get("best_model_name", "Logistic Regression")

st.markdown("""
<div style='padding:10px 0 20px'>
    <h1 style='font-size:1.8rem;font-weight:700;color:white;margin:0'>🔮 Machine Learning Churn Prediction Engine</h1>
    <p style='color:rgba(255,255,255,0.5);font-size:0.9rem;margin-top:4px'>Simulate individual customer scenarios or batch-score customer portfolios</p>
</div>
""", unsafe_allow_html=True)

# Tabs: Single Prediction, Batch Prediction, Model Governance
tab1, tab2, tab3 = st.tabs(["🎯 Single Customer Simulation", "📂 Batch CSV Inference", "🔬 Model Governance & Diagnostics"])

# =========================================================================
# TAB 1: SINGLE CUSTOMER SIMULATION
# =========================================================================
with tab1:
    st.markdown("<div class='section-header'>Interactive Customer Attribute Simulator</div>", unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (65+)" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=12)

    with col_in2:
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(monthly_charges * tenure))

    with col_in3:
        st.subheader("Subscribed Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    # Additional defaults for complete feature vector
    customer_dict = {
        "customer_id": "SIM-99999",
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": "No",
        "contract": contract,
        "paperless_billing": paperless,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Calculate Churn Probability & Strategy", type="primary", use_container_width=True):
        pred_res = predict_customer(customer_dict)
        prob = pred_res["churn_probability"]
        risk = pred_res["risk_category"]

        rec_res = get_recommendation({**customer_dict, "churn_probability": prob, "risk_category": risk})

        # Gauge Chart for Probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Customer Churn Risk: {risk}", 'font': {'size': 20, 'color': "white"}},
            number={'suffix': "%", 'font': {'size': 28, 'color': "white"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#F64E60" if risk == "High Risk" else ("#FFA800" if risk == "Medium Risk" else "#1BC5BD")},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 35], 'color': "rgba(27,197,189,0.25)"},
                    {'range': [35, 65], 'color': "rgba(255,168,0,0.25)"},
                    {'range': [65, 100], 'color': "rgba(246,78,96,0.25)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        res1, res2, res3 = st.columns(3)
        with res1:
            color_cls = "danger" if risk == "High Risk" else ("warning" if risk == "Medium Risk" else "success")
            st.markdown(kpi_card(f"{prob * 100:.1f}%", "Churn Probability", color_cls), unsafe_allow_html=True)
        with res2:
            st.markdown(kpi_card(risk, "Assigned Risk Tier", color_cls), unsafe_allow_html=True)
        with res3:
            st.markdown(kpi_card(rec_res.get("urgency", "Standard"), "Action Urgency"), unsafe_allow_html=True)

        st.markdown(f"""
        <div class='rec-box'>
            <h3 style='margin:0 0 8px;color:#6C63FF;font-size:1.15rem'>💡 Recommended Strategy: {rec_res.get('action', 'Standard Engagement')}</h3>
            <p style='margin:0 0 6px;color:rgba(255,255,255,0.9);font-size:0.95rem'><strong>Tactical Step:</strong> {rec_res.get('details', '')}</p>
            <p style='margin:0;color:rgba(255,255,255,0.6);font-size:0.85rem'><strong>Business Rationale:</strong> {rec_res.get('rationale', '')}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# TAB 2: BATCH CSV INFERENCE
# =========================================================================
with tab2:
    st.markdown("<div class='section-header'>Portfolio-Scale Batch Scoring</div>", unsafe_allow_html=True)
    st.write("Upload a CSV with customer records to score all accounts with ML probabilities and generate retention actions.")

    # One-click demo batch scoring
    col_demo1, col_demo2 = st.columns([1, 2])
    with col_demo1:
        run_demo_batch = st.button("⚡ Demo Mode: Score 25 Real Customers", use_container_width=True)

    uploaded_file = st.file_uploader("Or Upload Customer CSV File", type=["csv"])

    target_df = None
    if run_demo_batch and not df_eng.empty:
        target_df = df_eng.sample(25, random_state=42).copy()
        st.success(f"Loaded 25 customer records from production database.")
    elif uploaded_file is not None:
        try:
            target_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(target_df):,} customer records.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if target_df is not None:
        try:
            with st.spinner("Scoring customer accounts with production pipeline..."):
                scored_df = predict_batch(target_df)
                scored_df = apply_recommendations(scored_df)

            st.dataframe(scored_df[[
                "customer_id", "monthly_charges", "tenure", "contract",
                "churn_probability", "risk_category", "retention_action", "retention_urgency"
            ]].head(100), use_container_width=True)

            csv_scored = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=csv_scored,
                file_name="churn_predictions_batch_output.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Failed to process batch records: {e}")
    else:
        st.info("💡 You can test this using the processed dataset or download sample records from the Executive Overview page.")

# =========================================================================
# TAB 3: MODEL GOVERNANCE & DIAGNOSTICS
# =========================================================================
with tab3:
    st.markdown("<div class='section-header'>Model Comparison & Performance Benchmarking</div>", unsafe_allow_html=True)

    if all_metrics:
        # Comparison Table
        metrics_df = pd.DataFrame(all_metrics)[["model_name", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        metrics_df.columns = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        st.dataframe(metrics_df.style.highlight_max(axis=0, color="#2e3856"), use_container_width=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.plotly_chart(analytics.plot_model_comparison(all_metrics), use_container_width=True)
        with m_col2:
            st.plotly_chart(analytics.plot_roc_curve(all_metrics), use_container_width=True)

        # Feature Importance and Confusion Matrix
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            best_metric = next((m for m in all_metrics if m["model_name"] == best_model_name), all_metrics[0])
            st.plotly_chart(analytics.plot_confusion_matrix(best_metric["confusion_matrix"], best_model_name), use_container_width=True)
        with f_col2:
            imp_df = metadata.get("feature_importance", pd.DataFrame())
            if not imp_df.empty:
                st.plotly_chart(analytics.plot_feature_importance(imp_df, top_n=12), use_container_width=True)
            else:
                st.info("Feature importance data is calculated for tree models.")
    else:
        st.info("Train the model pipeline to view performance benchmarks.")
