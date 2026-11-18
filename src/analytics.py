"""
Analytics & Visualization Module
==================================
Generates all Plotly interactive charts used throughout the Streamlit dashboard.
All functions return plotly.graph_objs.Figure objects — never call fig.show().

Usage:
    from src.analytics import plot_churn_distribution, plot_churn_by_contract, ...
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color palette — consistent across all charts
# ---------------------------------------------------------------------------
COLORS = {
    "primary":    "#6C63FF",
    "secondary":  "#F64E60",
    "success":    "#1BC5BD",
    "warning":    "#FFA800",
    "info":       "#3699FF",
    "light_bg":   "#1e1e2e",
    "churned":    "#F64E60",
    "active":     "#1BC5BD",
    "high_risk":  "#F64E60",
    "med_risk":   "#FFA800",
    "low_risk":   "#1BC5BD",
}

CHURN_COLOR_MAP   = {"Yes": COLORS["churned"],  "No": COLORS["active"]}
RISK_COLOR_MAP    = {
    "High Risk":   COLORS["high_risk"],
    "Medium Risk": COLORS["med_risk"],
    "Low Risk":    COLORS["low_risk"],
}

CHART_TEMPLATE = "plotly_dark"


# ===========================================================================
# Overview Charts
# ===========================================================================

def plot_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing churned vs active customer ratio."""
    counts = df["churn"].value_counts()
    labels = ["Active" if c == "No" else "Churned" for c in counts.index]
    colors = [COLORS["active"] if c == "No" else COLORS["churned"] for c in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts.values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#1e1e2e", width=3)),
        textinfo="label+percent",
        textfont=dict(size=14),
    )])
    fig.update_layout(
        title="Customer Churn Distribution",
        template=CHART_TEMPLATE,
        showlegend=True,
        height=380,
        annotations=[dict(
            text=f"<b>{len(df):,}</b><br>Total",
            x=0.5, y=0.5, font_size=16, showarrow=False, font_color="white"
        )],
    )
    return fig


def plot_monthly_charges_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of monthly charges split by churn status."""
    fig = px.histogram(
        df, x="monthly_charges", color="churn",
        barmode="overlay", nbins=40,
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"monthly_charges": "Monthly Charges ($)", "churn": "Churn"},
        title="Monthly Charges Distribution by Churn Status",
        template=CHART_TEMPLATE,
        opacity=0.75,
    )
    fig.update_layout(height=380)
    return fig


def plot_tenure_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of customer tenure split by churn status."""
    fig = px.histogram(
        df, x="tenure", color="churn",
        barmode="overlay", nbins=36,
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"tenure": "Tenure (months)", "churn": "Churn"},
        title="Customer Tenure Distribution by Churn Status",
        template=CHART_TEMPLATE,
        opacity=0.75,
    )
    fig.update_layout(height=380)
    return fig


def plot_contract_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of customers by contract type, colored by churn."""
    grp = df.groupby(["contract", "churn"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="contract", y="count", color="churn",
        barmode="group",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"contract": "Contract Type", "count": "Customers", "churn": "Churn"},
        title="Contract Type Distribution",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=380)
    return fig


# ===========================================================================
# Churn Analysis Charts
# ===========================================================================

def plot_churn_rate_by_contract(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: churn rate % by contract type."""
    grp = df.groupby("contract")["churn_flag"].agg(["sum", "count"]).reset_index()
    grp.columns = ["contract", "churned", "total"]
    grp["churn_rate"] = (grp["churned"] / grp["total"] * 100).round(1)

    fig = px.bar(
        grp, x="contract", y="churn_rate",
        text=grp["churn_rate"].apply(lambda x: f"{x}%"),
        color="churn_rate",
        color_continuous_scale=["#1BC5BD", "#FFA800", "#F64E60"],
        labels={"contract": "Contract Type", "churn_rate": "Churn Rate (%)"},
        title="Churn Rate by Contract Type",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig


def plot_churn_by_payment_method(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: churn rate by payment method."""
    grp = df.groupby("payment_method")["churn_flag"].agg(["sum", "count"]).reset_index()
    grp.columns = ["payment_method", "churned", "total"]
    grp["churn_rate"] = (grp["churned"] / grp["total"] * 100).round(1)
    grp = grp.sort_values("churn_rate", ascending=True)

    fig = px.bar(
        grp, x="churn_rate", y="payment_method",
        orientation="h",
        text=grp["churn_rate"].apply(lambda x: f"{x}%"),
        color="churn_rate",
        color_continuous_scale=["#1BC5BD", "#FFA800", "#F64E60"],
        labels={"payment_method": "Payment Method", "churn_rate": "Churn Rate (%)"},
        title="Churn Rate by Payment Method",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig


def plot_churn_by_tenure_group(df: pd.DataFrame) -> go.Figure:
    """Bar chart: churn rate by tenure group."""
    grp = df.groupby("tenure_group")["churn_flag"].agg(["sum", "count"]).reset_index()
    grp.columns = ["tenure_group", "churned", "total"]
    grp["churn_rate"] = (grp["churned"] / grp["total"] * 100).round(1)

    fig = px.bar(
        grp, x="tenure_group", y="churn_rate",
        text=grp["churn_rate"].apply(lambda x: f"{x}%"),
        color="churn_rate",
        color_continuous_scale=["#1BC5BD", "#FFA800", "#F64E60"],
        labels={"tenure_group": "Tenure Group", "churn_rate": "Churn Rate (%)"},
        title="Churn Rate by Tenure Group",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, coloraxis_showscale=False)
    return fig


def plot_churn_by_internet_service(df: pd.DataFrame) -> go.Figure:
    """Donut charts for churn rate by internet service type."""
    grp = df.groupby("internet_service")["churn_flag"].agg(["sum", "count"]).reset_index()
    grp.columns = ["internet_service", "churned", "total"]
    grp["churn_rate"] = (grp["churned"] / grp["total"] * 100).round(1)

    fig = px.bar(
        grp, x="internet_service", y="churn_rate",
        text=grp["churn_rate"].apply(lambda x: f"{x}%"),
        color="internet_service",
        color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["success"]],
        labels={"internet_service": "Internet Service", "churn_rate": "Churn Rate (%)"},
        title="Churn Rate by Internet Service Type",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_churn_by_gender(df: pd.DataFrame) -> go.Figure:
    """Side-by-side bar chart for churn by gender."""
    grp = df.groupby(["gender", "churn"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="gender", y="count", color="churn",
        barmode="group",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"gender": "Gender", "count": "Customers", "churn": "Churn"},
        title="Churn by Gender",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=380)
    return fig


def plot_churn_by_senior_citizen(df: pd.DataFrame) -> go.Figure:
    """Bar chart: churn rate for senior vs non-senior customers."""
    grp = df.copy()
    grp["Senior"] = grp["senior_citizen"].map({1: "Senior", 0: "Non-Senior"})
    grp = grp.groupby(["Senior", "churn"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="Senior", y="count", color="churn",
        barmode="group",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"Senior": "Customer Type", "count": "Customers", "churn": "Churn"},
        title="Churn by Senior Citizen Status",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=380)
    return fig


def plot_charges_vs_tenure_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: monthly charges vs tenure, colored by churn."""
    sample = df.sample(min(1000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="tenure", y="monthly_charges", color="churn",
        color_discrete_map=CHURN_COLOR_MAP,
        opacity=0.65,
        labels={"tenure": "Tenure (months)", "monthly_charges": "Monthly Charges ($)", "churn": "Churn"},
        title="Monthly Charges vs. Tenure (sample of 1,000)",
        template=CHART_TEMPLATE,
        hover_data=["contract", "internet_service"],
    )
    fig.update_layout(height=420)
    return fig


# ===========================================================================
# Revenue Charts
# ===========================================================================

def plot_revenue_by_contract(df: pd.DataFrame) -> go.Figure:
    """Bar chart: total monthly revenue by contract type."""
    grp = df.groupby("contract")["monthly_charges"].sum().reset_index()
    grp.columns = ["contract", "total_monthly_revenue"]
    grp = grp.sort_values("total_monthly_revenue", ascending=False)

    fig = px.bar(
        grp, x="contract", y="total_monthly_revenue",
        text=grp["total_monthly_revenue"].apply(lambda x: f"${x:,.0f}"),
        color="contract",
        color_discrete_sequence=[COLORS["primary"], COLORS["info"], COLORS["success"]],
        labels={"contract": "Contract Type", "total_monthly_revenue": "Monthly Revenue ($)"},
        title="Monthly Revenue by Contract Type",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_revenue_by_risk_category(df_pred: pd.DataFrame) -> go.Figure:
    """Bar chart: monthly revenue distribution across ML risk categories."""
    if "risk_category" not in df_pred.columns:
        return go.Figure()

    grp = df_pred.groupby("risk_category")["monthly_charges"].agg(["sum", "count", "mean"]).reset_index()
    grp.columns = ["risk_category", "total_revenue", "customers", "avg_charges"]

    colors = [RISK_COLOR_MAP.get(r, COLORS["primary"]) for r in grp["risk_category"]]

    fig = px.bar(
        grp, x="risk_category", y="total_revenue",
        text=grp["total_revenue"].apply(lambda x: f"${x:,.0f}"),
        color="risk_category",
        color_discrete_map=RISK_COLOR_MAP,
        labels={"risk_category": "Risk Category", "total_revenue": "Monthly Revenue ($)"},
        title="Monthly Revenue by Predicted Risk Category",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_clv_by_tenure_group(df: pd.DataFrame) -> go.Figure:
    """Box plot: CLV indicator distribution per tenure group."""
    if "clv_indicator" not in df.columns:
        return go.Figure()

    fig = px.box(
        df, x="tenure_group", y="clv_indicator", color="churn",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"tenure_group": "Tenure Group", "clv_indicator": "CLV Indicator ($)", "churn": "Churn"},
        title="Customer Lifetime Value (Approx.) by Tenure Group",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=420)
    return fig


def plot_charge_group_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: customer count by charge group and churn."""
    grp = df.groupby(["charge_group", "churn"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="charge_group", y="count", color="churn",
        barmode="stack",
        color_discrete_map=CHURN_COLOR_MAP,
        labels={"charge_group": "Charge Group", "count": "Customers", "churn": "Churn"},
        title="Customer Distribution by Monthly Charge Group",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=400)
    return fig


# ===========================================================================
# ML / Model Charts
# ===========================================================================

def plot_model_comparison(all_metrics: list) -> go.Figure:
    """Grouped bar chart comparing model evaluation metrics."""
    if not all_metrics:
        return go.Figure()

    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models = [m["model_name"] for m in all_metrics]

    fig = go.Figure()
    metric_colors = [COLORS["primary"], COLORS["info"], COLORS["success"], COLORS["warning"], COLORS["secondary"]]

    for i, metric in enumerate(metric_cols):
        values = [m[metric] for m in all_metrics]
        fig.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=models,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            marker_color=metric_colors[i],
        ))

    fig.update_layout(
        barmode="group",
        title="Model Comparison — Evaluation Metrics",
        template=CHART_TEMPLATE,
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis_title="Model",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_confusion_matrix(cm: list, model_name: str = "") -> go.Figure:
    """Heatmap visualization of the confusion matrix."""
    labels = ["Active (0)", "Churned (1)"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#1e1e2e"], [1, COLORS["primary"]]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        showscale=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        template=CHART_TEMPLATE,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
    )
    return fig


def plot_roc_curve(all_metrics: list) -> go.Figure:
    """ROC curves for all models with AUC scores."""
    from sklearn.metrics import roc_curve

    fig = go.Figure()
    colors_list = [COLORS["primary"], COLORS["warning"], COLORS["success"]]

    for i, m in enumerate(all_metrics):
        if "y_test" not in m or "y_proba" not in m:
            continue
        fpr, tpr, _ = roc_curve(m["y_test"], m["y_proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{m['model_name']} (AUC={m['roc_auc']:.3f})",
            line=dict(color=colors_list[i % len(colors_list)], width=2.5),
        ))

    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line=dict(dash="dash", color="gray", width=1.5),
    ))

    fig.update_layout(
        title="ROC Curves — Model Comparison",
        template=CHART_TEMPLATE,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=440,
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart: top N feature importances."""
    if importance_df is None or importance_df.empty:
        return go.Figure()

    df_top = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig = px.bar(
        df_top, x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=[[0, COLORS["info"]], [1, COLORS["primary"]]],
        labels={"importance": "Importance Score", "feature": "Feature"},
        title=f"Top {top_n} Features by Importance",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(height=max(380, top_n * 28), coloraxis_showscale=False)
    return fig


def plot_risk_distribution(df_pred: pd.DataFrame) -> go.Figure:
    """Donut chart: distribution of customers across risk categories."""
    if "risk_category" not in df_pred.columns:
        return go.Figure()

    counts = df_pred["risk_category"].value_counts()
    colors = [RISK_COLOR_MAP.get(c, COLORS["primary"]) for c in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#1e1e2e", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13),
    )])
    fig.update_layout(
        title="Customer Risk Category Distribution",
        template=CHART_TEMPLATE,
        height=380,
    )
    return fig


def plot_churn_probability_histogram(df_pred: pd.DataFrame) -> go.Figure:
    """Histogram of predicted churn probabilities."""
    if "churn_probability" not in df_pred.columns:
        return go.Figure()

    fig = px.histogram(
        df_pred, x="churn_probability", nbins=40,
        color_discrete_sequence=[COLORS["primary"]],
        labels={"churn_probability": "Predicted Churn Probability"},
        title="Distribution of Predicted Churn Probabilities",
        template=CHART_TEMPLATE,
        opacity=0.8,
    )
    # Add threshold lines
    fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Medium Risk (35%)", annotation_position="top right")
    fig.add_vline(x=0.65, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text="High Risk (65%)", annotation_position="top right")
    fig.update_layout(height=400)
    return fig
