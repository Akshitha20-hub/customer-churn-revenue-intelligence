"""
Exports Module
==============
Generates clean CSV files for Power BI integration and external analysis.

Usage:
    from src.exports import generate_all_exports
    generate_all_exports(df_engineered, df_predictions)
"""

import os
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

EXPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


def _save(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(EXPORTS_DIR, filename)
    df.to_csv(path, index=False)
    logger.info(f"Exported: {path} ({len(df)} rows)")
    return path


def export_customers_cleaned(df: pd.DataFrame) -> str:
    """Export the cleaned, feature-engineered customer dataset."""
    return _save(df, "customers_cleaned.csv")


def export_churn_predictions(df_pred: pd.DataFrame) -> str:
    """Export ML prediction results."""
    cols = [
        "customer_id", "churn_probability", "risk_category", "predicted_churn",
        "contract", "tenure", "monthly_charges", "payment_method",
        "internet_service", "churn", "tenure_group",
    ]
    keep = [c for c in cols if c in df_pred.columns]
    return _save(df_pred[keep], "churn_predictions.csv")


def export_churn_by_contract(df: pd.DataFrame) -> str:
    """Export churn rate summary by contract type."""
    grp = df.groupby("contract").agg(
        total_customers=("customer_id", "count"),
        churned=("churn_flag", "sum"),
        avg_monthly_charges=("monthly_charges", "mean"),
        total_monthly_revenue=("monthly_charges", "sum"),
    ).reset_index()
    grp["churn_rate_pct"] = (grp["churned"] / grp["total_customers"] * 100).round(2)
    grp["avg_monthly_charges"] = grp["avg_monthly_charges"].round(2)
    grp["total_monthly_revenue"] = grp["total_monthly_revenue"].round(2)
    return _save(grp, "churn_by_contract.csv")


def export_revenue_by_segment(df: pd.DataFrame) -> str:
    """Export revenue summary by churn risk segment."""
    grp = df.groupby("churn_risk_segment").agg(
        total_customers=("customer_id", "count"),
        churned=("churn_flag", "sum"),
        total_monthly_revenue=("monthly_charges", "sum"),
        avg_monthly_charges=("monthly_charges", "mean"),
        total_clv=("clv_indicator", "sum"),
    ).reset_index()
    grp["churn_rate_pct"] = (grp["churned"] / grp["total_customers"] * 100).round(2)
    grp["avg_monthly_charges"] = grp["avg_monthly_charges"].round(2)
    grp["total_monthly_revenue"] = grp["total_monthly_revenue"].round(2)
    grp["total_clv"] = grp["total_clv"].round(2)
    return _save(grp, "revenue_by_segment.csv")


def export_risk_customers(df_pred: pd.DataFrame, risk_category: str = "High Risk") -> str:
    """Export high-risk customer list with recommendations."""
    df_risk = df_pred[df_pred["risk_category"] == risk_category].copy()
    df_risk = df_risk.sort_values("churn_probability", ascending=False)
    return _save(df_risk, "risk_customers.csv")


def generate_all_exports(df_engineered: pd.DataFrame, df_predictions: pd.DataFrame) -> dict:
    """
    Generate all 5 export files for Power BI and external analysis.

    Returns
    -------
    dict: filename -> file path
    """
    paths = {}
    paths["customers_cleaned"]  = export_customers_cleaned(df_engineered)
    paths["churn_predictions"]  = export_churn_predictions(df_predictions)
    paths["churn_by_contract"]  = export_churn_by_contract(df_engineered)
    paths["revenue_by_segment"] = export_revenue_by_segment(df_engineered)
    paths["risk_customers"]     = export_risk_customers(df_predictions)
    logger.info(f"All exports complete. Files saved to: {EXPORTS_DIR}")
    return paths
