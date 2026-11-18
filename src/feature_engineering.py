"""
Feature Engineering Module
==========================
Creates derived features from the cleaned customer dataset.
All features are business-meaningful and interpretable.

Usage:
    from src.feature_engineering import engineer_features
    df = engineer_features(df_clean)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by their tenure (months with the company).

    Groups:
    - New (0-12 months)
    - Developing (13-24 months)
    - Established (25-48 months)
    - Loyal (49-72 months)
    """
    df = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["New (0-12m)", "Developing (13-24m)", "Established (25-48m)", "Loyal (49-72m)"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    )
    df["tenure_group"] = df["tenure_group"].astype(str)
    return df


def create_charge_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by monthly charge bands.

    Groups:
    - Low (<$40)
    - Medium ($40-$70)
    - High ($70-$90)
    - Premium (>$90)
    """
    df = df.copy()
    bins = [0, 40, 70, 90, float("inf")]
    labels = ["Low (<$40)", "Medium ($40-70)", "High ($70-90)", "Premium (>$90)"]
    df["charge_group"] = pd.cut(
        df["monthly_charges"], bins=bins, labels=labels, include_lowest=True
    )
    df["charge_group"] = df["charge_group"].astype(str)
    return df


def count_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of active add-on services per customer.
    Services: OnlineSecurity, OnlineBackup, DeviceProtection,
              TechSupport, StreamingTV, StreamingMovies, MultipleLines.
    """
    df = df.copy()
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "multiple_lines",
    ]
    present_cols = [c for c in service_cols if c in df.columns]
    df["num_services"] = df[present_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    return df


def create_clv_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple Customer Lifetime Value indicator.
    CLV = monthly_charges * tenure
    This is a revenue-based approximation, not a predictive CLV model.
    """
    df = df.copy()
    df["clv_indicator"] = df["monthly_charges"] * df["tenure"]
    return df


def create_churn_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based pre-ML risk segment based on known churn predictors:
    - Month-to-month contract is a strong churn signal
    - Low tenure + high charges = elevated risk
    - Long tenure = reduced risk
    """
    df = df.copy()

    def classify(row):
        if row.get("contract") == "Month-to-month":
            if row.get("tenure", 100) <= 12:
                return "High Risk"
            else:
                return "Medium Risk"
        elif row.get("contract") == "One year":
            return "Medium Risk"
        else:
            return "Low Risk"

    df["churn_risk_segment"] = df.apply(classify, axis=1)
    return df


def create_contract_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simplified contract category feature.
    """
    df = df.copy()
    contract_map = {
        "Month-to-month": "Short-term",
        "One year": "Mid-term",
        "Two year": "Long-term",
    }
    df["contract_category"] = df["contract"].map(contract_map).fillna("Unknown")
    return df


def create_high_value_flag(df: pd.DataFrame, threshold_percentile: int = 75) -> pd.DataFrame:
    """
    Flag customers in the top N percentile of monthly charges as high-value.
    """
    df = df.copy()
    threshold = df["monthly_charges"].quantile(threshold_percentile / 100)
    df["high_value_customer"] = (df["monthly_charges"] >= threshold).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.

    Applies all feature engineering steps in order.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocess_data()

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame ready for ML and analytics.
    """
    logger.info("Starting feature engineering...")
    df = create_tenure_groups(df)
    df = create_charge_groups(df)
    df = count_services(df)
    df = create_clv_indicator(df)
    df = create_churn_risk_segment(df)
    df = create_contract_category(df)
    df = create_high_value_flag(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
