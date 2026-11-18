"""
Preprocessing Module
====================
Cleans and transforms the raw IBM Telco Customer Churn dataset.

Functions are designed to be stateless and composable so they can
be used both in standalone scripts and within ML pipelines.

Usage:
    from src.preprocessing import preprocess_data
    df_clean = preprocess_data(df_raw)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to their correct dtypes.
    - TotalCharges is stored as a string in the raw dataset.
    - SeniorCitizen is 0/1 integer — keep as int.
    """
    df = df.copy()

    # TotalCharges has whitespace strings instead of NaN for some rows
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Ensure numeric columns are float
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    logger.info("Data types fixed.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows and duplicate customerIDs.
    """
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["customerID"], keep="first")
    removed = before - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate records.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values:
    - TotalCharges NaN: impute with tenure * MonthlyCharges (logical estimate)
    - Any remaining NaN in numeric cols: fill with median
    - Any remaining NaN in categorical cols: fill with mode
    """
    df = df.copy()

    # TotalCharges imputation
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = df.loc[mask, "tenure"] * df.loc[mask, "MonthlyCharges"]
    logger.info(f"Imputed {mask.sum()} TotalCharges values.")

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info("Missing values handled.")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to snake_case for consistency.
    """
    rename_map = {
        "customerID": "customer_id",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
        "gender": "gender",
        "tenure": "tenure",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def create_churn_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary churn flag (1 = churned, 0 = active).
    """
    df = df.copy()
    df["churn_flag"] = (df["churn"].str.strip().str.lower() == "yes").astype(int)
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No binary columns to 1/0.
    Keeps originals for readability in the UI.
    """
    binary_cols = [
        "partner", "dependents", "phone_service", "paperless_billing",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    for col in binary_cols:
        if col in df.columns:
            new_col = f"{col}_enc"
            df[new_col] = df[col].map({"Yes": 1, "No": 0})
            df[new_col] = df[new_col].fillna(0).astype(int)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master preprocessing pipeline.

    Steps:
    1. Fix data types
    2. Remove duplicates
    3. Handle missing values
    4. Standardize column names
    5. Create churn flag
    6. Encode binary columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from load_data()

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    logger.info("Starting preprocessing pipeline...")
    df = fix_data_types(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = standardize_column_names(df)
    df = create_churn_flag(df)
    df = encode_binary_columns(df)
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df
