"""
Data Loader Module
==================
Loads the IBM Telco Customer Churn dataset.
Supports loading from local file or downloading from a public URL.

Usage:
    from src.data_loader import load_data
    df = load_data()
"""

import os
import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
LOCAL_FILE = os.path.join(DATA_DIR, "telco_churn.csv")

# GitHub mirror of the IBM Telco dataset (no authentication required)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def download_dataset(url: str = DATASET_URL, save_path: str = LOCAL_FILE) -> bool:
    """
    Download the dataset from a public URL and save it locally.

    Returns True on success, False on failure.
    """
    logger.info(f"Attempting to download dataset from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Dataset saved to: {save_path}")
        return True
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded DataFrame contains the expected columns.
    """
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning(f"Missing expected columns: {missing}")
        return False
    return True


def load_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load the IBM Telco Customer Churn dataset.

    Priority:
    1. Local file at data/raw/telco_churn.csv
    2. Download from public GitHub mirror
    3. Raise FileNotFoundError with manual instructions

    Parameters
    ----------
    force_download : bool
        If True, re-download even if local file exists.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed customer DataFrame.
    """
    if not force_download and os.path.exists(LOCAL_FILE):
        logger.info(f"Loading dataset from local file: {LOCAL_FILE}")
        df = pd.read_csv(LOCAL_FILE)
    else:
        logger.info("Local file not found. Attempting download...")
        success = download_dataset()
        if success and os.path.exists(LOCAL_FILE):
            df = pd.read_csv(LOCAL_FILE)
        else:
            raise FileNotFoundError(
                "Dataset not found. Please download it manually:\n"
                "1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
                "2. Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv'\n"
                "3. Rename it to 'telco_churn.csv'\n"
                f"4. Place it in: {DATA_DIR}"
            )

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if not validate_schema(df):
        logger.warning("Schema validation failed — some columns may be missing.")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Return basic information about the loaded dataset.
    """
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
