"""
Prediction Module
=================
Provides single-customer and batch churn prediction using the saved model.

Usage:
    from src.prediction import predict_customer, predict_batch, classify_risk
"""

import pandas as pd
import numpy as np
import logging
from src.churn_model import load_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk thresholds — configurable
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "high":   0.65,   # >= 65% churn probability → High Risk
    "medium": 0.35,   # 35–65% → Medium Risk
    # < 35% → Low Risk
}


def classify_risk(probability: float, thresholds: dict = None) -> str:
    """
    Convert churn probability to a risk category label.

    NOTE: Risk categories are model-based estimates, not guaranteed outcomes.
    """
    t = thresholds or RISK_THRESHOLDS
    if probability >= t["high"]:
        return "High Risk"
    elif probability >= t["medium"]:
        return "Medium Risk"
    else:
        return "Low Risk"


def predict_customer(customer_dict: dict, thresholds: dict = None) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_dict : dict
        Customer attributes matching feature column names.
    thresholds : dict, optional
        Custom risk thresholds.

    Returns
    -------
    dict: churn_probability, risk_category, predicted_churn
    """
    pipeline, metadata = load_model()
    if pipeline is None:
        return {"error": "Model not trained. Please train the model first."}

    all_features = metadata.get("all_features", [])
    input_df = pd.DataFrame([customer_dict])

    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0 if col in ["tenure", "monthly_charges", "total_charges",
                                          "num_services", "senior_citizen"] else "No"

    prob = pipeline.predict_proba(input_df[all_features])[0][1]
    risk = classify_risk(prob, thresholds)

    return {
        "churn_probability": round(float(prob), 4),
        "risk_category":     risk,
        "predicted_churn":   int(prob >= 0.5),
    }


def predict_batch(df: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Predict churn probability for a batch of customers.

    Returns original DataFrame with added prediction columns.
    """
    pipeline, metadata = load_model()
    if pipeline is None:
        df = df.copy()
        df["churn_probability"] = np.nan
        df["risk_category"]     = "Unknown"
        df["predicted_churn"]   = -1
        return df

    all_features = metadata.get("all_features", [])
    input_df = pd.DataFrame(index=df.index)

    for col in all_features:
        if col in df.columns:
            input_df[col] = df[col]
        else:
            input_df[col] = 0 if col in ["tenure", "monthly_charges", "total_charges",
                                           "num_services", "senior_citizen"] else "No"

    probs = pipeline.predict_proba(input_df[all_features])[:, 1]
    preds = (probs >= 0.5).astype(int)
    risks = [classify_risk(p, thresholds) for p in probs]

    result = df.copy()
    result["churn_probability"] = np.round(probs, 4)
    result["predicted_churn"]   = preds
    result["risk_category"]     = risks
    return result


def get_predictions_for_all(df_engineered: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """
    Run batch prediction on the full engineered dataset.
    Returns DataFrame with customer_id, predictions, and key attributes.
    """
    result = predict_batch(df_engineered, thresholds)

    output_cols = ["customer_id", "churn_probability", "risk_category", "predicted_churn"]
    extra_cols  = [
        "contract", "tenure", "monthly_charges", "payment_method",
        "internet_service", "churn", "tenure_group", "charge_group",
    ]
    keep = output_cols + [c for c in extra_cols if c in result.columns]
    return result[[c for c in keep if c in result.columns]]
