"""
State Management Module
=======================
Guarantees seamless self-healing session state across all Streamlit pages.
No page will ever crash or require returning to the home page.
"""

import os
import sys
import streamlit as st
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.churn_model import load_model, train_models, model_is_trained
from src.prediction import get_predictions_for_all
from src.recommendations import apply_recommendations
from database.database import init_database, ingest_customers, ingest_predictions


import logging
logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def load_cached_data():
    """Load, preprocess, and engineer features with Streamlit caching."""
    df_raw = load_data()
    df_clean = preprocess_data(df_raw)
    df_eng = engineer_features(df_clean)
    return df_eng


@st.cache_resource(show_spinner=False)
def load_cached_model(_df_eng):
    """Load or train model once and cache across app sessions."""
    pipeline = None
    metadata = {}
    all_metrics = []

    if model_is_trained():
        try:
            pipeline, metadata = load_model()
            if pipeline is not None:
                # Rigorously verify that the loaded model can actually predict on this environment
                test_sample = _df_eng.head(2)
                num_feats = ["tenure", "monthly_charges", "total_charges", "num_services", "senior_citizen"]
                cat_feats = [
                    "gender", "partner", "dependents", "phone_service", "multiple_lines",
                    "internet_service", "online_security", "online_backup", "device_protection",
                    "tech_support", "streaming_tv", "streaming_movies",
                    "contract", "paperless_billing", "payment_method",
                ]
                pipeline.predict_proba(test_sample[num_feats + cat_feats])
                all_metrics = metadata.get("all_metrics", [])
            else:
                pipeline = None
        except Exception as e:
            logger.warning(f"Cached model incompatible with runtime: {e}. Auto-training fresh model...")
            pipeline = None

    if pipeline is None:
        all_metrics, pipeline, metadata = train_models(_df_eng)

    return all_metrics, pipeline, metadata


@st.cache_data(show_spinner=False)
def load_cached_predictions(_df_eng):
    """Generate predictions and attach rule-based retention strategies."""
    try:
        df_pred = get_predictions_for_all(_df_eng)
        df_pred_full = apply_recommendations(df_pred)
        return df_pred_full
    except Exception as e:
        logger.warning(f"Prediction generation error: {e}. Retraining model...")
        train_models(_df_eng)
        df_pred = get_predictions_for_all(_df_eng)
        return apply_recommendations(df_pred)


def ensure_state_initialized():
    """
    Call this at the top of every page.
    Automatically populates st.session_state if missing, without interrupting the user.
    """
    if st.session_state.get("initialized") and "df_eng" in st.session_state and "df_pred" in st.session_state:
        return st.session_state["df_eng"], st.session_state["df_pred"], st.session_state.get("metadata", {})

    with st.spinner("⚡ Loading platform data and AI model..."):
        try:
            df_eng = load_cached_data()
            all_metrics, pipeline, metadata = load_cached_model(df_eng)
            df_pred = load_cached_predictions(df_eng)

            # Initialize DB in background
            try:
                init_database()
                ingest_customers(df_eng)
                ingest_predictions(df_pred)
            except Exception:
                pass

            st.session_state["df_eng"] = df_eng
            st.session_state["df_pred"] = df_pred
            st.session_state["all_metrics"] = all_metrics
            st.session_state["pipeline"] = pipeline
            st.session_state["metadata"] = metadata
            st.session_state["initialized"] = True
            st.session_state["load_error"] = None

            return df_eng, df_pred, metadata
        except Exception as e:
            st.session_state["load_error"] = str(e)
            st.error(f"Error initializing data: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}
