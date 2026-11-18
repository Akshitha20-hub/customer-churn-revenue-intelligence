"""
Churn Prediction Model Module
==============================
Trains, evaluates, and saves churn prediction models.

Models compared:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Usage:
    from src.churn_model import train_models, load_model
    results, best_model, metadata = train_models(df)
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_churn_model.pkl")
METADATA_PATH   = os.path.join(MODELS_DIR, "model_metadata.pkl")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "num_services", "senior_citizen"]
CATEGORICAL_FEATURES = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method",
]
TARGET = "churn_flag"


def get_feature_columns(df: pd.DataFrame) -> tuple:
    """Return present numeric and categorical feature lists."""
    num_feats = [f for f in NUMERIC_FEATURES if f in df.columns]
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in df.columns]
    return num_feats, cat_feats


def build_pipeline(model, num_feats: list, cat_feats: list) -> Pipeline:
    """
    Build a full sklearn Pipeline:
    - Numeric: impute median → scale
    - Categorical: impute most_frequent → one-hot encode
    - Classifier on top
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Evaluate a trained pipeline and return metrics dict."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name":  model_name,
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "precision":   round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":          round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "y_test":   y_test.tolist(),
        "y_proba":  y_proba.tolist(),
        "y_pred":   y_pred.tolist(),
    }
    logger.info(
        f"{model_name}: Acc={metrics['accuracy']}, "
        f"F1={metrics['f1']}, ROC-AUC={metrics['roc_auc']}"
    )
    return metrics


def get_feature_importance(pipeline: Pipeline, num_feats: list, cat_feats: list) -> pd.DataFrame:
    """
    Extract feature importance from the trained classifier.
    Works for tree-based models (feature_importances_) and
    linear models (coef_).
    """
    classifier   = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(cat_feats).tolist()
    except Exception:
        cat_feature_names = cat_feats

    all_features = num_feats + cat_feature_names

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame()

    min_len = min(len(importances), len(all_features))
    importances   = importances[:min_len]
    all_features  = all_features[:min_len]

    return pd.DataFrame({
        "feature":    all_features,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def train_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Train and compare Logistic Regression, Decision Tree, and Random Forest.

    Returns
    -------
    tuple: (list of metrics dicts, best pipeline, metadata dict)
    """
    logger.info("Starting model training...")

    num_feats, cat_feats = get_feature_columns(df)
    X = df[num_feats + cat_feats].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=random_state, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state,
            class_weight="balanced", n_jobs=-1
        ),
    }

    all_metrics = []
    trained_pipelines = {}

    for name, clf in models.items():
        logger.info(f"Training {name}...")
        pipeline = build_pipeline(clf, num_feats, cat_feats)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        all_metrics.append(metrics)
        trained_pipelines[name] = pipeline

    # Select best model by ROC-AUC (better for imbalanced classification)
    best          = max(all_metrics, key=lambda m: m["roc_auc"])
    best_pipeline = trained_pipelines[best["model_name"]]
    logger.info(f"Best model: {best['model_name']} (ROC-AUC={best['roc_auc']})")

    importance_df = get_feature_importance(best_pipeline, num_feats, cat_feats)

    metadata = {
        "best_model_name":  best["model_name"],
        "num_features":     num_feats,
        "cat_features":     cat_feats,
        "all_features":     num_feats + cat_feats,
        "feature_importance": importance_df,
        "test_size":        test_size,
        "random_state":     random_state,
        "training_date":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples":    len(X_train),
        "test_samples":     len(X_test),
        "all_metrics":      all_metrics,
    }

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)
    logger.info(f"Model saved to: {BEST_MODEL_PATH}")

    return all_metrics, best_pipeline, metadata


def load_model() -> tuple:
    """
    Load the saved best model and metadata.

    Returns
    -------
    tuple: (pipeline, metadata) or (None, None) if not found.
    """
    if not os.path.exists(BEST_MODEL_PATH):
        logger.warning("No trained model found. Please run train_models() first.")
        return None, None
    pipeline = joblib.load(BEST_MODEL_PATH)
    metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return pipeline, metadata


def model_is_trained() -> bool:
    """Check if a trained model exists on disk."""
    return os.path.exists(BEST_MODEL_PATH)
