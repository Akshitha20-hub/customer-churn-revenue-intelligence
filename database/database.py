"""
Database Module
===============
Manages the SQLite database connection and data ingestion
for the Customer Churn Intelligence Platform.

Usage:
    from database.database import init_database, ingest_data, get_connection
"""

import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "churn_intelligence.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection to the platform database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_database() -> None:
    """Create database tables from schema.sql if they don't exist."""
    if not os.path.exists(SCHEMA_PATH):
        logger.error(f"Schema file not found: {SCHEMA_PATH}")
        return
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r") as f:
            schema = f.read()
        conn.executescript(schema)
    logger.info("Database initialized.")


def ingest_customers(df: pd.DataFrame) -> int:
    """
    Insert or replace customer records into the customers table.
    Returns number of records ingested.
    """
    columns = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies", "contract", "paperless_billing",
        "payment_method", "monthly_charges", "total_charges", "churn",
        "churn_flag", "tenure_group", "charge_group", "num_services",
        "clv_indicator", "churn_risk_segment",
    ]
    present = [c for c in columns if c in df.columns]
    df_insert = df[present].copy()

    with get_connection() as conn:
        conn.execute("DELETE FROM churn_predictions")
        conn.execute("DELETE FROM customers")
        df_insert.to_sql(
            "customers", conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
    logger.info(f"Ingested {len(df_insert)} customer records.")
    return len(df_insert)


def ingest_predictions(predictions_df: pd.DataFrame, model_name: str = "best_model") -> int:
    """
    Insert churn prediction results into the churn_predictions table.
    """
    df = predictions_df.copy()
    df["model_name"] = model_name
    df["prediction_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [
        "customer_id", "churn_probability", "risk_category",
        "predicted_churn", "model_name", "prediction_date",
    ]
    present = [c for c in columns if c in df.columns]
    df_insert = df[present]

    with get_connection() as conn:
        conn.execute("DELETE FROM churn_predictions")
        df_insert.to_sql(
            "churn_predictions", conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )
    logger.info(f"Ingested {len(df_insert)} prediction records.")
    return len(df_insert)


def ingest_model_metrics(metrics: dict) -> None:
    """
    Insert model evaluation metrics.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO model_metrics
            (model_name, accuracy, precision_score, recall_score, f1_score, roc_auc, training_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.get("model_name"),
                metrics.get("accuracy"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
                metrics.get("roc_auc"),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    logger.info(f"Stored metrics for model: {metrics.get('model_name')}")


def query_to_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def table_exists(table_name: str) -> bool:
    """Check whether a table exists in the database."""
    with get_connection() as conn:
        result = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        ).fetchone()
    return result is not None


def get_database_stats() -> dict:
    """Return basic statistics about the database."""
    stats = {}
    with get_connection() as conn:
        for table in ["customers", "churn_predictions", "model_metrics"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            except Exception:
                stats[table] = 0
    return stats
