-- Customer Churn Intelligence Platform
-- Database Schema
-- SQLite

PRAGMA foreign_keys = ON;

-- ============================================================
-- Table: customers
-- Stores cleaned and feature-engineered customer records
-- ============================================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id         TEXT PRIMARY KEY,
    gender              TEXT,
    senior_citizen      INTEGER,
    partner             TEXT,
    dependents          TEXT,
    tenure              INTEGER,
    phone_service       TEXT,
    multiple_lines      TEXT,
    internet_service    TEXT,
    online_security     TEXT,
    online_backup       TEXT,
    device_protection   TEXT,
    tech_support        TEXT,
    streaming_tv        TEXT,
    streaming_movies    TEXT,
    contract            TEXT,
    paperless_billing   TEXT,
    payment_method      TEXT,
    monthly_charges     REAL,
    total_charges       REAL,
    churn               TEXT,
    churn_flag          INTEGER,
    tenure_group        TEXT,
    charge_group        TEXT,
    num_services        INTEGER,
    clv_indicator       REAL,
    churn_risk_segment  TEXT
);

-- ============================================================
-- Table: churn_predictions
-- Stores ML model prediction results
-- ============================================================
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         TEXT,
    churn_probability   REAL,
    risk_category       TEXT,
    predicted_churn     INTEGER,
    model_name          TEXT,
    prediction_date     TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ============================================================
-- Table: model_metrics
-- Stores ML model evaluation results
-- ============================================================
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name          TEXT,
    accuracy            REAL,
    precision_score     REAL,
    recall_score        REAL,
    f1_score            REAL,
    roc_auc             REAL,
    training_date       TEXT
);
