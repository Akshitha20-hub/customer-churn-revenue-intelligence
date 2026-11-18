"""
Platform Setup Script
======================
Run this once to verify the environment and kick off initial data processing.
This is separate from the Streamlit app and can be used for headless testing.

Usage:
    python setup.py
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    logger.info("=" * 60)
    logger.info("Customer Churn & Revenue Intelligence Platform — Setup")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("Step 1: Loading dataset...")
    from src.data_loader import load_data, get_dataset_info
    df_raw = load_data()
    info = get_dataset_info(df_raw)
    logger.info(f"  Rows: {info['total_rows']} | Columns: {info['total_columns']}")
    logger.info(f"  Missing values: {info['missing_values']} | Duplicates: {info['duplicate_rows']}")

    # Step 2: Preprocess
    logger.info("Step 2: Preprocessing...")
    from src.preprocessing import preprocess_data
    df_clean = preprocess_data(df_raw)
    logger.info(f"  Clean shape: {df_clean.shape}")

    # Step 3: Feature engineering
    logger.info("Step 3: Feature engineering...")
    from src.feature_engineering import engineer_features
    df_eng = engineer_features(df_clean)
    logger.info(f"  Engineered shape: {df_eng.shape}")

    # Save processed data
    processed_path = os.path.join(ROOT, "data", "processed", "customers_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_eng.to_csv(processed_path, index=False)
    logger.info(f"  Saved processed data to: {processed_path}")

    # Step 4: Database
    logger.info("Step 4: Initializing database...")
    from database.database import init_database, ingest_customers, get_database_stats
    init_database()
    n = ingest_customers(df_eng)
    logger.info(f"  Ingested {n} records. Stats: {get_database_stats()}")

    # Step 5: SQL validation
    logger.info("Step 5: Validating SQL queries...")
    from src.sql_queries import get_summary_kpis, get_churn_by_contract
    kpis = get_summary_kpis()
    logger.info(f"  KPIs: {kpis}")
    df_contract = get_churn_by_contract()
    logger.info(f"  Churn by contract:\n{df_contract.to_string(index=False)}")

    # Step 6: ML training
    logger.info("Step 6: Training ML models...")
    from src.churn_model import train_models
    all_metrics, pipeline, metadata = train_models(df_eng)
    for m in all_metrics:
        logger.info(
            f"  {m['model_name']}: Acc={m['accuracy']:.4f}, "
            f"F1={m['f1']:.4f}, ROC-AUC={m['roc_auc']:.4f}"
        )
    logger.info(f"  Best model: {metadata['best_model_name']}")

    # Step 7: Predictions
    logger.info("Step 7: Running batch predictions...")
    from src.prediction import get_predictions_for_all
    df_pred = get_predictions_for_all(df_eng)
    logger.info(f"  Predictions shape: {df_pred.shape}")
    if "risk_category" in df_pred.columns:
        risk_counts = df_pred["risk_category"].value_counts().to_dict()
        logger.info(f"  Risk distribution: {risk_counts}")

    # Step 8: Recommendations
    logger.info("Step 8: Applying retention recommendations...")
    from src.recommendations import apply_recommendations
    df_pred_full = apply_recommendations(df_pred)
    logger.info(f"  Recommendations applied. Sample actions: {df_pred_full['retention_action'].value_counts().head(3).to_dict()}")

    # Step 9: Ingest predictions to DB
    logger.info("Step 9: Ingesting predictions to database...")
    from database.database import ingest_predictions
    ingest_predictions(df_pred_full)

    # Step 10: Exports
    logger.info("Step 10: Generating Power BI exports...")
    from src.exports import generate_all_exports
    paths = generate_all_exports(df_eng, df_pred_full)
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    logger.info("=" * 60)
    logger.info("✅ Setup complete! Run the app with:")
    logger.info("   streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
