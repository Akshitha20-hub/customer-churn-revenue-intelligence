"""
SQL Queries Module
==================
Encapsulates reusable analytical SQL queries for the Customer Churn
& Revenue Intelligence Platform, connecting directly to SQLite.

Demonstrates SQL concepts:
- CTEs (Common Table Expressions)
- Window functions (RANK, NTILE, DENSE_RANK)
- Aggregations (GROUP BY, HAVING)
- Conditional logic (CASE WHEN)
- Analytical subqueries

Usage:
    from src.sql_queries import get_summary_kpis, get_churn_by_contract
"""

import sqlite3
import pandas as pd
import logging
from typing import Dict, Any, Optional

from database.database import get_connection

logger = logging.getLogger(__name__)


def run_query(query: str, params: tuple = ()) -> pd.DataFrame:
    """Execute an analytical SQL query and return results as a Pandas DataFrame."""
    try:
        with get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        logger.error(f"Error executing SQL query: {e}")
        return pd.DataFrame()


def get_summary_kpis() -> Dict[str, Any]:
    """
    Retrieve core executive KPIs using SQL aggregations.
    Returns:
        dict: total_customers, churned_customers, active_customers,
              churn_rate_pct, avg_monthly_charges, total_monthly_revenue, avg_tenure
    """
    query = """
    WITH customer_stats AS (
        SELECT
            COUNT(*) AS total_customers,
            SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
            SUM(CASE WHEN churn_flag = 0 THEN 1 ELSE 0 END) AS active_customers,
            ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
            ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
            ROUND(AVG(tenure), 1) AS avg_tenure
        FROM customers
    )
    SELECT
        total_customers,
        churned_customers,
        active_customers,
        ROUND((CAST(churned_customers AS REAL) / NULLIF(total_customers, 0)) * 100, 2) AS churn_rate_pct,
        avg_monthly_charges,
        total_monthly_revenue,
        avg_tenure
    FROM customer_stats;
    """
    df = run_query(query)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def get_churn_by_contract() -> pd.DataFrame:
    """Analyze churn rate and customer volume grouped by contract type."""
    query = """
    SELECT
        contract,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
        SUM(CASE WHEN churn_flag = 0 THEN 1 ELSE 0 END) AS active_customers,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(SUM(monthly_charges), 2) AS total_revenue,
        ROUND(AVG(tenure), 1) AS avg_tenure,
        ROUND((CAST(SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct
    FROM customers
    GROUP BY contract
    ORDER BY churn_rate_pct DESC;
    """
    return run_query(query)


def get_revenue_by_contract() -> pd.DataFrame:
    """Analyze monthly and total revenue contribution by contract."""
    query = """
    SELECT
        contract,
        COUNT(*) AS customer_count,
        ROUND(SUM(monthly_charges), 2) AS total_monthly_revenue,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(SUM(total_charges), 2) AS total_lifetime_revenue,
        ROUND((SUM(monthly_charges) * 100.0 / (SELECT SUM(monthly_charges) FROM customers)), 2) AS revenue_share_pct
    FROM customers
    GROUP BY contract
    ORDER BY total_monthly_revenue DESC;
    """
    return run_query(query)


def get_churn_by_payment_method() -> pd.DataFrame:
    """Analyze churn behavior across payment methods."""
    query = """
    SELECT
        payment_method,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
        ROUND((CAST(SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
    FROM customers
    GROUP BY payment_method
    ORDER BY churn_rate_pct DESC;
    """
    return run_query(query)


def get_churn_by_tenure_group() -> pd.DataFrame:
    """Analyze churn rate across tenure cohorts using ordered categorization."""
    query = """
    SELECT
        tenure_group,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
        ROUND((CAST(SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
    FROM customers
    GROUP BY tenure_group
    ORDER BY
        CASE tenure_group
            WHEN '0-12 Months' THEN 1
            WHEN '13-24 Months' THEN 2
            WHEN '25-48 Months' THEN 3
            WHEN '49-72 Months' THEN 4
            ELSE 5
        END;
    """
    return run_query(query)


def get_churn_by_internet_service() -> pd.DataFrame:
    """Analyze churn rates categorized by internet service technology."""
    query = """
    SELECT
        internet_service,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
        ROUND((CAST(SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
    FROM customers
    GROUP BY internet_service
    ORDER BY churn_rate_pct DESC;
    """
    return run_query(query)


def get_churn_by_gender() -> pd.DataFrame:
    """Analyze churn rates and counts across demographic gender."""
    query = """
    SELECT
        gender,
        COUNT(*) AS total_customers,
        SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
        ROUND((CAST(SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct
    FROM customers
    GROUP BY gender;
    """
    return run_query(query)


def get_risk_distribution() -> pd.DataFrame:
    """Analyze customer distribution across ML predicted risk tiers."""
    query = """
    SELECT
        COALESCE(p.risk_category, 'Unclassified') AS risk_tier,
        COUNT(c.customer_id) AS customer_count,
        ROUND(COUNT(c.customer_id) * 100.0 / (SELECT COUNT(*) FROM customers), 2) AS percentage,
        ROUND(AVG(c.monthly_charges), 2) AS avg_monthly_charges,
        ROUND(SUM(c.monthly_charges), 2) AS total_monthly_revenue,
        ROUND(AVG(c.tenure), 1) AS avg_tenure
    FROM customers c
    LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
    GROUP BY risk_tier
    ORDER BY
        CASE risk_tier
            WHEN 'High Risk' THEN 1
            WHEN 'Medium Risk' THEN 2
            WHEN 'Low Risk' THEN 3
            ELSE 4
        END;
    """
    return run_query(query)


def get_revenue_at_risk() -> Dict[str, Any]:
    """Calculate key revenue-at-risk metrics from ML predicted risk tiers."""
    query = """
    WITH risk_summary AS (
        SELECT
            c.customer_id,
            c.monthly_charges,
            COALESCE(p.risk_category, 'Low Risk') AS risk_category,
            COALESCE(p.churn_probability, 0.0) AS churn_probability
        FROM customers c
        LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
    )
    SELECT
        COUNT(*) AS total_evaluated,
        SUM(CASE WHEN risk_category = 'High Risk' THEN 1 ELSE 0 END) AS high_risk_customers,
        ROUND(SUM(CASE WHEN risk_category = 'High Risk' THEN monthly_charges ELSE 0 END), 2) AS high_risk_monthly_revenue,
        ROUND(AVG(CASE WHEN risk_category = 'High Risk' THEN monthly_charges ELSE NULL END), 2) AS high_risk_avg_charge,
        ROUND(SUM(monthly_charges * churn_probability), 2) AS expected_monthly_loss_exposure,
        ROUND(SUM(monthly_charges), 2) AS total_platform_monthly_revenue
    FROM risk_summary;
    """
    df = run_query(query)
    if df.empty:
        return {}
    res = df.iloc[0].to_dict()
    total_rev = res.get("total_platform_monthly_revenue") or 1.0
    high_rev = res.get("high_risk_monthly_revenue") or 0.0
    res["revenue_at_risk_pct"] = round((high_rev / total_rev) * 100, 2)
    return res


def get_high_value_customer_analysis(top_n: int = 50) -> pd.DataFrame:
    """
    Advanced SQL analysis: Identifies top high-value customers using Window Functions (RANK and NTILE)
    along with their churn risk status.
    """
    query = """
    WITH customer_ranks AS (
        SELECT
            c.customer_id,
            c.contract,
            c.tenure,
            c.monthly_charges,
            c.total_charges,
            c.num_services,
            c.churn,
            COALESCE(p.churn_probability, 0.0) AS churn_probability,
            COALESCE(p.risk_category, 'Low Risk') AS risk_category,
            RANK() OVER (ORDER BY c.monthly_charges DESC, c.total_charges DESC) AS charge_rank,
            NTILE(4) OVER (ORDER BY c.total_charges DESC) AS lifetime_value_quartile
        FROM customers c
        LEFT JOIN churn_predictions p ON c.customer_id = p.customer_id
    )
    SELECT
        charge_rank,
        customer_id,
        contract,
        tenure,
        monthly_charges,
        total_charges,
        num_services,
        risk_category,
        ROUND(churn_probability * 100, 1) AS churn_prob_pct,
        churn,
        CASE lifetime_value_quartile
            WHEN 1 THEN 'Top 25% (Platinum)'
            WHEN 2 THEN 'Top 50% (Gold)'
            WHEN 3 THEN 'Top 75% (Silver)'
            ELSE 'Bottom 25% (Bronze)'
        END AS value_tier
    FROM customer_ranks
    WHERE charge_rank <= ?
    ORDER BY charge_rank ASC;
    """
    return run_query(query, (top_n,))


def get_customer_segmentation_sql() -> pd.DataFrame:
    """
    Customer segmentation using CTEs and CASE WHEN logic:
    Segments customers into business cohorts (VIP Long-term, Tech Enthusiasts, At-Risk New Starters, etc.)
    """
    query = """
    WITH segmented_customers AS (
        SELECT
            customer_id,
            monthly_charges,
            tenure,
            contract,
            internet_service,
            num_services,
            churn_flag,
            CASE
                WHEN tenure <= 12 AND contract = 'Month-to-month' THEN 'New Month-to-Month (Vulnerable)'
                WHEN tenure > 24 AND contract IN ('One year', 'Two year') AND monthly_charges >= 70 THEN 'High-Value Loyal'
                WHEN tenure <= 12 AND monthly_charges >= 70 THEN 'High-Spend Early Adopter'
                WHEN tenure > 24 AND num_services >= 4 THEN 'Multi-Service Power User'
                WHEN internet_service = 'Fiber optic' AND contract = 'Month-to-month' THEN 'Fiber Flex (High Churn Risk)'
                ELSE 'Standard Core'
            END AS business_segment
        FROM customers
    )
    SELECT
        business_segment,
        COUNT(*) AS customer_count,
        ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
        ROUND(SUM(monthly_charges), 2) AS segment_monthly_revenue,
        ROUND(AVG(tenure), 1) AS avg_tenure,
        SUM(churn_flag) AS churned_count,
        ROUND((CAST(SUM(churn_flag) AS REAL) / COUNT(*)) * 100, 2) AS churn_rate_pct
    FROM segmented_customers
    GROUP BY business_segment
    ORDER BY churn_rate_pct DESC;
    """
    return run_query(query)
