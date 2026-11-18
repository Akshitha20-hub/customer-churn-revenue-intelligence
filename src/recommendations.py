"""
Retention Recommendation Engine
================================
Provides rule-based retention recommendations for customers
based on their attributes and predicted churn risk.

All recommendations are explainable and business-grounded.
They are strategic suggestions — not guaranteed interventions.

Usage:
    from src.recommendations import get_recommendation, apply_recommendations
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_recommendation(customer) -> dict:
    """
    Generate a retention recommendation for a single customer.

    Parameters
    ----------
    customer : dict or pd.Series

    Returns
    -------
    dict: action, reason, priority, strategy
    """
    if hasattr(customer, "to_dict"):
        customer = customer.to_dict()

    contract  = str(customer.get("contract", "")).strip()
    tenure    = int(customer.get("tenure", 0))
    monthly   = float(customer.get("monthly_charges", 0))
    payment   = str(customer.get("payment_method", "")).strip()
    internet  = str(customer.get("internet_service", "")).strip()
    risk      = str(customer.get("risk_category", "Low Risk")).strip()
    num_svc   = int(customer.get("num_services", 0))

    # Rule 1: Month-to-month + High Risk + New customer
    if contract == "Month-to-month" and risk == "High Risk" and tenure <= 12:
        return {
            "action":   "Onboarding & Contract Upgrade Offer",
            "reason":   "New customer on a flexible contract with high churn risk.",
            "priority": "High",
            "strategy": (
                "This customer is early in their lifecycle and has not yet committed to a long-term plan. "
                "Offer a discounted one-year contract with added benefits (free months, service bundle). "
                "Assign a dedicated onboarding specialist to improve early engagement."
            ),
        }

    # Rule 2: Month-to-month + High Risk + Long tenure
    if contract == "Month-to-month" and risk == "High Risk" and tenure > 24:
        return {
            "action":   "Loyalty Retention Intervention",
            "reason":   "Long-tenured customer on a flexible contract showing high churn risk.",
            "priority": "High",
            "strategy": (
                "This is a valuable long-term customer who has not committed to a contract. "
                "Contact proactively with a personalized loyalty offer: exclusive discount, "
                "service upgrade, or priority support enrollment. Acknowledge their tenure."
            ),
        }

    # Rule 3: High Monthly Charges + High Risk
    if monthly >= 80 and risk == "High Risk":
        return {
            "action":   "Personalized Pricing Review",
            "reason":   "High-value customer at risk — high charges may be a pain point.",
            "priority": "High",
            "strategy": (
                "Customer is paying premium rates and shows high churn probability. "
                "Conduct a personalized account review. Offer a tailored pricing plan or "
                "bundle discount. Highlight value delivered vs. cost. "
                "Consider a loyalty discount or service credit."
            ),
        }

    # Rule 4: Electronic Check + High Risk
    if "electronic check" in payment.lower() and risk == "High Risk":
        return {
            "action":   "Payment Experience Improvement",
            "reason":   "Electronic check payers show elevated churn rates in this dataset.",
            "priority": "High",
            "strategy": (
                "Research shows electronic check customers churn more frequently. "
                "Offer incentives to switch to auto-pay (bank transfer or credit card). "
                "A small monthly discount for auto-pay enrollment can reduce friction "
                "and improve payment reliability."
            ),
        }

    # Rule 5: Fiber Optic + High Risk
    if internet == "Fiber optic" and risk == "High Risk":
        return {
            "action":   "Service Quality & Value Communication",
            "reason":   "Fiber optic customers have above-average churn despite premium pricing.",
            "priority": "High",
            "strategy": (
                "Fiber optic customers pay the highest rates and show significant churn. "
                "Conduct a satisfaction survey. Proactively address service quality concerns. "
                "Offer a free service upgrade or additional streaming/security add-ons "
                "to reinforce value perception."
            ),
        }

    # Rule 6: Low services + Medium or High Risk
    if num_svc <= 1 and risk in ["High Risk", "Medium Risk"]:
        return {
            "action":   "Service Bundle Upsell Campaign",
            "reason":   "Low service adoption correlates with lower engagement and higher churn risk.",
            "priority": "Medium",
            "strategy": (
                "Customer uses very few services. Low product adoption often leads to low stickiness. "
                "Run a targeted campaign offering a free trial of TechSupport, "
                "OnlineSecurity, or StreamingTV to increase platform engagement and switching cost."
            ),
        }

    # Rule 7: Month-to-month + Medium Risk
    if contract == "Month-to-month" and risk == "Medium Risk":
        return {
            "action":   "Contract Upgrade Incentive",
            "reason":   "Month-to-month contract with moderate churn risk.",
            "priority": "Medium",
            "strategy": (
                "Customer is on a flexible contract with moderate churn probability. "
                "Offer a one-year contract incentive: first month free, "
                "locked-in rate, or service bundle. "
                "Long-term contracts significantly reduce churn probability."
            ),
        }

    # Rule 8: One-year contract + High Risk
    if contract == "One year" and risk == "High Risk":
        return {
            "action":   "Mid-Cycle Retention Outreach",
            "reason":   "Mid-term contract customer showing unexpected high churn probability.",
            "priority": "High",
            "strategy": (
                "Despite a one-year contract, this customer shows high churn risk. "
                "Reach out before contract renewal with a renewal incentive. "
                "Offer an upgrade to a two-year plan with added benefits. "
                "Understand if there is an underlying service issue."
            ),
        }

    # Rule 9: Long tenure + Low Risk
    if tenure >= 48 and risk == "Low Risk":
        return {
            "action":   "Loyalty Recognition Program",
            "reason":   "Long-tenured, low-risk customer — reinforce satisfaction.",
            "priority": "Low",
            "strategy": (
                "This is a loyal, stable customer. Acknowledge their loyalty with a "
                "thank-you message, loyalty badge, or small account credit. "
                "Encourage referrals. These customers can become brand advocates."
            ),
        }

    # Default
    return {
        "action":   "Standard Monitoring",
        "reason":   "Customer shows low churn probability at this time.",
        "priority": "Low",
        "strategy": (
            "Customer is currently low-risk. Continue standard engagement: "
            "monthly newsletters, product updates, and satisfaction surveys. "
            "Re-evaluate if usage patterns change significantly."
        ),
    }


def apply_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply get_recommendation to each row in a DataFrame.
    Returns DataFrame with added recommendation columns.
    """
    recs = df.apply(get_recommendation, axis=1)
    df = df.copy()
    df["retention_action"]   = recs.apply(lambda r: r["action"])
    df["retention_reason"]   = recs.apply(lambda r: r["reason"])
    df["retention_priority"] = recs.apply(lambda r: r["priority"])
    df["retention_strategy"] = recs.apply(lambda r: r["strategy"])
    return df
