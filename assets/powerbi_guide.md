# Power BI Dashboard Blueprint & Integration Guide
### Customer Churn & Revenue Intelligence Platform

This guide provides step-by-step instructions for importing the exported platform CSV datasets into Microsoft Power BI and building an executive-grade dashboard.

---

## 1. Export Files Overview

The platform automatically produces 5 clean, normalized, and pre-aggregated CSV files located in the `exports/` folder:

| Filename | Records | Purpose in Power BI |
| :--- | :--- | :--- |
| `customers_cleaned.csv` | Full Dataset (~7,043) | Primary Fact Table: customer demographics, billing, services, and tenure |
| `churn_predictions.csv` | Scored Accounts (~7,043) | ML Prediction Table: probabilities, risk categories, model outputs |
| `churn_by_contract.csv` | Aggregated Summary | Pre-computed benchmark table by contract tier |
| `revenue_by_segment.csv` | Aggregated Summary | Revenue distribution across customer segments |
| `risk_customers.csv` | High-Risk Accounts | Operational target queue for immediate customer retention |

---

## 2. Importing Data into Power BI

1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Navigate to your project directory: `<path-to-project>/exports/`.
4. Select `customers_cleaned.csv` and click **Load**.
5. Repeat for `churn_predictions.csv`, `churn_by_contract.csv`, and `revenue_by_segment.csv`.
6. Go to the **Model View** tab in Power BI:
   - Create a **1-to-1 or 1-to-Many Relationship** between:
     `customers_cleaned[customer_id]` (Primary Key) <---> `churn_predictions[customer_id]`
   - Set Cross-filter direction to **Both**.

---

## 3. Recommended DAX Measures

Create a new table named `_Measures` and add the following DAX calculations:

```dax
// Total Customer Count
Total Customers = COUNTROWS('customers_cleaned')

// Total Churned Customers
Total Churned = CALCULATE(COUNTROWS('customers_cleaned'), 'customers_cleaned'[churn_flag] = 1)

// Overall Churn Rate (%)
Churn Rate % = DIVIDE([Total Churned], [Total Customers], 0) * 100

// Total Monthly Recurring Revenue (MRR)
Total MRR = SUM('customers_cleaned'[monthly_charges])

// High-Risk Customer Count
High Risk Customers = CALCULATE(COUNTROWS('churn_predictions'), 'churn_predictions'[risk_category] = "High Risk")

// Monthly Revenue at Risk ($)
Revenue at Risk = CALCULATE(SUM('customers_cleaned'[monthly_charges]), 'churn_predictions'[risk_category] = "High Risk")

// Revenue at Risk Percentage (%)
Revenue at Risk % = DIVIDE([Revenue at Risk], [Total MRR], 0) * 100

// Average Customer Lifetime Value (CLV Indicator)
Avg CLV = AVERAGE('customers_cleaned'[clv_indicator])
```

---

## 4. Recommended Dashboard Page Layouts

### Page 1: Executive Churn & Revenue Overview
* **Top Ribbon (Cards)**:
  * `Total Customers` | `Churn Rate %` | `Total MRR` | `Revenue at Risk` | `High Risk Customers`
* **Visual 1 (Donut Chart)**:
  * Legend: `customers_cleaned[contract]`
  * Values: `[Total Customers]`
* **Visual 2 (Clustered Column Chart)**:
  * X-Axis: `customers_cleaned[contract]`
  * Y-Axis: `[Churn Rate %]` and `[Revenue at Risk]`
* **Visual 3 (Stacked Bar Chart)**:
  * Y-Axis: `customers_cleaned[tenure_group]`
  * X-Axis: `[Total Customers]`
  * Legend: `customers_cleaned[churn]`
* **Visual 4 (Scatter Plot)**:
  * X-Axis: `customers_cleaned[tenure]`
  * Y-Axis: `customers_cleaned[monthly_charges]`
  * Legend: `churn_predictions[risk_category]`

### Page 2: Risk Scoring & Retention Operations
* **Visual 1 (Gauge or Treemap)**:
  * Grouping: `churn_predictions[risk_category]`
  * Values: `[Revenue at Risk]`
* **Visual 2 (Decomposition Tree)**:
  * Analyze: `[Revenue at Risk]`
  * Explain by: `contract` -> `internet_service` -> `payment_method`
* **Visual 3 (Operational Table / Matrix)**:
  * Columns: `customer_id`, `contract`, `tenure`, `monthly_charges`, `churn_probability`, `risk_category`
  * Conditional Formatting: Highlight `churn_probability > 0.65` in soft red.
* **Slicers**:
  * Contract, Payment Method, Risk Category, Internet Service.

---

## 5. Notes & Governance
* **Predictive Exposure vs Realized Loss:** Clearly label `Revenue at Risk` as modeled risk exposure rather than booked financial loss.
* **Refresh Frequency:** Whenever the Streamlit app or `setup.py` pipeline runs, exports in `exports/` are updated. Clicking **Refresh** in Power BI automatically updates the reports.
