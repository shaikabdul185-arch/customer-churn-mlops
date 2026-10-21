# customer-churn-mlops
End-to-End # End-to-End Customer Churn Prediction & MLOps Engine

An end-to-end machine learning engineering and MLOps repository designed to identify high-risk subscription customers before they cancel, explain individual risk drivers, optimize retention campaign ROI, and monitor production data drift.

## 1. Business Problem & Cost-Sensitive Framing

Subscription telecom and SaaS companies lose significant recurring revenue to customer churn, while blanket retention offers waste marketing budget on customers who would have stayed anyway.

* **Customer Lifetime Value (LTV) at risk per churner:** $800 (with an expected 40% save rate when proactively contacted, yielding **$275 net value saved** after deducting the $45 retention offer cost)
* **Outreach cost per false alarm (False Positive):** **$45** per contacted customer
* **Optimization objective:** Rather than defaulting to a naive `0.50` probability cutoff or optimizing raw accuracy on an imbalanced dataset, this pipeline tunes the classification decision threshold (`0.22`) to maximize **Expected Net Retention Savings ($ USD)**.

## 2. System Architecture & Repository Structure

```text
customer-churn-mlops/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI pipeline (training + unit tests)
├── app/
│   └── main.py                    # FastAPI REST service (/health, /predict, /predict/batch)
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Realistic synthetic dataset generator with messy data artifacts
│   ├── preprocessing.py           # Fit/transform leakage-free imputation, capping & domain features
│   ├── models.py                  # L2 Logistic Regression, GBDT, Stacked Ensemble & ROI optimizer
│   └── explainability.py          # Permutation importance, local risk attribution & PSI drift checks
├── tests/
│   └── test_pipeline.py           # 6 unit & integration tests (imputation, AUC, ROI, PSI, FastAPI)
├── Dockerfile                     # Production container image definition
├── requirements.txt               # Pinned runtime dependencies
├── train_and_evaluate.py          # End-to-end training, evaluation, and plot generation entry point
└── README.md
```

## 3. Holdout Test Set Performance (N = 700)

All models are evaluated on a stratified 20% holdout test split (`N = 700` customers) after fitting preprocessing statistics strictly on the 80% training split (`N = 2,800`):

| Model | Decision Threshold | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Brier Score | Net Retention Savings ($ USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | 0.50 | 0.8707 | 0.8509 | 0.7868 | 0.7180 | 0.8703 | 0.1534 | $70,765.00 |
| **Gradient Boosted Trees (GBDT)** | 0.50 | 0.8562 | 0.8308 | 0.7504 | 0.7744 | 0.7278 | 0.1580 | $60,235.00 |
| **Stacked Ensemble (Production)** | **0.22** | **0.8673** | **0.8456** | **0.7056** | **0.5530** | **0.9747** | **0.1518** | **$73,495.00** |

## 4. Feature Engineering & Explainability

The `ChurnFeaturePipeline` imputes missing values (`total_charges`, `support_tickets`) using training medians, caps extreme billing outliers at the 1st/99th percentiles, and engineers 20 standardized predictors including domain interaction ratios:

* **`contract_month_to_month`** (Mean ROC-AUC permutation drop: `0.06760`)
* **`tenure_months`** (Mean ROC-AUC permutation drop: `0.06584`)
* **`usage_change_pct`** (Mean ROC-AUC permutation drop: `0.03302`)
* **`monthly_charges`** (Mean ROC-AUC permutation drop: `0.01996`)
* **`support_tickets`** (Mean ROC-AUC permutation drop: `0.01696`)

## 5. Production Data Drift Monitoring (PSI)

To guard against silent model degradation when customer pricing or behavior shifts, `src/explainability.py` computes the **Population Stability Index (PSI)** across core numeric features:

| Feature | Simulated Production PSI | Drift Status | Operational Action |
| :--- | :---: | :---: | :--- |
| `tenure_months` | 0.0189 | `STABLE` | Normal serving |
| `monthly_charges` (simulated price hike) | 1.1940 | `SEVERE_DRIFT` | Trigger automated retraining |
| `total_charges` | 0.0075 | `STABLE` | Normal serving |
| `support_tickets` | 0.0162 | `STABLE` | Normal serving |

## 6. Quickstart & Local Execution

```bash
pip install -r requirements.txt
python3 train_and_evaluate.py
python3 -m unittest discover -s tests -v
uvicorn app.main:app --reload --port 8000
```
