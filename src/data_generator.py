"""Synthetic realistic Telecom/SaaS Customer Churn dataset generator with messy real-world artifacts."""
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 3500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = rng.integers(1, 73, size=n_samples).astype(float)
    contract_type = rng.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = rng.choice(["Electronic Check", "Credit Card", "Bank Transfer", "Mailed Check"], size=n_samples, p=[0.38, 0.28, 0.22, 0.12])
    internet_service = rng.choice(["Fiber Optic", "DSL", "No Internet"], size=n_samples, p=[0.48, 0.35, 0.17])
    tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])

    base_charge = np.where(
        internet_service == "Fiber Optic",
        rng.normal(85.0, 14.0, size=n_samples),
        np.where(internet_service == "DSL", rng.normal(55.0, 10.0, size=n_samples), rng.normal(25.0, 5.0, size=n_samples)),
    )
    monthly_charges = np.clip(base_charge, 18.0, 160.0).round(2)
    outlier_idx = rng.choice(n_samples, size=int(0.015 * n_samples), replace=False)
    monthly_charges[outlier_idx] = (monthly_charges[outlier_idx] * 1.65).round(2)
    total_charges = (tenure_months * monthly_charges * rng.uniform(0.92, 1.08, size=n_samples)).round(2)
    support_tickets = rng.poisson(lam=np.where(tech_support == "No", 2.4, 1.1) + np.where(tenure_months < 6, 1.5, 0.3), size=n_samples).astype(float)
    usage_change_pct = np.clip(rng.normal(loc=-2.0, scale=18.0, size=n_samples) - np.where(contract_type == "Month-to-Month", 6.5, 0.0), -65.0, 45.0).round(2)
    late_payments_last_year = rng.binomial(4, p=np.where(payment_method == "Electronic Check", 0.28, 0.08), size=n_samples)

    logit = (
        -1.55
        + 1.35 * (contract_type == "Month-to-Month").astype(float)
        - 0.95 * (contract_type == "Two-Year").astype(float)
        + 0.65 * (internet_service == "Fiber Optic").astype(float)
        + 0.55 * (payment_method == "Electronic Check").astype(float)
        + 0.45 * (tech_support == "No").astype(float)
        - 0.042 * tenure_months
        + 0.018 * (monthly_charges - 65.0)
        + 0.32 * support_tickets
        - 0.038 * usage_change_pct
        + 0.42 * late_payments_last_year
        + 0.55 * ((tenure_months <= 6) & (monthly_charges > 80.0)).astype(float)
        + 0.48 * ((support_tickets >= 4) & (tech_support == "No")).astype(float)
        + rng.normal(0.0, 0.45, size=n_samples)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(0.0, 1.0, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids, "tenure_months": tenure_months, "contract_type": contract_type,
        "payment_method": payment_method, "internet_service": internet_service, "tech_support": tech_support,
        "monthly_charges": monthly_charges, "total_charges": total_charges, "support_tickets": support_tickets,
        "usage_change_pct": usage_change_pct, "late_payments_last_year": late_payments_last_year, "churn": churn,
    })
    miss_tc = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    miss_st = rng.choice(n_samples, size=int(0.025 * n_samples), replace=False)
    df.loc[miss_tc, "total_charges"] = np.nan
    df.loc[miss_st, "support_tickets"] = np.nan
    return df
