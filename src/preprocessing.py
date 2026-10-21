"""Leakage-free preprocessing and domain feature engineering pipeline."""
from typing import Dict, List
import numpy as np
import pandas as pd


class ChurnFeaturePipeline:
    NUMERIC_COLS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets", "usage_change_pct", "late_payments_last_year"]
    CATEGORICAL_COLS = ["contract_type", "payment_method", "internet_service", "tech_support"]

    def __init__(self) -> None:
        self.medians_: Dict[str, float] = {}
        self.upper_caps_: Dict[str, float] = {}
        self.lower_caps_: Dict[str, float] = {}
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def _engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.NUMERIC_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(self.medians_[col])
            out[col] = np.clip(out[col], self.lower_caps_[col], self.upper_caps_[col])
        expected_total = np.maximum(out["tenure_months"] * out["monthly_charges"], 1.0)
        out["billing_overage_ratio"] = out["total_charges"] / expected_total
        out["support_friction_rate"] = out["support_tickets"] / np.maximum(out["tenure_months"], 1.0)
        out["early_tenure_high_bill"] = ((out["tenure_months"] <= 6) & (out["monthly_charges"] > 80.0)).astype(float)
        out["unresolved_support_risk"] = ((out["support_tickets"] >= 3) & (out["tech_support"] == "No")).astype(float)
        out["engagement_decay_risk"] = np.where(out["usage_change_pct"] < -10.0, -out["usage_change_pct"], 0.0)
        out["contract_month_to_month"] = (out["contract_type"] == "Month-to-Month").astype(float)
        out["contract_one_year"] = (out["contract_type"] == "One-Year").astype(float)
        out["contract_two_year"] = (out["contract_type"] == "Two-Year").astype(float)
        out["pay_electronic_check"] = (out["payment_method"] == "Electronic Check").astype(float)
        out["pay_credit_card"] = (out["payment_method"] == "Credit Card").astype(float)
        out["pay_bank_transfer"] = (out["payment_method"] == "Bank Transfer").astype(float)
        out["net_fiber_optic"] = (out["internet_service"] == "Fiber Optic").astype(float)
        out["net_dsl"] = (out["internet_service"] == "DSL").astype(float)
        out["has_tech_support"] = (out["tech_support"] == "Yes").astype(float)
        return out

    def fit(self, df: pd.DataFrame) -> "ChurnFeaturePipeline":
        for col in self.NUMERIC_COLS:
            series = pd.to_numeric(df[col], errors="coerce")
            self.medians_[col] = float(series.median())
            filled = series.fillna(self.medians_[col])
            self.lower_caps_[col] = float(np.percentile(filled, 1.0))
            self.upper_caps_[col] = float(np.percentile(filled, 99.0))
        engineered = self._engineer_domain_features(df)
        continuous_cols = self.NUMERIC_COLS + ["billing_overage_ratio", "support_friction_rate", "engagement_decay_risk"]
        binary_cols = ["early_tenure_high_bill", "unresolved_support_risk", "contract_month_to_month", "contract_one_year", "contract_two_year", "pay_electronic_check", "pay_credit_card", "pay_bank_transfer", "net_fiber_optic", "net_dsl", "has_tech_support"]
        self.feature_names_ = continuous_cols + binary_cols
        for col in continuous_cols:
            self.means_[col] = float(engineered[col].mean())
            std = float(engineered[col].std())
            self.stds_[col] = std if std > 1e-8 else 1.0
        for col in binary_cols:
            self.means_[col] = 0.0
            self.stds_[col] = 1.0
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("ChurnFeaturePipeline must be fitted before calling transform().")
        engineered = self._engineer_domain_features(df)
        X = np.zeros((len(engineered), len(self.feature_names_)), dtype=float)
        for idx, col in enumerate(self.feature_names_):
            X[:, idx] = (engineered[col].values - self.means_[col]) / self.stds_[col]
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
