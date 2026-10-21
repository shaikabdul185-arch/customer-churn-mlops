"""SHAP-style marginal attribution explainability and Population Stability Index (PSI) drift detection."""
from typing import Dict, List
import numpy as np


def explain_single_prediction(model, x_row: np.ndarray, feature_names: List[str], top_k: int = 4) -> List[Dict[str, float]]:
    base_prob = float(model.predict_proba(x_row.reshape(1, -1))[0])
    contributions = []
    for idx, feat in enumerate(feature_names):
        x_zeroed = x_row.copy()
        x_zeroed[idx] = 0.0
        without_prob = float(model.predict_proba(x_zeroed.reshape(1, -1))[0])
        impact = round(base_prob - without_prob, 4)
        contributions.append({"feature": feat, "standardized_value": round(float(x_row[idx]), 3), "probability_impact": impact})
    contributions.sort(key=lambda item: abs(item["probability_impact"]), reverse=True)
    return contributions[:top_k]


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)
    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-4, 1.0)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-4, 1.0)
    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)
