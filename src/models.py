"""Production classification models, cross-validation, and business cost threshold optimization."""
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import minimize


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class RegularizedLogisticClassifier:
    def __init__(self, l2_penalty: float = 1.0, class_weight_pos: float = 1.8) -> None:
        self.l2_penalty = l2_penalty
        self.class_weight_pos = class_weight_pos
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedLogisticClassifier":
        n_samples, n_features = X.shape
        weights = np.where(y == 1, self.class_weight_pos, 1.0)
        def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
            w, b = params[:-1], params[-1]
            p = sigmoid(X @ w + b)
            eps = 1e-12
            bce = -np.mean(weights * (y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
            reg = 0.5 * (self.l2_penalty / n_samples) * float(np.sum(w ** 2))
            diff = weights * (p - y)
            grad_w = (X.T @ diff) / n_samples + (self.l2_penalty / n_samples) * w
            return bce + reg, np.append(grad_w, float(np.mean(diff)))
        res = minimize(loss_and_grad, np.zeros(n_features + 1, dtype=float), jac=True, method="L-BFGS-B")
        self.coef_, self.intercept_ = res.x[:-1], float(res.x[-1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.coef_ + self.intercept_)
