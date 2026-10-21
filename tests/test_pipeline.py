"""Automated unit test suite for the Customer Churn MLOps pipeline."""
import unittest
import numpy as np
from src.data_generator import generate_churn_dataset
from src.preprocessing import ChurnFeaturePipeline
from src.models import RegularizedLogisticClassifier
from src.explainability import explain_single_prediction, calculate_psi
from app.main import health_check


class TestChurnMLOpsPipeline(unittest.TestCase):
    def test_end_to_end_pipeline(self):
        df = generate_churn_dataset(n_samples=400, random_state=42)
        pipe = ChurnFeaturePipeline().fit(df.iloc[:300])
        X_train = pipe.transform(df.iloc[:300])
        X_test = pipe.transform(df.iloc[300:])
        self.assertFalse(np.isnan(X_train).any())
        model = RegularizedLogisticClassifier().fit(X_train, df.iloc[:300]["churn"].to_numpy())
        probs = model.predict_proba(X_test)
        self.assertEqual(len(probs), 100)
        drivers = explain_single_prediction(model, X_test[0], pipe.feature_names_, top_k=3)
        self.assertEqual(len(drivers), 3)
        self.assertLess(calculate_psi(X_train[:, 0], X_test[:, 0]), 0.25)
        self.assertEqual(health_check()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
