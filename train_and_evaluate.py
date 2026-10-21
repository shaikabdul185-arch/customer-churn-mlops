"""End-to-end training, evaluation, and drift analysis entry point."""
from src.data_generator import generate_churn_dataset
from src.preprocessing import ChurnFeaturePipeline
from src.models import RegularizedLogisticClassifier
from src.explainability import explain_single_prediction, calculate_psi


def run_pipeline():
    df = generate_churn_dataset(n_samples=3500, random_state=42)
    train_df = df.iloc[:2800].reset_index(drop=True)
    test_df = df.iloc[2800:].reset_index(drop=True)

    pipeline = ChurnFeaturePipeline().fit(train_df)
    X_train = pipeline.transform(train_df)
    y_train = train_df["churn"].to_numpy()
    X_test = pipeline.transform(test_df)
    y_test = test_df["churn"].to_numpy()

    model = RegularizedLogisticClassifier(l2_penalty=1.0, class_weight_pos=1.6).fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    drivers = explain_single_prediction(model, X_test[0], pipeline.feature_names_, top_k=4)
    psi_val = calculate_psi(X_train[:, 1], X_test[:, 1])
    print(f"Scored {len(probs)} test customers | Sample PSI: {psi_val:.4f} | Top driver: {drivers[0]['feature']}")


if __name__ == "__main__":
    run_pipeline()
