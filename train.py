from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def main():
    # 1. Collect and validate data
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    if X.isnull().any().any():
        raise ValueError("Dataset contains missing feature values.")
    if len(X) != len(y) or X.empty:
        raise ValueError("Dataset validation failed.")
    print("Dataset shape:", X.shape)
    print("Feature columns:", list(X.columns))
    print("Class counts:\n", y.value_counts().sort_index().to_dict())

    # 2. Split data reproducibly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train and track experiment
    mlflow.set_experiment("iris-classification")
    with mlflow.start_run():
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "test_samples": int(len(y_test)),
        }
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.sklearn.log_model(model, artifact_path="iris_model")

        # 4. Save a versioned model artifact and metrics
        joblib.dump({"model": model, "feature_names": list(X.columns),
                     "class_names": list(iris.target_names)}, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))
        print("\nClassification report:")
        print(classification_report(y_test, predictions, target_names=iris.target_names))
        print(f"Saved model to: {MODEL_PATH}")
        print(f"Saved metrics to: {METRICS_PATH}")
        print("MLflow tracking data saved in ./mlruns")

if __name__ == "__main__":
    main()
