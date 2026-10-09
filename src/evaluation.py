import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test, test_ids):
    # Calculate held-out metrics and return one row per prediction.
    predictions = model.predict(X_test)
    class_one_index = list(model.classes_).index(1)
    churn_probability = model.predict_proba(X_test)[:, class_one_index]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    prediction_output = pd.DataFrame(
        {
            "customerID": test_ids.to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "Churn_Probability": churn_probability,
        }
    )
    return metrics, prediction_output


def save_outputs(metrics: dict, predictions: pd.DataFrame, output_dir: str | Path) -> None:
    # Write held-out predictions and metrics to the output directory.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path / "predictions.csv", index=False)