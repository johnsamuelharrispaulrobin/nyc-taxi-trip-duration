"""
Linear Regression baseline model for NYC taxi trip duration prediction.

Linear models serve as interpretable baselines that reveal linear relationships
between features and duration. They establish whether simple relationships
(e.g., distance + time-of-day effects) explain most of the variance before
resorting to complex tree-based models.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Train Linear Regression baseline and evaluate on test set.

    Returns dict with model, predictions, and metrics.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)

    y_pred_seconds = np.expm1(y_pred_log)
    y_true_seconds = np.expm1(y_test)

    rmse_seconds = np.sqrt(mean_squared_error(y_true_seconds, y_pred_seconds))
    mae_seconds = mean_absolute_error(y_true_seconds, y_pred_seconds)
    r2 = r2_score(y_true_seconds, y_pred_seconds)

    return {
        "model": model,
        "rmse_seconds": rmse_seconds,
        "mae_seconds": mae_seconds,
        "r2_score": r2,
        "y_pred_seconds": y_pred_seconds,
        "y_true_seconds": y_true_seconds
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = pd.read_parquet("data/processed/features.parquet")
    y = df["log_trip_duration_seconds"]
    X = df.drop(columns=["log_trip_duration_seconds"])

    LINEAR_FEATURES = [c for c in X.columns if c not in ["PULocationID", "DOLocationID"]]
    print(f"Using {len(LINEAR_FEATURES)} linear features: {LINEAR_FEATURES}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X[LINEAR_FEATURES], y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    start_time = time.time()
    results = train_baseline(X_train, X_test, y_train, y_test)
    training_time = time.time() - start_time

    print("\n" + "="*50)
    print("BASELINE MODEL: Linear Regression")
    print("="*50)
    print(f"RMSE (seconds): {results['rmse_seconds']:.2f}")
    print(f"MAE (seconds):  {results['mae_seconds']:.2f}")
    print(f"R² Score:       {results['r2_score']:.4f}")
    print(f"Training time:  {training_time:.2f}s")

    joblib.dump(results["model"], "models/baseline.pkl")
    print("\nSaved model to models/baseline.pkl")