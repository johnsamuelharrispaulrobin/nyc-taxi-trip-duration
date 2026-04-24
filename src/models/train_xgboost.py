"""
XGBoost default model for NYC taxi trip duration prediction.

XGBoost with default hyperparameters establishes a strong tree-based baseline
before tuning. Tree models capture non-linear relationships and feature
interactions that linear models miss, particularly useful for complex urban
mobility patterns.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


TREE_FEATURES = None


def train_xgboost_default(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Train XGBoost with default parameters and evaluate on test set.
    """
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=False)

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
        "training_time": None
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = pd.read_parquet("data/processed/features.parquet")
    y = df["log_trip_duration_seconds"]
    X = df.drop(columns=["log_trip_duration_seconds"])

    TREE_FEATURES = X.columns.tolist()
    LINEAR_FEATURES = [c for c in TREE_FEATURES if c not in ["PULocationID", "DOLocationID"]]
    print(f"Using {len(TREE_FEATURES)} tree features: {TREE_FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X[TREE_FEATURES], y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    start_time = time.time()
    results = train_xgboost_default(X_train, X_test, y_train, y_test)
    training_time = time.time() - start_time
    results["training_time"] = training_time

    print("\n" + "="*50)
    print("XGBOOST DEFAULT MODEL")
    print("="*50)
    print(f"RMSE (seconds): {results['rmse_seconds']:.2f}")
    print(f"MAE (seconds):  {results['mae_seconds']:.2f}")
    print(f"R² Score:       {results['r2_score']:.4f}")
    print(f"Training time:  {training_time:.2f}s")

    joblib.dump(results["model"], "models/xgboost_default.pkl")
    print("\nSaved model to models/xgboost_default.pkl")

    baseline_model = joblib.load("models/baseline.pkl")
    y_pred_baseline = np.expm1(baseline_model.predict(X_test[LINEAR_FEATURES]))
    y_true_seconds = np.expm1(y_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_true_seconds, y_pred_baseline))
    baseline_mae = mean_absolute_error(y_true_seconds, y_pred_baseline)
    baseline_r2 = r2_score(y_true_seconds, y_pred_baseline)

    print("\n" + "="*50)
    print("COMPARISON TABLE")
    print("="*50)
    print(f"{'Model':<20} {'RMSE (s)':<12} {'MAE (s)':<12} {'R²':<10}")
    print("-"*50)
    print(f"{'Linear Regression':<20} {baseline_rmse:<12.2f} {baseline_mae:<12.2f} {baseline_r2:<10.4f}")
    print(f"{'XGBoost Default':<20} {results['rmse_seconds']:<12.2f} {results['mae_seconds']:<12.2f} {results['r2_score']:<10.4f}")