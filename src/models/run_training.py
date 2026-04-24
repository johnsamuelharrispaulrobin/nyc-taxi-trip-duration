"""
MLflow experiment tracking for NYC taxi trip duration prediction.

Sequentially runs baseline, XGBoost default, and XGBoost tuned experiments
with full parameter and metric logging for model comparison.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


mlflow.set_tracking_uri("sqlite:///mlflow.db")
try:
    exp_id = mlflow.create_experiment("nyc_taxi_trip_duration")
except Exception:
    exp_id = mlflow.get_experiment_by_name("nyc_taxi_trip_duration").experiment_id
mlflow.set_experiment(exp_id)


def evaluate(y_pred_log: np.ndarray, y_test: pd.Series) -> dict:
    """Convert predictions to real seconds and compute metrics."""
    y_pred_seconds = np.expm1(y_pred_log)
    y_true_seconds = np.expm1(y_test)
    return {
        "rmse_seconds": np.sqrt(mean_squared_error(y_true_seconds, y_pred_seconds)),
        "mae_seconds": mean_absolute_error(y_true_seconds, y_pred_seconds),
        "r2_score": r2_score(y_true_seconds, y_pred_seconds)
    }


def run_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series) -> dict:
    """Run MLflow baseline (Linear Regression) experiment."""
    with mlflow.start_run(run_name="baseline_linear_regression"):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        start_time = time.time()
        model = LinearRegression()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred_log = model.predict(X_test)
        metrics = evaluate(y_pred_log, y_test)
        metrics["training_time_seconds"] = training_time

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("src/models/baseline.py")

        print(f"  RMSE: {metrics['rmse_seconds']:.2f}s, MAE: {metrics['mae_seconds']:.2f}s, "
              f"R²: {metrics['r2_score']:.4f}, Time: {training_time:.2f}s")

        return {"model": model, **metrics}


def run_xgboost_default(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_test: pd.Series) -> dict:
    """Run MLflow XGBoost default experiment."""
    with mlflow.start_run(run_name="xgboost_default"):
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1
        }

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_features", X_train.shape[1])
        for k, v in params.items():
            mlflow.log_param(k, v)

        start_time = time.time()
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        training_time = time.time() - start_time

        y_pred_log = model.predict(X_test)
        metrics = evaluate(y_pred_log, y_test)
        metrics["training_time_seconds"] = training_time

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact("src/models/train_xgboost.py")

        print(f"  RMSE: {metrics['rmse_seconds']:.2f}s, MAE: {metrics['mae_seconds']:.2f}s, "
              f"R²: {metrics['r2_score']:.4f}, Time: {training_time:.2f}s")

        return {"model": model, **metrics}


def run_xgboost_tuned(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      y_train: pd.Series, y_test: pd.Series) -> dict:
    """Run MLflow XGBoost tuned experiment using saved best params."""
    with mlflow.start_run(run_name="xgboost_tuned"):
        with open("models/best_params.json") as f:
            best_params = json.load(f)

        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("n_features", X_train.shape[1])
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        start_time = time.time()
        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)
        training_time = time.time() - start_time

        y_pred_log = model.predict(X_test)
        metrics = evaluate(y_pred_log, y_test)
        metrics["training_time_seconds"] = training_time

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact("src/models/tuning.py")

        print(f"  RMSE: {metrics['rmse_seconds']:.2f}s, MAE: {metrics['mae_seconds']:.2f}s, "
              f"R²: {metrics['r2_score']:.4f}, Time: {training_time:.2f}s")

        return {"model": model, **metrics}


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = pd.read_parquet("data/processed/features.parquet")
    y = df["log_trip_duration_seconds"]
    X = df.drop(columns=["log_trip_duration_seconds"])

    LINEAR_FEATURES = [c for c in X.columns if c not in ["PULocationID", "DOLocationID"]]
    TREE_FEATURES = X.columns.tolist()

    X_train_lin, X_test_lin, y_train, y_test = train_test_split(
        X[LINEAR_FEATURES], y, test_size=0.2, random_state=42
    )
    X_train_tree, X_test_tree, _, _ = train_test_split(
        X[TREE_FEATURES], y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print("="*60)
    print("MLFLOW EXPERIMENT: nyc_taxi_trip_duration")
    print("="*60)

    print("\n[1/3] Running baseline (Linear Regression)...")
    baseline_results = run_baseline(X_train_lin, X_test_lin, y_train, y_test)

    print("\n[2/3] Running XGBoost default...")
    xgb_default_results = run_xgboost_default(X_train_tree, X_test_tree, y_train, y_test)

    print("\n[3/3] Running XGBoost tuned (Optuna 30 trials)...")
    xgb_tuned_results = run_xgboost_tuned(X_train_tree, X_test_tree, y_train, y_test)

    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Model':<20} {'RMSE (s)':<12} {'MAE (s)':<12} {'R²':<10} {'Time (s)':<10}")
    print("-"*60)
    print(f"{'Linear Regression':<20} {baseline_results['rmse_seconds']:<12.2f} "
          f"{baseline_results['mae_seconds']:<12.2f} {baseline_results['r2_score']:<10.4f} "
          f"{baseline_results['training_time_seconds']:<10.2f}")
    print(f"{'XGBoost Default':<20} {xgb_default_results['rmse_seconds']:<12.2f} "
          f"{xgb_default_results['mae_seconds']:<12.2f} {xgb_default_results['r2_score']:<10.4f} "
          f"{xgb_default_results['training_time_seconds']:<10.2f}")
    print(f"{'XGBoost Tuned':<20} {xgb_tuned_results['rmse_seconds']:<12.2f} "
          f"{xgb_tuned_results['mae_seconds']:<12.2f} {xgb_tuned_results['r2_score']:<10.4f} "
          f"{xgb_tuned_results['training_time_seconds']:<10.2f}")

    joblib.dump(xgb_tuned_results["model"], "models/production_model.pkl")
    print("\nSaved production model to models/production_model.pkl")