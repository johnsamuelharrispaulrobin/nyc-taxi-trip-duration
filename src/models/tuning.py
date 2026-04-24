"""
XGBoost hyperparameter tuning with Optuna for NYC taxi trip duration prediction.

Hyperparameter tuning optimizes the model to balance bias-variance tradeoff.
Using cross-validation during tuning ensures parameters generalize well to
unseen data rather than overfitting the test set.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


TREE_FEATURES = None


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Optuna objective function: minimize RMSE on validation split.

    Uses a single train/val split for speed with large dataset.
    Returns positive RMSE so Optuna (minimizing) finds optimal hyperparameters.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "random_state": 42,
        "n_jobs": -1
    }

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse


def tune_and_train(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> dict:
    """
    Run Optuna tuning, train final model, and evaluate on test set.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=10,
        show_progress_bar=True
    )

    best_params = study.best_params
    print(f"\nBest CV RMSE (log scale): {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train, verbose=False)

    y_pred_log = final_model.predict(X_test)
    y_pred_seconds = np.expm1(y_pred_log)
    y_true_seconds = np.expm1(y_test)

    rmse_seconds = np.sqrt(mean_squared_error(y_true_seconds, y_pred_seconds))
    mae_seconds = mean_absolute_error(y_true_seconds, y_pred_seconds)
    r2 = r2_score(y_true_seconds, y_pred_seconds)

    return {
        "model": final_model,
        "best_params": best_params,
        "best_cv_rmse": study.best_value,
        "rmse_seconds": rmse_seconds,
        "mae_seconds": mae_seconds,
        "r2_score": r2,
        "study": study
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    df = pd.read_parquet("data/processed/features.parquet")
    y = df["log_trip_duration_seconds"]
    X = df.drop(columns=["log_trip_duration_seconds"])

    TREE_FEATURES = X.columns.tolist()
    print(f"Using {len(TREE_FEATURES)} features: {TREE_FEATURES}")

    X_train, X_test, y_train, y_test = train_test_split(
        X[TREE_FEATURES], y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\n" + "="*50)
    print("Starting Optuna hyperparameter tuning (10 trials)...")
    print("="*50)

    start_time = time.time()
    results = tune_and_train(X_train, X_test, y_train, y_test)
    training_time = time.time() - start_time

    print("\n" + "="*50)
    print("TUNED XGBOOST MODEL")
    print("="*50)
    print(f"RMSE (seconds): {results['rmse_seconds']:.2f}")
    print(f"MAE (seconds):  {results['mae_seconds']:.2f}")
    print(f"R² Score:       {results['r2_score']:.4f}")
    print(f"Training time:  {training_time:.2f}s")

    with open("models/best_params.json", "w") as f:
        json.dump(results["best_params"], f, indent=2)
    print("\nSaved best params to models/best_params.json")

    joblib.dump(results["model"], "models/tuned_xgboost.pkl")
    print("Saved model to models/tuned_xgboost.pkl")