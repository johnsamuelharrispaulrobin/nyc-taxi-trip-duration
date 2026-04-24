"""
Feature engineering for NYC taxi trip duration prediction.

This module transforms cleaned taxi data into features optimized for
trip duration modeling, incorporating temporal, distance, location,
and interaction features based on domain knowledge.
"""

import time
import numpy as np
import pandas as pd
import joblib


def create_features(df: pd.DataFrame, inference_mode: bool = False) -> pd.DataFrame:
    """
    Create engineered features from cleaned taxi dataframe.

    Args:
        df: DataFrame with raw columns (PULocationID, DOLocationID, trip_distance, tpep_pickup_datetime)
        inference_mode: If True, skips target engineering (no trip_duration_seconds needed).
                        If False, expects trip_duration_seconds for training target.
    """
    df = df.copy()

    if not inference_mode:
        df["log_trip_duration_seconds"] = np.log1p(df["trip_duration_seconds"])

    # =============================================================================
    # TEMPORAL FEATURES (from tpep_pickup_datetime)
    # Traffic patterns vary systematically by time of day and day of week.
    # Hour of day captures rush hour effects, weekday/weekend differences,
    # and night-time traffic patterns. Cyclical encoding preserves the
    # continuity between e.g. hour 23 and hour 0.
    # =============================================================================
    pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])

    df["pickup_hour"] = pickup_dt.dt.hour
    df["pickup_dayofweek"] = pickup_dt.dt.dayofweek
    df["pickup_month"] = pickup_dt.dt.month
    df["pickup_day"] = pickup_dt.dt.day

    df["is_rush_hour"] = df["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_night"] = df["pickup_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)

    # =============================================================================
    # DISTANCE FEATURES
    # Trip distance is the primary driver of duration. Log-transform handles
    # the skewed distribution, while squared term captures non-linear effects
    # where longer trips may face compounding delays.
    # =============================================================================
    df["log_trip_distance"] = np.log1p(df["trip_distance"])
    df["trip_distance_squared"] = df["trip_distance"] ** 2

    # =============================================================================
    # LOCATION FEATURES (join taxi_zone_lookup.csv on LocationID)
    # Geographic context matters: airport trips have distinct traffic patterns,
    # inter-borough trips cross congested boundaries, and zone-level characteristics
    # (traffic density, accessibility) influence duration.
    # =============================================================================
    zone_lookup = pd.read_csv("data/raw/taxi_zone_lookup.csv")

    df = df.merge(
        zone_lookup.rename(columns={
            "LocationID": "PULocationID",
            "Borough": "pu_borough",
            "Zone": "pu_zone",
            "service_zone": "pu_service_zone"
        }),
        on="PULocationID",
        how="left"
    )

    df = df.merge(
        zone_lookup.rename(columns={
            "LocationID": "DOLocationID",
            "Borough": "do_borough",
            "Zone": "do_zone",
            "service_zone": "do_service_zone"
        }),
        on="DOLocationID",
        how="left"
    )

    df["is_airport_trip"] = (
        (df["pu_service_zone"] == "Airports") |
        (df["do_service_zone"] == "Airports")
    ).astype(int)

    df["same_borough"] = (df["pu_borough"] == df["do_borough"]).astype(int)
    df["pu_do_pair"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    # =============================================================================
    # INTERACTION FEATURES
    # Distance interacts with conditions: rush hour amplifies the effect of
    # longer distances due to congestion, night trips may be faster per mile,
    # and airport trips have unique access/egress patterns.
    # =============================================================================
    df["distance_rush_interaction"] = df["log_trip_distance"] * df["is_rush_hour"]
    df["distance_night_interaction"] = df["log_trip_distance"] * df["is_night"]
    df["airport_distance_interaction"] = df["log_trip_distance"] * df["is_airport_trip"]

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and encode features for model training.

    Drops high-cardinality identifiers and encodes categorical variables
    as integers for ML model compatibility.
    """
    drop_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_duration_seconds",
        "pu_zone",
        "do_zone",
        "pu_do_pair",
        "PULocationID",
        "DOLocationID"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical_cols = ["pu_borough", "do_borough", "pu_service_zone", "do_service_zone", "VendorID"]
    cat_mappings = {}

    for col in categorical_cols:
        if col in df.columns:
            cat = pd.Categorical(df[col])
            cat_mappings[col] = {val: code for val, code in zip(cat.categories, cat.codes)}
            df[col] = cat.codes

    joblib.dump(cat_mappings, "models/category_mappings.pkl")
    print(f"Saved category mappings to models/category_mappings.pkl")

    return df


def encode_for_inference(df: pd.DataFrame, cat_mappings: dict = None) -> pd.DataFrame:
    """
    Encode categorical columns using saved mappings for inference.

    Args:
        df: DataFrame with string/obj categorical columns
        cat_mappings: Dict of {col: {value: code}}. If None, loads from disk.
    """
    if cat_mappings is None:
        cat_mappings = joblib.load("models/category_mappings.pkl")

    df = df.copy()

    drop_cols = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_duration_seconds",
        "pu_zone",
        "do_zone",
        "pu_do_pair",
        "PULocationID",
        "DOLocationID"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for col, mapping in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


if __name__ == "__main__":
    import joblib
    start_time = time.time()

    df = pd.read_parquet("data/processed/cleaned.parquet")
    print(f"Shape after loading cleaned data: {df.shape}")

    df = create_features(df, inference_mode=False)
    print(f"Shape after feature engineering: {df.shape}")

    df = select_features(df)
    print(f"Shape after feature selection: {df.shape}")

    feature_cols = [c for c in df.columns if c != "log_trip_duration_seconds"]
    joblib.dump(feature_cols, "models/feature_columns.pkl")
    print(f"\nSaved {len(feature_cols)} feature columns to models/feature_columns.pkl")

    y = df["log_trip_duration_seconds"]
    df = df.drop(columns=["log_trip_duration_seconds"])

    print(f"\nFinal feature columns ({len(df.columns)}):")
    print(df.columns.tolist())

    print(f"\nSample of 5 rows:")
    print(df.head())

    nan_mask = df.isna().any()
    inf_mask = df.select_dtypes(include=np.number).apply(lambda x: np.isinf(x).any())
    if nan_mask.any() or inf_mask.any():
        raise ValueError(f"Found NaN or inf values in: NaN={nan_mask.sum()}, Inf={inf_mask.sum()}")

    df.to_parquet("data/processed/features.parquet")
    elapsed = time.time() - start_time
    print(f"\nSaved to data/processed/features.parquet")
    print(f"Elapsed time: {elapsed:.2f}s")