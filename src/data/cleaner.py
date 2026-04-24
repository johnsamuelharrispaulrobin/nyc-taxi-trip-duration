import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.quality import check_data_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    logger.info(f"Starting data cleaning with {initial_rows} rows")

    df = df.copy()
    df["trip_duration_seconds"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds()
        .astype(int)
    )

    dropped = _drop_corrupt_and_invalid(df)
    df = dropped["df"]
    logger.info(f"Rows after dropping corrupt/invalid: {len(df)}")

    df = _handle_nulls(df)
    logger.info(f"Rows after handling nulls: {len(df)}")

    df = _drop_outcome_columns(df)
    logger.info(f"Columns after dropping outcome columns: {list(df.columns)}")

    df = _remove_duplicates(df)
    logger.info(f"Rows after removing duplicates: {len(df)}")

    df = _final_dtype_cleanup(df)

    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path / "cleaned.parquet", index=False)
    logger.info(f"Saved cleaned data to {output_path / 'cleaned.parquet'}")

    final_rows = len(df)
    logger.info(f"Cleaning complete: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} dropped)")

    return df


def _drop_corrupt_and_invalid(df: pd.DataFrame) -> dict:
    result = {"df": df.copy(), "dropped_counts": {}}

    checks = [
        ("fare_amount < -100", lambda d: d["fare_amount"] < -100),
        ("trip_distance > 500", lambda d: d["trip_distance"] > 500),
        ("fare_amount < 0", lambda d: d["fare_amount"] < 0),
        ("total_amount < 0", lambda d: d["total_amount"] < 0),
        ("trip_duration_seconds <= 0", lambda d: d["trip_duration_seconds"] <= 0),
        ("trip_duration_seconds < 60", lambda d: d["trip_duration_seconds"] < 60),
        ("trip_duration_seconds > 10800", lambda d: d["trip_duration_seconds"] > 10800),
        ("trip_distance == 0", lambda d: d["trip_distance"] == 0),
    ]

    for condition_name, condition_fn in checks:
        mask = condition_fn(result["df"])
        count = mask.sum()
        if count > 0:
            result["dropped_counts"][condition_name] = int(count)
            logger.info(f"Dropping {count} rows where {condition_name}")
            result["df"] = result["df"][~mask]

    return result


def _handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["RatecodeID", "store_and_fwd_flag", "congestion_surcharge", "Airport_fee"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping columns with >30% nulls: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    if "passenger_count" in df.columns:
        median_val = df["passenger_count"].median()
        null_count = df["passenger_count"].isna().sum()
        if null_count > 0:
            logger.info(f"Filling {null_count} nulls in passenger_count with median {median_val}")
            df["passenger_count"] = df["passenger_count"].fillna(median_val)

    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    dropped = rows_before - rows_after
    if dropped > 0:
        logger.info(f"Dropping {dropped} rows with remaining nulls")

    return df


def _drop_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    outcome_cols = [
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "total_amount",
        "cbd_congestion_fee",
        "payment_type",
    ]
    cols_to_drop = [c for c in outcome_cols if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping outcome columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    rows_before = len(df)
    df = df.drop_duplicates(keep="first")
    rows_after = len(df)
    dropped = rows_before - rows_after
    if dropped > 0:
        logger.info(f"Dropping {dropped} duplicate rows")
    return df


def _final_dtype_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    int32_cols = ["passenger_count", "PULocationID", "DOLocationID", "VendorID"]
    for col in int32_cols:
        if col in df.columns:
            df[col] = df[col].astype("int32")
    return df


if __name__ == "__main__":
    df = pd.read_parquet("data/raw/yellow_tripdata.parquet")
    print(f"Before shape: {df.shape}")
    cleaned_df = clean_data(df)
    print(f"After shape: {cleaned_df.shape}")
    print("\nData quality check on cleaned data:")
    print(check_data_quality(cleaned_df, schema_type="cleaned"))
    print("\nTrip duration statistics:")
    print(cleaned_df["trip_duration_seconds"].describe())