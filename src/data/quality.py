import logging
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_REQUIRED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
]

CLEANED_REQUIRED_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "trip_duration_seconds",
    "passenger_count",
]


def check_data_quality(df: pd.DataFrame, schema_type: str = "raw") -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    statistics: dict[str, Any] = {
        "total_rows": len(df),
        "null_counts": {},
        "negative_fares": 0,
        "zero_distance_trips": 0,
        "zero_duration_trips": 0,
    }

    schema_errors = _check_schema(df, schema_type)
    failures.extend(schema_errors.get("failures", []))
    warnings.extend(schema_errors.get("warnings", []))

    row_errors = _check_row_count(df)
    failures.extend(row_errors.get("failures", []))
    warnings.extend(row_errors.get("warnings", []))

    null_errors = _check_null_rates(df)
    failures.extend(null_errors.get("failures", []))
    warnings.extend(null_errors.get("warnings", []))
    statistics["null_counts"] = null_errors.get("statistics", {}).get("null_counts", {})

    value_errors = _check_value_ranges(df, schema_type)
    failures.extend(value_errors.get("failures", []))
    warnings.extend(value_errors.get("warnings", []))
    statistics["negative_fares"] = value_errors.get("statistics", {}).get("negative_fares", 0)
    statistics["zero_distance_trips"] = value_errors.get("statistics", {}).get("zero_distance_trips", 0)

    duration_errors = _check_trip_duration(df)
    failures.extend(duration_errors.get("failures", []))
    warnings.extend(duration_errors.get("warnings", []))
    statistics["zero_duration_trips"] = duration_errors.get("statistics", {}).get("zero_duration_trips", 0)

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": statistics,
    }


def _check_schema(df: pd.DataFrame, schema_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {"failures": [], "warnings": [], "statistics": {}}
    required_cols = RAW_REQUIRED_COLUMNS if schema_type == "raw" else CLEANED_REQUIRED_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        result["failures"].append(f"Missing required columns: {missing_cols}")
    if "tpep_pickup_datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["tpep_pickup_datetime"]
    ):
        result["failures"].append("tpep_pickup_datetime must be datetime dtype")
    if "tpep_dropoff_datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["tpep_dropoff_datetime"]
    ):
        result["failures"].append("tpep_dropoff_datetime must be datetime dtype")
    return result


def _check_row_count(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"failures": [], "warnings": [], "statistics": {}}
    total_rows = len(df)
    if total_rows < 100:
        result["failures"].append(f"Row count {total_rows} is less than 100")
    elif total_rows < 1000:
        result["warnings"].append(f"Row count {total_rows} is less than 1000")
    return result


def _check_null_rates(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"failures": [], "warnings": [], "statistics": {}}
    total_rows = len(df)
    null_counts: dict[str, int] = {}
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_counts[col] = null_count
        null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
        if null_pct > 50:
            result["failures"].append(f"Column '{col}' has {null_pct:.2f}% nulls (>50% threshold)")
        elif null_pct > 20:
            result["warnings"].append(f"Column '{col}' has {null_pct:.2f}% nulls (>20% threshold)")
    result["statistics"]["null_counts"] = null_counts
    return result


def _check_value_ranges(df: pd.DataFrame, schema_type: str) -> dict[str, Any]:
    result: dict[str, Any] = {"failures": [], "warnings": [], "statistics": {}}
    total_rows = len(df)
    negative_fares = 0
    zero_distance_trips = 0

    if schema_type == "raw":
        if "fare_amount" in df.columns:
            negative_fares = int((df["fare_amount"] < 0).sum())
            result["statistics"]["negative_fares"] = negative_fares
            corrupt_fares = int((df["fare_amount"] < -100).sum())
            if corrupt_fares > 0:
                result["failures"].append(f"fare_amount has {corrupt_fares} values below -100 (corrupt records)")
            if negative_fares > 0:
                result["warnings"].append(f"fare_amount has {negative_fares} negative values")

        if "total_amount" in df.columns:
            negative_total = int((df["total_amount"] < 0).sum())
            if negative_total > 0:
                result["warnings"].append(f"total_amount has {negative_total} negative values")
    else:
        if "trip_duration_seconds" in df.columns:
            invalid_duration = int((df["trip_duration_seconds"] <= 0).sum())
            if invalid_duration > 0:
                result["failures"].append(f"trip_duration_seconds has {invalid_duration} values <= 0")

    if "trip_distance" in df.columns:
        zero_distance_trips = int((df["trip_distance"] == 0).sum())
        result["statistics"]["zero_distance_trips"] = zero_distance_trips
        zero_pct = (zero_distance_trips / total_rows) * 100 if total_rows > 0 else 0
        if zero_pct > 5:
            result["failures"].append(f"trip_distance has {zero_pct:.2f}% zero-distance trips (>5% threshold)")
        impossible_trips = int((df["trip_distance"] > 500).sum())
        if impossible_trips > 0:
            result["failures"].append(f"trip_distance has {impossible_trips} values above 500 miles (impossible trips)")

    return result


def _check_trip_duration(df: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"failures": [], "warnings": [], "statistics": {}}
    total_rows = len(df)
    zero_duration_trips = 0

    if "trip_duration_seconds" in df.columns:
        duration = df["trip_duration_seconds"]
        zero_duration_trips = int((duration <= 0).sum())
        result["statistics"]["zero_duration_trips"] = zero_duration_trips
        zero_pct = (zero_duration_trips / total_rows) * 100 if total_rows > 0 else 0
        if zero_pct > 5:
            result["failures"].append(
                f"{zero_pct:.2f}% of trips have duration <= 0 seconds (>5% threshold)"
            )
        long_trips = int((duration > 10800).sum())
        if long_trips > 0:
            result["warnings"].append(f"{long_trips} trips have duration > 10800 seconds (3 hours)")
        short_trips = int((duration < 60).sum())
        if short_trips > 0:
            result["warnings"].append(f"{short_trips} trips have duration < 60 seconds")
    elif "tpep_pickup_datetime" in df.columns and "tpep_dropoff_datetime" in df.columns:
        pickup = df["tpep_pickup_datetime"]
        dropoff = df["tpep_dropoff_datetime"]
        duration = (dropoff - pickup).dt.total_seconds()
        zero_duration_trips = int((duration <= 0).sum())
        result["statistics"]["zero_duration_trips"] = zero_duration_trips
        zero_pct = (zero_duration_trips / total_rows) * 100 if total_rows > 0 else 0
        if zero_pct > 5:
            result["failures"].append(
                f"{zero_pct:.2f}% of trips have duration <= 0 seconds (>5% threshold)"
            )
        long_trips = int((duration > 10800).sum())
        if long_trips > 0:
            result["warnings"].append(f"{long_trips} trips have duration > 10800 seconds (3 hours)")
        short_trips = int((duration < 60).sum())
        if short_trips > 0:
            result["warnings"].append(f"{short_trips} trips have duration < 60 seconds")

    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data.cleaner import clean_data

    raw_df = pd.read_parquet("data/raw/yellow_tripdata.parquet")
    print("=" * 50)
    print("RAW DATA QUALITY CHECK")
    print("=" * 50)
    print(check_data_quality(raw_df, schema_type="raw"))

    print("\n" + "=" * 50)
    print("CLEANED DATA QUALITY CHECK")
    print("=" * 50)
    cleaned_df = clean_data(raw_df)
    print(check_data_quality(cleaned_df, schema_type="cleaned"))