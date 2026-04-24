import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading Parquet file: {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Successfully loaded {len(df)} rows")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Inspect a DataFrame and print various statistics."""
    logger.info(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    logger.info("Column names and data types:")
    for col in df.columns:
        logger.info(f"  {col}: {df[col].dtype}")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        logger.info("Summary statistics for numeric columns:")
        print(df[numeric_cols].describe().to_string())
    else:
        logger.info("No numeric columns found.")

    logger.info("Missing values:")
    for col in df.columns:
        missing = df[col].isna().sum()
        pct = (missing / len(df)) * 100
        logger.info(f"  {col}: {missing} ({pct:.2f}%)")

    logger.info("First 5 rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    df = load_data("data/raw/yellow_tripdata.parquet")
    inspect_data(df)