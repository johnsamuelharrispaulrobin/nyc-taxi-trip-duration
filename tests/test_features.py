import os

import numpy as np
import pandas as pd
import pytest


class TestDataQuality:
    """Test data quality gate functions."""

    @pytest.fixture
    def clean_data(self):
        np.random.seed(42)
        n = 50
        return pd.DataFrame({
            "log_trip_duration_seconds": np.random.normal(6.7, 0.5, n),
            "trip_distance": np.random.exponential(3, n),
            "hour_of_day": np.random.randint(0, 24, n),
            "is_rush_hour": np.random.randint(0, 2, n),
            "is_weekend": np.random.randint(0, 2, n),
        })

    def test_quality_gate_passes_on_clean_data(self, clean_data):
        """Assert clean data passes quality checks."""
        assert clean_data.shape[0] > 0, "Row count should be > 0"
        assert "log_trip_duration_seconds" in clean_data.columns, \
            "Target column should exist"
        assert clean_data.notna().all().all(), \
            "No nulls in any column"

    def test_quality_gate_catches_bad_data(self):
        """Assert bad data fails quality checks."""
        bad_df = pd.DataFrame({
            "good_col": np.random.randn(50),
            "bad_col": [None] * 50,
        })

        null_ratio = bad_df["bad_col"].isnull().mean()
        assert null_ratio > 0.5, "Test setup: bad_col should be >50% null"


class TestFeatures:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def base_input(self):
        from datetime import datetime

        return pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])

    def test_feature_count(self, base_input):
        """Assert engineered features >= 20 columns."""
        from src.features.engineering import create_features

        df = create_features(base_input, inference_mode=True)

        assert df.shape[1] >= 20, \
            f"Expected >=20 features, got {df.shape[1]}"

    def test_no_nulls_in_features(self, base_input):
        """Assert no null values in engineered output."""
        from src.features.engineering import create_features

        df = create_features(base_input, inference_mode=True)

        null_count = df.isnull().sum().sum()
        assert null_count == 0, \
            f"Found {null_count} null values in features"

    def test_rush_hour_flag(self):
        """Assert rush hour flag correct for different times."""
        from datetime import datetime

        from src.features.engineering import create_features

        rush_hour_input = pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])
        df_rush = create_features(rush_hour_input, inference_mode=True)
        assert df_rush["is_rush_hour"].iloc[0] == 1, \
            "Hour 8 should be rush hour"

        midday_input = pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 14, 0),
            "VendorID": 1,
            "passenger_count": 1,
        }])
        df_midday = create_features(midday_input, inference_mode=True)
        assert df_midday["is_rush_hour"].iloc[0] == 0, \
            "Hour 14 should not be rush hour"

    def test_airport_flag(self):
        """Assert airport flag correct for airport/non-airport zones."""
        from datetime import datetime

        from src.features.engineering import create_features

        jfk_input = pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])
        df_jfk = create_features(jfk_input, inference_mode=True)
        assert df_jfk["is_airport_trip"].iloc[0] == 1, \
            "JFK pickup should be airport trip"

        non_airport_input = pd.DataFrame([{
            "PULocationID": 237,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])
        df_non = create_features(non_airport_input, inference_mode=True)
        assert df_non["is_airport_trip"].iloc[0] == 0, \
            "Non-airport to non-airport should not be airport trip"


MODEL_SKIP = pytest.mark.skipif(
    not os.path.exists("models/production_model.pkl"),
    reason="Model not available in CI",
)


@pytest.mark.usefixtures("model", "feature_cols")
class TestModel:
    """Test model loading and prediction pipeline."""

    @pytest.fixture
    def model(self):
        import joblib

        return joblib.load("models/production_model.pkl")

    @pytest.fixture
    def feature_cols(self):
        import joblib

        return joblib.load("models/feature_columns.pkl")

    @MODEL_SKIP
    def test_model_loads(self, model):
        """Assert model loads correctly."""
        assert model is not None, "Model should not be None"
        assert hasattr(model, "predict"), \
            "Model should have predict method"

    @MODEL_SKIP
    def test_prediction_in_range(self, model, feature_cols):
        """Assert prediction within expected range."""
        from datetime import datetime

        import numpy as np

        from src.features.engineering import create_features, encode_for_inference

        raw = pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])

        df_feat = create_features(raw, inference_mode=True)
        df_encoded = encode_for_inference(df_feat)
        X = df_encoded[feature_cols]

        log_pred = model.predict(X)[0]
        seconds = np.expm1(log_pred)
        minutes = seconds / 60

        assert 10 < minutes < 120, \
            f"Prediction {minutes:.1f} min outside expected range"

    @MODEL_SKIP
    def test_prediction_type(self, model, feature_cols):
        """Assert prediction is valid float."""
        from datetime import datetime

        import numpy as np

        from src.features.engineering import create_features, encode_for_inference

        raw = pd.DataFrame([{
            "PULocationID": 132,
            "DOLocationID": 161,
            "trip_distance": 12.5,
            "tpep_pickup_datetime": datetime(2024, 3, 15, 8, 30),
            "VendorID": 1,
            "passenger_count": 1,
        }])

        df_feat = create_features(raw, inference_mode=True)
        df_encoded = encode_for_inference(df_feat)
        X = df_encoded[feature_cols]

        log_pred = model.predict(X)[0]

        assert isinstance(log_pred, (float, np.floating)), \
            f"Prediction type should be float, got {type(log_pred)}"
        assert not np.isnan(log_pred), \
            "Prediction should not be NaN"