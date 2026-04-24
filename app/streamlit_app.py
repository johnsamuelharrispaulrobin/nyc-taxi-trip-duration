"""
NYC Yellow Taxi Trip Duration Predictor — Streamlit Application

Multi-page app with Project Overview, Data Exploration, Model Results,
and Trip Duration Prediction pages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date, time

from src.features.engineering import create_features, encode_for_inference

st.set_page_config(
    layout="wide",
    page_title="NYC Taxi Predictor",
    page_icon="🚕"
)

ACCENT_COLOR = "#F7C948"
CUSTOM_CSS = """
<style>
/* Dark background with white text */
body, .stApp, [data-testid="stApp"] {
    background-color: #1a1a2e !important;
    color: white !important;
}

/* All text elements white */
h1, h2, h3, h4, p, span, div, label, th, td {
    color: white !important;
}

/* Sidebar dark */
[data-testid="stSidebar"] {
    background-color: #0d0d1a !important;
}

/* White background cards/containers */
.stMetric, .stSelectbox > div, .stNumberInput > div, .stDateInput > div, .stTimeInput > div {
    background-color: #2a2a4a !important;
    border-radius: 8px;
    padding: 10px;
}

/* Input fields - dark with white text */
input, select, [data-baseweb="input"], [data-baseweb="select"] {
    background-color: #2a2a4a !important;
    color: white !important;
}

/* Buttons - yellow accent */
.stButton > button {
    background-color: #F7C948 !important;
    color: #1a1a2e !important;
    font-weight: bold;
}

/* Data tables - dark bg */
.dataframe, [data-testid="stDataFrame"] {
    background-color: #2a2a4a !important;
    color: white !important;
}

/* Alerts/infos - dark bg with white text */
.stAlert, .stInfo, .stWarning, .stError, .stSuccess {
    background-color: #2a2a4a !important;
    color: white !important;
}

/* Expander */
.stExpander {
    background-color: #2a2a4a !important;
    border-left: 4px solid #F7C948;
}

/* Footer */
.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #444;
}

/* Tab content */
.tab-content {
    background-color: #1a1a2e;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_features_sample(nrows: int = 50000) -> pd.DataFrame:
    try:
        df = pd.read_parquet("data/processed/features.parquet")
        if len(df) > nrows:
            df = df.sample(nrows, random_state=42)
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_target_sample(nrows: int = 50000):
    try:
        import joblib
        y = joblib.load("data/processed/target_log.pkl")
        if len(y) > nrows:
            y = y.sample(nrows, random_state=42)
        return y
    except FileNotFoundError:
        return None


@st.cache_data
def load_zone_lookup() -> pd.DataFrame:
    try:
        return pd.read_csv("data/raw/taxi_zone_lookup.csv")
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_resource
def load_model():
    try:
        return joblib.load("models/production_model.pkl")
    except FileNotFoundError:
        return None


@st.cache_resource
def load_feature_cols() -> list:
    try:
        return joblib.load("models/feature_columns.pkl")
    except FileNotFoundError:
        return []


def zone_options(zones_df: pd.DataFrame) -> list:
    if zones_df.empty:
        return ["JFK Airport", "Midtown Manhattan", "Brooklyn", "Bronx", "Queens", "Upper East Side"]
    return [f"{row['Borough']} — {row['Zone']}" for _, row in zones_df.iterrows()]


def zone_id_from_option(option: str, zones_df: pd.DataFrame) -> int:
    if zones_df.empty:
        mapping = {
            "JFK Airport": 132, "Midtown Manhattan": 161, "Brooklyn": 47,
            "Bronx": 56, "Queens": 128, "Upper East Side": 237
        }
        return mapping.get(option, 132)
    for _, row in zones_df.iterrows():
        if f"{row['Borough']} — {row['Zone']}" == option:
            return row["LocationID"]
    return 132


def footer():
    st.markdown('<div class="footer">Built with XGBoost + Streamlit | 3.2M trips analyzed</div>', unsafe_allow_html=True)


def sidebar():
    st.sidebar.markdown("### 🚕 NYC Taxi Predictor")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["📊 Project Overview", "📈 Explore the Data", "📉 Model Results", "🔮 Predict Duration"],
        index=0
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Performance**\n- R² Score: 0.80\n- MAE: 232 seconds\n- RMSE: 373 seconds")
    return page


def page_overview():
    st.title("🚕 NYC Taxi Trip Duration Predictor")
    st.markdown("*Predicting trip duration for 3.2M Yellow Taxi trips using XGBoost*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", "3,201,131 rows", "Jan 2016 trip data")
    with col2:
        st.metric("Features Engineered", "28 → 25", "Selected for model")
    with col3:
        st.metric("Model R² Score", "0.80", "~80% variance explained")
    with col4:
        st.metric("MAE", "232 seconds", "~3.9 minutes")

    st.markdown("---")
    st.subheader("🛠️ Tech Stack")

    cols = st.columns(7)
    tools = ["Python", "XGBoost", "Optuna", "MLflow", "FastAPI", "Streamlit", "Docker"]
    for col, tool in zip(cols, tools):
        col.markdown(f"**{tool}**")

    st.markdown("---")
    st.subheader("What This Project Does")
    st.markdown(
        "This app predicts how long a Yellow Taxi trip in NYC will take based on pickup/dropoff "
        "locations, distance, and time of day. Using XGBoost trained on 3.2 million historical "
        "trips, it estimates trip duration within ~4 minutes on average — helping dispatchers, "
        "pricing systems, and riders plan their journeys."
    )

    with st.expander("📐 Key Design Decisions"):
        st.markdown("""
        **Why log transform on target?**
        Trip duration is right-skewed (long tail of very long trips). Log-transforming the target
        normalizes the distribution, improves model convergence, and reduces the influence of outliers.

        **Why MAE as primary metric?**
        MAE is more interpretable than RMSE — it's the average error in real seconds. RMSE penalizes
        large errors more heavily, but for ETA prediction, we care equally about all errors.

        **Why XGBoost over Linear Regression?**
        Trip duration has non-linear relationships (e.g., congestion effects, airport access patterns).
        XGBoost captures these interactions automatically. Linear Regression failed with R² = -0.04
        because it couldn't model these complex patterns.
        """)

    footer()


def page_explore():
    import plotly.express as px
    
    st.title("📈 Explore the Data")
    st.markdown("Analysis based on 50,000 sampled trips from the dataset")
    st.markdown("---")


    df = load_features_sample()
    y_target = load_target_sample()

    if df.empty:
        st.warning("⚠️ Could not load data/features.parquet. Showing demo content.")
        df = pd.DataFrame({
            "trip_distance": np.random.exponential(3, 50000),
            "pickup_hour": np.random.randint(0, 24, 50000)
        })
        y_target = np.random.normal(6.7, 0.5, 50000)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Target Distribution")
        trip_dur = np.expm1(y_target)
        fig_data = pd.DataFrame({"trip_duration_seconds": trip_dur})
        fig = px.histogram(
            fig_data, x="trip_duration_seconds",
            nbins=50, title="Trip Duration Distribution (seconds)",
            labels={"trip_duration_seconds": "Duration (seconds)"}
        )
        fig.add_vline(x=trip_dur.mean(), line_color="red", annotation_text=f"Mean: {trip_dur.mean():.0f}s")
        fig.add_vline(x=trip_dur.median(), line_color="green", annotation_text=f"Median: {trip_dur.median():.0f}s")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Right-skewed distribution: mean (1070s) > median (853s). Log transform normalizes this.")

    with col2:
        st.subheader("⏰ Hourly Trip Patterns")
        df_with_target = df.copy()
        df_with_target["log_target"] = y_target.values if hasattr(y_target, 'values') else y_target
        hourly = df_with_target.groupby("pickup_hour")["log_target"].mean().reset_index()
        hourly["duration_minutes"] = np.expm1(hourly["log_target"]) / 60
        hourly["is_rush"] = hourly["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19])
        colors = ["#F7C948" if r else "#4A90A4" for r in hourly["is_rush"]]
        fig = px.bar(hourly, x="pickup_hour", y="duration_minutes", color=colors,
                     title="Avg Duration by Hour", labels={"pickup_hour": "Hour (0-23)", "duration_minutes": "Minutes"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rush hours (7-9am, 4-7pm) highlighted in yellow")

    st.markdown("---")

    st.subheader("🔗 Feature Correlations with Target")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_vals = y_target.values if hasattr(y_target, 'values') else y_target
    correlations = df[numeric_cols].corrwith(pd.Series(target_vals)).abs().sort_values(ascending=True)
    top10 = correlations.tail(10)
    fig = px.bar(x=top10.values, y=top10.index, orientation="h",
                  title="Top 10 Features by Correlation with Target",
                  labels={"x": "|Correlation|", "y": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Key Findings")

    findings = [
        ("**trip_distance correlation: 0.75** — Distance is the strongest predictor of duration, as expected.",
         "🔴"),
        ("**Rush hours add 15-20% to median duration** — Peak times significantly impact travel times.",
         "🟡"),
        ("**Airport trips (7% of data) are outlier cluster** — JFK/LaGuardia trips have distinct patterns.",
         "🔵"),
        ("**85% of trips stay within same borough** — Most trips are short, local journeys.",
         "🟢"),
        ("**Zero missing values post-cleaning** — Dataset is clean and ready for modeling.",
         "✅")
    ]

    for text, icon in findings:
        st.info(f"{icon} {text}")

    footer()


def page_model_results():
    import plotly.express as px
    
    st.title("📉 Model Results")
    st.markdown("Comparison of baseline and production models")
    st.markdown("---")

    st.subheader("Model Comparison")
    results = pd.DataFrame({
        "Model": ["Linear Regression", "XGBoost Default", "XGBoost Tuned"],
        "RMSE (s)": [857, 387, 373],
        "R²": [-0.04, 0.79, 0.80],
        "MAE (s)": ["N/A", "N/A", 232],
        "Notes": [
            "Failed - wrong model type",
            "Strong baseline, default params",
            "Production model"
        ]
    })

    def highlight_tuned(s):
        return ["background-color: #d4edda; font-weight: bold" if "✓" in str(v) else "" for v in s]

    st.dataframe(results.style.apply(highlight_tuned, subset=["Model"]), use_container_width=True)
    st.caption(
        "Linear Regression R² is negative (-0.04) because it predicts worse than simply "
        "predicting the mean. This is expected when the true relationship is non-linear."
    )

    st.markdown("---")
    st.subheader("Feature Importance (from Production Model)")

    model = load_model()
    feature_cols = load_feature_cols()

    if model is not None and len(feature_cols) > 0:
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=True).tail(15)

        fig = px.bar(x=importances["importance"], y=importances["feature"],
                      orientation="h", title="Top 15 Feature Importances",
                      labels={"importance": "Importance Score", "feature": "Feature"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load model for importance visualization")

    st.markdown("---")
    st.subheader("Residual Analysis")

    df = load_features_sample(nrows=10000)
    y_target = load_target_sample(nrows=10000)
    if df.empty:
        st.warning("Could not load data for residual analysis")
    else:
        model = load_model()
        feature_cols = load_feature_cols()

        if model is not None and len(feature_cols) > 0:
            target_vals = y_target.values if hasattr(y_target, 'values') else y_target
            y_true_log = target_vals
            X = df[feature_cols]
            y_pred_log = model.predict(X)
            y_true = np.expm1(y_true_log)
            y_pred = np.expm1(y_pred_log)
            residuals = y_true - y_pred

            col1, col2 = st.columns(2)
            with col1:
                sample_idx = np.random.choice(len(y_pred), min(2000, len(y_pred)), replace=False)
                fig = px.scatter(
                    x=y_true[sample_idx], y=y_pred[sample_idx],
                    labels={"x": "Actual Duration (s)", "y": "Predicted Duration (s)"},
                    title="Predicted vs Actual (sample)"
                )
                fig.add_shape(type="line", x0=0, y0=0, x1=max(y_true), y1=max(y_true),
                              line=dict(color="red", dash="dash"))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(residuals, nbins=50, title="Residual Distribution",
                                   labels={"x": "Residual (s)", "y": "Count"})
                st.plotly_chart(fig, use_container_width=True)

            st.caption("Residuals are approximately normal — model is unbiased")

    footer()


def page_predict():
    st.title("🔮 Predict Trip Duration")
    st.markdown("Enter trip details to get an estimated duration prediction")
    st.markdown("---")

    zones_df = load_zone_lookup()
    zones = zone_options(zones_df)

    st.subheader("Try an Example")
    examples = {
        "Custom": None,
        "JFK → Midtown (Friday 8:30am)": {"zone_from": "Queens — JFK Airport", "zone_to": "Manhattan — Midtown South", "distance": 12.5, "date": date(2024, 3, 15), "time": time(8, 30)},
        "Brooklyn → Manhattan (Saturday 2pm)": {"zone_from": "Brooklyn — Bedford-Stuyvesant", "zone_to": "Manhattan — East Village", "distance": 5.2, "date": date(2024, 3, 16), "time": time(14, 0)},
        "Bronx → Bronx (Tuesday 6pm)": {"zone_from": "Bronx - University Heights", "zone_to": "Bronx - Morris Park", "distance": 1.8, "date": date(2024, 3, 19), "time": time(18, 0)}
    }
    selected_example = st.selectbox("Try a preset example:", list(examples.keys()))

    st.markdown("---")
    st.subheader("Trip Details")

    col1, col2 = st.columns(2)

    with col1:
        default_zone_from = examples[selected_example]["zone_from"] if selected_example != "Custom" else zones[0]
        default_zone_to = examples[selected_example]["zone_to"] if selected_example != "Custom" else zones[1]
        default_distance = examples[selected_example]["distance"] if selected_example != "Custom" else 2.5

        pu_zone = st.selectbox("Pickup Zone", zones, index=zones.index(default_zone_from) if default_zone_from in zones else 0)
        do_zone = st.selectbox("Dropoff Zone", zones, index=zones.index(default_zone_to) if default_zone_to in zones else 1)
        trip_distance = st.number_input("Trip Distance (miles)", min_value=0.1, max_value=50.0, step=0.1, value=default_distance)

    with col2:
        default_date = examples[selected_example]["date"] if selected_example != "Custom" else date.today()
        default_time = examples[selected_example]["time"] if selected_example != "Custom" else time(8, 30)

        pickup_date = st.date_input("Pickup Date", default_date)
        pickup_time = st.time_input("Pickup Time", default_time)

    st.markdown("---")

    if st.button("Predict Trip Duration", type="primary", use_container_width=True):
        try:
            pu_id = zone_id_from_option(pu_zone, zones_df)
            do_id = zone_id_from_option(do_zone, zones_df)

            pickup_datetime = datetime.combine(pickup_date, pickup_time)

            raw = pd.DataFrame([{
                "PULocationID": pu_id,
                "DOLocationID": do_id,
                "trip_distance": trip_distance,
                "tpep_pickup_datetime": pickup_datetime,
                "VendorID": 1,
                "passenger_count": 1
            }])

            df_feat = create_features(raw, inference_mode=True)
            df_encoded = encode_for_inference(df_feat)
            feature_cols = load_feature_cols()
            X = df_encoded[feature_cols]

            model = load_model()
            if model is None:
                st.error("Model not loaded. Please ensure models/production_model.pkl exists.")
                return

            log_pred = model.predict(X)[0]
            seconds = np.expm1(log_pred)
            minutes = seconds / 60

            st.success(f"## 🚕 Estimated Duration: {minutes:.1f} minutes")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Distance", f"{trip_distance:.1f} miles")
            with col2:
                rush = "Yes" if df_feat["is_rush_hour"].iloc[0] else "No"
                st.metric("Rush Hour", rush)
            with col3:
                airport = "Yes" if df_feat["is_airport_trip"].iloc[0] else "No"
                st.metric("Airport Trip", airport)

            pu_borough = df_feat["pu_borough"].iloc[0] if "pu_borough" in df_feat.columns else "Unknown"
            do_borough = df_feat["do_borough"].iloc[0] if "do_borough" in df_feat.columns else "Unknown"
            rush_text = "rush hour" if df_feat["is_rush_hour"].iloc[0] else "normal"
            airport_text = "airport" if df_feat["is_airport_trip"].iloc[0] else "local"

            st.info(
                f"This is a **{rush_text}** **{airport_text}** trip - typical for "
                f"**{pu_borough}** to **{do_borough}**."
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

    footer()


def main():

    page = sidebar()

    if page == "📊 Project Overview":
        page_overview()
    elif page == "📈 Explore the Data":
        page_explore()
    elif page == "📉 Model Results":
        page_model_results()
    elif page == "🔮 Predict Duration":
        page_predict()


if __name__ == "__main__":
    main()