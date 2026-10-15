from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_FORECAST_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SYNTHETIC_CSV,
    DEFAULT_TEST_PREDICTIONS_PATH,
    snowflake_config_available,
)
from src.features import build_feature_frame
from src.generate_data import generate_synthetic_sales_dataframe
from src.snowflake_client import fetch_sales_data
from src.train import calculate_metrics


st.set_page_config(
    page_title="Retail Sales Forecasting | ML Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef3f9 100%);
        border: 1px solid #e3eaf3; padding: 1rem 1.1rem; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] {color: #52657c;}
    .hero {
        padding: 1.5rem 1.7rem; border-radius: 16px; margin: .5rem 0 1.2rem;
        color: #f7fbff; background: linear-gradient(115deg, #102a43, #176b87);
    }
    .hero h1 {font-size: 2rem; margin: 0 0 .35rem; color: #fff;}
    .hero p {margin: 0; color: #d9eaf2;}
    .section-note {color: #64748b; font-size: .9rem; margin-top: -.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def get_synthetic_data() -> pd.DataFrame:
    if not DEFAULT_SYNTHETIC_CSV.exists():
        DEFAULT_SYNTHETIC_CSV.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df = generate_synthetic_sales_dataframe()
        synthetic_df[["SALE_DATE", "REVENUE", "QUANTITY"]].to_csv(
            DEFAULT_SYNTHETIC_CSV, index=False
        )
    return pd.read_csv(DEFAULT_SYNTHETIC_CSV)


@st.cache_data(ttl=300)
def get_snowflake_data() -> pd.DataFrame:
    return fetch_sales_data()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def money(value: float | int | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def render_dashboard() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Retail Sales Forecasting</h1>
          <p>Explore daily sales, evaluate a machine-learning model against a seasonal baseline,
          and inspect a configurable 14-day outlook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Dashboard controls")
    source_options = ["Synthetic CSV"]
    if snowflake_config_available():
        source_options.append("Snowflake")
    selected_source = st.sidebar.selectbox("Data source", source_options)
    if st.sidebar.button("Refresh data", use_container_width=True):
        get_synthetic_data.clear()
        get_snowflake_data.clear()
        st.rerun()

    try:
        if selected_source == "Synthetic CSV":
            historical = get_synthetic_data()
            source_label = "Synthetic CSV"
        else:
            historical = get_snowflake_data()
            source_label = "Snowflake"
    except Exception as exc:
        st.error(f"Could not load {selected_source} data. {exc}")
        if selected_source == "Snowflake":
            st.info(
                "Check the Snowflake values in `.env`, confirm your role can read "
                "`RETAIL_ML.ANALYTICS.SALES_RAW`, and try refreshing."
            )
        st.stop()

    historical = historical.copy()
    historical["SALE_DATE"] = pd.to_datetime(historical["SALE_DATE"], errors="raise")
    historical["REVENUE"] = pd.to_numeric(historical["REVENUE"], errors="raise")
    historical = historical.sort_values("SALE_DATE").reset_index(drop=True)
    if historical.empty:
        st.error("The selected data source returned no sales observations.")
        st.stop()

    metrics = load_json(DEFAULT_METRICS_PATH)
    test_predictions = load_csv(DEFAULT_TEST_PREDICTIONS_PATH)
    forecast = load_csv(DEFAULT_FORECAST_PATH)
    if test_predictions is not None:
        test_predictions["SALE_DATE"] = pd.to_datetime(test_predictions["SALE_DATE"])
    if forecast is not None:
        forecast["SALE_DATE"] = pd.to_datetime(forecast["SALE_DATE"])

    st.sidebar.markdown("---")
    history_days = st.sidebar.slider(
        "Historical chart window (days)",
        min_value=1,
        max_value=len(historical),
        value=min(len(historical), 90),
        help="Limit the historical chart and summary to the most recent days.",
    )
    forecast_days = 0
    if forecast is not None and not forecast.empty:
        forecast_days = st.sidebar.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=min(14, len(forecast)),
            value=min(14, len(forecast)),
            help="Show only the selected number of days from the saved recursive forecast.",
        )

    st.sidebar.caption(
        "Forecast horizon can be selected up to the 14 days generated by "
        "`python -m src.train`."
    )

    st.caption(
        f"**Data source:** {source_label}  ·  "
        f"**Coverage:** {historical['SALE_DATE'].min():%b %d, %Y} – "
        f"{historical['SALE_DATE'].max():%b %d, %Y}"
    )
    if source_label == "Synthetic CSV":
        st.info(
            "Demo dataset: these sales are synthetic and do not represent a real retailer. "
            "Use this mode to explore the workflow without Snowflake credentials.",
            icon="🧪",
        )

    recent_history = historical.tail(history_days)
    latest_revenue = historical.iloc[-1]["REVENUE"]
    model_metrics = (metrics or {}).get("model_metrics", {})
    baseline_metrics = (metrics or {}).get("naive_baseline_metrics", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("Observations in view", f"{len(recent_history):,}")
    metric_columns[1].metric("Latest actual revenue", money(latest_revenue))
    metric_columns[2].metric("Holdout MAE", money(model_metrics.get("MAE")))
    metric_columns[3].metric("Holdout RMSE", money(model_metrics.get("RMSE")))

    if metrics:
        evaluation_source = metrics.get("data_source", "unknown")
        st.caption(
            f"Model metrics are from the **{evaluation_source}** chronological holdout "
            f"({metrics.get('evaluation_period', {}).get('start', 'unknown')} to "
            f"{metrics.get('evaluation_period', {}).get('end', 'unknown')})."
        )
        if evaluation_source.lower() != source_label.lower().replace(" csv", ""):
            st.warning(
                f"The displayed model evaluation was generated from {evaluation_source} data, "
                f"not the currently selected {source_label} source."
            )
    elif not Path(DEFAULT_MODEL_PATH).exists():
        st.warning(
            "Evaluation and forecast outputs are not available yet. Run "
            "`python -m src.train` to create the model, holdout metrics, and forecast."
        )

    historical_tab, evaluation_tab, forecast_tab, data_tab = st.tabs(
        ["Sales history", "Model evaluation", "Forecast", "Data quality"]
    )

    with historical_tab:
        st.subheader("Daily revenue")
        st.markdown(
            f'<p class="section-note">Showing the latest {len(recent_history):,} of '
            f'{len(historical):,} observations.</p>',
            unsafe_allow_html=True,
        )
        st.line_chart(
            recent_history.set_index("SALE_DATE")[["REVENUE"]],
            color=["#176b87"],
            y_label="Revenue",
        )
        if "QUANTITY" in recent_history.columns:
            with st.expander("View sales records"):
                st.dataframe(recent_history, width="stretch", hide_index=True)

    with evaluation_tab:
        st.subheader("Chronological holdout performance")
        if test_predictions is None or test_predictions.empty:
            st.info("No holdout predictions found. Run `python -m src.train` to generate them.")
        else:
            evaluation_days = st.slider(
                "Holdout days to inspect",
                min_value=1,
                max_value=len(test_predictions),
                value=min(len(test_predictions), 90),
                key="evaluation_days",
                help="The most recent selected holdout observations are shown and scored below.",
            )
            visible_test = test_predictions.tail(evaluation_days).copy()
            st.line_chart(
                visible_test.set_index("SALE_DATE")[
                    ["ACTUAL_REVENUE", "PREDICTED_REVENUE", "NAIVE_7D_REVENUE"]
                ],
                color=["#176b87", "#e07a3f", "#8795a1"],
                y_label="Revenue",
            )
            selected_metrics = calculate_metrics(
                visible_test["ACTUAL_REVENUE"].to_numpy(),
                visible_test["PREDICTED_REVENUE"].to_numpy(),
            )
            selected_baseline = calculate_metrics(
                visible_test["ACTUAL_REVENUE"].to_numpy(),
                visible_test["NAIVE_7D_REVENUE"].to_numpy(),
            )
            score_columns = st.columns(3)
            score_columns[0].metric("Selected-period MAE", money(selected_metrics["MAE"]))
            score_columns[1].metric("Selected-period RMSE", money(selected_metrics["RMSE"]))
            score_columns[2].metric(
                "Selected-period WAPE",
                "N/A" if selected_metrics["WAPE"] is None else f"{selected_metrics['WAPE']:.1%}",
            )
            if selected_metrics["MAE"] > selected_baseline["MAE"]:
                st.warning(
                    "The model's MAE is higher than the naive 7-day baseline for this "
                    "selected period. The baseline is currently the stronger forecast."
                )
            else:
                st.success(
                    "The model's MAE is lower than the naive 7-day baseline for this "
                    "selected period."
                )
            with st.expander("Saved full holdout summary"):
                st.json(metrics or {"message": "Saved evaluation metrics are unavailable."})

    with forecast_tab:
        st.subheader("Future revenue outlook")
        if forecast is None or forecast.empty:
            st.info(
                "No future forecast is available. Run `python -m src.train` to generate "
                "the model and 14-day forecast."
            )
        else:
            visible_forecast = forecast.head(forecast_days).copy()
            forecast_total = visible_forecast["REVENUE"].sum()
            forecast_average = visible_forecast["REVENUE"].mean()
            forecast_columns = st.columns(3)
            forecast_columns[0].metric("Days displayed", len(visible_forecast))
            forecast_columns[1].metric("Forecast total", money(forecast_total))
            forecast_columns[2].metric("Daily average", money(forecast_average))
            st.line_chart(
                visible_forecast.set_index("SALE_DATE")[["REVENUE"]],
                color=["#6a5acd"],
                y_label="Forecast revenue",
            )
            st.dataframe(
                visible_forecast[["SALE_DATE", "REVENUE"]].rename(
                    columns={"SALE_DATE": "Forecast date", "REVENUE": "Predicted revenue"}
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Predicted revenue": st.column_config.NumberColumn(format="$%.2f")
                },
            )
            st.caption(
                "Forecasts are experimental. Recursive predictions use earlier predicted "
                "values as inputs, so errors can accumulate over time."
            )

    with data_tab:
        st.subheader("Data validation")
        try:
            build_feature_frame(historical)
            st.success(
                "No duplicate dates or missing daily observations detected. "
                "Lag and rolling features can be created."
            )
        except (ValueError, KeyError) as exc:
            st.warning(str(exc))
            st.write(
                "Investigate missing or duplicate dates against the source system before "
                "training. This dashboard does not fill gaps or invent revenue values."
            )
        st.write(
            f"**Selected source:** {source_label}  \n"
            f"**Rows:** {len(historical):,}  \n"
            f"**Date range:** {historical['SALE_DATE'].min():%Y-%m-%d} to "
            f"{historical['SALE_DATE'].max():%Y-%m-%d}"
        )
        if metrics:
            st.subheader("Model run details")
            st.json(
                {
                    "model": metrics.get("model"),
                    "features": metrics.get("feature_list"),
                    "evaluation_period": metrics.get("evaluation_period"),
                    "model_metrics": model_metrics,
                    "naive_baseline_metrics": baseline_metrics,
                    "random_seed": metrics.get("random_seed"),
                }
            )

    st.markdown("---")
    st.caption(
        "Portfolio project · Python · scikit-learn · Snowflake · Streamlit · "
        "Synthetic demo data is not real retailer performance."
    )


render_dashboard()
