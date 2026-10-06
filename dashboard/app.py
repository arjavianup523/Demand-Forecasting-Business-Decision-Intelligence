from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORECAST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_data.csv"
)

FORECAST_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_results.csv"
)

ANOMALIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_anomalies.csv"
)

SCENARIOS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "what_if_scenarios.csv"
)


st.set_page_config(
    page_title="Demand Forecasting & Business Intelligence",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_data():
    forecast_data = pd.read_csv(
        FORECAST_DATA_FILE,
        parse_dates=["date"],
    )

    forecast_results = pd.read_csv(
        FORECAST_RESULTS_FILE,
        parse_dates=["date"],
    )

    anomalies = pd.read_csv(
        ANOMALIES_FILE,
        parse_dates=["date"],
    )

    scenarios = pd.read_csv(
        SCENARIOS_FILE,
    )

    return (
        forecast_data,
        forecast_results,
        anomalies,
        scenarios,
    )


forecast_data, forecast_results, anomalies, scenarios = (
    load_data()
)


st.title(
    "📈 Demand Forecasting & Business Decision Intelligence"
)

st.caption(
    "End-to-end demand forecasting, anomaly detection, "
    "and business scenario analysis"
)


st.sidebar.header(
    "Dashboard Filters"
)


min_date = forecast_data["date"].min().date()

max_date = forecast_data["date"].max().date()


selected_dates = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)


if len(selected_dates) == 2:

    start_date = pd.Timestamp(
        selected_dates[0]
    )

    end_date = pd.Timestamp(
        selected_dates[1]
    )

    filtered_data = forecast_data[
        (
            forecast_data["date"]
            >= start_date
        )
        &
        (
            forecast_data["date"]
            <= end_date
        )
    ].copy()

else:

    filtered_data = forecast_data.copy()


total_demand = filtered_data[
    "total_demand"
].sum()


total_revenue = filtered_data[
    "total_revenue"
].sum()


average_daily_demand = filtered_data[
    "total_demand"
].mean()


anomaly_count = len(
    anomalies[
        (
            anomalies["date"]
            >= filtered_data["date"].min()
        )
        &
        (
            anomalies["date"]
            <= filtered_data["date"].max()
        )
    ]
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Demand",
        f"{total_demand:,.0f}",
    )


with col2:

    st.metric(
        "Total Revenue",
        f"₹{total_revenue / 1_000_000:,.1f}M",
    )


with col3:

    st.metric(
        "Average Daily Demand",
        f"{average_daily_demand:,.0f}",
    )


with col4:

    st.metric(
        "Detected Anomalies",
        f"{anomaly_count:,}",
    )


st.divider()


st.header(
    "📊 Demand Trend"
)


daily_trend = (
    filtered_data[
        [
            "date",
            "total_demand",
        ]
    ]
    .set_index("date")
)


st.line_chart(
    daily_trend
)


st.divider()


st.header(
    "🔮 Forecast Performance"
)


forecast_display = forecast_results.copy()


st.line_chart(
    forecast_display[
        [
            "date",
            "total_demand",
            "naive_forecast",
            "seasonal_naive_forecast",
            "moving_average_forecast",
        ]
    ].set_index("date")
)


st.subheader(
    "Forecast Model Comparison"
)


model_metrics = []


for model_name, column in [
    (
        "Naive",
        "naive_forecast",
    ),
    (
        "Seasonal Naive",
        "seasonal_naive_forecast",
    ),
    (
        "7-Day Moving Average",
        "moving_average_forecast",
    ),
]:

    actual = forecast_results[
        "total_demand"
    ]

    predicted = forecast_results[
        column
    ]

    mae = (
        actual - predicted
    ).abs().mean()

    rmse = (
        (
            (actual - predicted) ** 2
        ).mean()
        ** 0.5
    )

    mape = (
        (
            (
                actual - predicted
            ).abs()
            / actual
        ).mean()
        * 100
    )

    model_metrics.append(
        {
            "Model": model_name,
            "MAE": round(
                mae,
                2,
            ),
            "RMSE": round(
                rmse,
                2,
            ),
            "MAPE (%)": round(
                mape,
                2,
            ),
        }
    )


metrics_df = pd.DataFrame(
    model_metrics
)


st.dataframe(
    metrics_df,
    use_container_width=True,
    hide_index=True,
)


st.divider()


st.header(
    "🚨 Demand Anomalies"
)


if len(anomalies) > 0:

    st.dataframe(
        anomalies[
            [
                "date",
                "total_demand",
                "rolling_mean",
                "z_score",
                "anomaly_type",
            ]
        ].sort_values(
            "date",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No demand anomalies detected."
    )


st.divider()


st.header(
    "🎯 What-If Demand Scenarios"
)


st.dataframe(
    scenarios,
    use_container_width=True,
    hide_index=True,
)


st.bar_chart(
    scenarios.set_index(
        "scenario"
    )[
        "scenario_demand"
    ]
)


st.divider()


st.header(
    "💡 Business Recommendations"
)


st.markdown(
    """
### Inventory Planning

Use the demand forecast to prepare inventory
before expected demand increases.

### Seasonal Planning

Demand is strongest during the final quarter,
so inventory and supply capacity should be
increased before October.

### Promotion Planning

Promotional periods historically generate
higher demand and should be coordinated with
inventory availability.

### Anomaly Monitoring

Large demand spikes or drops should trigger
investigation into promotions, supply issues,
pricing changes, or unusual market conditions.

### Scenario Planning

A sudden 10–20% demand increase can create
significant supply pressure. Scenario analysis
helps management prepare capacity and inventory
before the increase occurs.
"""
)


st.divider()


st.caption(
    "Demand Forecasting & Business Decision Intelligence Platform"
)