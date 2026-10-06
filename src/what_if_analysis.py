from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORECAST_RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_results.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "what_if_scenarios.csv"
)


SCENARIOS = {
    "Baseline": 0.00,
    "Demand +10%": 0.10,
    "Demand +15%": 0.15,
    "Demand +20%": 0.20,
    "Demand -10%": -0.10,
}


def load_forecast_data():
    df = pd.read_csv(
        FORECAST_RESULTS_FILE,
        parse_dates=["date"],
    )

    return df


def select_best_forecast(df):
    model_columns = {
        "Naive": "naive_forecast",
        "Seasonal Naive": "seasonal_naive_forecast",
        "7-Day Moving Average": (
            "moving_average_forecast"
        ),
    }

    errors = {}

    actual = df["total_demand"]

    for model_name, column in model_columns.items():
        error = (
            (actual - df[column])
            .abs()
            .mean()
        )

        errors[model_name] = error

    best_model = min(
        errors,
        key=errors.get,
    )

    best_column = model_columns[
        best_model
    ]

    return (
        best_model,
        best_column,
    )


def create_scenarios(
    df,
    forecast_column,
):
    total_forecast = (
        df[forecast_column].sum()
    )

    rows = []

    for scenario, change in SCENARIOS.items():

        scenario_demand = (
            total_forecast
            * (1 + change)
        )

        additional_demand = (
            scenario_demand
            - total_forecast
        )

        rows.append(
            {
                "scenario": scenario,
                "demand_change_percent": (
                    change * 100
                ),
                "forecast_demand": round(
                    total_forecast,
                    2,
                ),
                "scenario_demand": round(
                    scenario_demand,
                    2,
                ),
                "additional_demand": round(
                    additional_demand,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print(
        "Starting what-if demand analysis..."
    )

    df = load_forecast_data()

    (
        best_model,
        forecast_column,
    ) = select_best_forecast(df)

    print(
        f"Selected forecast model: "
        f"{best_model}"
    )

    scenario_data = create_scenarios(
        df,
        forecast_column,
    )

    scenario_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "WHAT-IF SCENARIO RESULTS"
    )

    print(
        "========================================"
    )

    print(
        scenario_data.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )

    print(
        "\nBusiness interpretation:"
    )

    print(
        "• +10% demand → prepare for 10% additional volume."
    )

    print(
        "• +15% demand → increase inventory and supply capacity."
    )

    print(
        "• +20% demand → evaluate major supply-chain pressure."
    )

    print(
        "• -10% demand → reduce inventory and avoid overstocking."
    )

    print(
        "\n========================================"
    )

    print(
        "WHAT-IF ANALYSIS COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()