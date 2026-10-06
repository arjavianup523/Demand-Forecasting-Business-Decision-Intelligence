from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_test.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast_results.csv"
)


SEASONAL_PERIOD = 7


def load_data():
    train_data = pd.read_csv(
        TRAIN_FILE,
        parse_dates=["date"],
    )

    test_data = pd.read_csv(
        TEST_FILE,
        parse_dates=["date"],
    )

    train_data = train_data.sort_values(
        "date"
    )

    test_data = test_data.sort_values(
        "date"
    )

    return train_data, test_data


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mape = (
        np.mean(
            np.abs(
                (actual - predicted)
                / actual
            )
        )
        * 100
    )

    return mae, rmse, mape


def naive_forecast(
    train_data,
    test_data,
):
    last_value = train_data[
        "total_demand"
    ].iloc[-1]

    predictions = np.repeat(
        last_value,
        len(test_data),
    )

    return predictions


def seasonal_naive_forecast(
    train_data,
    test_data,
):
    train_values = train_data[
        "total_demand"
    ].to_numpy()

    predictions = []

    for i in range(len(test_data)):
        source_index = (
            len(train_values)
            - SEASONAL_PERIOD
            + (
                i
                % SEASONAL_PERIOD
            )
        )

        predictions.append(
            train_values[source_index]
        )

    return np.array(predictions)


def moving_average_forecast(
    train_data,
    test_data,
):
    train_values = train_data[
        "total_demand"
    ].to_numpy()

    window_size = 7

    predictions = []

    history = list(train_values)

    for _ in range(len(test_data)):
        recent_values = history[
            -window_size:
        ]

        prediction = np.mean(
            recent_values
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.array(predictions)


def evaluate_model(
    model_name,
    actual,
    predictions,
):
    mae, rmse, mape = calculate_metrics(
        actual,
        predictions,
    )

    return {
        "model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_percent": round(mape, 2),
    }


def main():
    print(
        "Starting demand forecasting..."
    )

    train_data, test_data = load_data()

    actual = test_data[
        "total_demand"
    ].to_numpy()

    print(
        f"Training observations: "
        f"{len(train_data):,}"
    )

    print(
        f"Testing observations: "
        f"{len(test_data):,}"
    )

    print(
        "\nRunning Naive Forecast..."
    )

    naive_predictions = naive_forecast(
        train_data,
        test_data,
    )

    print(
        "Running Seasonal Naive Forecast..."
    )

    seasonal_predictions = (
        seasonal_naive_forecast(
            train_data,
            test_data,
        )
    )

    print(
        "Running 7-Day Moving Average Forecast..."
    )

    moving_average_predictions = (
        moving_average_forecast(
            train_data,
            test_data,
        )
    )

    results = []

    results.append(
        evaluate_model(
            "Naive",
            actual,
            naive_predictions,
        )
    )

    results.append(
        evaluate_model(
            "Seasonal Naive",
            actual,
            seasonal_predictions,
        )
    )

    results.append(
        evaluate_model(
            "7-Day Moving Average",
            actual,
            moving_average_predictions,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAPE_percent"
    )

    forecast_output = test_data[
        [
            "date",
            "total_demand",
        ]
    ].copy()

    forecast_output[
        "naive_forecast"
    ] = naive_predictions

    forecast_output[
        "seasonal_naive_forecast"
    ] = seasonal_predictions

    forecast_output[
        "moving_average_forecast"
    ] = moving_average_predictions

    forecast_output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "FORECAST MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    best_model = results_df.iloc[0]

    print(
        "\n========================================"
    )

    print(
        "BEST MODEL"
    )

    print(
        "========================================"
    )

    print(
        f"Model: {best_model['model']}"
    )

    print(
        f"MAE: {best_model['MAE']:,.2f}"
    )

    print(
        f"RMSE: {best_model['RMSE']:,.2f}"
    )

    print(
        f"MAPE: "
        f"{best_model['MAPE_percent']:.2f}%"
    )

    print(
        f"\nForecast results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\n========================================"
    )

    print(
        "DEMAND FORECASTING COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()