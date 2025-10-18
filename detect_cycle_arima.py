"""Utility to test for cyclical patterns in time-series data using ARIMA models.

The script compares the best-fitting non-seasonal ARIMA model against a seasonal
variant (SARIMA) for a user-specified cycle length. A markedly better
seasonal model suggests the presence of a repeating cycle in the data.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ModelSelection:
    """Holds the outcome of a model search."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect cyclical patterns in a time series using ARIMA/SARIMA models.",
    )
    parser.add_argument(
        "data",
        help="Path to a CSV file containing the time series.",
    )
    parser.add_argument(
        "--column",
        help="Name of the column containing the target series. Defaults to the first numeric column.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=12,
        help="Cycle length to test for. Common values are 7 (weekly), 12 (monthly), etc. Default: 12.",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=2,
        help="Maximum AR and MA order to explore during model selection. Default: 2.",
    )
    parser.add_argument(
        "--aic-threshold",
        type=float,
        default=2.0,
        help="Minimum AIC improvement required for the seasonal model to be considered superior.",
    )
    parser.add_argument(
        "--date-column",
        help="Optional name of the column that should be parsed as a datetime index.",
    )
    return parser.parse_args()


def load_series(path: str, column: Optional[str], date_column: Optional[str]) -> pd.Series:
    df = pd.read_csv(path)

    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()

    target_col = column
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available. Please specify --column explicitly.")
        target_col = numeric_cols[0]

    series = df[target_col].dropna()
    if series.empty:
        raise ValueError("The selected column has no data after dropping missing values.")

    series = series.astype(float)

    if isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        inferred_freq = pd.infer_freq(series.index)
        if inferred_freq is not None:
            series = series.asfreq(inferred_freq)

    return series


def suppress_statsmodels_warnings() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings(
        "ignore",
        message=".*No frequency information was provided.*",
        category=ValueWarning,
    )


def candidate_orders(max_order: int, d_values: Iterable[int]) -> Iterable[tuple[int, int, int]]:
    for d in d_values:
        for p in range(max_order + 1):
            for q in range(max_order + 1):
                yield (p, d, q)


def candidate_seasonal_orders(period: int, max_order: int) -> Iterable[tuple[int, int, int, int]]:
    for D in (0, 1):
        for P in range(min(max_order, 1) + 1):
            for Q in range(min(max_order, 1) + 1):
                yield (P, D, Q, period)


def fit_best_model(series: pd.Series, seasonal: bool, period: int, max_order: int) -> ModelSelection:
    best_model: Optional[ModelSelection] = None

    d_candidates = [0, 1]
    seasonal_orders = [(0, 0, 0, 0)] if not seasonal else list(candidate_seasonal_orders(period, max_order))

    for order in candidate_orders(max_order, d_candidates):
        for seas_order in seasonal_orders:
            # Skip seasonal order when it degenerates to non-seasonal ARIMA with d > 1 and no seasonality
            if seasonal and period <= 1:
                continue
            try:
                model = SARIMAX(
                    series,
                    order=order,
                    seasonal_order=seas_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)
            except Exception:
                continue

            if not np.isfinite(result.aic):
                continue

            if best_model is None or result.aic < best_model.aic:
                best_model = ModelSelection(order, seas_order, result.aic)

    if best_model is None:
        raise RuntimeError("Unable to fit any ARIMA models. Consider adjusting max_order or preprocessing data.")

    return best_model


def main() -> None:
    args = parse_args()

    if not args.show_warnings:
        suppress_statsmodels_warnings()

    series = load_series(args.data, args.column, args.date_column)

    if args.period < 2:
        raise ValueError("Period must be at least 2 to evaluate cyclical behavior.")

    non_seasonal = fit_best_model(series, seasonal=False, period=args.period, max_order=args.max_order)
    seasonal = fit_best_model(series, seasonal=True, period=args.period, max_order=args.max_order)

    improvement = non_seasonal.aic - seasonal.aic

    print("Non-seasonal ARIMA best model:")
    print(f"  order={non_seasonal.order}, seasonal_order={non_seasonal.seasonal_order}, AIC={non_seasonal.aic:.2f}")

    print("\nSeasonal ARIMA best model:")
    print(f"  order={seasonal.order}, seasonal_order={seasonal.seasonal_order}, AIC={seasonal.aic:.2f}")

    print("\nAssessment:")
    if improvement > args.aic_threshold:
        print(
            "  The seasonal ARIMA model provides a significantly better fit (AIC improvement "
            f"of {improvement:.2f}). This suggests a cyclical pattern with period {args.period}."
        )
    else:
        print(
            "  Seasonal ARIMA does not substantially outperform the non-seasonal model "
            "based on AIC. A repeating cycle of the specified period is not strongly supported."
        )


if __name__ == "__main__":
    main()
