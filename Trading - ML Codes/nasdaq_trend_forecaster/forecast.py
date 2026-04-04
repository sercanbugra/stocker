from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .analysis import OscillationStats, TrendMetrics, compute_oscillation, fit_linear, is_uptrend


@dataclass
class ForecastResult:
    symbol: str
    slope: float
    r2: float
    next_min_date: Optional[pd.Timestamp]
    predicted_min_price: Optional[float]
    next_max_date: Optional[pd.Timestamp]
    predicted_max_price: Optional[float]
    suggested_buy_price: Optional[float]
    note: str


def _linear_forecast_by_time(series: pd.Series, target_date: pd.Timestamp) -> float:
    """
    Fit a linear model over calendar days and predict at target_date.

    This keeps the "step size" tied to days instead of event order.
    """
    x = (series.index - series.index[0]).days.astype(float)
    y = series.to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    delta_days = (target_date - series.index[0]).days
    return float(slope * delta_days + intercept)


def _forecast_extrema_dates(series: pd.Series, stats: OscillationStats):
    next_min_date = None
    next_max_date = None
    if stats.minima and stats.mean_min_cycle_days:
        next_min_date = stats.minima[-1] + pd.Timedelta(days=int(round(stats.mean_min_cycle_days)))
    if stats.maxima and stats.mean_max_cycle_days:
        next_max_date = stats.maxima[-1] + pd.Timedelta(days=int(round(stats.mean_max_cycle_days)))
    return next_min_date, next_max_date


def _forecast_extrema_prices(series: pd.Series, stats: OscillationStats, next_min_date, next_max_date):
    predicted_min_price = None
    predicted_max_price = None

    if next_min_date is not None and len(stats.minima) >= 2:
        minima_series = series.loc[stats.minima]
        predicted_min_price = _linear_forecast_by_time(minima_series, next_min_date)
    if next_max_date is not None and len(stats.maxima) >= 2:
        maxima_series = series.loc[stats.maxima]
        predicted_max_price = _linear_forecast_by_time(maxima_series, next_max_date)

    # Fallback: use full-trend regression if extrema models are not usable.
    if predicted_min_price is None and next_min_date is not None:
        predicted_min_price = _linear_forecast_by_time(series, next_min_date)
    if predicted_max_price is None and next_max_date is not None:
        predicted_max_price = _linear_forecast_by_time(series, next_max_date)
    return predicted_min_price, predicted_max_price


def make_forecast(
    symbol: str,
    price_history: pd.DataFrame,
    *,
    min_points: int = 20,
    min_slope: float = 0.0,
    min_r2: float = 0.05,
) -> Optional[ForecastResult]:
    """
    Generate a forecast for the given symbol using recent price history.

    Returns None if there is not enough data or the trend quality is too low.
    """
    if price_history.empty or len(price_history) < min_points:
        return None

    close = price_history["Close"] if "Close" in price_history else price_history.iloc[:, 0]
    close = close.dropna()
    trend = fit_linear(close)
    if not is_uptrend(trend, min_slope=min_slope, min_r2=min_r2):
        return None

    osc = compute_oscillation(close, window=3)
    next_min_date, next_max_date = _forecast_extrema_dates(close, osc)
    predicted_min_price, predicted_max_price = _forecast_extrema_prices(
        close, osc, next_min_date, next_max_date
    )

    # Suggested buy trigger: midpoint between projected max/min keeps risk in check.
    suggested_buy_price = None
    if predicted_min_price and predicted_max_price:
        suggested_buy_price = (predicted_min_price + predicted_max_price) / 2

    note_parts = []
    if osc.mean_amplitude:
        note_parts.append(f"typical swing ~{osc.mean_amplitude:.2f}")
    if osc.mean_max_cycle_days:
        note_parts.append(f"cycle ~{osc.mean_max_cycle_days:.0f}d")
    if not note_parts:
        note_parts.append("limited oscillation history")

    return ForecastResult(
        symbol=symbol,
        slope=trend.slope,
        r2=trend.r2,
        next_min_date=next_min_date,
        predicted_min_price=predicted_min_price,
        next_max_date=next_max_date,
        predicted_max_price=predicted_max_price,
        suggested_buy_price=suggested_buy_price,
        note="; ".join(note_parts),
    )
