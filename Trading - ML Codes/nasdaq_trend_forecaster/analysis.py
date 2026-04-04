from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrendMetrics:
    slope: float
    intercept: float
    r2: float
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class OscillationStats:
    minima: List[pd.Timestamp]
    maxima: List[pd.Timestamp]
    mean_min_cycle_days: Optional[float]
    mean_max_cycle_days: Optional[float]
    mean_amplitude: Optional[float]


def fit_linear(series: pd.Series) -> TrendMetrics:
    """Return slope/intercept/R^2 for the provided series."""
    y = series.to_numpy()
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return TrendMetrics(
        slope=slope,
        intercept=intercept,
        r2=r2,
        start=series.index[0],
        end=series.index[-1],
    )


def is_uptrend(metrics: TrendMetrics, min_slope: float = 0.0, min_r2: float = 0.2) -> bool:
    """
    Decide if the series is in an upward linear trend.

    min_slope: minimum slope in price units per observation.
    min_r2: minimum R^2 for the linear fit.
    """
    return metrics.slope >= min_slope and metrics.r2 >= min_r2


def _extract_local_extrema(series: pd.Series, window: int = 3) -> Tuple[List[int], List[int]]:
    """
    Identify local minima and maxima indices using a symmetric window.

    The rule is simple but noise-resistant: a point must be the strict min/max inside
    its window to be considered an extremum.
    """
    values = series.to_numpy()
    minima_idx: List[int] = []
    maxima_idx: List[int] = []
    n = len(values)
    for i in range(window, n - window):
        segment = values[i - window : i + window + 1]
        center = values[i]
        if center == segment.min() and (segment < center).sum() >= 1:
            minima_idx.append(i)
        if center == segment.max() and (segment > center).sum() >= 1:
            maxima_idx.append(i)
    return minima_idx, maxima_idx


def compute_oscillation(series: pd.Series, window: int = 3) -> OscillationStats:
    """Detect local minima/maxima and derive typical cycle length and amplitude."""
    minima_idx, maxima_idx = _extract_local_extrema(series, window=window)
    minima = [series.index[i] for i in minima_idx]
    maxima = [series.index[i] for i in maxima_idx]

    def _mean_diff_days(points: Sequence[pd.Timestamp]) -> Optional[float]:
        if len(points) < 2:
            return None
        diffs = np.diff(pd.to_datetime(points)).astype("timedelta64[D]").astype(int)
        return float(np.mean(diffs)) if len(diffs) else None

    mean_min_cycle_days = _mean_diff_days(minima)
    mean_max_cycle_days = _mean_diff_days(maxima)

    amplitudes: List[float] = []
    paired = zip(minima_idx, maxima_idx) if minima_idx and maxima_idx else []
    for min_i, max_i in paired:
        amplitudes.append(abs(series.iloc[max_i] - series.iloc[min_i]))
    mean_amplitude = float(np.mean(amplitudes)) if amplitudes else None

    return OscillationStats(
        minima=minima,
        maxima=maxima,
        mean_min_cycle_days=mean_min_cycle_days,
        mean_max_cycle_days=mean_max_cycle_days,
        mean_amplitude=mean_amplitude,
    )


def last_values(series: pd.Series, count: int = 2) -> List[float]:
    """Helper to get last N values as list."""
    if series.empty:
        return []
    return list(series.iloc[-count:])
