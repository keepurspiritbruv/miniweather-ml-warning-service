"""Calendar-aligned sliding window with observed/missing masks.

Implements the masked reconstruction-error approach:
- the window is a fixed calendar span (N days) ending at the latest day,
- missing days/values become NaN with mask = 0,
- imputed values are only used as model input and never counted in the error.

Pure pandas/numpy helpers so they can be unit-tested without TensorFlow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Derived feature -> (dependency base columns, rolling window length).
# A derived feature is observed only if all its dependencies are observed
# across the whole rolling window.
FEATURE_DEPS: dict[str, tuple[tuple[str, ...], int]] = {
    "temp_range": (("Tx", "Tn"), 1),
    "rain_3d": (("RR",), 3),
    "rain_7d": (("RR",), 7),
    "rain_change_1d": (("RR",), 2),
    "wind_change_1d": (("ff_x",), 2),
    "ddd_x_sin": (("ddd_x",), 1),
    "ddd_x_cos": (("ddd_x",), 1),
}


def to_calendar_window(
    daily_df: pd.DataFrame,
    window_days: int,
    date_col: str = "date",
) -> pd.DataFrame:
    """Reindex daily features onto N consecutive calendar days ending at the last date.

    Dates without any record become NaN rows instead of being dropped.
    """
    if daily_df.empty:
        return daily_df.copy()

    frame = daily_df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    frame = (
        frame.sort_values(date_col)
        .drop_duplicates(date_col, keep="last")
        .reset_index(drop=True)
    )

    end = frame[date_col].max()
    start = end - pd.Timedelta(days=window_days - 1)
    index = pd.date_range(start, end, freq="D")
    window = frame.set_index(date_col).reindex(index)
    window.index.name = date_col
    return window.reset_index()


def build_feature_masks(
    window_df: pd.DataFrame,
    feature_names: list[str],
    date_col: str = "date",
) -> pd.DataFrame:
    """Return a 0/1 mask per (day, feature) for the given window."""
    value_cols = [c for c in window_df.columns if c != date_col]
    if value_cols:
        observed_matrix = window_df[value_cols].notna().to_numpy()
        any_observed = observed_matrix.any(axis=1).tolist()
    else:
        any_observed = [False] * len(window_df)
    present = pd.Series(any_observed, index=window_df.index, dtype=bool)
    present_int = present.astype(int)

    masks: dict[str, pd.Series] = {}
    for feature in feature_names:
        if feature not in window_df.columns:
            masks[feature] = present_int
            continue
        deps = FEATURE_DEPS.get(feature)
        if deps is None:
            masks[feature] = (window_df[feature].notna() & present).astype(int)
            continue
        dep_cols, span = deps
        base = pd.Series(1, index=window_df.index)
        for dep in dep_cols:
            if dep in window_df.columns:
                base = base & window_df[dep].notna().astype(int)
            else:
                base = base & present_int
        rolled = base.rolling(span, min_periods=span).sum()
        masks[feature] = (rolled == span).astype(int)

    return pd.DataFrame(masks, index=window_df.index)


def impute_window(
    window_df: pd.DataFrame,
    feature_names: list[str],
    max_gap_days: int,
    date_col: str = "date",
) -> pd.DataFrame:
    """Fill NaN gaps for model input only (mask must still come from the raw window)."""
    frame = window_df.copy()
    ordered = frame.sort_values(date_col).set_index(date_col)

    flag_cols = [c for c in feature_names if c.startswith("missing_")]
    value_cols = [c for c in feature_names if c not in flag_cols]

    if value_cols:
        ordered[value_cols] = ordered[value_cols].interpolate(
            method="time",
            limit=max_gap_days,
            limit_direction="both",
        )
    if flag_cols:
        # A fully missing day is genuinely "missing", so the flag becomes 1.
        ordered[flag_cols] = ordered[flag_cols].fillna(1.0)

    ordered[feature_names] = ordered[feature_names].fillna(0.0)
    return ordered.reset_index()


def observed_ratio(mask_df: pd.DataFrame) -> float:
    """Fraction of observed (mask=1) elements in the window."""
    if mask_df.empty or mask_df.size == 0:
        return 0.0
    return float(mask_df.to_numpy().sum() / mask_df.size)


def masked_weighted_error(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[float | None, dict[int, float | None]]:
    """Masked, feature-weighted reconstruction error.

    error = sum(mask * weights * (x_true - x_pred)^2) / sum(mask * weights)

    Returns (total_error, {feature_index: error}). Masked-out elements never
    contribute, so imputed values do not affect the anomaly score.
    """
    x_true = np.asarray(x_true, dtype=np.float64)
    x_pred = np.asarray(x_pred, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x_true.ndim == 3:
        x_true = x_true[0]
        x_pred = x_pred[0]
    if mask.shape != x_true.shape:
        raise ValueError("mask shape must match data shape")

    squared = np.square(x_true - x_pred)
    weighted = squared * weights.reshape(1, -1)

    numerator = (weighted * mask).sum(axis=0)
    denominator = (weights.reshape(1, -1) * mask).sum(axis=0)

    per_feature: dict[int, float | None] = {}
    for idx in range(x_true.shape[1]):
        if denominator[idx] > 0:
            per_feature[idx] = float(numerator[idx] / denominator[idx])
        else:
            per_feature[idx] = None

    total_denominator = denominator.sum()
    if total_denominator <= 0:
        return None, per_feature

    return float(numerator.sum() / total_denominator), per_feature
