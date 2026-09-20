import numpy as np
import pandas as pd

from app.masking import (
    build_feature_masks,
    impute_window,
    masked_weighted_error,
    observed_ratio,
    to_calendar_window,
)


def _daily_frame(dates):
    rr = [1.0] * len(dates)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "RR": rr,
            "rain_3d": pd.Series(rr).rolling(3, min_periods=1).sum(),
            "rain_7d": pd.Series(rr).rolling(7, min_periods=1).sum(),
            "rain_change_1d": pd.Series(rr).diff().fillna(0.0),
            "missing_RR": [0.0] * len(dates),
        }
    )
    return frame


def test_to_calendar_window_fills_missing_dates_with_nan():
    dates = pd.date_range("2026-09-01", periods=5)
    frame = _daily_frame(dates)
    frame = frame[frame["date"] != pd.Timestamp("2026-09-03")]

    window = to_calendar_window(frame, 5)

    assert len(window) == 5
    missing_row = window[window["date"] == pd.Timestamp("2026-09-03")]
    assert len(missing_row) == 1
    assert missing_row["RR"].isna().all()


def test_masks_mark_missing_day_and_propagate_to_derived_features():
    dates = pd.date_range("2026-09-01", periods=10)
    frame = _daily_frame(dates)
    frame = frame[frame["date"] != pd.Timestamp("2026-09-05")]

    window = to_calendar_window(frame, 10)
    masks = build_feature_masks(window, list(frame.columns.drop("date")))

    gap_day = window["date"] == pd.Timestamp("2026-09-05")
    assert masks.loc[gap_day, "RR"].iloc[0] == 0
    assert masks.loc[gap_day, "missing_RR"].iloc[0] == 0

    # rain_7d is observed only once no gap falls inside its 7-day rolling window.
    day_after_gap = window["date"] == pd.Timestamp("2026-09-06")
    assert masks.loc[day_after_gap, "rain_7d"].iloc[0] == 0


def test_observed_ratio_of_one_missing_day_out_of_30():
    dates = pd.date_range("2026-09-01", periods=30)
    frame = _daily_frame(dates)
    frame = frame[frame["date"] != pd.Timestamp("2026-09-15")]

    window = to_calendar_window(frame, 30)
    masks = build_feature_masks(window, ["RR", "missing_RR"])

    ratio = observed_ratio(masks)
    assert 0.0 < ratio < 1.0
    # 29 observed days of 30, averaged over two columns (RR, missing_RR).
    assert round(ratio, 4) == round(29 / 30, 4)


def test_masked_error_ignores_masked_elements():
    x_true = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    x_pred = np.array([[[1.0, 1.0], [9.0, 1.0]]])  # big error only at (day 1, feature 0)
    weights = np.array([1.0, 1.0])
    mask = np.array([[1, 1], [0, 1]])  # (day 1, feature 0) is masked out

    total, per_feature = masked_weighted_error(x_true[0], x_pred[0], weights, mask)

    assert total == 0.0
    assert per_feature[0] in (None, 0.0) or per_feature[0] == 0.0
    assert per_feature[1] == 0.0


def test_masked_error_counts_only_observed():
    x_true = np.array([[2.0, 2.0]])
    x_pred = np.array([[1.0, 1.0]])
    weights = np.array([3.0, 1.0])
    mask = np.array([[1, 0]])

    total, per_feature = masked_weighted_error(x_true, x_pred, weights, mask)

    assert total == 1.0  # only feature 0 observed: (2-1)^2 * 3 / (3)
    assert per_feature[0] == 1.0
    assert per_feature[1] is None


def test_impute_window_interpolates_gap_and_flags_missing():
    dates = pd.date_range("2026-09-01", periods=5)
    frame = _daily_frame(dates)
    frame = frame[frame["date"] != pd.Timestamp("2026-09-03")]

    window = to_calendar_window(frame, 5)
    imputed = impute_window(window, ["RR", "missing_RR"], max_gap_days=3)

    assert not imputed["RR"].isna().any()
    gap_row = imputed[imputed["date"] == pd.Timestamp("2026-09-03")]
    assert gap_row["RR"].iloc[0] == 1.0  # interpolated between two 1.0 days
    assert gap_row["missing_RR"].iloc[0] == 1.0
