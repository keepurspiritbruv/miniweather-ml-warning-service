import pandas as pd

from app.model_runner import has_enough_history, has_consecutive_window


def test_has_enough_history():
    df = pd.DataFrame({"RR": range(7)})

    assert has_enough_history(df, 7) is True
    assert has_enough_history(df, 8) is False


def test_has_consecutive_window_accepts_contiguous_days():
    df = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=7)})

    assert has_consecutive_window(df, 7) is True


def test_has_consecutive_window_rejects_gap_in_window():
    dates = list(pd.date_range("2026-06-28", periods=4)) + list(
        pd.date_range("2026-07-03", periods=3)
    )
    df = pd.DataFrame({"date": dates})

    assert has_consecutive_window(df, 7) is False


def test_has_consecutive_window_rejects_too_few_rows():
    df = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=6)})

    assert has_consecutive_window(df, 7) is False


def test_has_consecutive_window_ignores_old_gap_before_window():
    dates = list(pd.date_range("2026-06-01", periods=3)) + list(
        pd.date_range("2026-07-01", periods=7)
    )
    df = pd.DataFrame({"date": dates})

    assert has_consecutive_window(df, 7) is True
