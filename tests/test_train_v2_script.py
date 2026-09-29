import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_deployment_v2.py"
FEATURES = ["RR", "rain_3d", "rain_7d", "rain_change_1d", "missing_RR"]


def load_script():
    spec = importlib.util.spec_from_file_location("train_deployment_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def processed_frame() -> pd.DataFrame:
    frames = []
    for station in ["96851", "96855"]:
        dates = pd.date_range("2026-01-01", periods=40)
        rr = np.linspace(0.0, 5.0, 40)
        frame = pd.DataFrame(
            {
                "date": dates,
                "station_id": station,
                "RR": rr,
                "rain_3d": pd.Series(rr).rolling(3, min_periods=1).sum(),
                "rain_7d": pd.Series(rr).rolling(7, min_periods=1).sum(),
                "rain_change_1d": pd.Series(rr).diff().fillna(0.0),
                "missing_RR": 0.0,
            }
        )
        if station == "96851":
            frame = frame[frame["date"] != pd.Timestamp("2026-01-20")]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_chronological_split_returns_timestamp_sets():
    module = load_script()

    train, _validation, _test = module.chronological_split_dates(processed_frame())

    assert isinstance(next(iter(train)), pd.Timestamp)
    assert pd.Timestamp("2026-01-01") in train


def test_build_windows_returns_calendar_windows_with_missing_day():
    module = load_script()
    data = processed_frame()

    frames = module.calendar_frames(data, FEATURES, impute_max_gap_days=3)
    date_sets = module.chronological_split_dates(data)

    train_rows = []
    for _station_id, (dates, values, _masks) in frames.items():
        in_train = pd.to_datetime(dates).isin(date_sets[0]).to_numpy()
        if in_train.any():
            train_rows.append(values.to_numpy()[in_train])
    scaler = RobustScaler().fit(pd.DataFrame(np.vstack(train_rows), columns=FEATURES))

    x, masks, meta = module.build_windows(frames, FEATURES, 7, scaler, date_sets)

    assert x.shape[0] > 0
    assert x.shape[1] == 7
    assert x.shape[2] == len(FEATURES)
    assert masks.shape == x.shape
    assert set(meta["split"]).issubset({"train", "validation", "test"})
