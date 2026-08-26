from datetime import datetime, timezone

import numpy as np

from app.features import build_daily_features
from app.scylla_client import WeatherRow


def row(day: int, rain: float | None, wind: float | None, direction: float | None) -> WeatherRow:
    return WeatherRow(
        row_id=f"row-{day}",
        collection_id="91e2e000-cb17-494c-8a45-c6182b2a89ac",
        created_by="0194fd5b-1768-7533-ac8a-1200c9d6748c",
        updated_at=datetime(2026, 7, day, 12, 0, tzinfo=timezone.utc),
        curah_hujan=rain,
        kecepatan_angin=wind,
        arah_angin=direction,
    )


def test_build_daily_features_creates_rain_and_wind_columns():
    rows = [
        row(1, 1.0, 2.0, 0.0),
        row(2, 3.0, 4.0, 90.0),
        row(3, None, None, None),
    ]

    df = build_daily_features(rows, timezone_name="Asia/Jakarta")

    assert list(df["RR"]) == [1.0, 3.0, 0.0]
    assert list(df["rain_3d"]) == [1.0, 4.0, 4.0]
    assert list(df["rain_change_1d"]) == [0.0, 2.0, -3.0]
    assert list(df["missing_RR"]) == [0.0, 0.0, 1.0]
    assert list(df["ff_x"]) == [2.0, 4.0, 0.0]
    assert list(df["ff_avg"]) == [2.0, 4.0, 0.0]
    assert list(df["missing_ff_x"]) == [0.0, 0.0, 1.0]
    assert np.isclose(df.loc[0, "ddd_x_sin"], 0.0)
    assert np.isclose(df.loc[0, "ddd_x_cos"], 1.0)
    assert np.isclose(df.loc[1, "ddd_x_sin"], 1.0)
    assert np.isclose(df.loc[1, "ddd_x_cos"], 0.0, atol=1e-7)
