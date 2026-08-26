from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.scylla_client import WeatherRow


def build_daily_features(rows: Sequence[WeatherRow], timezone_name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(
        {
            "timestamp": [row.updated_at for row in rows],
            "curah_hujan": [row.curah_hujan for row in rows],
            "kecepatan_angin": [row.kecepatan_angin for row in rows],
            "arah_angin": [row.arah_angin for row in rows],
        }
    )
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert(timezone_name)
    raw["date"] = raw["timestamp"].dt.date

    grouped = raw.groupby("date", as_index=False).agg(
        RR=("curah_hujan", "sum"),
        rainfall_count=("curah_hujan", "count"),
        ff_x=("kecepatan_angin", "max"),
        ff_avg=("kecepatan_angin", "mean"),
        wind_count=("kecepatan_angin", "count"),
        ddd_x=("arah_angin", "mean"),
    )

    grouped["missing_RR"] = (grouped["rainfall_count"] == 0).astype(float)
    grouped["missing_ff_x"] = (grouped["wind_count"] == 0).astype(float)
    grouped["missing_ff_avg"] = (grouped["wind_count"] == 0).astype(float)

    grouped["RR"] = grouped["RR"].fillna(0.0)
    grouped["ff_x"] = grouped["ff_x"].fillna(0.0)
    grouped["ff_avg"] = grouped["ff_avg"].fillna(0.0)
    grouped["ddd_x"] = grouped["ddd_x"].fillna(0.0)

    grouped["rain_3d"] = grouped["RR"].rolling(window=3, min_periods=1).sum()
    grouped["rain_7d"] = grouped["RR"].rolling(window=7, min_periods=1).sum()
    grouped["rain_change_1d"] = grouped["RR"].diff().fillna(0.0)
    grouped["wind_change_1d"] = grouped["ff_avg"].diff().fillna(0.0)

    radians = np.deg2rad(grouped["ddd_x"])
    grouped["ddd_x_sin"] = np.sin(radians)
    grouped["ddd_x_cos"] = np.cos(radians)

    return grouped.drop(columns=["rainfall_count", "wind_count", "ddd_x"])
