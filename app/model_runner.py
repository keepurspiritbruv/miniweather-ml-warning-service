from dataclasses import dataclass, field
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from app.masking import (
    build_feature_masks,
    impute_window,
    masked_weighted_error,
    observed_ratio,
    to_calendar_window,
)


@dataclass(frozen=True)
class ModelResult:
    experiment_id: str
    hazard: str
    status: str
    score: float | None
    thresholds: dict[str, float]
    reason: str
    feature_errors: dict[str, float | None] = field(default_factory=dict)
    missing_ratio: float | None = None


def has_enough_history(daily_df: pd.DataFrame, sequence_length: int) -> bool:
    return len(daily_df) >= sequence_length


def has_consecutive_window(daily_df: pd.DataFrame, sequence_length: int) -> bool:
    if "date" not in daily_df.columns or len(daily_df) < sequence_length:
        return False
    ordered = daily_df.sort_values("date")
    tail_dates = pd.to_datetime(ordered["date"]).tail(sequence_length)
    day_diffs = tail_dates.diff().dt.days.iloc[1:]
    return bool(len(day_diffs) == sequence_length - 1 and (day_diffs == 1).all())


class LSTMAutoencoderRunner:
    def __init__(self, artifact_dir: Path):
        from tensorflow.keras.models import load_model

        self.artifact_dir = artifact_dir
        with (artifact_dir / "feature_config.json").open("r", encoding="utf-8") as file:
            self.feature_config = json.load(file)
        with (artifact_dir / "threshold.json").open("r", encoding="utf-8") as file:
            raw_thresholds = json.load(file)

        self.experiment_id = self.feature_config["experiment_id"]
        self.hazard = self.feature_config["hazard"]
        self.sequence_length = int(self.feature_config["sequence_length"])
        self.model_features = list(self.feature_config["model_features"])
        self.feature_weights = np.array(
            [float(self.feature_config["feature_weights"][name]) for name in self.model_features],
            dtype=np.float32,
        )

        # v1 artifacts store a flat {p95,p99,p995}; v2 store {"total": {...}, "per_feature": {...}}.
        if "p95" in raw_thresholds:
            self.thresholds = {key: float(value) for key, value in raw_thresholds.items()}
            self.per_feature_thresholds: dict[str, dict[str, float]] = {}
        else:
            self.thresholds = {
                key: float(value) for key, value in raw_thresholds.get("total", {}).items()
            }
            self.per_feature_thresholds = {
                name: {key: float(value) for key, value in values.items()}
                for name, values in raw_thresholds.get("per_feature", {}).items()
            }

        # Masked scoring activates only for artifacts that declare a mask policy.
        self.masked_scoring = bool(self.feature_config.get("mask_policy"))
        self.min_observed_ratio = float(self.feature_config.get("min_observed_ratio", 0.7))
        self.impute_max_gap_days = int(self.feature_config.get("impute_max_gap_days", 3))

        self.scaler = joblib.load(artifact_dir / "scaler.pkl")
        self.model = load_model(artifact_dir / "model.keras")

    def _waiting(self, reason: str) -> ModelResult:
        return ModelResult(
            experiment_id=self.experiment_id,
            hazard=self.hazard,
            status="WAITING_FOR_HISTORY",
            score=None,
            thresholds=self.thresholds,
            reason=reason,
        )

    def _insufficient(self, reason: str) -> ModelResult:
        return ModelResult(
            experiment_id=self.experiment_id,
            hazard=self.hazard,
            status="INSUFFICIENT_DATA",
            score=None,
            thresholds=self.thresholds,
            reason=reason,
        )

    def score_latest(self, daily_df: pd.DataFrame) -> ModelResult:
        if not has_enough_history(daily_df, self.sequence_length):
            return self._waiting(
                f"Need {self.sequence_length} daily rows, got {len(daily_df)}"
            )

        missing_features = [
            name for name in self.model_features if name not in daily_df.columns
        ]
        if missing_features:
            return ModelResult(
                experiment_id=self.experiment_id,
                hazard=self.hazard,
                status="ERROR",
                score=None,
                thresholds=self.thresholds,
                reason=f"Missing features: {', '.join(missing_features)}",
            )

        if self.masked_scoring:
            return self._score_masked(daily_df)

        if not has_consecutive_window(daily_df, self.sequence_length):
            return self._waiting(
                f"Last {self.sequence_length} daily rows are not "
                "consecutive calendar days"
            )

        latest = daily_df.tail(self.sequence_length)
        values = latest[self.model_features].astype(float)
        scaled = self.scaler.transform(values)
        sequence = np.expand_dims(scaled, axis=0)
        reconstructed = self.model.predict(sequence, verbose=0)
        error = np.square(sequence - reconstructed)
        weighted_error = error * self.feature_weights.reshape(1, 1, -1)
        score = float(np.mean(weighted_error))

        return ModelResult(
            experiment_id=self.experiment_id,
            hazard=self.hazard,
            status="SCORED",
            score=score,
            thresholds=self.thresholds,
            reason="OK",
        )

    def _score_masked(self, daily_df: pd.DataFrame) -> ModelResult:
        window = to_calendar_window(daily_df, self.sequence_length)
        masks = build_feature_masks(window, self.model_features)
        ratio = observed_ratio(masks)
        missing_ratio = 1.0 - ratio

        if ratio < self.min_observed_ratio:
            return self._insufficient(
                f"Observed ratio {ratio:.2f} < {self.min_observed_ratio:.2f}"
            )

        imputed = impute_window(window, self.model_features, self.impute_max_gap_days)
        if imputed[self.model_features].isna().to_numpy().any():
            return self._insufficient("Not enough observed data to impute window")

        values = imputed[self.model_features].astype(float)
        scaled = self.scaler.transform(values)
        sequence = np.expand_dims(scaled, axis=0)
        reconstructed = self.model.predict(sequence, verbose=0)

        total, per_feature_index = masked_weighted_error(
            sequence[0], reconstructed[0], self.feature_weights, masks.to_numpy()
        )
        if total is None:
            return self._insufficient("No observed elements in window")

        feature_errors = {
            name: per_feature_index.get(index)
            for index, name in enumerate(self.model_features)
        }

        return ModelResult(
            experiment_id=self.experiment_id,
            hazard=self.hazard,
            status="SCORED",
            score=total,
            thresholds=self.thresholds,
            reason="OK",
            feature_errors=feature_errors,
            missing_ratio=missing_ratio,
        )
