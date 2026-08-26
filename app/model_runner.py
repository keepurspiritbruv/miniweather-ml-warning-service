from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelResult:
    experiment_id: str
    hazard: str
    status: str
    score: float | None
    thresholds: dict[str, float]
    reason: str


def has_enough_history(daily_df: pd.DataFrame, sequence_length: int) -> bool:
    return len(daily_df) >= sequence_length


class LSTMAutoencoderRunner:
    def __init__(self, artifact_dir: Path):
        from tensorflow.keras.models import load_model

        self.artifact_dir = artifact_dir
        with (artifact_dir / "feature_config.json").open("r", encoding="utf-8") as file:
            self.feature_config = json.load(file)
        with (artifact_dir / "threshold.json").open("r", encoding="utf-8") as file:
            self.thresholds = {key: float(value) for key, value in json.load(file).items()}

        self.experiment_id = self.feature_config["experiment_id"]
        self.hazard = self.feature_config["hazard"]
        self.sequence_length = int(self.feature_config["sequence_length"])
        self.model_features = list(self.feature_config["model_features"])
        self.feature_weights = np.array(
            [float(self.feature_config["feature_weights"][name]) for name in self.model_features],
            dtype=np.float32,
        )
        self.scaler = joblib.load(artifact_dir / "scaler.pkl")
        self.model = load_model(artifact_dir / "model.keras")

    def score_latest(self, daily_df: pd.DataFrame) -> ModelResult:
        if not has_enough_history(daily_df, self.sequence_length):
            return ModelResult(
                experiment_id=self.experiment_id,
                hazard=self.hazard,
                status="WAITING_FOR_HISTORY",
                score=None,
                thresholds=self.thresholds,
                reason=f"Need {self.sequence_length} daily rows, got {len(daily_df)}",
            )

        missing_features = [name for name in self.model_features if name not in daily_df.columns]
        if missing_features:
            return ModelResult(
                experiment_id=self.experiment_id,
                hazard=self.hazard,
                status="ERROR",
                score=None,
                thresholds=self.thresholds,
                reason=f"Missing features: {', '.join(missing_features)}",
            )

        latest = daily_df.tail(self.sequence_length)
        values = latest[self.model_features].astype(float).to_numpy()
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
