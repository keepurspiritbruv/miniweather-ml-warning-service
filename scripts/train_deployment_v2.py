"""Train masked (v2) deployment artifacts for the Miniweather ML warning service.

This is the "Fase 2" tool for the masked sliding-window approach. Run it where
TensorFlow is available (Colab / Jupyter server) after notebook 01 produced the
processed feature table.

It reuses `app.masking` so training masks and inference masks follow exactly the
same rules (train/serve parity).

Example:
    python scripts/train_deployment_v2.py \
        --processed data/processed/yogyakarta_weather_features.csv \
        --out artifacts/deployment_v2

Outputs per model (rain_7d, wind_30d):
    model.keras, scaler.pkl, threshold.json (v2: total + per_feature),
    feature_config.json (mask_policy, min_observed_ratio, impute_max_gap_days,
    version=2), training_report.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from app.masking import build_feature_masks, impute_window, masked_weighted_error, to_calendar_window

RAIN_WEIGHTS = {"RR": 3.0, "rain_3d": 3.0, "rain_7d": 3.0, "rain_change_1d": 2.0, "missing_RR": 0.5}
WIND_WEIGHTS = {
    "ff_x": 3.0,
    "ff_avg": 3.0,
    "wind_change_1d": 3.0,
    "ddd_x_sin": 1.0,
    "ddd_x_cos": 1.0,
    "missing_ff_x": 0.5,
    "missing_ff_avg": 0.5,
}

DEPLOYMENT_MODELS = [
    {
        "experiment_id": "rain_7d",
        "hazard": "curah_hujan_tinggi",
        "sequence_length": 7,
        "feature_group": "rain_only",
        "features": ["RR", "rain_3d", "rain_7d", "rain_change_1d", "missing_RR"],
        "weights": RAIN_WEIGHTS,
    },
    {
        "experiment_id": "wind_30d",
        "hazard": "angin_kencang",
        "sequence_length": 30,
        "feature_group": "wind_only",
        "features": ["ff_x", "ff_avg", "wind_change_1d", "ddd_x_sin", "ddd_x_cos", "missing_ff_x", "missing_ff_avg"],
        "weights": WIND_WEIGHTS,
    },
]


def set_seed(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_compact_lstm_autoencoder(sequence_length: int, n_features: int, learning_rate: float):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.optimizers import Adam

    inputs = Input(shape=(sequence_length, n_features))
    encoded = LSTM(32, activation="tanh", return_sequences=True)(inputs)
    encoded = Dropout(0.15)(encoded)
    encoded = LSTM(16, activation="tanh", return_sequences=False)(encoded)
    decoded = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(16, activation="tanh", return_sequences=True)(decoded)
    decoded = Dropout(0.15)(decoded)
    decoded = LSTM(32, activation="tanh", return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def normalised_weights(features: list[str], source: dict[str, float]) -> np.ndarray:
    raw = np.array([source.get(name, 1.0) for name in features], dtype=np.float32)
    return raw / raw.mean()


def calendar_frames(data: pd.DataFrame, features: list[str], impute_max_gap_days: int):
    """Return {station_id: (dates, scaled_values, masks)} with a contiguous calendar index."""
    frames = {}
    for station_id, group in data.groupby("station_id", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        span = (group["date"].max() - group["date"].min()).days + 1
        window = to_calendar_window(group, span)
        masks = build_feature_masks(window, features)
        imputed = impute_window(window, features, impute_max_gap_days)
        frames[str(station_id)] = (window["date"], imputed[features].astype(float), masks)
    return frames


def chronological_split_dates(data: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.85):
    dates = np.array(sorted(pd.to_datetime(data["date"]).dt.normalize().unique()))
    train_end = int(len(dates) * train_ratio)
    val_end = int(len(dates) * val_ratio)
    return set(dates[:train_end]), set(dates[train_end:val_end]), set(dates[val_end:])


def build_windows(frames, features, sequence_length, scaler, date_sets):
    """Slide a calendar window of `sequence_length` days over each station frame."""
    x_parts, m_parts, meta = [], [], []
    train_dates, val_dates, test_dates = date_sets

    for station_id, (dates, values, masks) in frames.items():
        scaled = pd.DataFrame(scaler.transform(values), columns=features)
        for end in range(sequence_length - 1, len(dates)):
            end_date = pd.Timestamp(dates.iloc[end])
            start = end - sequence_length + 1
            if end_date in train_dates:
                split = "train"
            elif end_date in val_dates:
                split = "validation"
            elif end_date in test_dates:
                split = "test"
            else:
                continue
            x_parts.append(scaled.iloc[start:end + 1].to_numpy(dtype=np.float32))
            m_parts.append(masks.iloc[start:end + 1].to_numpy(dtype=np.float32))
            meta.append({"split": split, "date": end_date, "station_id": station_id})

    if not x_parts:
        raise ValueError("No windows built; check the processed data and split.")

    return np.stack(x_parts), np.stack(m_parts), pd.DataFrame(meta)


def masked_loss_values(x, recon, mask, weights):
    import tensorflow as tf

    weights_t = tf.reshape(tf.constant(weights, dtype=tf.float32), (1, 1, -1))
    squared = tf.square(x - recon)
    numerator = tf.reduce_sum(squared * weights_t * mask, axis=[1, 2])
    denominator = tf.reduce_sum(weights_t * mask, axis=[1, 2]) + 1e-8
    return numerator / denominator


def evaluate_masked_loss(model, x, m, weights) -> float:
    import tensorflow as tf

    recon = model(x, training=False)
    return float(tf.reduce_mean(masked_loss_values(x, recon, m, weights)))


def train_masked(
    model,
    x,
    m,
    weights,
    epochs,
    patience,
    batch_size,
    learning_rate,
    x_val=None,
    m_val=None,
):
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    x = tf.constant(x)
    m = tf.constant(m)
    x_val = None if x_val is None else tf.constant(x_val)
    m_val = None if m_val is None else tf.constant(m_val)
    n = int(x.shape[0])
    best = float("inf")
    best_weights = None
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, mb = tf.gather(x, idx), tf.gather(m, idx)
            with tf.GradientTape() as tape:
                recon = model(xb, training=True)
                loss = tf.reduce_mean(masked_loss_values(xb, recon, mb, weights))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            batch_losses.append(float(loss))
        epoch_loss = float(np.mean(batch_losses))

        if x_val is not None:
            monitored = evaluate_masked_loss(model, x_val, m_val, weights)
        else:
            monitored = epoch_loss
        history.append({"epoch": epoch, "train_loss": epoch_loss, "val_loss": monitored})
        print(f"  epoch {epoch:3d} | train {epoch_loss:.6f} | val {monitored:.6f}")

        if monitored < best - 1e-6:
            best = monitored
            best_weights = [v.numpy().copy() for v in model.trainable_variables]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch} (patience {patience})")
                break

    if best_weights is not None:
        for variable, value in zip(model.trainable_variables, best_weights):
            variable.assign(value)
    return history


def evaluate_errors(model, x, m, weights) -> tuple[np.ndarray, np.ndarray]:
    recon = model.predict(x, verbose=0)
    totals, per_feature = [], []
    for i in range(x.shape[0]):
        total, feature_errors = masked_weighted_error(x[i], recon[i], weights, m[i])
        totals.append(np.nan if total is None else total)
        per_feature.append([np.nan if feature_errors[j] is None else feature_errors[j] for j in range(x.shape[2])])
    return np.array(totals), np.array(per_feature)


def thresholds_from(values: np.ndarray) -> dict:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        raise ValueError("No observed values for threshold calculation")
    return {
        "p95": float(np.percentile(clean, 95)),
        "p99": float(np.percentile(clean, 99)),
        "p995": float(np.percentile(clean, 99.5)),
    }


def run(config, data, args, out_root: Path) -> dict:
    features = config["features"]
    weights = normalised_weights(features, config["weights"])
    frames = calendar_frames(data, features, args.impute_max_gap_days)
    date_sets = chronological_split_dates(data)

    # Fit the scaler on train-split rows (observed + imputed), matching inference scaling.
    train_dates = date_sets[0]
    train_rows = []
    for station_id, (dates, values, _) in frames.items():
        mask = pd.to_datetime(dates).isin(train_dates).to_numpy()
        if mask.any():
            train_rows.append(values.to_numpy()[mask])
    if not train_rows:
        raise ValueError("No train rows for the scaler")
    scaler = RobustScaler().fit(np.vstack(train_rows))

    x, m, meta = build_windows(frames, features, config["sequence_length"], scaler, date_sets)
    train = (meta["split"] == "train").to_numpy()
    validation = (meta["split"] == "validation").to_numpy()

    set_seed(args.seed)
    model = build_compact_lstm_autoencoder(config["sequence_length"], len(features), args.learning_rate)
    print(f"[{config['experiment_id']}] train windows={int(train.sum())} validation={int(validation.sum())}")
    history = train_masked(
        model,
        x[train],
        m[train],
        weights,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        x_val=x[validation],
        m_val=m[validation],
    )

    total_val, per_feature_val = evaluate_errors(model, x[validation], m[validation], weights)
    total_thresholds = thresholds_from(total_val)
    per_feature_thresholds = {
        name: thresholds_from(per_feature_val[:, index])
        for index, name in enumerate(features)
    }

    output_dir = out_root / config["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    joblib.dump(scaler, output_dir / "scaler.pkl")
    (output_dir / "threshold.json").write_text(
        json.dumps({"total": total_thresholds, "per_feature": per_feature_thresholds}, indent=2),
        encoding="utf-8",
    )
    feature_config = {
        "experiment_id": config["experiment_id"],
        "hazard": config["hazard"],
        "sequence_length": config["sequence_length"],
        "feature_group": config["feature_group"],
        "model_features": features,
        "feature_weights": {name: float(w) for name, w in zip(features, weights)},
        "mask_policy": "observed_mask_v1",
        "min_observed_ratio": args.min_observed_ratio,
        "impute_max_gap_days": args.impute_max_gap_days,
        "version": 2,
        "source_notebook": "scripts/train_deployment_v2.py",
        "notes": "Masked sliding-window deployment artifact.",
    }
    (output_dir / "feature_config.json").write_text(json.dumps(feature_config, indent=2), encoding="utf-8")
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "train_windows": int(train.sum()),
                "validation_windows": int(validation.sum()),
                "test_windows": int((meta["split"] == "test").sum()),
                "seed": args.seed,
                "epochs_ran": len(history),
                "train_masked_loss_history": history,
                "thresholds": {"total": total_thresholds, "per_feature": per_feature_thresholds},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{config['experiment_id']}] saved to {output_dir}")

    return {
        "experiment_id": config["experiment_id"],
        "hazard": config["hazard"],
        "sequence_length": config["sequence_length"],
        "feature_group": config["feature_group"],
        "n_features": len(features),
        "train_windows": int(train.sum()),
        "validation_windows": int(validation.sum()),
        "test_windows": int((meta["split"] == "test").sum()),
        "total_p95": total_thresholds["p95"],
        "total_p99": total_thresholds["p99"],
        "total_p995": total_thresholds["p995"],
        "artifact_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train masked (v2) deployment artifacts")
    parser.add_argument("--processed", default="data/processed/yogyakarta_weather_features.csv")
    parser.add_argument("--out", default="artifacts/deployment_v2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-observed-ratio", type=float, default=0.7)
    parser.add_argument("--impute-max-gap-days", type=int, default=3)
    args = parser.parse_args()

    processed = Path(args.processed)
    if not processed.exists():
        raise FileNotFoundError(
            f"Processed features not found: {processed}. Run notebook 01 first."
        )

    data = pd.read_csv(processed, parse_dates=["date"], dtype={"station_id": "string"})
    data = data.sort_values(["station_id", "date"]).reset_index(drop=True)

    out_root = Path(args.out)
    rows = [run(config, data, args, out_root) for config in DEPLOYMENT_MODELS]
    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_root / "deployment_model_export_summary.csv", index=False)
    print("Done. Summary:", out_root / "deployment_model_export_summary.csv")


if __name__ == "__main__":
    main()
