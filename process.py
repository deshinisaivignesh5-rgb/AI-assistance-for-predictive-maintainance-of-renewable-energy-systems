from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
from matplotlib import pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "static" / "plots"

WIND_SOURCE = "https://raw.githubusercontent.com/n-sapkota/Fault-Detection-wind-turbine/main/wt.csv"
VISION_SOURCE = "https://github.com/zae-bayern/elpv-dataset"
REPORT_BENCHMARK = {
    "specificity": 0.91,
    "sensitivity": 0.89,
    "context": "Values cited in the seminar report for the CNN-based inspection study.",
}
VISION_PROFILE = {
    "profile_id": "solar_elpv",
    "display_name": "Dual-purpose renewable inspection module",
    "dataset_name": "ELPV solar electroluminescence dataset",
    "dataset_note": (
        "The platform supports both wind operational screening and image-based inspection. "
        "The bundled public image demo uses solar electroluminescence imagery because that is "
        "the available open dataset in this workspace."
    ),
    "task": "Binary renewable-energy image defect classification",
    "image_size": [64, 64],
    "color_mode": "grayscale",
    "threshold": 0.5,
    "label_map": {
        "0": "Healthy / low-defect asset image",
        "1": "Defective asset image",
    },
}
RANDOM_STATE = 42
RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)


def reset_plot_directory() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in PLOT_DIR.glob("*.png"):
        file_path.unlink()


def save_plot(filename: str) -> str:
    output_path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return f"plots/{filename}"


def plot_item(title: str, filename: str, description: str) -> dict:
    return {"title": title, "path": filename, "description": description}


def load_wind_dataframe() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "wind_turbine_scada.csv")
    df["timestamp"] = pd.to_datetime(df["ttimestamplocal"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["unitlocation", "timestamp"]).reset_index(drop=True)
    df["generation_delta"] = (
        df.groupby("unitlocation")["generation"].diff().fillna(0).clip(lower=0, upper=2500)
    )
    df["power_delta"] = df.groupby("unitlocation")["power"].diff().fillna(0).clip(-1200, 1200)
    df["temp_gap"] = df["wheel hub temperature"] - df["ambient Temperature"]
    df["tower_gap"] = df["Tower bottom ambient temperature"] - df["ambient Temperature"]
    windspeed_safe = np.clip(df["windspeed"], 0.5, None)
    df["power_to_speed"] = df["power"] / windspeed_safe
    df["rpm_to_speed"] = df["rtr_rpm"] / windspeed_safe
    df["hour_of_day"] = df["timestamp"].dt.hour + (df["timestamp"].dt.minute / 60.0)
    return df


def wind_band_counts(scores: np.ndarray, thresholds: dict[str, float]) -> dict[str, int]:
    return {
        "High": int(np.sum(scores <= thresholds["high"])),
        "Elevated": int(np.sum((scores > thresholds["high"]) & (scores <= thresholds["elevated"]))),
        "Watch": int(np.sum((scores > thresholds["elevated"]) & (scores <= thresholds["watch"]))),
        "Stable": int(np.sum(scores > thresholds["watch"])),
    }


def plot_wind_score_distribution(scores: np.ndarray, thresholds: dict[str, float]) -> dict:
    plt.figure(figsize=(8, 4.5))
    plt.hist(scores, bins=45, color="#2c7a54", alpha=0.82, edgecolor="#ffffff")
    plt.axvline(thresholds["high"], color="#842029", linestyle="--", linewidth=2, label="High risk threshold")
    plt.axvline(thresholds["elevated"], color="#a84e1a", linestyle="--", linewidth=2, label="Elevated threshold")
    plt.axvline(thresholds["watch"], color="#c28a1f", linestyle="--", linewidth=2, label="Watch threshold")
    plt.title("Wind Anomaly Score Distribution")
    plt.xlabel("Anomaly score")
    plt.ylabel("Record count")
    plt.legend(frameon=False)
    filename = save_plot("wind_score_distribution.png")
    return plot_item(
        "Wind Score Distribution",
        filename,
        "Histogram of wind anomaly scores with the three operational alert thresholds.",
    )


def plot_wind_band_distribution(counts: dict[str, int]) -> dict:
    colors = ["#842029", "#a84e1a", "#c28a1f", "#2c7a54"]
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(list(counts.keys()), list(counts.values()), color=colors)
    plt.title("Wind Risk Band Distribution")
    plt.ylabel("Record count")
    filename = save_plot("wind_risk_bands.png")
    return plot_item(
        "Wind Risk Bands",
        filename,
        "Count of records that fall into stable, watch, elevated, and high-risk monitoring bands.",
    )


def train_wind_model() -> tuple[dict, dict, list[dict]]:
    df = load_wind_dataframe()
    feature_columns = [
        "windspeed",
        "power",
        "wind direction Angle",
        "rtr_rpm",
        "pitch Angle",
        "wheel hub temperature",
        "ambient Temperature",
        "Tower bottom ambient temperature",
        "generation_delta",
        "power_delta",
        "temp_gap",
        "tower_gap",
        "power_to_speed",
        "rpm_to_speed",
        "hour_of_day",
    ]
    feature_df = df[feature_columns].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    normal_mask = (
        feature_df["windspeed"].between(2.0, 18.0)
        & feature_df["rtr_rpm"].between(1.5, 13.0)
        & feature_df["pitch Angle"].between(-1.2, 18.0)
        & feature_df["wheel hub temperature"].between(-8.0, 42.0)
        & feature_df["power"].between(-10.0, 2300.0)
        & feature_df["generation_delta"].between(0.0, 850.0)
    )
    train_frame = feature_df.loc[normal_mask, feature_columns]
    if len(train_frame) > 30000:
        train_frame = train_frame.sample(30000, random_state=RANDOM_STATE)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("detector", IsolationForest(n_estimators=250, contamination=0.08, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(train_frame)

    full_scores = pipeline.named_steps["detector"].decision_function(
        pipeline.named_steps["scaler"].transform(feature_df[feature_columns])
    )
    thresholds = {
        "high": float(np.quantile(full_scores, 0.08)),
        "elevated": float(np.quantile(full_scores, 0.25)),
        "watch": float(np.quantile(full_scores, 0.55)),
    }
    band_counts = wind_band_counts(full_scores, thresholds)
    alert_rate = float((full_scores <= thresholds["high"]).mean())

    artifact = {
        "pipeline": pipeline,
        "features": feature_columns,
        "thresholds": thresholds,
        "defaults": {column: float(feature_df[column].median()) for column in feature_columns},
        "ranges": {
            column: {
                "min": float(feature_df[column].quantile(0.02)),
                "max": float(feature_df[column].quantile(0.98)),
            }
            for column in feature_columns
        },
    }
    metrics = {
        "training_rows": int(len(train_frame)),
        "inference_rows": int(len(feature_df)),
        "feature_count": len(feature_columns),
        "estimated_high_risk_rate": round(alert_rate, 4),
        "risk_band_counts": band_counts,
    }
    plots = [
        plot_wind_score_distribution(full_scores, thresholds),
        plot_wind_band_distribution(band_counts),
    ]
    return artifact, metrics, plots


def load_vision_rows() -> pd.DataFrame:
    labels_path = RAW_DIR / "elpv-dataset" / "src" / "elpv_dataset" / "data" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError("Image dataset labels were not found. Run scripts/download_datasets.py first.")

    image_root = labels_path.parent
    frame = pd.read_csv(
        labels_path,
        sep=r"\s+",
        names=["relative_path", "probability", "module_type"],
        engine="python",
    )
    frame["path"] = frame["relative_path"].map(lambda value: str(image_root / value))
    frame["probability"] = frame["probability"].astype(float)
    frame["label"] = (frame["probability"] >= 0.5).astype(np.int32)
    return frame[["path", "module_type", "probability", "label"]]


def preprocess_image_bytes(image_bytes: bytes, size: tuple[int, int], color_mode: str) -> np.ndarray:
    mode = "L" if color_mode == "grayscale" else "RGB"
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert(mode).resize(size, RESAMPLE)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if color_mode == "grayscale":
        array = array[..., np.newaxis]
    return array


def build_vision_arrays(paths: pd.Series, image_size: tuple[int, int], color_mode: str) -> np.ndarray:
    arrays = [
        preprocess_image_bytes(Path(path).read_bytes(), image_size, color_mode)
        for path in paths
    ]
    return np.stack(arrays).astype(np.float32)


def build_class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = np.bincount(labels)
    total = float(labels.shape[0])
    return {class_id: total / (2.0 * count) for class_id, count in enumerate(counts) if count > 0}


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(0.15, 0.85, 29):
        predictions = (probabilities >= threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1_value = float(f1_score(labels, predictions, zero_division=0))
        score = specificity + sensitivity
        if score > best_score or (np.isclose(score, best_score) and f1_value > best_f1):
            best_threshold = float(threshold)
            best_score = score
            best_f1 = f1_value
    return round(best_threshold, 4)


def build_vision_model(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="inspection_image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.25, fill_mode="nearest")(x)

    for index, filters in enumerate((32, 64, 96, 128), start=1):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv_{index}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{index}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{index}")(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool_{index}")(x)
        if index >= 2:
            x = tf.keras.layers.Dropout(0.12 + (index * 0.04), name=f"dropout_{index}")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(96, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dense_dropout_1")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.15, name="dense_dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="fault_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="vision_inspection_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def plot_vision_class_distribution(frame: pd.DataFrame) -> dict:
    counts = frame["label"].value_counts().sort_index()
    labels = ["Healthy", "Defective"]
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, counts.values, color=["#2c7a54", "#a84e1a"])
    plt.title("Vision Dataset Class Distribution")
    plt.ylabel("Image count")
    filename = save_plot("vision_class_distribution.png")
    return plot_item(
        "Vision Class Distribution",
        filename,
        "Healthy versus defective image counts in the bundled public inspection dataset.",
    )


def plot_training_history(history: dict[str, list[float]]) -> dict:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(epochs, history["loss"], label="Train loss", color="#2c7a54")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss", color="#a84e1a")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)

    train_auc = history.get("auc") or history.get("accuracy", [])
    val_auc = history.get("val_auc") or history.get("val_accuracy", [])
    axes[1].plot(epochs, train_auc, label="Train score", color="#2c7a54")
    axes[1].plot(epochs, val_auc, label="Validation score", color="#a84e1a")
    axes[1].set_title("Training Progress over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)

    filename = save_plot("vision_training_history.png")
    return plot_item(
        "Vision Training over Epochs",
        filename,
        "Training and validation curves across epochs so you can inspect convergence behavior.",
    )


def plot_confusion_matrix_figure(cm: np.ndarray) -> dict:
    labels = ["Healthy", "Defective"]
    plt.figure(figsize=(5.2, 4.8))
    plt.imshow(cm, cmap="YlGn")
    plt.title("Vision Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            plt.text(column, row, int(cm[row, column]), ha="center", va="center", color="#19211c")
    plt.colorbar(fraction=0.046, pad=0.04)
    filename = save_plot("vision_confusion_matrix.png")
    return plot_item(
        "Vision Confusion Matrix",
        filename,
        "True versus predicted defect decisions on the held-out image test set.",
    )


def plot_roc_curve_figure(labels: np.ndarray, probabilities: np.ndarray, roc_auc: float) -> dict:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(fpr, tpr, color="#2c7a54", linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#9aa39d")
    plt.title("Vision ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(frameon=False)
    filename = save_plot("vision_roc_curve.png")
    return plot_item(
        "Vision ROC Curve",
        filename,
        "Receiver operating characteristic curve for the image classifier on the test split.",
    )


def plot_metric_overview(metrics: dict) -> dict:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity", "sensitivity"]
    display_names = ["Acc", "Prec", "Recall", "F1", "ROC AUC", "Spec", "Sens"]
    values = [metrics[name] for name in metric_names]
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(display_names, values, color="#2c7a54")
    plt.ylim(0, 1.05)
    plt.title("Vision Metric Overview")
    plt.ylabel("Score")
    filename = save_plot("vision_metric_overview.png")
    return plot_item(
        "Vision Metric Overview",
        filename,
        "Side-by-side view of the main evaluation metrics for the current image model.",
    )


def plot_report_comparison(metrics: dict) -> dict:
    categories = ["Specificity", "Sensitivity"]
    current_values = [metrics["specificity"], metrics["sensitivity"]]
    report_values = [REPORT_BENCHMARK["specificity"], REPORT_BENCHMARK["sensitivity"]]
    positions = np.arange(len(categories))
    width = 0.34

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(positions - width / 2, current_values, width=width, color="#2c7a54", label="Current model")
    plt.bar(positions + width / 2, report_values, width=width, color="#c28a1f", label="Report benchmark")
    plt.xticks(positions, categories)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Current Vision Model vs Report Benchmark")
    plt.legend(frameon=False)
    filename = save_plot("vision_report_comparison.png")
    return plot_item(
        "Vision vs Report Benchmark",
        filename,
        "Comparison of the current model against the report's cited specificity and sensitivity values.",
    )


def train_vision_model() -> tuple[dict, dict, list[dict]]:
    frame = load_vision_rows()
    image_size = tuple(VISION_PROFILE["image_size"])
    color_mode = VISION_PROFILE["color_mode"]

    train_eval_frame, test_frame = train_test_split(
        frame,
        test_size=0.2,
        stratify=frame["label"],
        random_state=RANDOM_STATE,
    )
    train_frame, validation_frame = train_test_split(
        train_eval_frame,
        test_size=0.15,
        stratify=train_eval_frame["label"],
        random_state=RANDOM_STATE,
    )

    x_train = build_vision_arrays(train_frame["path"], image_size, color_mode)
    x_validation = build_vision_arrays(validation_frame["path"], image_size, color_mode)
    x_test = build_vision_arrays(test_frame["path"], image_size, color_mode)
    y_train = train_frame["label"].to_numpy(dtype=np.int32)
    y_validation = validation_frame["label"].to_numpy(dtype=np.int32)
    y_test = test_frame["label"].to_numpy(dtype=np.int32)

    tf.keras.backend.clear_session()
    model = build_vision_model(x_train.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=3,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=1,
            mode="max",
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=12,
        batch_size=64,
        class_weight=build_class_weights(y_train),
        callbacks=callbacks,
        verbose=2,
    )

    validation_probabilities = model.predict(x_validation, verbose=0).reshape(-1)
    threshold = select_threshold(y_validation, validation_probabilities)
    probabilities = model.predict(x_test, verbose=0).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int32)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    roc_auc = float(roc_auc_score(y_test, probabilities))
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "specificity": round(specificity, 4),
        "sensitivity": round(sensitivity, 4),
        "train_images": int(len(train_frame)),
        "validation_images": int(len(validation_frame)),
        "test_images": int(len(test_frame)),
        "epochs_ran": int(len(history.history["loss"])),
        "threshold": threshold,
        "backend": "tensorflow",
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    artifact = {
        "model": model,
        "model_filename": "vision_inspection_model.h5",
        "metadata_filename": "vision_inspection_metadata.json",
        "image_size": list(image_size),
        "color_mode": color_mode,
        "threshold": threshold,
        "profile": VISION_PROFILE,
        "label_map": VISION_PROFILE["label_map"],
        "defaults": {"sample_image": str(test_frame.iloc[0]["path"])},
        "metrics": metrics,
    }
    plots = [
        plot_vision_class_distribution(frame),
        plot_training_history(history.history),
        plot_confusion_matrix_figure(cm),
        plot_roc_curve_figure(y_test, probabilities, roc_auc),
        plot_metric_overview(metrics),
        plot_report_comparison(metrics),
    ]
    return artifact, metrics, plots


def write_metrics_file(wind_metrics: dict, vision_metrics: dict, plot_groups: dict[str, list[dict]]) -> None:
    metrics = {
        "project_title": "Renewable AI Maintenance Assistant",
        "report_alignment": {
            "summary": [
                "This is a dual-purpose renewable maintenance platform with both wind analytics and visual inspection.",
                "The wind side screens SCADA behavior for maintenance risk, and the vision side runs CNN-based defect inspection.",
                "AI provides triage support while humans keep the final maintenance decision.",
                "The public visual dataset is solar electroluminescence, but the platform structure is reusable for other renewable asset images.",
            ],
            "seminar_reference": REPORT_BENCHMARK,
            "current_scope": VISION_PROFILE["dataset_note"],
        },
        "datasets": {
            "wind": {
                "source": WIND_SOURCE,
                "rows": 85066,
                "task": "Wind turbine operational anomaly detection",
            },
            "vision": {
                "source": VISION_SOURCE,
                "images": vision_metrics["train_images"] + vision_metrics["validation_images"] + vision_metrics["test_images"],
                "task": VISION_PROFILE["task"],
                "profile_id": VISION_PROFILE["profile_id"],
                "display_name": VISION_PROFILE["dataset_name"],
            },
        },
        "models": {
            "wind": wind_metrics,
            "vision": vision_metrics,
        },
        "plots": plot_groups,
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    reset_plot_directory()

    stale_model = MODEL_DIR / "vision_inspection_model.keras"
    if stale_model.exists():
        try:
            stale_model.unlink()
        except PermissionError:
            pass

    wind_artifact, wind_metrics, wind_plots = train_wind_model()
    joblib.dump(wind_artifact, MODEL_DIR / "wind_risk_model.joblib")

    vision_artifact, vision_metrics, vision_plots = train_vision_model()
    model_path = MODEL_DIR / vision_artifact["model_filename"]
    metadata_path = MODEL_DIR / vision_artifact["metadata_filename"]
    if model_path.exists():
        model_path.unlink()
    vision_artifact["model"].save(model_path, include_optimizer=False)

    metadata = {key: value for key, value in vision_artifact.items() if key != "model"}
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    write_metrics_file(
        wind_metrics,
        vision_metrics,
        {"wind": wind_plots, "vision": vision_plots},
    )

    print("Saved wind model ->", MODEL_DIR / "wind_risk_model.joblib")
    print("Saved vision model ->", model_path)
    print("Saved vision metadata ->", metadata_path)
    print("Saved metrics ->", MODEL_DIR / "metrics.json")
    print("Saved plots ->", PLOT_DIR)


if __name__ == "__main__":
    main()
