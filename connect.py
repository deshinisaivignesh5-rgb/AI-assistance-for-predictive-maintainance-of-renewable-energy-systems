from __future__ import annotations

import base64
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from flask import Flask, render_template, request

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
WIND_MODEL_PATH = MODEL_DIR / "wind_risk_model.joblib"
VISION_MODEL_PATH = MODEL_DIR / "vision_inspection_model.h5"
VISION_METADATA_PATH = MODEL_DIR / "vision_inspection_metadata.json"
ALLOWED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

app = Flask(__name__)

MODE_OPTIONS = [
    ("wind", "Wind Operational Risk"),
    ("vision", "Image Defect Inspection"),
]
WIND_FORM_FIELDS = [
    ("windspeed", "Wind Speed (m/s)"),
    ("power", "Generated Power (kW)"),
    ("wind direction Angle", "Wind Direction (deg)"),
    ("rtr_rpm", "Rotor RPM"),
    ("pitch Angle", "Pitch Angle (deg)"),
    ("wheel hub temperature", "Wheel Hub Temperature (C)"),
    ("ambient Temperature", "Ambient Temperature (C)"),
    ("Tower bottom ambient temperature", "Tower Bottom Temperature (C)"),
    ("generation_delta", "Generation Delta"),
    ("power_delta", "Power Delta"),
    ("temp_gap", "Hub / Ambient Gap"),
    ("tower_gap", "Tower / Ambient Gap"),
    ("power_to_speed", "Power-to-Speed Ratio"),
    ("rpm_to_speed", "RPM-to-Speed Ratio"),
    ("hour_of_day", "Hour of Day"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_wind_artifact() -> dict | None:
    return joblib.load(WIND_MODEL_PATH) if WIND_MODEL_PATH.exists() else None


@lru_cache(maxsize=1)
def load_vision_metadata() -> dict:
    return load_json(VISION_METADATA_PATH)


@lru_cache(maxsize=1)
def load_vision_model():
    if tf is None or not VISION_MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(VISION_MODEL_PATH, compile=False)


def derive_wind_features(values: dict[str, float]) -> dict[str, float]:
    windspeed_safe = max(values["windspeed"], 0.5)
    values["temp_gap"] = values["wheel hub temperature"] - values["ambient Temperature"]
    values["tower_gap"] = values["Tower bottom ambient temperature"] - values["ambient Temperature"]
    values["power_to_speed"] = values["power"] / windspeed_safe
    values["rpm_to_speed"] = values["rtr_rpm"] / windspeed_safe
    return values


def default_wind_values() -> dict[str, float]:
    wind_artifact = load_wind_artifact()
    if wind_artifact is None:
        return derive_wind_features(
            {
                "windspeed": 4.47,
                "power": 361.86,
                "wind direction Angle": 180.54,
                "rtr_rpm": 7.43,
                "pitch Angle": -0.50,
                "wheel hub temperature": 24.04,
                "ambient Temperature": 17.59,
                "Tower bottom ambient temperature": 30.14,
                "generation_delta": 42.0,
                "power_delta": 0.0,
                "temp_gap": 6.45,
                "tower_gap": 12.55,
                "power_to_speed": 80.96,
                "rpm_to_speed": 1.66,
                "hour_of_day": 12.0,
            }
        )
    defaults = {key: round(value, 2) for key, value in wind_artifact["defaults"].items()}
    return derive_wind_features(defaults)


def default_page_values() -> dict:
    defaults = default_wind_values()
    return {
        "analysis_mode": "wind",
        "vision_filename": "",
        "wind_form": defaults,
    }


def get_form_values(source) -> dict:
    values = default_page_values()
    values["analysis_mode"] = source.get("analysis_mode", values["analysis_mode"])
    for key, _label in WIND_FORM_FIELDS:
        values["wind_form"][key] = source.get(key, values["wind_form"][key])
    return values


def render_page(**context):
    metrics = load_json(METRICS_PATH)
    page_context = {
        **default_page_values(),
        **context,
        "metrics": metrics,
        "plot_groups": metrics.get("plots", {}),
        "mode_options": MODE_OPTIONS,
        "wind_fields": WIND_FORM_FIELDS,
        "vision_profile": load_vision_metadata().get("profile", {}),
        "models_ready": (
            load_wind_artifact() is not None
            and bool(load_vision_metadata())
            and load_vision_model() is not None
        ),
    }
    return render_template("index.html", **page_context)


def preprocess_image_bytes(image_bytes: bytes, size: tuple[int, int], color_mode: str) -> np.ndarray:
    mode = "L" if color_mode == "grayscale" else "RGB"
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert(mode).resize(size, RESAMPLE)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if color_mode == "grayscale":
        array = array[..., np.newaxis]
    return array


def build_preview(raw_bytes: bytes, mimetype: str | None) -> str:
    safe_mimetype = mimetype if mimetype and mimetype.startswith("image/") else "image/png"
    return f"data:{safe_mimetype};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"


def read_upload(upload) -> tuple[bytes | None, str | None]:
    if not upload or not upload.filename:
        return None, "Please upload an inspection image before predicting."
    suffix = Path(upload.filename).suffix.lower()
    if suffix and suffix not in ALLOWED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
        return None, f"Unsupported image type. Use one of: {allowed}."
    raw_bytes = upload.read()
    if not raw_bytes:
        return None, "The uploaded file is empty."
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        return None, "The uploaded image is too large. Keep it below 8 MB."
    return raw_bytes, None


def wind_risk_label(score: float, thresholds: dict) -> tuple[str, str]:
    if score <= thresholds["high"]:
        return "High Maintenance Risk", "high"
    if score <= thresholds["elevated"]:
        return "Elevated Maintenance Risk", "high"
    if score <= thresholds["watch"]:
        return "Watch Condition", "medium"
    return "Stable Operating Band", "low"


def vision_risk_level(probability: float, threshold: float) -> str:
    if probability >= max(0.8, threshold + 0.2):
        return "high"
    if probability >= threshold:
        return "medium"
    return "low"


def build_metric(label: str, value) -> dict:
    return {"label": label, "value": value}


def build_summary(label: str, value) -> dict:
    return {"label": label, "value": value}


def predict_wind(form_values: dict) -> dict:
    wind_artifact = load_wind_artifact()
    wind_values = {key: float(form_values["wind_form"][key]) for key, _label in WIND_FORM_FIELDS}
    wind_values = derive_wind_features(wind_values)
    sample = pd.DataFrame([wind_values])[wind_artifact["features"]]
    score = float(wind_artifact["pipeline"].decision_function(sample)[0])
    label, risk_level = wind_risk_label(score, wind_artifact["thresholds"])

    return {
        "prediction_text": label,
        "risk_level": risk_level,
        "reason_text": (
            f"The wind anomaly model compared your values with learned maintenance bands. "
            f"Current anomaly score: {score:.4f}."
        ),
        "result_metrics": [
            build_metric("Anomaly Score", round(score, 4)),
            build_metric("High Threshold", round(wind_artifact["thresholds"]["high"], 4)),
            build_metric("Analysis Mode", "Wind Operational Risk"),
            build_metric("Feature Count", len(wind_artifact["features"])),
        ],
        "summary_items": [
            build_summary("Wind Speed", f"{wind_values['windspeed']:.2f} m/s"),
            build_summary("Power", f"{wind_values['power']:.2f} kW"),
            build_summary("Rotor RPM", f"{wind_values['rtr_rpm']:.2f}"),
            build_summary("Pitch Angle", f"{wind_values['pitch Angle']:.2f} deg"),
            build_summary("Ambient Temperature", f"{wind_values['ambient Temperature']:.2f} C"),
            build_summary("Hub Temperature", f"{wind_values['wheel hub temperature']:.2f} C"),
            build_summary("Temp Gap", f"{wind_values['temp_gap']:.2f}"),
            build_summary("Power-to-Speed", f"{wind_values['power_to_speed']:.2f}"),
        ],
        "wind_form": wind_values,
    }


def predict_vision(form_values: dict, upload) -> dict:
    raw_bytes, upload_error = read_upload(upload)
    if upload_error:
        raise ValueError(upload_error)

    metadata = load_vision_metadata()
    model = load_vision_model()
    image_size = tuple(metadata.get("image_size", [64, 64]))
    color_mode = metadata.get("color_mode", "grayscale")
    threshold = float(metadata.get("threshold", 0.5))
    label_map = metadata.get("label_map", {})

    image_array = preprocess_image_bytes(raw_bytes, image_size, color_mode)
    probability = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
    prediction = int(probability >= threshold)
    prediction_text = label_map.get(str(prediction), "Image Inspection Result")
    risk_level = vision_risk_level(probability, threshold)

    return {
        "prediction_text": prediction_text,
        "risk_level": risk_level,
        "reason_text": (
            f"The TensorFlow vision model estimated a defect probability of {probability * 100:.2f}% "
            f"against a saved decision threshold of {threshold * 100:.2f}%."
        ),
        "result_metrics": [
            build_metric("Model Probability", f"{probability * 100:.2f}%"),
            build_metric("Decision Threshold", f"{threshold * 100:.2f}%"),
            build_metric("Analysis Mode", "Image Defect Inspection"),
            build_metric("Image Size", "64 x 64"),
        ],
        "summary_items": [
            build_summary("Uploaded File", upload.filename),
            build_summary("Image Profile", metadata.get("profile", {}).get("dataset_name", "Vision Dataset")),
            build_summary("Color Mode", color_mode),
            build_summary("Decision", "Defective" if prediction == 1 else "Healthy"),
            build_summary("Threshold", f"{threshold:.2f}"),
            build_summary("Prediction Score", f"{probability:.4f}"),
        ],
        "vision_preview": build_preview(raw_bytes, upload.mimetype),
        "vision_filename": upload.filename,
        "analysis_mode": form_values["analysis_mode"],
        "wind_form": form_values["wind_form"],
    }


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    form_values = get_form_values(request.form)

    try:
        if form_values["analysis_mode"] == "vision":
            result = predict_vision(form_values, request.files.get("vision_image"))
        else:
            result = predict_wind(form_values)
        return render_page(**result)
    except Exception as exc:
        return render_page(
            **form_values,
            prediction_text=f"Error: {exc}",
            risk_level="high",
            reason_text="Please review the inputs and try again.",
            result_metrics=[
                build_metric("Status", "Prediction failed"),
                build_metric("Analysis Mode", form_values["analysis_mode"].title()),
            ],
            summary_items=[],
        )


if __name__ == "__main__":
    app.run(debug=True)
