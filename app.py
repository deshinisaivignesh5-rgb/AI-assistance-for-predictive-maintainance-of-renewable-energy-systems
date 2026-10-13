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
except Exception:  # pragma: no cover - TensorFlow is optional at import time
    tf = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
WIND_MODEL_PATH = MODEL_DIR / "wind_risk_model.joblib"
VISION_MODEL_PATH = MODEL_DIR / "vision_inspection_model.h5"
VISION_METADATA_PATH = MODEL_DIR / "vision_inspection_metadata.json"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

WIND_FORM_FIELDS = [
    ("windspeed", "Wind speed (m/s)"),
    ("power", "Generated power (kW)"),
    ("wind direction Angle", "Wind direction (deg)"),
    ("rtr_rpm", "Rotor RPM"),
    ("pitch Angle", "Pitch angle (deg)"),
    ("wheel hub temperature", "Wheel hub temp (C)"),
    ("ambient Temperature", "Ambient temp (C)"),
    ("Tower bottom ambient temperature", "Tower bottom temp (C)"),
    ("generation_delta", "Generation delta"),
    ("power_delta", "Power delta"),
    ("temp_gap", "Hub/ambient temp gap"),
    ("tower_gap", "Tower/ambient temp gap"),
    ("power_to_speed", "Power-to-speed ratio"),
    ("rpm_to_speed", "RPM-to-speed ratio"),
    ("hour_of_day", "Hour of day"),
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_joblib(path: Path) -> dict | None:
    return joblib.load(path) if path.exists() else None


@lru_cache(maxsize=1)
def load_vision_model():
    if tf is None or not VISION_MODEL_PATH.exists():
        return None
    return tf.keras.models.load_model(VISION_MODEL_PATH, compile=False)


def load_vision_metadata() -> dict:
    return load_json(VISION_METADATA_PATH)


def preprocess_image_bytes(image_bytes: bytes, size: tuple[int, int], color_mode: str) -> np.ndarray:
    mode = "L" if color_mode == "grayscale" else "RGB"
    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert(mode).resize(size, RESAMPLE)
        array = np.asarray(image, dtype=np.float32) / 255.0
    if color_mode == "grayscale":
        array = array[..., np.newaxis]
    return array


def read_upload_bytes(upload) -> tuple[bytes | None, str | None]:
    if not upload or not upload.filename:
        return None, "Please upload an inspection image before running visual prediction."

    suffix = Path(upload.filename).suffix.lower()
    if suffix and suffix not in ALLOWED_IMAGE_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_SUFFIXES))
        return None, f"Unsupported image type. Use one of: {allowed}."

    raw_bytes = upload.read()
    if not raw_bytes:
        return None, "The uploaded file is empty."
    if len(raw_bytes) > MAX_IMAGE_BYTES:
        return None, "The uploaded image is too large. Keep inspection images below 8 MB."
    return raw_bytes, None


def build_preview_data_uri(raw_bytes: bytes, mimetype: str | None) -> str:
    safe_mimetype = mimetype if mimetype and mimetype.startswith("image/") else "image/png"
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{safe_mimetype};base64,{encoded}"


def derive_wind_features(values: dict[str, float]) -> dict[str, float]:
    windspeed_safe = max(values["windspeed"], 0.5)
    values["temp_gap"] = values["wheel hub temperature"] - values["ambient Temperature"]
    values["tower_gap"] = values["Tower bottom ambient temperature"] - values["ambient Temperature"]
    values["power_to_speed"] = values["power"] / windspeed_safe
    values["rpm_to_speed"] = values["rtr_rpm"] / windspeed_safe
    return values


def build_default_wind_form(wind_artifact: dict | None) -> dict[str, float]:
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


def wind_risk_label(score: float, thresholds: dict) -> tuple[str, str]:
    if score <= thresholds["high"]:
        return "High maintenance risk", "critical"
    if score <= thresholds["elevated"]:
        return "Elevated maintenance risk", "warn"
    if score <= thresholds["watch"]:
        return "Watch condition", "watch"
    return "Stable operating band", "ok"


def wind_actions(level: str) -> list[str]:
    if level == "critical":
        return [
            "Schedule a detailed gearbox and bearing inspection immediately.",
            "Review recent temperature spikes, pitch changes, and power drops.",
            "Keep a human engineer in the loop before maintenance decisions.",
        ]
    if level == "warn":
        return [
            "Plan an inspection in the next maintenance window.",
            "Compare this reading against recent turbine history.",
            "Watch temperature gap and rotor-speed stability closely.",
        ]
    if level == "watch":
        return [
            "Track the next few operating cycles for trend changes.",
            "Use this score as an AI triage signal, not a final diagnosis.",
            "Escalate if power loss or abnormal heating continues.",
        ]
    return [
        "Continue routine monitoring.",
        "Use AI assistance for periodic screening and early warning.",
        "Maintain regular inspection cadence from the report workflow.",
    ]


def resolve_label(label_map: dict, prediction: int) -> str:
    return label_map.get(str(prediction)) or label_map.get(prediction) or (
        "Defective asset image" if prediction == 1 else "Healthy asset image"
    )


def vision_badge(probability: float, threshold: float) -> str:
    if probability >= max(0.8, threshold + 0.2):
        return "critical"
    if probability >= threshold:
        return "warn"
    if probability >= max(0.35, threshold - 0.1):
        return "watch"
    return "ok"


def vision_actions(prediction: int, badge: str) -> list[str]:
    if prediction == 1:
        if badge == "critical":
            return [
                "Escalate the image to a specialist for manual review immediately.",
                "Compare the frame against neighboring inspections before planning maintenance.",
                "Capture additional images if angle, focus, or lighting quality is inconsistent.",
            ]
        return [
            "Queue this asset for human review in the next inspection cycle.",
            "Confirm the defect signal with a second image when possible.",
            "Treat the model output as triage support rather than a final diagnosis.",
        ]
    if badge == "watch":
        return [
            "This frame is near the review threshold, so a second image is recommended.",
            "Keep the image in the maintenance record for later comparison.",
            "Escalate if repeated frames stay close to the defect threshold.",
        ]
    return [
        "The current image does not show a strong defect signal.",
        "Retain the frame in the maintenance record for later comparison.",
        "Continue periodic inspection with a human in the final decision loop.",
    ]


def predict_vision_result(model, metadata: dict, image_bytes: bytes) -> dict:
    image_size = tuple(metadata.get("image_size", [64, 64]))
    color_mode = metadata.get("color_mode", "grayscale")
    threshold = float(metadata.get("threshold", 0.5))
    label_map = metadata.get("label_map", {})
    image_array = preprocess_image_bytes(image_bytes, image_size, color_mode)
    probability = float(model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0][0])
    prediction = int(probability >= threshold)
    badge = vision_badge(probability, threshold)
    return {
        "label": resolve_label(label_map, prediction),
        "fault_probability_pct": round(probability * 100, 2),
        "confidence_pct": round((probability if prediction == 1 else 1 - probability) * 100, 2),
        "badge": badge,
        "actions": vision_actions(prediction, badge),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    metrics = load_json(METRICS_PATH)
    wind_artifact = load_joblib(WIND_MODEL_PATH)
    vision_metadata = load_vision_metadata()
    vision_model = load_vision_model()

    error_message = None
    if tf is None and VISION_MODEL_PATH.exists():
        error_message = "TensorFlow is required to load the vision model in this app."
    elif VISION_MODEL_PATH.exists() and vision_model is None:
        error_message = "The vision model file exists but could not be loaded."

    models_ready = (
        wind_artifact is not None
        and bool(vision_metadata)
        and vision_model is not None
        and not error_message
    )
    wind_form = build_default_wind_form(wind_artifact)
    wind_result = None
    vision_result = None
    vision_preview = None

    if request.method == "POST" and models_ready:
        asset_type = request.form.get("asset_type", "")

        if asset_type == "wind":
            try:
                for key, _label in WIND_FORM_FIELDS:
                    wind_form[key] = float(request.form.get(key, wind_form[key]))
                wind_form = derive_wind_features(wind_form)
                sample = pd.DataFrame([wind_form])[wind_artifact["features"]]
                score = float(wind_artifact["pipeline"].decision_function(sample)[0])
                label, badge = wind_risk_label(score, wind_artifact["thresholds"])
                wind_result = {
                    "score": round(score, 4),
                    "label": label,
                    "badge": badge,
                    "actions": wind_actions(badge),
                }
            except Exception as exc:
                error_message = f"Wind prediction failed: {exc}"

        if asset_type == "vision":
            upload = request.files.get("vision_image")
            raw_bytes, upload_error = read_upload_bytes(upload)
            if upload_error:
                error_message = upload_error
            else:
                try:
                    vision_result = predict_vision_result(vision_model, vision_metadata, raw_bytes)
                    vision_preview = build_preview_data_uri(raw_bytes, upload.mimetype)
                except Exception as exc:
                    error_message = f"Vision prediction failed: {exc}"

    return render_template(
        "index.html",
        metrics=metrics,
        models_ready=models_ready,
        wind_fields=WIND_FORM_FIELDS,
        wind_form=wind_form,
        wind_result=wind_result,
        vision_result=vision_result,
        vision_preview=vision_preview,
        vision_profile=vision_metadata.get("profile", {}),
        error_message=error_message,
        plot_groups=metrics.get("plots", {}),
    )


if __name__ == "__main__":
    app.run(debug=True)
