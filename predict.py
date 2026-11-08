"""Load and validate the QuickBite late-order model for inference."""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "late_orders.joblib"
INFO_PATH = ROOT / "models" / "model_info.json"

# Load both artifacts once when this module is imported.
with INFO_PATH.open(encoding="utf-8") as file:
    MODEL_INFO = json.load(file)

MODEL = joblib.load(MODEL_PATH)
THRESHOLD = float(MODEL_INFO["threshold"])

NUMERIC_LIMITS = {
    "distance_km": (0.1, 30.0),
    "items": (1, 20),
    "hour": (10, 22),
    "courier_months": (0, 120),
    "prep_minutes": (1, 60),
}
INTEGER_FIELDS = {"items", "hour"}
CHOICES = {
    "weather": {"clear", "rain", "snow"},
    "day_type": {"weekday", "weekend"},
    "vehicle": {"bike", "scooter", "car"},
}


def validate_order(order: dict[str, Any]) -> list[str]:
    """Validate an order and return warnings for accepted missing values."""
    if not isinstance(order, dict):
        raise ValueError("order: expected a dictionary of field values")

    expected = MODEL_INFO["numeric_features"] + MODEL_INFO["categorical_features"]
    warnings: list[str] = []

    for field in expected:
        if field not in order:
            raise ValueError(f"{field}: this field is required")

    for field, (minimum, maximum) in NUMERIC_LIMITS.items():
        value = order[field]
        if field == "prep_minutes" and value is None:
            warnings.append(
                "prep_minutes is empty; the model pipeline will fill it "
                "with the training median."
            )
            continue
        if value is None or isinstance(value, bool):
            raise ValueError(f"{field}: enter a number from {minimum} to {maximum}")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field}: enter a number from {minimum} to {maximum}"
            ) from exc
        if not minimum <= number <= maximum:
            raise ValueError(f"{field}: value must be from {minimum} to {maximum}")
        if field in INTEGER_FIELDS and not number.is_integer():
            raise ValueError(
                f"{field}: enter a whole number from {minimum} to {maximum}"
            )

    for field, allowed in CHOICES.items():
        value = order[field]
        if not isinstance(value, str) or value not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(f"{field}: choose one of {options}")

    return warnings


def should_offer_voucher(probability: float, threshold: float | None = None) -> bool:
    """Apply the saved cost-based threshold; equality counts as a flag."""
    cutoff = THRESHOLD if threshold is None else float(threshold)
    if not 0 <= probability <= 1:
        raise ValueError("probability: must be between 0 and 1")
    if not 0 <= cutoff <= 1:
        raise ValueError("threshold: must be between 0 and 1")
    return probability >= cutoff


def predict_order(order: dict[str, Any]) -> dict[str, Any]:
    """Validate and score one order, returning its probability and action."""
    warnings = validate_order(order)
    columns = MODEL_INFO["numeric_features"] + MODEL_INFO["categorical_features"]
    row = {column: order[column] for column in columns}
    frame = pd.DataFrame([row])
    probability = float(MODEL.predict_proba(frame)[0, 1])
    offer_voucher = should_offer_voucher(probability)

    return {
        "probability": probability,
        "probability_on_time": 1 - probability,
        "offer_voucher": offer_voucher,
        "threshold": THRESHOLD,
        "warnings": warnings,
        "message": (
            "Offer the $4 voucher now."
            if offer_voucher
            else "Do not offer a voucher based on this prediction."
        ),
    }
