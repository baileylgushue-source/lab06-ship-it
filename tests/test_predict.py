"""Tests for predict.py: probabilities, the threshold and input checks."""

import importlib
import json
from pathlib import Path

import joblib
import pytest

import predict

INFO_PATH = Path(__file__).resolve().parents[1] / "models" / "model_info.json"


# --- probabilities -----------------------------------------------------------


def test_probabilities_are_valid(valid_order):
    result = predict.predict_order(valid_order)

    assert 0 <= result["probability"] <= 1
    assert result["probability_on_time"] == pytest.approx(1 - result["probability"])
    assert isinstance(result["offer_voucher"], bool)
    assert result["warnings"] == []


def test_riskier_order_scores_higher(safe_order, risky_order):
    safe = predict.predict_order(safe_order)["probability"]
    risky = predict.predict_order(risky_order)["probability"]

    assert risky > safe


def test_risky_order_gets_voucher_and_safe_order_does_not(safe_order, risky_order):
    assert predict.predict_order(risky_order)["offer_voucher"] is True
    assert predict.predict_order(safe_order)["offer_voucher"] is False


# --- threshold -----------------------------------------------------------------


def test_threshold_comes_from_model_info(valid_order):
    saved = json.loads(INFO_PATH.read_text(encoding="utf-8"))

    assert predict.THRESHOLD == saved["threshold"]
    assert predict.predict_order(valid_order)["threshold"] == saved["threshold"]


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(-0.01, False), (0.0, True), (0.01, True)],
    ids=["just_below", "exactly_at", "just_above"],
)
def test_voucher_decision_at_threshold_boundary(offset, expected):
    probability = predict.THRESHOLD + offset

    assert predict.should_offer_voucher(probability) is expected


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_impossible_probability_is_refused(probability):
    with pytest.raises(ValueError, match="^probability"):
        predict.should_offer_voucher(probability)


# --- missing and invalid numbers ---------------------------------------------


def test_missing_prep_minutes_is_accepted_with_warning(valid_order):
    valid_order["prep_minutes"] = None

    result = predict.predict_order(valid_order)

    assert 0 <= result["probability"] <= 1
    assert len(result["warnings"]) == 1
    assert "prep_minutes" in result["warnings"][0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_km", 0.0),
        ("distance_km", 31),
        ("distance_km", None),
        ("items", 0),
        ("items", 21),
        ("items", 2.5),
        ("hour", 9),
        ("hour", 23),
        ("courier_months", -1),
        ("courier_months", 121),
        ("prep_minutes", 0),
        ("prep_minutes", 61),
        ("prep_minutes", "abc"),
        ("prep_minutes", True),
    ],
)
def test_invalid_number_is_refused_and_names_the_field(valid_order, field, value):
    valid_order[field] = value

    with pytest.raises(ValueError, match=f"^{field}"):
        predict.predict_order(valid_order)


def test_missing_field_is_refused(valid_order):
    del valid_order["hour"]

    with pytest.raises(ValueError, match="^hour"):
        predict.predict_order(valid_order)


# --- invalid categories --------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weather", "Snow"),
        ("weather", "hail"),
        ("weather", ""),
        ("day_type", "holiday"),
        ("vehicle", "truck"),
        ("vehicle", None),
    ],
)
def test_unseen_category_is_refused(valid_order, field, value):
    valid_order[field] = value

    with pytest.raises(ValueError, match=f"^{field}"):
        predict.predict_order(valid_order)


# --- the model is loaded once --------------------------------------------------


def test_model_is_loaded_only_once(monkeypatch, valid_order):
    calls = []
    real_load = joblib.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    importlib.reload(predict)  # import the module again, as the app would

    for _ in range(3):
        predict.predict_order(valid_order)

    assert len(calls) == 1
