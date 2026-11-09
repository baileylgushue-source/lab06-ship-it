"""Shared fixtures for the prediction tests."""

import pytest


@pytest.fixture
def valid_order():
    """A normal order with every field filled in and inside its limits."""
    return {
        "distance_km": 4.0,
        "items": 3,
        "hour": 15,
        "courier_months": 24,
        "prep_minutes": 12.0,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "scooter",
    }


@pytest.fixture
def safe_order():
    """An order that should almost never be late: short, quiet and experienced."""
    return {
        "distance_km": 0.5,
        "items": 1,
        "hour": 15,
        "courier_months": 59,
        "prep_minutes": 5.0,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "car",
    }


@pytest.fixture
def risky_order():
    """An order that should often be late: far, big, rush hour, snow, new courier."""
    return {
        "distance_km": 14.0,
        "items": 8,
        "hour": 19,
        "courier_months": 0,
        "prep_minutes": 35.0,
        "weather": "snow",
        "day_type": "weekend",
        "vehicle": "bike",
    }
