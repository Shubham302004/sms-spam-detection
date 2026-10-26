"""
API tests for backend/app.py.

Run from the repository root:

    pip install -r requirements-dev.txt
    pytest
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import app as app_module  # noqa: E402  (needs sys.path set up first)

SPAM = "Congratulations! You have won a FREE prize. Call now to claim"
HAM = "hey, are we still meeting for lunch tomorrow?"


@pytest.fixture()
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def predict(client, payload=None, raw=None, content_type="application/json", headers=None):
    """POST to /predict, either with a dict payload or a raw body string."""
    if raw is not None:
        return client.post("/predict", data=raw, content_type=content_type, headers=headers)
    return client.post("/predict", json=payload if payload is not None else {}, headers=headers)


# --------------------------------------------------------------------------
# Startup & Health
# --------------------------------------------------------------------------
def test_model_artifacts_are_loaded():
    assert app_module.model is not None, "backend/spam_model.pkl failed to load"
    assert app_module.vectorizer is not None, "backend/vectorizer.pkl failed to load"


def test_health_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ready"
    assert body["service"] == "SMS Spam Detector API"
    assert body["model"] == "loaded"
    assert "database" in body


# --------------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        (SPAM, "Spam"),
        (HAM, "Ham"),
        ("WINNER!! Text WIN to 80085 to claim your prize", "Spam"),
        ("Can you pick up milk on your way home?", "Ham"),
    ],
)
def test_predict_returns_label_and_confidence(client, message, expected):
    res = predict(client, {"message": message})
    assert res.status_code == 200
    body = res.get_json()
    assert body["prediction"] == expected
    assert body["message"] == message
    assert 0 < body["confidence"] <= 100


def test_message_is_trimmed(client):
    res = predict(client, {"message": f"   {HAM}   "})
    assert res.status_code == 200
    assert res.get_json()["message"] == HAM


def test_cors_header_present(client):
    res = predict(client, {"message": HAM}, headers={"Origin": "http://localhost:3000"})
    assert res.headers.get("Access-Control-Allow-Origin") is not None


# --------------------------------------------------------------------------
# UI & Endpoints
# --------------------------------------------------------------------------
def test_ui_endpoint(client):
    res = client.get("/ui")
    assert res.status_code == 200
    assert "text/html" in res.content_type
    assert b"SMS Spam Detector" in res.data


def test_predictions_endpoint_when_db_disabled(client, monkeypatch):
    monkeypatch.setattr(app_module, "DB_HOST", None)
    res = client.get("/predictions")
    assert res.status_code == 200
    body = res.get_json()
    assert "predictions" in body
    assert body["status"] == "disabled"


def test_predictions_limit_query_param_sanitized(client, monkeypatch):
    monkeypatch.setattr(app_module, "DB_HOST", None)
    for invalid_limit in ["-5", "0", "abc", "9999"]:
        res = client.get(f"/predictions?limit={invalid_limit}")
        assert res.status_code == 200


# --------------------------------------------------------------------------
# Validation — every one of these must be a 400, never a 500
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload,description",
    [
        ({}, "missing 'message' key"),
        ({"message": ""}, "empty message"),
        ({"message": "     "}, "whitespace-only message"),
        ({"message": None}, "null message"),
        ({"message": 12345}, "numeric message"),
        ({"message": ["not", "a", "string"]}, "array message"),
        ({"message": "a" * 1001}, "message over 1000 characters"),
    ],
)
def test_invalid_payloads_return_400(client, payload, description):
    res = predict(client, payload)
    assert res.status_code == 400, f"{description}: expected 400, got {res.status_code}"
    assert "error" in res.get_json(), f"{description}: response must be JSON with an error key"


@pytest.mark.parametrize(
    "raw,description",
    [
        ("not json at all", "malformed JSON"),
        ('["not", "an", "object"]', "JSON array body"),
        ('"just a string"', "JSON string body"),
        ("42", "JSON number body"),
        ("null", "JSON null body"),
        ("", "empty body"),
    ],
)
def test_non_object_bodies_return_400_not_500(client, raw, description):
    """Regression: a JSON array body used to raise AttributeError -> HTTP 500."""
    res = predict(client, raw=raw)
    assert res.status_code == 400, f"{description}: expected 400, got {res.status_code}"
    assert res.content_type.startswith("application/json"), f"{description}: must answer JSON, not an HTML error page"
    assert "error" in res.get_json()


def test_message_length_boundary(client):
    assert predict(client, {"message": "a" * 1000}).status_code == 200  # allowed
    assert predict(client, {"message": "a" * 1001}).status_code == 400  # rejected


# --------------------------------------------------------------------------
# Database resilience
# --------------------------------------------------------------------------
def test_log_prediction_handles_exceptions_gracefully(monkeypatch):
    """Database connection drops must never raise an unhandled exception."""
    monkeypatch.setattr(app_module, "DB_HOST", "invalid-host")

    class FailingPsycopg2:
        @staticmethod
        def connect(*args, **kwargs):
            raise RuntimeError("Database connection refused")

    monkeypatch.setattr(app_module, "psycopg2", FailingPsycopg2)
    # Must run and complete without raising
    app_module.log_prediction("Test message", "Spam", 99.0)


def test_predict_succeeds_even_when_db_fails(client, monkeypatch):
    """Predict API must return 200 and ML result even if database is unreachable."""
    monkeypatch.setattr(app_module, "DB_HOST", "invalid-host")

    class FailingPsycopg2:
        @staticmethod
        def connect(*args, **kwargs):
            raise Exception("Database connection refused")

    monkeypatch.setattr(app_module, "psycopg2", FailingPsycopg2)
    res = predict(client, {"message": HAM})
    assert res.status_code == 200
    assert res.get_json()["prediction"] == "Ham"


# --------------------------------------------------------------------------
# Degraded mode and routing
# --------------------------------------------------------------------------
def test_returns_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    res = predict(client, {"message": HAM})
    assert res.status_code == 503
    assert "not loaded" in res.get_json()["error"]


def test_returns_503_when_vectorizer_missing(client, monkeypatch):
    monkeypatch.setattr(app_module, "vectorizer", None)
    res = predict(client, {"message": HAM})
    assert res.status_code == 503


def test_unknown_route_returns_404(client):
    res = client.get("/does-not-exist")
    assert res.status_code == 404
