"""
Unit and Integration Tests for FastAPI Customer Churn Service.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, ml_resources


@pytest.fixture
def client():
    """TestClient fixture with lifespan context enabled."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    """Verify health probe returns status 200 and indicates model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_churn_high_risk_profile(client):
    """Test prediction for high-risk customer profile."""
    payload = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50,
        "TotalCharges": 179.00,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["is_churn"], bool)
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert "recommendation" in data


def test_predict_churn_low_risk_profile(client):
    """Test prediction for low-risk loyal subscriber."""
    payload = {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.00,
        "TotalCharges": 3575.00,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] < 0.50
    assert data["risk_tier"] in ["Low", "Medium"]


def test_predict_invalid_boundary_tenure(client):
    """Validation test: tenure exceeding boundary (120) should trigger HTTP 422."""
    invalid_payload = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 150,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.00,
        "TotalCharges": 140.00,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_model_unavailable_returns_503(client):
    """Verify graceful 503 error handling when model artifact is unavailable."""
    original_state = ml_resources["model_loaded"]
    try:
        ml_resources["model_loaded"] = False
        sample = {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "No",
            "Dependents": "No",
            "tenure": 10,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 45.0,
            "TotalCharges": 450.0,
        }
        res = client.post("/predict", json=sample)
        assert res.status_code == 503
        assert "unavailable" in res.json()["detail"].lower()
    finally:
        ml_resources["model_loaded"] = original_state
