"""
Unit and Integration Tests for FastAPI Customer Churn Service.
Covers: health, predict, explain, model-info, metrics, and input validation.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, ml_resources


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """TestClient fixture with lifespan context enabled."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_high_risk_payload():
    return {
        "gender": "Female", "SeniorCitizen": "0",
        "Partner": "No", "Dependents": "No", "tenure": 2,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.50, "TotalCharges": 179.00,
    }


@pytest.fixture
def valid_low_risk_payload():
    return {
        "gender": "Male", "SeniorCitizen": "0",
        "Partner": "Yes", "Dependents": "Yes", "tenure": 65,
        "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.00, "TotalCharges": 3575.00,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    """Health probe returns 200 with status and model_loaded keys."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "explainer_loaded" in data
    assert "version" in data


def test_health_status_is_string(client):
    """Status field must be a string (healthy/degraded)."""
    data = client.get("/health").json()
    assert isinstance(data["status"], str)
    assert data["status"] in ("healthy", "degraded")


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

def test_model_info_endpoint(client):
    """/model-info returns versioning metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "algorithm" in data
    assert "model_name" in data
    assert "features" in data
    assert isinstance(data["features"], int)


def test_model_info_version_format(client):
    """Version follows semver format."""
    data = client.get("/model-info").json()
    parts = data["version"].split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint(client):
    """/metrics returns list of model performance rows."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "best_model" in data
    assert len(data["models"]) > 0


def test_metrics_fields(client):
    """Each metrics row has required performance fields."""
    data = client.get("/metrics").json()
    for row in data["models"]:
        assert "accuracy" in row
        assert "precision" in row
        assert "recall" in row
        assert "f1_score" in row
        assert "roc_auc" in row
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["roc_auc"] <= 1.0


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_predict_churn_high_risk_profile(client, valid_high_risk_payload):
    """Prediction returns valid churn response for high-risk profile."""
    response = client.post("/predict", json=valid_high_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["is_churn"], bool)
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert "recommendation" in data
    assert "decision_threshold" in data


def test_predict_churn_low_risk_profile(client, valid_low_risk_payload):
    """Long-tenure, two-year-contract subscriber should be low/medium risk."""
    response = client.post("/predict", json=valid_low_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_probability"] < 0.60
    assert data["risk_tier"] in ["Low", "Medium"]


def test_predict_invalid_tenure_boundary(client, valid_high_risk_payload):
    """Tenure > 120 must trigger HTTP 422 (Pydantic validation)."""
    payload = {**valid_high_risk_payload, "tenure": 150}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_tenure(client, valid_high_risk_payload):
    """Negative tenure must trigger HTTP 422."""
    payload = {**valid_high_risk_payload, "tenure": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_contract_enum(client, valid_high_risk_payload):
    """Invalid enum value for Contract triggers HTTP 422."""
    payload = {**valid_high_risk_payload, "Contract": "Weekly"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_unavailable_returns_503(client):
    """Graceful 503 when model is flagged as not loaded."""
    original_state = ml_resources["model_loaded"]
    try:
        ml_resources["model_loaded"] = False
        payload = {
            "gender": "Female", "SeniorCitizen": "0",
            "Partner": "No", "Dependents": "No", "tenure": 10,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 45.0, "TotalCharges": 450.0,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 503
        assert "unavailable" in res.json()["detail"].lower()
    finally:
        ml_resources["model_loaded"] = original_state


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

def test_explain_returns_feature_impacts(client, valid_high_risk_payload):
    """/explain returns prediction, probability, and explanation list."""
    response = client.post("/explain", json=valid_high_risk_payload)
    if response.status_code == 503:
        pytest.skip("SHAP explainer not available in this environment")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "explanation" in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0


def test_explain_explanation_structure(client, valid_high_risk_payload):
    """Each explanation item must have 'feature' (str) and 'impact' (float)."""
    response = client.post("/explain", json=valid_high_risk_payload)
    if response.status_code == 503:
        pytest.skip("SHAP explainer not available in this environment")
    assert response.status_code == 200
    for item in response.json()["explanation"]:
        assert "feature" in item
        assert "impact" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["impact"], float)


def test_explain_probability_in_range(client, valid_high_risk_payload):
    """Probability returned by /explain must be in [0, 1]."""
    response = client.post("/explain", json=valid_high_risk_payload)
    if response.status_code == 503:
        pytest.skip("SHAP explainer not available in this environment")
    assert response.status_code == 200
    prob = response.json()["probability"]
    assert 0.0 <= prob <= 1.0


def test_explain_invalid_input_rejected(client, valid_high_risk_payload):
    """/explain rejects invalid payloads with 422 — same validation as /predict."""
    payload = {**valid_high_risk_payload, "MonthlyCharges": -10.0}
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Security & Enterprise Endpoints
# ---------------------------------------------------------------------------

def test_login_valid_credentials(client):
    """POST /auth/login returns bearer token on valid credentials."""
    payload = {"username": "admin@churnguard.ai", "password": "AdminPass123!"}
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "admin"


def test_login_invalid_credentials(client):
    """POST /auth/login returns 401 on incorrect password."""
    payload = {"username": "admin@churnguard.ai", "password": "WrongPassword!"}
    response = client.post("/auth/login", json=payload)
    assert response.status_code == 401


def test_dashboard_endpoint(client):
    """GET /dashboard serves the executive dashboard."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_analytics_overview(client):
    """GET /analytics returns portfolio summary overview."""
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_customers" in data
    assert "at_risk_count" in data
    assert "segment_distribution" in data
    assert "risk_distribution" in data


def test_model_metrics_cv(client):
    """GET /model/metrics returns benchmark comparison."""
    response = client.get("/model/metrics")
    assert response.status_code == 200
    assert isinstance(response.json(), list) or isinstance(response.json(), dict)


def test_monitoring_status(client):
    """GET /monitoring returns live telemetry metrics."""
    response = client.get("/monitoring")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "requests_processed" in data
    assert "data_drift_status" in data


def test_predict_batch_csv(client):
    """POST /predict/batch scores uploaded CSV and returns prioritized CSV."""
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,"
        "MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,"
        "TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "CUST-1,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,85.50,171.00\n"
        "CUST-2,Male,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Bank transfer (automatic),55.00,3300.00\n"
    )
    files = {"file": ("test_customers.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    assert "text/csv" in response.headers.get("content-type", "")
    content = response.text
    assert "CustomerID" in content
    assert "ChurnProbability" in content
    assert "RiskTier" in content
    assert "CUST-1" in content


def test_customer_by_id_endpoint(client):
    """GET /customers/{customer_id} returns full customer retention analysis."""
    response = client.get("/customers/7590-VHVEG")
    assert response.status_code == 200
    data = response.json()
    assert "customer_id" in data
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert "segmentation" in data
    assert "top_churn_drivers" in data
    assert "retention_playbook" in data


def test_model_info_slash_route(client):
    """GET /model/info alias returns algorithm and version metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "algorithm" in data
    assert "optimal_threshold" in data


def test_monitoring_report_endpoint(client):
    """GET /monitoring/report returns latest drift evaluation."""
    response = client.get("/monitoring/report")
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert "retraining_recommended" in data

