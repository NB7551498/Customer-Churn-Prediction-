"""
ChurnGuard AI — Pydantic Validation & Serialization Schemas.
Defines strict input boundaries, security token schemas, SHAP factor attributions,
segmentation metadata, and executive customer retention responses.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class LoginPayload(BaseModel):
    """User credentials for JWT authentication."""
    username: str = Field(..., json_schema_extra={"example": "admin@churnguard.ai"})
    password: str = Field(..., json_schema_extra={"example": "AdminPass123!"})


class TokenResponse(BaseModel):
    """JWT authorization token return payload."""
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class CustomerInput(BaseModel):
    """
    Subscribed customer profile with realistic boundary validation.
    """
    gender: Literal["Female", "Male"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal["0", "1"] = Field(..., description="Senior citizen indicator ('0' or '1')")
    Partner: Literal["Yes", "No"] = Field(..., description="Has partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months active as subscriber (0-120)")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Phone service subscription")
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(..., description="Multiple phone lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet connection technology")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., description="Online security package")
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(..., description="Cloud backup add-on")
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(..., description="Device warranty plan")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., description="Dedicated technical support")
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming television tier")
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming movies tier")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract duration commitment")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Enrolled in e-billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Billing settlement channel")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly recurring rate in USD")
    TotalCharges: float = Field(..., ge=0.0, le=50000.0, description="Cumulative lifetime spend in USD")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
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
                "TotalCharges": 268.50,
            }
        }
    }


class SHAPDriver(BaseModel):
    """Attribution item explaining prediction impact."""
    factor: str
    impact_score: float
    relative_pct: str


class CustomerSegment(BaseModel):
    """Customer persona identification."""
    segment_id: int
    segment_name: str
    description: str
    risk_profile: str
    icon: str


class RetentionPlaybook(BaseModel):
    """AI-powered intervention recommendations."""
    primary_issue: str
    segment: str
    recommended_actions: List[str]
    projected_risk_reduction: str
    estimated_arr_at_risk: str
    projected_clv_preserved: str
    llm_outreach_draft: Dict[str, str]


class ChurnPredictionResponse(BaseModel):
    """Comprehensive single-customer risk scoring response."""
    customer_id: Optional[str] = "C-ANON"
    churn_probability: float = Field(..., description="Calibrated churn probability [0.0 - 1.0]")
    is_churn: bool = Field(..., description="Threshold-based binary prediction")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "Low", "Medium", "High"] = Field(..., description="Categorical risk tier")
    recommendation: Optional[str] = Field(None, description="Primary retention strategy")
    decision_threshold: float = Field(..., description="Financial decision cutoff applied")
    segmentation: CustomerSegment
    top_churn_drivers: List[SHAPDriver]
    top_retention_factors: List[SHAPDriver]
    retention_playbook: RetentionPlaybook


class BatchPredictionSummary(BaseModel):
    """Summary metrics of a bulk CSV scoring run."""
    total_customers_evaluated: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    portfolio_churn_rate_pct: float
    annual_revenue_at_risk_usd: float


class AnalyticsOverview(BaseModel):
    """Portfolio analytics overview."""
    total_customers: int
    at_risk_count: int
    avg_churn_risk_pct: float
    total_mrr_at_risk_usd: float
    segment_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]


class HealthResponse(BaseModel):
    """Service status health response."""
    status: str
    model_loaded: bool
    segmenter_loaded: bool
    explainer_loaded: bool
    version: str = "2.0.0"
