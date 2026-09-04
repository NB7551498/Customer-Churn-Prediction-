"""
Pydantic Schemas for the Customer Churn Serving API.
Enforces strict input validation, realistic field boundaries, and typed responses.
"""

from typing import Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """
    Incoming subscriber profile payload with strict validation and boundaries.
    """
    gender: Literal["Female", "Male"] = Field(..., description="Customer gender")
    SeniorCitizen: Literal["0", "1"] = Field(..., description="Whether subscriber is a senior citizen ('0' or '1')")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether subscriber has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether subscriber has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Tenure in months (0 to 120)")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Subscribed to phone service")
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(..., description="Multiple telephone lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet connection type")
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(..., description="Online cloud backup add-on")
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(..., description="Device protection plan")
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(..., description="Dedicated tech support")
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming TV package")
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(..., description="Streaming movies package")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contractual agreement terms")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Enrolled in paperless e-billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Billing payment gateway/method")
    MonthlyCharges: float = Field(..., ge=0.0, le=500.0, description="Monthly recurring charge in USD (0 to 500)")
    TotalCharges: float = Field(..., ge=0.0, le=50000.0, description="Cumulative historical charges in USD")

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


class ChurnPredictionResponse(BaseModel):
    """
    Response schema returning churn probability, threshold decision, and risk tier.
    """
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churning")
    is_churn: bool = Field(..., description="True if probability >= optimized business threshold")
    risk_tier: Literal["Low", "Medium", "High"] = Field(..., description="Qualitative churn risk level")
    decision_threshold: float = Field(..., description="Classification cutoff applied")
    recommendation: str = Field(..., description="Actionable retention intervention strategy")


class HealthResponse(BaseModel):
    """API health status schema."""
    status: str = Field(..., description="Application operational status")
    model_loaded: bool = Field(..., description="Whether ML pipeline is active in memory")
    version: str = Field(default="1.0.0", description="API version")
