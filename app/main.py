"""
FastAPI Serving Layer for Customer Churn Prediction.
Loads the serialized Scikit-Learn pipeline and serves real-time REST predictions
with graceful error handling and lifespan state management.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from app.schemas import ChurnPredictionResponse, CustomerInput, HealthResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn.api")

# Paths
MODEL_PATH: Path = Path("models/pipeline.joblib")
FALLBACK_MODEL_PATH: Path = Path("models/best_model.pkl")
THRESHOLD_CONFIG_PATH: Path = Path("models/optimal_threshold.json")

# Global application state
ml_resources: Dict[str, Any] = {
    "pipeline": None,
    "threshold": 0.50,
    "model_loaded": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load ML artifacts at application startup
    and perform cleanup on shutdown.
    """
    logger.info("Initializing FastAPI application and loading ML pipeline...")

    # Load Model Pipeline
    chosen_path: Optional[Path] = None
    if MODEL_PATH.exists():
        chosen_path = MODEL_PATH
    elif FALLBACK_MODEL_PATH.exists():
        chosen_path = FALLBACK_MODEL_PATH

    if chosen_path is not None:
        try:
            ml_resources["pipeline"] = joblib.load(chosen_path)
            ml_resources["model_loaded"] = True
            logger.info("ML pipeline successfully loaded from %s", chosen_path)
        except Exception as exc:
            logger.error("Failed to deserialize model pipeline from %s: %s", chosen_path, exc)
            ml_resources["model_loaded"] = False
    else:
        logger.warning("No model artifact found at %s or %s", MODEL_PATH, FALLBACK_MODEL_PATH)
        ml_resources["model_loaded"] = False

    # Load Optimal Decision Threshold if available
    if THRESHOLD_CONFIG_PATH.exists():
        try:
            with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                ml_resources["threshold"] = float(config.get("optimal_threshold", 0.50))
            logger.info("Loaded financially optimized threshold: %.2f", ml_resources["threshold"])
        except Exception as exc:
            logger.warning("Could not parse %s, defaulting to 0.50: %s", THRESHOLD_CONFIG_PATH, exc)
            ml_resources["threshold"] = 0.50

    yield

    logger.info("Shutting down API and releasing ML resources.")
    ml_resources.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST service for real-time customer attrition forecasting and retention decisioning.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Liveness and readiness health probe."""
    is_ready = ml_resources.get("model_loaded", False)
    status_text = "healthy" if is_ready else "degraded"
    return HealthResponse(
        status=status_text,
        model_loaded=is_ready,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Predict customer churn probability and risk tier",
)
async def predict_churn(payload: CustomerInput):
    """
    Accept validated customer subscription profile, run model inference,
    and return predicted probability, binary decision, and risk tier.
    """
    pipeline = ml_resources.get("pipeline")
    if not ml_resources.get("model_loaded") or pipeline is None:
        logger.error("Inference request rejected: ML model pipeline is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is currently unavailable. Service is not ready to process predictions.",
        )

    try:
        # Convert Pydantic payload to single-row DataFrame
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Execute inference through complete ColumnTransformer + Classifier pipeline
        prob_churn = float(pipeline.predict_proba(input_df)[0, 1])
        threshold = float(ml_resources.get("threshold", 0.50))
        is_churn = prob_churn >= threshold

        # Assign risk tier
        if prob_churn < 0.30:
            risk_tier = "Low"
            rec = "Customer exhibits high stability. Maintain standard engagement."
        elif prob_churn < 0.60:
            risk_tier = "Medium"
            rec = "Moderate churn risk detected. Consider offering customer loyalty rewards or checking support satisfaction."
        else:
            risk_tier = "High"
            rec = "High churn risk! Immediate intervention required: deploy annual contract discount and dedicated support bundle."

        return ChurnPredictionResponse(
            churn_probability=round(prob_churn, 4),
            is_churn=is_churn,
            risk_tier=risk_tier,
            decision_threshold=threshold,
            recommendation=rec,
        )

    except Exception as err:
        logger.exception("Inference execution failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during model prediction: {str(err)}",
        )
