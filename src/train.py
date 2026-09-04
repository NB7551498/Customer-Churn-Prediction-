"""
Production-grade Training Pipeline for Customer Churn Prediction.
Encapsulates data loading, cleaning, stratified splitting, preprocessing,
model fitting, and artifact serialization with strict type hints and logging.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Graceful import of XGBClassifier with fallback to HistGradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    DEFAULT_CLASSIFIER = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    CLASSIFIER_NAME = "XGBClassifier"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    DEFAULT_CLASSIFIER = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
    )
    CLASSIFIER_NAME = "HistGradientBoostingClassifier"


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("churn.train")

# Feature declarations
NUMERICAL_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN: str = "Churn"
DEFAULT_DATA_PATH: Path = Path("data/customer_churn.csv")
DEFAULT_MODEL_DIR: Path = Path("models")


def load_and_preprocess_raw_data(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset from CSV, clean TotalCharges anomalies, drop arbitrary identifiers,
    and separate features from the binary target.

    Args:
        data_path: Path to raw customer churn CSV.

    Returns:
        Tuple of (X: feature matrix DataFrame, y: binary target Series).
    """
    logger.info("Loading raw dataset from %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path.resolve()}")

    df: pd.DataFrame = pd.read_csv(data_path)
    logger.info("Raw dataset loaded: %d rows, %d columns", len(df), len(df.columns))

    # TotalCharges contains whitespace strings for tenure=0 records
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)

    # Drop non-predictive identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Cast SeniorCitizen to string category
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    y: pd.Series = (df[TARGET_COLUMN] == "Yes").astype(int)
    X: pd.DataFrame = df.drop(columns=[TARGET_COLUMN])

    churn_rate: float = float(y.mean() * 100)
    logger.info("Features extracted: %d columns | Churn prevalence: %.2f%%", X.shape[1], churn_rate)
    return X, y


def build_pipeline() -> Pipeline:
    """
    Build leak-proof Scikit-Learn Pipeline combining ColumnTransformer
    (StandardScaler + OneHotEncoder) with the gradient boosted classifier.

    Returns:
        Assembled Pipeline ready for fitting.
    """
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DEFAULT_CLASSIFIER),
        ]
    )
    logger.info("Assembled Pipeline with ColumnTransformer and %s", CLASSIFIER_NAME)
    return pipeline


def run_training_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Execute full training workflow: load, split, 5-fold CV, fit, and serialize.

    Args:
        data_path: Path to customer CSV.
        model_dir: Target directory to save serialized pipeline.
        test_size: Ratio for train-test split (default 0.20).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (fitted Pipeline, cross-validation metrics dict).
    """
    X, y = load_and_preprocess_raw_data(data_path)

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info("Split completed: %d train samples, %d test samples", len(X_train), len(X_test))

    # Cache split partitions for evaluate.py
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(processed_dir / "X_test.parquet", index=False)
    y_test.to_frame("Churn").to_parquet(processed_dir / "y_test.parquet", index=False)

    pipeline: Pipeline = build_pipeline()

    # 5-Fold Stratified Cross-Validation
    logger.info("Running 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring: Dict[str, str] = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    metrics: Dict[str, float] = {
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_scores["test_accuracy"])),
        "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
        "cv_recall_std": float(np.std(cv_scores["test_recall"])),
        "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
        "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
    }

    logger.info(
        "CV Results: ROC-AUC: %.4f (+/- %.4f) | Recall: %.4f (+/- %.4f) | F1: %.4f",
        metrics["cv_roc_auc_mean"],
        metrics["cv_roc_auc_std"],
        metrics["cv_recall_mean"],
        metrics["cv_recall_std"],
        metrics["cv_f1_mean"],
    )

    # Fit final pipeline on complete training set
    logger.info("Fitting final pipeline on all %d training samples...", len(X_train))
    pipeline.fit(X_train, y_train)

    # Serialize single pipeline artifact
    model_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path: Path = model_dir / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    # Also save as best_model.pkl for compatibility
    joblib.dump(pipeline, model_dir / "best_model.pkl")

    logger.info("Model pipeline successfully serialized to %s", pipeline_path.resolve())
    return pipeline, metrics


if __name__ == "__main__":
    run_training_pipeline()
