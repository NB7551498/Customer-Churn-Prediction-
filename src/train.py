"""
ChurnGuard AI — Production Training & Multi-Model Benchmarking Pipeline.
Implements domain feature engineering, leak-proof ColumnTransformer,
5-fold stratified cross-validation across 5 algorithms (Logistic Regression,
Random Forest, LightGBM, HistGradientBoosting, GradientBoosting), PR-AUC
evaluation, and single-artifact pipeline serialization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Try importing LightGBM and XGBoost, fallback to GradientBoosting if needed
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.mlops import tracker

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("churn.train")

# Base feature groups
RAW_NUMERICAL: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED_NUMERICAL: List[str] = [
    "tenure_monthly_ratio",
    "avg_monthly_spend",
    "support_protection_index",
]
ALL_NUMERICAL: List[str] = RAW_NUMERICAL + ENGINEERED_NUMERICAL

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
    "high_risk_combo",
]

TARGET_COLUMN: str = "Churn"
DEFAULT_DATA_PATH: Path = Path("data/customer_churn.csv")
DEFAULT_MODEL_DIR: Path = Path("models")
REPORTS_DIR: Path = Path("reports")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-driven feature engineering to uncover non-linear churn drivers:
    - tenure_monthly_ratio: interaction between tenure and monthly billing
    - avg_monthly_spend: TotalCharges normalized by tenure
    - support_protection_index: count of protective add-on services
    - high_risk_combo: indicator for Month-to-month + Fiber Optic + Electronic check
    """
    df = df.copy()

    # Tenure x Monthly Charges interaction
    df["tenure_monthly_ratio"] = df["tenure"] * df["MonthlyCharges"]

    # Average monthly spend across historical tenure
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1.0)

    # Support & Security Protection Index (0 to 4)
    security_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["support_protection_index"] = 0
    for col in security_services:
        if col in df.columns:
            df["support_protection_index"] += (df[col] == "Yes").astype(int)

    # High-Risk Vulnerability Flag: Month-to-month contract with Electronic Check
    is_m2m = df["Contract"] == "Month-to-month"
    is_echeck = df["PaymentMethod"] == "Electronic check"
    is_fiber = df["InternetService"] == "Fiber optic"
    df["high_risk_combo"] = np.where(is_m2m & (is_echeck | is_fiber), "HighRiskCombo", "Standard")

    return df


def load_and_preprocess_raw_data(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw CSV, clean anomalies, engineer domain features, and separate target.
    """
    logger.info("Loading raw dataset from %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path.resolve()}")

    df: pd.DataFrame = pd.read_csv(data_path)

    # Clean whitespace strings in TotalCharges (tenure == 0)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)

    # Drop non-predictive customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Cast SeniorCitizen to string
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Apply Feature Engineering
    df = engineer_features(df)

    y: pd.Series = (df[TARGET_COLUMN] == "Yes").astype(int)
    X: pd.DataFrame = df.drop(columns=[TARGET_COLUMN])

    logger.info("Dataset prepared: %d rows, %d features | Churn rate: %.2f%%", X.shape[0], X.shape[1], y.mean() * 100)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build leak-proof ColumnTransformer for numeric scaling and one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERICAL),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier: object = None) -> Pipeline:
    """Build a complete ML pipeline containing ColumnTransformer preprocessor and classifier."""
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
        )
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def get_candidate_models() -> Dict[str, object]:
    """Instantiate the 5 candidate classification algorithms."""
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.1,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
        ),
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=2.7,
            eval_metric="logloss",
            random_state=42,
        )

    return models


def evaluate_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """
    Perform 5-Fold Stratified Cross-Validation across all candidate models,
    computing Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC.
    """
    logger.info("Executing 5-Fold Stratified CV across candidate models...")
    models = get_candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results_list: List[Dict[str, object]] = []
    best_score: float = -1.0
    best_model_name: str = ""

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        accs, precs, recs, f1s, aucs, praucs = [], [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            y_prob = pipeline.predict_proba(X_val)[:, 1]

            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0))
            recs.append(recall_score(y_val, y_pred))
            f1s.append(f1_score(y_val, y_pred))
            aucs.append(roc_auc_score(y_val, y_prob))
            praucs.append(average_precision_score(y_val, y_prob))

        mean_auc = float(np.mean(aucs))
        mean_recall = float(np.mean(recs))
        mean_f1 = float(np.mean(f1s))

        # Balanced ranking criterion: 0.5 * ROC-AUC + 0.3 * F1 + 0.2 * PR-AUC
        composite_score = 0.5 * mean_auc + 0.3 * mean_f1 + 0.2 * float(np.mean(praucs))

        if composite_score > best_score:
            best_score = composite_score
            best_model_name = name

        results_list.append({
            "Model": name,
            "Accuracy": round(float(np.mean(accs)), 4),
            "Precision": round(float(np.mean(precs)), 4),
            "Recall": round(mean_recall, 4),
            "F1-Score": round(mean_f1, 4),
            "ROC-AUC": round(mean_auc, 4),
            "PR-AUC": round(float(np.mean(praucs)), 4),
            "ROC-AUC Std": round(float(np.std(aucs)), 4),
        })

        logger.info(
            "[%s] ROC-AUC: %.4f | Recall: %.4f | F1: %.4f | PR-AUC: %.4f",
            name, mean_auc, mean_recall, mean_f1, float(np.mean(praucs))
        )

        # Log fold evaluation to MLOps tracker
        tracker.log_run(
            run_name=f"CV_{name.replace(' ', '_')}",
            model_name=name,
            params={"model_type": name, "cv_folds": 5},
            metrics={
                "accuracy": round(float(np.mean(accs)), 4),
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(mean_recall, 4),
                "f1_score": round(mean_f1, 4),
                "roc_auc": round(mean_auc, 4),
                "pr_auc": round(float(np.mean(praucs)), 4),
            },
        )

    df_results = pd.DataFrame(results_list).sort_values(by="ROC-AUC", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(REPORTS_DIR / "model_benchmark_comparison.csv", index=False)
    logger.info("Saved model benchmark scorecard to %s", REPORTS_DIR / "model_benchmark_comparison.csv")
    logger.info("Selected Top Model: %s (Composite Score: %.4f)", best_model_name, best_score)
    return df_results, best_model_name


def run_training_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Execute end-to-end training and serialization workflow."""
    X, y = load_and_preprocess_raw_data(data_path)

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Cache split data
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(processed_dir / "X_test.parquet", index=False)
    y_test.to_frame("Churn").to_parquet(processed_dir / "y_test.parquet", index=False)
    X_train.to_parquet(processed_dir / "X_train.parquet", index=False)
    y_train.to_frame("Churn").to_parquet(processed_dir / "y_train.parquet", index=False)

    preprocessor = build_preprocessor()
    benchmark_df, best_name = evaluate_candidate_models(X_train, y_train, preprocessor)

    # Instantiate best model
    candidate_models = get_candidate_models()
    best_classifier = candidate_models[best_name]

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_classifier),
    ])

    logger.info("Fitting final production pipeline (%s) on complete training set (%d rows)...", best_name, len(X_train))
    final_pipeline.fit(X_train, y_train)

    # Fit and serialize all candidate pipelines for comparison
    all_models: Dict[str, Pipeline] = {}
    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        all_models[name] = pipe

    # Serialize artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_dir / "pipeline.joblib")
    joblib.dump(final_pipeline, model_dir / "best_model.pkl")
    joblib.dump(all_models, model_dir / "all_models.pkl")
    logger.info("Pipelines serialized to %s, %s, and %s", model_dir / "pipeline.joblib", model_dir / "best_model.pkl", model_dir / "all_models.pkl")

    # Record production release to MLOps tracker
    tracker.log_run(
        run_name=f"PROD_{best_name.replace(' ', '_')}",
        model_name=best_name,
        params={"model_type": best_name, "stage": "production"},
        metrics={"status": 1.0},
        artifacts=[model_dir / "pipeline.joblib", model_dir / "all_models.pkl"],
    )

    return final_pipeline, benchmark_df


if __name__ == "__main__":
    run_training_pipeline()
