"""
Model Training Module for Customer Churn Prediction
Implements Logistic Regression, Random Forest, 5-Fold Stratified Cross-Validation,
Hyperparameter Optimization (GridSearchCV), and Final Model Serialization.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from data_preprocessing import load_raw_data, clean_data, get_preprocessor, split_data


MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def get_base_models(preprocessor):
    """Define candidate model pipelines."""
    return {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42))
        ])
    }


def perform_cross_validation(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Perform 5-Fold Stratified Cross-Validation on training data.
    Evaluates Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_summary = []
    print("=" * 65)
    print("STEP 10: 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 65)

    for name, pipeline in models.items():
        print(f"Evaluating {name} with 5-Fold Stratified CV...")
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        row = {
            'Model': name,
            'CV_Accuracy_Mean': scores['test_accuracy'].mean(),
            'CV_Accuracy_Std': scores['test_accuracy'].std(),
            'CV_Precision_Mean': scores['test_precision'].mean(),
            'CV_Precision_Std': scores['test_precision'].std(),
            'CV_Recall_Mean': scores['test_recall'].mean(),
            'CV_Recall_Std': scores['test_recall'].std(),
            'CV_F1_Mean': scores['test_f1'].mean(),
            'CV_F1_Std': scores['test_f1'].std(),
            'CV_ROC_AUC_Mean': scores['test_roc_auc'].mean(),
            'CV_ROC_AUC_Std': scores['test_roc_auc'].std(),
        }
        cv_summary.append(row)
        print(f"  -> Accuracy : {row['CV_Accuracy_Mean']:.4f} (+/- {row['CV_Accuracy_Std']:.4f})")
        print(f"  -> Precision: {row['CV_Precision_Mean']:.4f} (+/- {row['CV_Precision_Std']:.4f})")
        print(f"  -> Recall   : {row['CV_Recall_Mean']:.4f} (+/- {row['CV_Recall_Std']:.4f})")
        print(f"  -> F1-Score : {row['CV_F1_Mean']:.4f} (+/- {row['CV_F1_Std']:.4f})")
        print(f"  -> ROC-AUC  : {row['CV_ROC_AUC_Mean']:.4f} (+/- {row['CV_ROC_AUC_Std']:.4f})")

    cv_df = pd.DataFrame(cv_summary)
    cv_df.to_csv(os.path.join(REPORTS_DIR, "cv_metrics_summary.csv"), index=False)
    return cv_df


def tune_random_forest(preprocessor, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tune Random Forest hyperparameters using GridSearchCV with Stratified 5-Fold CV.
    Target metric: ROC-AUC.
    """
    print("\n" + "=" * 65)
    print("STEP 12: HYPERPARAMETER TUNING (RANDOM FOREST)")
    print("=" * 65)

    rf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [6, 8, 12],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print("Optimal Parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_save():
    """Main training orchestration."""
    # 1. Load and split
    df_raw = load_raw_data("data/customer_churn.csv")
    df_clean, X, y = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    # Cache split datasets for evaluate.py
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_parquet("data/processed/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/X_test.parquet", index=False)
    y_train.to_frame('Churn').to_parquet("data/processed/y_train.parquet", index=False)
    y_test.to_frame('Churn').to_parquet("data/processed/y_test.parquet", index=False)

    preprocessor = get_preprocessor()
    base_models = get_base_models(preprocessor)

    # 2. Cross-Validation on Baseline Models
    cv_results = perform_cross_validation(base_models, X_train, y_train)

    # 3. Fit base models on training data and serialize for evaluation
    trained_base_models = {}
    for name, pipe in base_models.items():
        pipe.fit(X_train, y_train)
        trained_base_models[name] = pipe

    # 4. Hyperparameter Tuning
    best_model, best_params, best_cv_auc = tune_random_forest(preprocessor, X_train, y_train)

    # 5. Fit best model on complete training data (Pipeline fits preprocessor + model together)
    print("\nFitting final optimized pipeline on complete training set...")
    best_model.fit(X_train, y_train)

    # 6. Save final model & baseline models
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"Successfully serialized final best pipeline to: {best_model_path}")

    # Also save dictionary of models for evaluation scripts
    all_models = {
        'Logistic Regression': trained_base_models['Logistic Regression'],
        'Random Forest (Baseline)': trained_base_models['Random Forest'],
        'Tuned Random Forest': best_model
    }
    joblib.dump(all_models, os.path.join(MODELS_DIR, "all_models.pkl"))
    print("Training phase complete.")


if __name__ == '__main__':
    train_and_save()
