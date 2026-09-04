"""
Model Evaluation and Diagnostic Module for Customer Churn Prediction.
Evaluates model performance on the untouched test set, generates diagnostic visualizations
(Confusion Matrices, ROC Curves, Feature Importance, Model Comparison Table), and logs results.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)

from data_preprocessing import load_raw_data, clean_data, split_data


plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = os.path.join("reports", "figures")
REPORTS_DIR = "reports"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_models():
    # 1. Load test data
    if os.path.exists("data/processed/X_test.parquet") and os.path.exists("data/processed/y_test.parquet"):
        X_test = pd.read_parquet("data/processed/X_test.parquet")
        y_test = pd.read_parquet("data/processed/y_test.parquet")['Churn']
    else:
        df_raw = load_raw_data("data/customer_churn.csv")
        df_clean, X, y = clean_data(df_raw)
        _, X_test, _, y_test = split_data(X, y, test_size=0.20, random_state=42)

    # 2. Load trained models
    models_dict = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))

    print("=" * 70)
    print("STEP 11 & 15: MODEL EVALUATION ON UNTOUCHED TEST SET (N = 1,409)")
    print("=" * 70)

    test_results = []
    test_preds = {}
    test_probs = {}

    for name, pipeline in models_dict.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        test_preds[name] = y_pred
        test_probs[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\nModel: {name}")
        print("-" * 35)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churn (1)']))

        test_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc
        })

    results_df = pd.DataFrame(test_results)
    results_df.to_csv(os.path.join(REPORTS_DIR, "test_metrics_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 3. Generate Confusion Matrices Plot
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5.5 * len(models_dict), 4.5))
    if len(models_dict) == 1:
        axes = [axes]

    for idx, (name, _) in enumerate(models_dict.items()):
        cm = confusion_matrix(y_test, test_preds[name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Retained (0)', 'Churn (1)'],
                    yticklabels=['Retained (0)', 'Churn (1)'],
                    annot_kws={'size': 13, 'weight': 'bold'})
        axes[idx].set_title(f"{name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrices to: {cm_path}")

    # 4. Generate Comparative ROC Curves
    plt.figure(figsize=(8.5, 6.5))
    palette = ['#2b5c8f', '#2ca02c', '#d95f02', '#9467bd']

    for idx, (name, _) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, test_probs[name])
        auc = roc_auc_score(y_test, test_probs[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=palette[idx % len(palette)], lw=2.5)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Comparative ROC Curves (Test Set)", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to: {roc_path}")

    # 5. Feature Importance Analysis
    best_pipe = models_dict['Tuned Random Forest']
    preprocessor = best_pipe.named_steps['preprocessor']
    classifier = best_pipe.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()

    importances = classifier.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    feat_df.to_csv(os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 7))
    top15 = feat_df.head(15)
    sns.barplot(data=top15, x='Importance', y='Feature', palette='mako')
    plt.title("Top 15 Predictive Features (Tuned Random Forest)", fontsize=14, fontweight='bold')
    plt.xlabel("Gini Feature Importance", fontsize=12)
    plt.ylabel("Engineered Feature", fontsize=12)
    for p in plt.gca().patches:
        w = p.get_width()
        plt.gca().text(w + 0.002, p.get_y() + p.get_height()/2, f"{w:.3f}",
                       va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    fi_path = os.path.join(FIGURES_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to: {fi_path}")

    # 6. Logistic Regression Coefficients Analysis
    lr_pipe = models_dict['Logistic Regression']
    lr_clf = lr_pipe.named_steps['classifier']
    lr_coefs = lr_clf.coef_[0]
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr_coefs
    }).sort_values(by='Coefficient', ascending=False)

    top_pos = coef_df.head(7)
    top_neg = coef_df.tail(7)
    key_coefs = pd.concat([top_pos, top_neg]).sort_values(by='Coefficient')

    plt.figure(figsize=(10, 7))
    colors = ['#2ca02c' if c < 0 else '#d95f02' for c in key_coefs['Coefficient']]
    plt.barh(key_coefs['Feature'], key_coefs['Coefficient'], color=colors)
    plt.title("Key Logistic Regression Coefficients (Log-Odds Impact on Churn)", fontsize=13, fontweight='bold')
    plt.xlabel("Coefficient Value (Positive = Increases Churn, Negative = Reduces Churn)", fontsize=11)
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.tight_layout()
    lr_coef_path = os.path.join(FIGURES_DIR, "logistic_coefficients.png")
    plt.savefig(lr_coef_path, dpi=300)
    plt.close()
    print(f"Saved Logistic Regression coefficients to: {lr_coef_path}")

    return results_df


if __name__ == '__main__':
    evaluate_models()
