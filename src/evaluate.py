"""
ChurnGuard AI — Financial Threshold Optimization & Comprehensive Evaluation Suite.
Evaluates model performance through both statistical metrics (Accuracy, Precision,
Recall, F1, ROC-AUC, PR-AUC, Brier Calibration Score) and an executive financial lens,
determining the decision cutoff that maximizes net portfolio return.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("churn.evaluate")

FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")

# Financial Cost-Benefit Assumptions
VALUE_TRUE_POSITIVE: float = 550.0   # Net customer lifetime value saved from successful retention
COST_FALSE_POSITIVE: float = -50.0   # Cost of proactive retention voucher offered to retained user
LOSS_FALSE_NEGATIVE: float = -600.0  # Gross lost annual recurring revenue from undetected churner
VALUE_TRUE_NEGATIVE: float = 0.0     # Retained customer uncontacted (no intervention, zero cost)


def compute_net_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    val_tp: float = VALUE_TRUE_POSITIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    loss_fn: float = LOSS_FALSE_NEGATIVE,
) -> Tuple[float, int, int, int, int]:
    """
    Compute net financial return for a specific decision threshold:
    Net Profit = (TP * $550) + (FP * -$50) + (FN * -$600) + (TN * $0)
    """
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    net_profit: float = float(tp * val_tp + fp * cost_fp + fn * loss_fn)
    return net_profit, int(tp), int(fp), int(fn), int(tn)


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Iterate through probability thresholds from 0.05 to 0.95 to identify
    the decision cutoff that maximizes financial portfolio value.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    logger.info("Evaluating financial outcomes across %d candidate cutoffs...", len(thresholds))

    results: List[Dict[str, float]] = []
    best_profit: float = -float("inf")
    best_threshold: float = 0.50
    best_cm: Tuple[int, int, int, int] = (0, 0, 0, 0)

    for t in thresholds:
        profit, tp, fp, fn, tn = compute_net_profit(y_true, y_prob, float(t))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": float(t),
            "net_profit": profit,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
        })

        if profit > best_profit:
            best_profit = profit
            best_threshold = float(t)
            best_cm = (tp, fp, fn, tn)

    # Baseline 0.50 cutoff comparison
    default_profit, def_tp, def_fp, def_fn, def_tn = compute_net_profit(y_true, y_prob, 0.50)
    profit_delta = best_profit - default_profit

    logger.info("=" * 68)
    logger.info("FINANCIAL THRESHOLD OPTIMIZATION REPORT")
    logger.info("=" * 68)
    logger.info("Naive Cutoff (0.50) Net Value:   $%s (TP: %d, FP: %d, FN: %d)", f"{default_profit:,.2f}", def_tp, def_fp, def_fn)
    logger.info("Optimal Cutoff (%.2f) Net Value: $%s (TP: %d, FP: %d, FN: %d)", best_threshold, f"{best_profit:,.2f}", best_cm[0], best_cm[1], best_cm[2])
    logger.info("Net Financial Gain from Tuning: +$%s", f"{profit_delta:,.2f}")
    logger.info("=" * 68)

    return {
        "best_threshold": best_threshold,
        "best_profit": best_profit,
        "best_tp": best_cm[0],
        "best_fp": best_cm[1],
        "best_fn": best_cm[2],
        "best_tn": best_cm[3],
        "default_threshold": 0.50,
        "default_profit": default_profit,
        "profit_delta": profit_delta,
        "threshold_curve": results,
    }


def compute_comprehensive_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Calculate the complete suite of statistical classification and probability metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    opt_results: Dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> None:
    """Generate and save publication-grade visual artifacts across all evaluation dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_t = float(opt_results["best_threshold"])

    # 1. Financial Profit vs Threshold Curve
    plt.figure(figsize=(9, 5))
    df_curve = pd.DataFrame(opt_results["threshold_curve"])
    plt.plot(df_curve["threshold"], df_curve["net_profit"], color="#1e3a8a", lw=2.5, label="Net Portfolio Value ($)")
    plt.axvline(best_t, color="#10b981", linestyle="--", lw=2, label=f"Optimal Cutoff (t={best_t:.2f}, ${opt_results['best_profit']:,.0f})")
    plt.axvline(0.50, color="#ef4444", linestyle=":", lw=2, label=f"Default Cutoff (t=0.50, ${opt_results['default_profit']:,.0f})")
    plt.title("Business Net Value vs. Classification Threshold", fontsize=12, fontweight="bold")
    plt.xlabel("Probability Threshold", fontsize=10)
    plt.ylabel("Portfolio Return ($USD)", fontsize=10)
    plt.legend(loc="lower center")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "financial_threshold_curve.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Model ROC (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", lw=1.5, label="Random Guessing (AUC = 0.500)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    baseline_churn_rate = float(np.mean(y_true))
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, color="#7c3aed", lw=2.5, label=f"Model PR Curve (PR-AUC = {pr_auc_val:.3f})")
    plt.axhline(baseline_churn_rate, color="#9ca3af", linestyle="--", lw=1.5, label=f"Baseline Churn Rate ({baseline_churn_rate:.1%})")
    plt.title("Precision-Recall Curve (PR-AUC)", fontsize=12, fontweight="bold")
    plt.xlabel("Recall", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="#059669", lw=2, label=f"Calibrated Model (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Perfectly Calibrated")
    plt.title("Reliability Diagram / Calibration Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=10)
    plt.ylabel("Fraction of True Churners", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=300)
    plt.close()

    # 5. Confusion Matrix at Optimal Threshold
    y_pred_opt = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Optimal Cutoff t = {best_t:.2f})", fontsize=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Retained (0)", "Churned (1)"])
    plt.yticks(tick_marks, ["Retained (0)", "Churned (1)"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            plt.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.ylabel("True Class", fontsize=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    logger.info("Saved 5 publication-ready diagnostic charts to %s", output_dir)


def run_evaluation(
    model_path: Path = Path("models/pipeline.joblib"),
    test_features_path: Path = Path("data/processed/X_test.parquet"),
    test_target_path: Path = Path("data/processed/y_test.parquet"),
) -> Dict[str, Any]:
    """Execute end-to-end evaluation, financial optimization, plot generation, and metric exports."""
    logger.info("Loading model from %s", model_path)
    if not model_path.exists():
        fallback = Path("models/best_model.pkl")
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline: Pipeline = joblib.load(model_path)
    X_test: pd.DataFrame = pd.read_parquet(test_features_path)
    y_test: np.ndarray = pd.read_parquet(test_target_path)["Churn"].values

    y_prob: np.ndarray = pipeline.predict_proba(X_test)[:, 1]

    # Compute metrics at default (0.50) and optimal thresholds
    metrics_default = compute_comprehensive_metrics(y_test, y_prob, threshold=0.50)
    opt_results = optimize_threshold(y_test, y_prob)
    metrics_optimal = compute_comprehensive_metrics(y_test, y_prob, threshold=float(opt_results["best_threshold"]))

    evaluation_report = {
        "test_sample_size": len(y_test),
        "actual_churners": int(np.sum(y_test)),
        "baseline_churn_rate": round(float(np.mean(y_test)), 4),
        "metrics_at_default_threshold_0_50": metrics_default,
        "metrics_at_optimal_threshold": metrics_optimal,
        "financial_optimization": {
            "optimal_threshold": opt_results["best_threshold"],
            "optimal_net_profit_usd": opt_results["best_profit"],
            "default_net_profit_usd": opt_results["default_profit"],
            "net_gain_from_threshold_tuning_usd": opt_results["profit_delta"],
            "optimal_confusion_matrix": {
                "true_positives": opt_results["best_tp"],
                "false_positives": opt_results["best_fp"],
                "false_negatives": opt_results["best_fn"],
                "true_negatives": opt_results["best_tn"],
            },
        },
    }

    # Save threshold configuration for serving layer
    config_path = Path("models/optimal_threshold.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": opt_results["best_threshold"],
            "optimal_profit": opt_results["best_profit"],
            "default_profit": opt_results["default_profit"],
            "profit_gain": opt_results["profit_delta"],
            "roc_auc": metrics_default["roc_auc"],
            "pr_auc": metrics_default["pr_auc"],
            "brier_score": metrics_default["brier_score"],
        }, f, indent=2)

    # Save comprehensive evaluation scorecard
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    # Generate all visual charts
    generate_evaluation_plots(y_test, y_prob, opt_results)

    logger.info("Evaluation complete! Metrics written to %s and models/optimal_threshold.json", REPORTS_DIR / "evaluation_metrics.json")
    return evaluation_report


if __name__ == "__main__":
    run_evaluation()
