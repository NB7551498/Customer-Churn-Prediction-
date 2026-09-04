"""
Financial Threshold Optimization & Evaluation Module.
Evaluates model performance through a financial cost-benefit lens, determining
the decision threshold that maximizes net dollar value rather than naive accuracy.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("churn.evaluate")

# Financial Cost-Benefit Assumptions
VALUE_TRUE_POSITIVE: float = 550.0   # Net customer lifetime value saved from retention
COST_FALSE_POSITIVE: float = -50.0    # Cost of retention incentive offered to retained customer
LOSS_FALSE_NEGATIVE: float = -600.0  # Gross lost recurring revenue from undetected churner
VALUE_TRUE_NEGATIVE: float = 0.0     # Retained customer uncontacted (no intervention, no cost)


def compute_net_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    val_tp: float = VALUE_TRUE_POSITIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    loss_fn: float = LOSS_FALSE_NEGATIVE,
) -> Tuple[float, int, int, int, int]:
    """
    Compute net financial return for a specific decision threshold.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted churn probabilities in range [0, 1].
        threshold: Decision cutoff threshold.
        val_tp: Monetary gain for a True Positive.
        cost_fp: Monetary penalty (negative) for a False Positive.
        loss_fn: Monetary loss (negative) for a False Negative.

    Returns:
        Tuple of (net_profit, TP, FP, FN, TN).
    """
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Net Profit formula: TP * 550 + FP * (-50) + FN * (-600)
    net_profit: float = float(tp * val_tp + fp * cost_fp + fn * loss_fn)
    return net_profit, int(tp), int(fp), int(fn), int(tn)


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Iterate through probability thresholds from 0.10 to 0.90 to find the
    financially optimal cutoff maximizing business profit.

    Args:
        y_true: Actual ground truth binary labels.
        y_prob: Predicted churn probabilities.
        thresholds: Array of thresholds to evaluate (defaults to 0.1 to 0.9 in 0.01 steps).

    Returns:
        Dictionary containing optimization results, best threshold, and financial comparisons.
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)

    logger.info("Evaluating financial outcomes across %d candidate thresholds (0.10 to 0.90)...", len(thresholds))

    results: List[Dict[str, float]] = []
    best_profit: float = -float("inf")
    best_threshold: float = 0.50
    best_cm: Tuple[int, int, int, int] = (0, 0, 0, 0)

    for t in thresholds:
        profit, tp, fp, fn, tn = compute_net_profit(y_true, y_prob, float(t))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results.append({
            "threshold": float(t),
            "net_profit": profit,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": recall,
            "precision": precision,
        })

        if profit > best_profit:
            best_profit = profit
            best_threshold = float(t)
            best_cm = (tp, fp, fn, tn)

    # Baseline 0.50 evaluation for comparison
    default_profit, def_tp, def_fp, def_fn, def_tn = compute_net_profit(y_true, y_prob, 0.50)
    profit_delta = best_profit - default_profit

    logger.info("=" * 65)
    logger.info("FINANCIAL THRESHOLD OPTIMIZATION RESULTS")
    logger.info("=" * 65)
    logger.info("Default Threshold (0.50) Net Value: $%.2f (TP: %d, FP: %d, FN: %d)", default_profit, def_tp, def_fp, def_fn)
    logger.info("Optimal Threshold (%.2f) Net Value: $%.2f (TP: %d, FP: %d, FN: %d)", best_threshold, best_profit, best_cm[0], best_cm[1], best_cm[2])
    logger.info("Net Financial Gain from Tuning: +$%.2f", profit_delta)
    logger.info("=" * 65)

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


def plot_profit_curve(optimization_results: Dict[str, object], output_path: Path) -> None:
    """Plot and save financial profit vs threshold optimization curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve_data: List[Dict[str, float]] = optimization_results["threshold_curve"]  # type: ignore
    df_curve = pd.DataFrame(curve_data)

    plt.figure(figsize=(10, 6))
    plt.plot(df_curve["threshold"], df_curve["net_profit"], color="#1e3a8a", lw=2.5, label="Net Financial Value ($)")

    best_t: float = optimization_results["best_threshold"]  # type: ignore
    best_p: float = optimization_results["best_profit"]      # type: ignore
    def_p: float = optimization_results["default_profit"]    # type: ignore

    plt.axvline(best_t, color="#10b981", linestyle="--", lw=2, label=f"Optimal Cutoff (t = {best_t:.2f}, ${best_p:,.0f})")
    plt.axvline(0.50, color="#ef4444", linestyle=":", lw=2, label=f"Default Naive Cutoff (t = 0.50, ${def_p:,.0f})")

    plt.title("Business Value vs. Classification Probability Threshold", fontsize=13, fontweight="bold")
    plt.xlabel("Decision Probability Threshold", fontsize=11)
    plt.ylabel("Net Portfolio Return ($USD)", fontsize=11)
    plt.legend(loc="lower center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved financial threshold optimization chart to %s", output_path)


def run_evaluation(
    model_path: Path = Path("models/pipeline.joblib"),
    test_features_path: Path = Path("data/processed/X_test.parquet"),
    test_target_path: Path = Path("data/processed/y_test.parquet"),
) -> Dict[str, object]:
    """Load model and test set, compute ROC-AUC, and execute financial threshold optimization."""
    logger.info("Loading model from %s", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    pipeline: Pipeline = joblib.load(model_path)
    X_test: pd.DataFrame = pd.read_parquet(test_features_path)
    y_test: np.ndarray = pd.read_parquet(test_target_path)["Churn"].values

    y_prob: np.ndarray = pipeline.predict_proba(X_test)[:, 1]
    auc_score: float = float(roc_auc_score(y_test, y_prob))
    logger.info("Evaluated Test Set (N = %d) | ROC-AUC: %.4f", len(y_test), auc_score)

    opt_results = optimize_threshold(y_test, y_prob)
    opt_results["test_roc_auc"] = auc_score

    # Save threshold configuration for serving layer
    config_path = Path("models/optimal_threshold.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": opt_results["best_threshold"],
            "optimal_profit": opt_results["best_profit"],
            "default_profit": opt_results["default_profit"],
            "profit_gain": opt_results["profit_delta"],
            "roc_auc": auc_score,
        }, f, indent=2)
    logger.info("Saved optimal threshold configuration to %s", config_path)

    plot_profit_curve(opt_results, Path("reports/figures/financial_threshold_curve.png"))
    return opt_results


if __name__ == "__main__":
    run_evaluation()
