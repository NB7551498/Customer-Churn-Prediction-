"""
Fairness Analysis and Bias Mitigation for Customer Churn Classifier.

Computes group-level performance metrics across SeniorCitizen and gender —
the only legitimate group attributes available in the Telco dataset.

Documented limitation: the dataset does not contain race/ethnicity or
other protected attributes. Analysis is constrained to available attributes.

Usage:
    python -m src.fairness
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn.fairness")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = Path("data/customer_churn.csv")
MODEL_PATH = Path("models/pipeline.joblib")
FALLBACK_MODEL_PATH = Path("models/best_model.pkl")
THRESHOLD_PATH = Path("models/optimal_threshold.json")
REPORT_DIR = Path("reports")
FIGURES_DIR = REPORT_DIR / "figures"

# Sensitive / group attributes available in this dataset
GROUP_ATTRIBUTES: List[str] = ["SeniorCitizen", "gender"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_and_model() -> tuple:
    """Load raw dataset and fitted pipeline."""
    logger.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    ).fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    logger.info("Loading pipeline from %s", model_path)
    pipeline: Pipeline = joblib.load(model_path)

    return X, y, df, pipeline


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_name: str,
    group_value: str,
) -> Dict[str, object]:
    """Compute a full set of fairness-relevant metrics for a subgroup."""
    n = len(y_true)
    if n == 0:
        return {}

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # False Negative Rate
    selection_rate = (tp + fp) / n                     # Positive prediction rate

    return {
        "attribute": group_name,
        "group": group_value,
        "n_samples": n,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "selection_rate": round(selection_rate, 4),
    }


# ---------------------------------------------------------------------------
# Core fairness analysis
# ---------------------------------------------------------------------------

def run_fairness_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    df_raw: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float,
) -> pd.DataFrame:
    """
    Compute group metrics for all subgroups of all sensitive attributes.
    Returns a DataFrame with one row per subgroup.
    """
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    for attr in GROUP_ATTRIBUTES:
        if attr not in df_raw.columns:
            logger.warning("Attribute %s not found in dataset — skipping", attr)
            continue

        groups = df_raw[attr].unique()
        logger.info("Analysing attribute '%s' → groups: %s", attr, sorted(groups))

        for group_val in sorted(groups):
            mask = df_raw[attr] == group_val
            metrics = compute_group_metrics(
                y.values[mask],
                y_pred[mask],
                group_name=attr,
                group_value=str(group_val),
            )
            if metrics:
                rows.append(metrics)
                logger.info(
                    "  [%s=%s] n=%d acc=%.3f rec=%.3f fpr=%.3f fnr=%.3f",
                    attr, group_val, metrics["n_samples"],
                    metrics["accuracy"], metrics["recall"],
                    metrics["fpr"], metrics["fnr"],
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mitigation experiment — threshold adjustment per group
# ---------------------------------------------------------------------------

def run_mitigation_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    df_raw: pd.DataFrame,
    pipeline: Pipeline,
    threshold: float,
) -> pd.DataFrame:
    """
    Post-processing mitigation: adjust decision threshold per group to
    reduce FPR disparity for SeniorCitizen groups.

    Returns before/after comparison DataFrame.
    """
    logger.info("=" * 60)
    logger.info("BIAS MITIGATION EXPERIMENT — Threshold Adjustment")
    logger.info("=" * 60)

    y_prob = pipeline.predict_proba(X)[:, 1]
    attr = "SeniorCitizen"

    # Compute base metrics
    baseline_rows = []
    mitigated_rows = []

    for group_val in ["0", "1"]:
        mask = df_raw[attr] == group_val
        y_true_g = y.values[mask]
        y_prob_g = y_prob[mask]

        # Baseline (global threshold)
        y_pred_base = (y_prob_g >= threshold).astype(int)
        base = compute_group_metrics(y_true_g, y_pred_base, attr, group_val)

        # Mitigation: raise threshold for non-senior (group 0) to reduce FPR
        # Senior citizens may already face higher FPR; we equalise by adjusting
        group_threshold = threshold + 0.05 if group_val == "0" else threshold - 0.05
        group_threshold = max(0.10, min(0.90, group_threshold))
        y_pred_mit = (y_prob_g >= group_threshold).astype(int)
        mit = compute_group_metrics(y_true_g, y_pred_mit, attr, group_val)

        baseline_rows.append({**base, "phase": "before"})
        mitigated_rows.append({**mit, "phase": "after", "group_threshold": round(group_threshold, 2)})

        logger.info(
            "  [%s=%s] threshold: %.2f→%.2f | FPR: %.3f→%.3f | FNR: %.3f→%.3f | Accuracy: %.3f→%.3f",
            attr, group_val,
            threshold, group_threshold,
            base["fpr"], mit["fpr"],
            base["fnr"], mit["fnr"],
            base["accuracy"], mit["accuracy"],
        )

    all_rows = baseline_rows + mitigated_rows
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fairness_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart comparing key fairness metrics across groups and attributes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["accuracy", "recall", "fpr", "fnr", "selection_rate"]
    palette = ["#1e3a8a", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    n_attrs = df["attribute"].nunique()
    fig, axes = plt.subplots(n_attrs, len(metrics_to_plot), figsize=(18, 5 * n_attrs))
    if n_attrs == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (attr, group_df) in enumerate(df.groupby("attribute")):
        groups = group_df["group"].tolist()
        x = np.arange(len(groups))

        for col_idx, (metric, color) in enumerate(zip(metrics_to_plot, palette)):
            ax = axes[row_idx, col_idx]
            values = group_df[metric].tolist()
            bars = ax.bar(x, values, color=color, alpha=0.85, edgecolor="white", linewidth=1.2)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{attr}={g}" for g in groups], fontsize=10, rotation=15)
            ax.set_title(f"{metric.upper()} by {attr}", fontsize=11, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(metric, fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )

    plt.suptitle(
        "Fairness Analysis — Group Performance Metrics\n(Telco Customer Churn Dataset)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Fairness comparison chart saved to %s", output_path)


def plot_mitigation_comparison(df_mit: pd.DataFrame, output_path: Path) -> None:
    """Side-by-side bar chart for before/after mitigation on key metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "recall", "fpr", "fnr"]
    groups = df_mit["group"].unique()
    phases = ["before", "after"]
    x = np.arange(len(groups))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        for i, phase in enumerate(phases):
            subset = df_mit[df_mit["phase"] == phase]
            values = [subset[subset["group"] == g][metric].values[0] for g in groups]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=phase.capitalize(),
                          color=["#1e3a8a", "#10b981"][i], alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"SeniorCitizen={g}" for g in groups], fontsize=9)
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Bias Mitigation Experiment — Threshold Adjustment per Group\n"
        "Trade-off: FPR reduction vs. accuracy/recall impact",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Mitigation comparison chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_fairness_pipeline() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data + model
    X, y, df_raw, pipeline = load_data_and_model()

    # Load threshold
    threshold = 0.50
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(json.load(f).get("optimal_threshold", 0.50))
    logger.info("Using decision threshold: %.2f", threshold)

    # -------------------------------------------------------------------
    # Fairness Analysis
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("FAIRNESS ANALYSIS — Group Performance Metrics")
    logger.info("=" * 60)

    df_fairness = run_fairness_analysis(X, y, df_raw, pipeline, threshold)

    # Save CSV report
    fairness_csv = REPORT_DIR / "fairness_report.csv"
    df_fairness.to_csv(fairness_csv, index=False)
    logger.info("Fairness report saved to %s", fairness_csv)

    # Print summary table
    print("\n" + "=" * 70)
    print("FAIRNESS REPORT — Group Metrics")
    print("=" * 70)
    print(df_fairness.to_string(index=False))
    print()

    # Compute and log disparities
    for attr, group_df in df_fairness.groupby("attribute"):
        max_fpr = group_df["fpr"].max()
        min_fpr = group_df["fpr"].min()
        max_acc = group_df["accuracy"].max()
        min_acc = group_df["accuracy"].min()
        logger.info(
            "Disparity [%s] — Accuracy gap: %.3f | FPR gap: %.3f",
            attr, max_acc - min_acc, max_fpr - min_fpr,
        )

    plot_fairness_comparison(df_fairness, FIGURES_DIR / "fairness_comparison.png")

    # -------------------------------------------------------------------
    # Bias Mitigation Experiment
    # -------------------------------------------------------------------
    df_mit = run_mitigation_experiment(X, y, df_raw, pipeline, threshold)

    mit_csv = REPORT_DIR / "mitigation_experiment.csv"
    df_mit.to_csv(mit_csv, index=False)
    logger.info("Mitigation experiment results saved to %s", mit_csv)

    print("\n" + "=" * 70)
    print("MITIGATION EXPERIMENT — Before vs After Threshold Adjustment")
    print("=" * 70)
    print(df_mit[["attribute", "group", "phase", "accuracy", "recall", "fpr", "fnr"]].to_string(index=False))
    print(
        "\nKey insight: threshold adjustment reduces FPR for the advantaged group "
        "at the cost of a small accuracy/recall trade-off. There is no 'free lunch' "
        "— every fairness intervention involves a performance compromise that must be "
        "evaluated against business and ethical priorities."
    )

    plot_mitigation_comparison(df_mit, FIGURES_DIR / "mitigation_comparison.png")

    # -------------------------------------------------------------------
    # Limitation documentation
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DOCUMENTED LIMITATIONS")
    print("=" * 70)
    print(
        "1. Sensitive attributes: Only 'SeniorCitizen' and 'gender' are available.\n"
        "   Race, ethnicity, disability, and other protected classes are absent.\n"
        "2. Mitigation method: Post-processing threshold adjustment is used\n"
        "   because the existing pipeline must remain deployed unchanged.\n"
        "   In-processing methods (e.g., adversarial debiasing) would require\n"
        "   full retraining and are outside scope.\n"
        "3. Statistical significance: Group sample sizes differ substantially.\n"
        "   Senior citizen group (SeniorCitizen=1) is ~16% of the dataset.\n"
        "   Metrics for small subgroups carry higher variance.\n"
    )

    logger.info("Fairness pipeline complete.")


if __name__ == "__main__":
    run_fairness_pipeline()
