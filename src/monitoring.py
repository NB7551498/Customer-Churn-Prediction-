"""
ChurnGuard AI — Data & Model Drift Monitoring System.
Implements Population Stability Index (PSI), Kolmogorov-Smirnov (KS) two-sample
tests for numerical drift, Chi-square/frequency tests for categorical shifts,
and automated retraining alert generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("churn.monitoring")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current samples.

    Threshold interpretations:
    - PSI < 0.10: No significant drift (Baseline stable)
    - 0.10 <= PSI < 0.25: Moderate drift (Investigate feature)
    - PSI >= 0.25: Significant distribution shift (Trigger retraining alert)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    raw_bins = np.percentile(expected, percentiles)
    bins = np.unique(raw_bins)

    if len(bins) < 2:
        return 0.0

    # Ensure edge coverage
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = np.maximum(expected_counts / len(expected), epsilon)
    actual_pct = np.maximum(actual_counts / len(actual), epsilon)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))


def detect_numerical_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    p_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Perform Kolmogorov-Smirnov two-sample test on continuous numerical data."""
    ks_stat, p_val = stats.ks_2samp(baseline, current)
    psi = calculate_psi(baseline, current)
    drift_detected = p_val < p_threshold or psi >= 0.25
    return {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": float(p_val),
        "psi": psi,
        "drift_detected": bool(drift_detected),
    }


def detect_categorical_drift(
    baseline: pd.Series,
    current: pd.Series,
    shift_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Evaluate categorical distribution shift between baseline and production batches."""
    ref_dist = baseline.value_counts(normalize=True).to_dict()
    cur_dist = current.value_counts(normalize=True).to_dict()
    all_keys = set(ref_dist.keys()) | set(cur_dist.keys())
    max_shift = max(abs(ref_dist.get(k, 0.0) - cur_dist.get(k, 0.0)) for k in all_keys) if all_keys else 0.0
    return {
        "max_class_shift": round(float(max_shift), 4),
        "drift_detected": bool(max_shift > shift_threshold),
    }


class DataDriftDetector:
    """Evaluates real-time and batch production data against baseline reference."""

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        reference_df: Optional[pd.DataFrame] = None,
        reference_path: Path = Path("data/customer_churn.csv"),
    ):
        ref = reference_data if reference_data is not None else reference_df
        if ref is not None:
            self.reference_df = ref.copy()
        elif reference_path.exists():
            self.reference_df = pd.read_csv(reference_path)
        else:
            self.reference_df = pd.DataFrame()

        self._preprocess_reference()

    def _preprocess_reference(self) -> None:
        if self.reference_df.empty:
            return
        if "TotalCharges" in self.reference_df.columns:
            self.reference_df["TotalCharges"] = pd.to_numeric(
                self.reference_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)

    def evaluate_drift(
        self,
        current_df: pd.DataFrame,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full statistical drift between reference and production batch.
        Returns overall status and feature-level metrics dict.
        """
        if self.reference_df.empty:
            return {
                "overall_status": "UNAVAILABLE",
                "retraining_recommended": False,
                "message": "Reference baseline dataset is missing.",
                "feature_metrics": {},
            }

        num_cols = numerical_features or [
            c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in self.reference_df.columns and c in current_df.columns
        ]
        cat_cols = categorical_features or [
            c for c in ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"]
            if c in self.reference_df.columns and c in current_df.columns
        ]

        feature_metrics: Dict[str, Dict[str, Any]] = {}
        drifted_features: List[str] = []
        max_psi: float = 0.0

        # 1. Numerical Drift
        for col in num_cols:
            ref_vals = pd.to_numeric(self.reference_df[col], errors="coerce").dropna().values
            cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna().values
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                res = detect_numerical_drift(ref_vals, cur_vals)
                max_psi = max(max_psi, res["psi"])
                if res["drift_detected"]:
                    drifted_features.append(col)
                feature_metrics[col] = {
                    "type": "numerical",
                    "psi": res["psi"],
                    "ks_statistic": res["ks_statistic"],
                    "p_value": res["p_value"],
                    "drift_detected": res["drift_detected"],
                    "status": "CRITICAL" if res["psi"] >= 0.25 else ("MODERATE" if res["psi"] >= 0.10 else "STABLE"),
                }

        # 2. Categorical Drift
        for col in cat_cols:
            res = detect_categorical_drift(self.reference_df[col].dropna(), current_df[col].dropna())
            if res["drift_detected"]:
                drifted_features.append(col)
            feature_metrics[col] = {
                "type": "categorical",
                "max_class_shift": res["max_class_shift"],
                "drift_detected": res["drift_detected"],
                "status": "HIGH" if res["drift_detected"] else "STABLE",
            }

        # 3. Overall Verdict & Retraining Alert
        if len(drifted_features) >= 2 or max_psi >= 0.25:
            overall_status = "CRITICAL_DRIFT"
            alert_message = f"Significant data drift detected on {len(drifted_features)} features ({', '.join(drifted_features)}). Retraining recommended."
            retraining_recommended = True
        elif len(drifted_features) == 1 or max_psi >= 0.10:
            overall_status = "MODERATE_DRIFT"
            alert_message = f"Moderate distribution shift detected on {', '.join(drifted_features)}. Continue monitoring."
            retraining_recommended = False
        else:
            overall_status = "STABLE"
            alert_message = "All monitored features within normal statistical variance."
            retraining_recommended = False

        missing_rate = float(current_df.isnull().sum().sum() / max(1, current_df.size))

        return {
            "overall_status": overall_status,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "retraining_recommended": retraining_recommended,
            "missing_value_rate_pct": round(missing_rate * 100, 2),
            "alert_message": alert_message,
            "feature_metrics": feature_metrics,
            "features_evaluated": [
                {"feature": k, **v} for k, v in feature_metrics.items()
            ],
        }

    def evaluate_batch_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Convenience alias for evaluate_drift."""
        return self.evaluate_drift(current_df)


_global_monitor: Optional[DataDriftDetector] = None


def get_global_monitor(reference_path: Path = Path("data/customer_churn.csv")) -> DataDriftDetector:
    """Singleton getter for the global DataDriftDetector."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = DataDriftDetector(reference_path=reference_path)
    return _global_monitor
