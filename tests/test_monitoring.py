"""
Unit tests for ChurnGuard AI Data & Model Monitoring Module (PSI, KS-Test, Retraining Alerts).
"""

import numpy as np
import pandas as pd
from src.monitoring import DataDriftDetector, calculate_psi, detect_numerical_drift


def test_psi_identical_distributions():
    """Identical distributions should have PSI ~ 0.0 (< 0.05)."""
    np.random.seed(42)
    baseline = np.random.normal(50, 10, 1000)
    current = baseline.copy()

    psi = calculate_psi(baseline, current, num_buckets=10)
    assert psi >= 0.0
    assert psi < 0.05, f"Expected near-zero PSI, got {psi}"


def test_psi_moderate_drift():
    """Slightly shifted distributions should produce non-zero PSI."""
    np.random.seed(42)
    baseline = np.random.normal(50, 10, 1000)
    current = np.random.normal(55, 10, 1000)

    psi = calculate_psi(baseline, current, num_buckets=10)
    assert psi > 0.02


def test_psi_severe_drift():
    """Heavily shifted distribution should produce PSI >= 0.20."""
    np.random.seed(42)
    baseline = np.random.normal(20, 5, 1000)
    current = np.random.normal(80, 5, 1000)

    psi = calculate_psi(baseline, current, num_buckets=10)
    assert psi >= 0.20


def test_numerical_drift_ks_test():
    """KS-test detects statistical distribution divergence."""
    np.random.seed(42)
    base = np.random.uniform(10, 100, 500)
    curr_same = np.random.uniform(10, 100, 500)
    curr_drifted = np.random.uniform(70, 150, 500)

    res_same = detect_numerical_drift(base, curr_same)
    assert res_same["drift_detected"] is False

    res_drift = detect_numerical_drift(base, curr_drifted)
    assert res_drift["drift_detected"] is True
    assert res_drift["p_value"] < 0.05


def test_data_drift_detector_evaluation():
    """DataDriftDetector runs complete drift suite on DataFrame inputs."""
    np.random.seed(42)
    n = 300
    ref_df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
    })

    # Simulating a current dataset with strong tenure drop (lots of new customers)
    curr_df = pd.DataFrame({
        "tenure": np.random.randint(1, 10, n),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
    })

    detector = DataDriftDetector(reference_df=ref_df)
    report = detector.evaluate_drift(curr_df)

    assert "overall_status" in report
    assert "retraining_recommended" in report
    assert "feature_metrics" in report
    assert "tenure" in report["feature_metrics"]
    assert report["feature_metrics"]["tenure"]["drift_detected"] is True
