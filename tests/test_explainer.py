"""
Unit tests for ChurnExplainer in isolation.
Uses a minimal mock pipeline so tests pass without loading the real model.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_pipeline():
    """Build a minimal mock sklearn Pipeline with preprocessor + classifier."""
    # Preprocessor mock
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.zeros((1, 5))
    preprocessor.get_feature_names_out.return_value = [
        "tenure", "MonthlyCharges", "TotalCharges", "Contract_Two year", "InternetService_Fiber optic"
    ]

    # Classifier mock
    classifier = MagicMock()

    # shap_values returns list of arrays (binary classification)
    shap_vals = [np.zeros((1, 5)), np.array([[0.1, -0.2, 0.05, -0.3, 0.15]])]

    pipeline = MagicMock()
    pipeline.named_steps = {
        "preprocessor": preprocessor,
        "classifier": classifier,
    }

    return pipeline, shap_vals


def test_explainer_initialises(mock_pipeline):
    """ChurnExplainer can be instantiated from a mock pipeline."""
    pipeline, shap_vals = mock_pipeline

    with patch("shap.TreeExplainer") as MockExplainer:
        mock_exp = MagicMock()
        mock_exp.shap_values.return_value = shap_vals
        MockExplainer.return_value = mock_exp

        from app.explainer import ChurnExplainer
        explainer = ChurnExplainer(pipeline)

        assert explainer._feature_names == [
            "tenure", "MonthlyCharges", "TotalCharges",
            "Contract_Two year", "InternetService_Fiber optic",
        ]


def test_explain_returns_sorted_by_abs_impact(mock_pipeline):
    """explain() returns features sorted by absolute SHAP value descending."""
    pipeline, shap_vals = mock_pipeline

    with patch("shap.TreeExplainer") as MockExplainer:
        mock_exp = MagicMock()
        mock_exp.shap_values.return_value = shap_vals
        MockExplainer.return_value = mock_exp

        from app.explainer import ChurnExplainer
        explainer = ChurnExplainer(pipeline)

        input_df = pd.DataFrame([{"col": 1}])  # content doesn't matter with mock
        result = explainer.explain(input_df, top_n=5)

        # Verify sorted by |impact| descending
        abs_impacts = [abs(r["impact"]) for r in result]
        assert abs_impacts == sorted(abs_impacts, reverse=True)


def test_explain_returns_feature_and_impact_keys(mock_pipeline):
    """Each item in explain() result has 'feature' and 'impact' keys."""
    pipeline, shap_vals = mock_pipeline

    with patch("shap.TreeExplainer") as MockExplainer:
        mock_exp = MagicMock()
        mock_exp.shap_values.return_value = shap_vals
        MockExplainer.return_value = mock_exp

        from app.explainer import ChurnExplainer
        explainer = ChurnExplainer(pipeline)

        input_df = pd.DataFrame([{"col": 1}])
        result = explainer.explain(input_df)

        for item in result:
            assert "feature" in item
            assert "impact" in item
            assert isinstance(item["feature"], str)
            assert isinstance(item["impact"], float)


def test_explain_top_n_limits_results(mock_pipeline):
    """top_n parameter correctly limits the returned list length."""
    pipeline, shap_vals = mock_pipeline

    with patch("shap.TreeExplainer") as MockExplainer:
        mock_exp = MagicMock()
        mock_exp.shap_values.return_value = shap_vals
        MockExplainer.return_value = mock_exp

        from app.explainer import ChurnExplainer
        explainer = ChurnExplainer(pipeline)

        input_df = pd.DataFrame([{"col": 1}])
        result = explainer.explain(input_df, top_n=3)
        assert len(result) <= 3


def test_explain_handles_single_array_shap_output(mock_pipeline):
    """explain() works when shap_values returns a single array (not a list)."""
    pipeline, _ = mock_pipeline
    single_array_shap = np.array([[0.1, -0.2, 0.05, -0.3, 0.15]])

    with patch("shap.TreeExplainer") as MockExplainer:
        mock_exp = MagicMock()
        mock_exp.shap_values.return_value = single_array_shap  # not a list
        MockExplainer.return_value = mock_exp

        from app.explainer import ChurnExplainer
        explainer = ChurnExplainer(pipeline)

        input_df = pd.DataFrame([{"col": 1}])
        result = explainer.explain(input_df)

        assert len(result) > 0
        assert all("feature" in r and "impact" in r for r in result)
