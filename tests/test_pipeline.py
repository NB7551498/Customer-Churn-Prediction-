"""
Unit tests for data preprocessing and pipeline logic.
"""

from pathlib import Path
import numpy as np
from src.train import (
    load_and_preprocess_raw_data,
    build_pipeline,
)


def test_data_loading_and_cleaning():
    """Verify raw data loader cleans TotalCharges and extracts features properly."""
    data_path = Path("data/customer_churn.csv")
    X, y = load_and_preprocess_raw_data(data_path)

    assert len(X) == 7043
    assert len(y) == 7043
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert X["TotalCharges"].dtype in [np.float64, float]
    assert not X["TotalCharges"].isna().any()


def test_pipeline_assembly():
    """Verify pipeline contains ColumnTransformer and classifier."""
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
