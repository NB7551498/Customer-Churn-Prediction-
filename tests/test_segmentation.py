"""
Unit tests for CustomerSegmenter (K-Means customer personas).
"""

import pandas as pd
import pytest
from src.segmentation import CustomerSegmenter


@pytest.fixture
def sample_customer_df():
    return pd.DataFrame({
        "tenure": [2, 60, 24, 5],
        "MonthlyCharges": [85.0, 95.0, 45.0, 20.0],
        "TotalCharges": [170.0, 5700.0, 1080.0, 100.0],
        "support_protection_index": [0, 4, 2, 0],
    })


def test_segmenter_fit_predict(sample_customer_df):
    """Verify segmenter trains and returns valid persona dictionary."""
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(sample_customer_df)

    assert segmenter.is_fitted is True

    # Predict single instance
    single_record = sample_customer_df.iloc[[0]]
    result = segmenter.predict_segment(single_record)

    assert "segment_id" in result
    assert "segment_name" in result
    assert "description" in result
    assert "risk_profile" in result
    assert "icon" in result
    assert isinstance(result["segment_id"], int)
    assert 0 <= result["segment_id"] <= 3


def test_segmenter_batch_assign(sample_customer_df):
    """Verify batch persona assignment attaches persona names to DataFrame."""
    segmenter = CustomerSegmenter(n_clusters=4, random_state=42)
    segmenter.fit(sample_customer_df)

    df_segmented = segmenter.assign_personas(sample_customer_df)
    assert "segment_id" in df_segmented.columns
    assert "persona" in df_segmented.columns
    assert len(df_segmented) == len(sample_customer_df)
