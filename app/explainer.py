"""
SHAP Explainability Engine for Customer Churn Prediction.

Wraps shap.TreeExplainer around the fitted classifier extracted from
the sklearn Pipeline. Feature names are recovered from the preprocessor's
get_feature_names_out() so that SHAP values are labelled meaningfully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as _shap_err:  # covers ImportError and any numba/llvm issues
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    logging.getLogger("churn.explainer").warning(
        "SHAP not available (%s). /explain endpoint will return 503.", _shap_err
    )

logger = logging.getLogger("churn.explainer")

# How many features to return in the explanation (top by |SHAP value|)
TOP_N_FEATURES = 10


class ChurnExplainer:
    """
    Stateful SHAP explainer that is initialised once at app startup and
    reused for every /explain request.
    """

    def __init__(self, pipeline: Any) -> None:
        if not SHAP_AVAILABLE:
            raise RuntimeError("SHAP is not available. Install shap and numba.")
        self._pipeline = pipeline
        self._preprocessor = pipeline.named_steps["preprocessor"]
        self._classifier = pipeline.named_steps["classifier"]
        self._feature_names: List[str] = list(
            self._preprocessor.get_feature_names_out()
        )
        # TreeExplainer is fast and exact for XGBoost / Random Forest
        self._explainer = shap.TreeExplainer(self._classifier)
        logger.info(
            "ChurnExplainer initialised — %d features, model type: %s",
            len(self._feature_names),
            type(self._classifier).__name__,
        )

    def explain(
        self,
        input_df: pd.DataFrame,
        top_n: int = TOP_N_FEATURES,
    ) -> List[Dict[str, Any]]:
        """
        Compute SHAP values for a single input row.

        Args:
            input_df: One-row DataFrame with raw (pre-transform) features,
                      exactly as received from the API payload.
            top_n:    Maximum number of features to return, ranked by
                      absolute SHAP impact.

        Returns:
            List of dicts: [{"feature": str, "impact": float}, ...]
            sorted by descending absolute impact.
        """
        # Transform raw input through the fitted preprocessor
        X_transformed: np.ndarray = self._preprocessor.transform(input_df)

        # Compute SHAP values — shape: (1, n_features) for binary class 1
        shap_values = self._explainer.shap_values(X_transformed)

        # shap_values may be a list (one array per class) or a single ndarray
        if isinstance(shap_values, list):
            # Index 1 = probability of churn (class 1)
            row_shap: np.ndarray = shap_values[1][0]
        else:
            row_shap = shap_values[0]

        # Build ranked list
        impacts = [
            {"feature": name, "impact": float(value)}
            for name, value in zip(self._feature_names, row_shap)
        ]
        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return impacts[:top_n]


# Module-level singleton — populated by main.py at startup
_explainer_instance: Optional[ChurnExplainer] = None


def init_explainer(pipeline: Any) -> None:
    """Initialise the module-level explainer singleton."""
    global _explainer_instance
    _explainer_instance = ChurnExplainer(pipeline)


def get_explainer() -> Optional[ChurnExplainer]:
    """Return the initialised explainer, or None if not ready."""
    return _explainer_instance
